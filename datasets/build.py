from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import utils
from .tsv import TSVDataset
from .tsv_openimage import TSVOpenImageDataset

from .samplers import DistributedChunkSampler
from .comm import comm
from pathlib import Path

def build_dataloader(args, is_train=True):

    if args.aug_opt == 'deit_aug':
        transform = DataAugmentationDEIT(args)

    elif args.aug_opt == 'dino_aug':
        transform = DataAugmentationDINO(
            args.global_crops_scale,
            args.local_crops_scale,
            args.local_crops_number,
            args.local_crops_size,
        )

    if 'imagenet1k' in args.dataset:
        if args.zip_mode:
            from .zipdata import ZipData
            if is_train:
                datapath = os.path.join(args.data_path, 'train.zip')
                data_map = os.path.join(args.data_path, 'train_map.txt')

            dataset = ZipData(
                datapath, data_map,
                transform
            )
        elif args.tsv_mode:
            map_file = None
            dataset = TSVDataset(
                os.path.join(args.data_path, 'train.tsv'),
                transform=transform,
                map_file=map_file
            )
        else:
            dataset = datasets.ImageFolder(args.data_path, transform=transform)
    elif 'imagenet22k' in args.dataset:
        dataset = _build_vis_dataset(args, transforms=transform, is_train=True)
    elif 'webvision1' in args.dataset:
        dataset = webvision_dataset(args, transform=transform, is_train=True)
    elif 'openimages_v4' in args.dataset:
        dataset = _build_openimage_dataset(args, transforms=transform, is_train=True)

    else:
        # only support folder format for other datasets
        dataset = datasets.ImageFolder(args.data_path, transform=transform)

    if args.sampler == 'distributed':
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    elif args.sampler == 'chunk':
        chunk_sizes = dataset.get_chunk_sizes() \
            if hasattr(dataset, 'get_chunk_sizes') else None
        sampler = DistributedChunkSampler(
            dataset, shuffle=True, chunk_sizes=chunk_sizes
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    return data_loader



def _build_vis_dataset(args, transforms, is_train=True):
    if comm.is_main_process():
        phase = 'train' if is_train else 'test'
        print('{} transforms: {}'.format(phase, transforms))

    dataset_name = 'train' if is_train else 'val'
    if args.tsv_mode:

        if args.dataset == 'imagenet22k':
            map_file = os.path.join(args.data_path, 'labelmap_22k_reorder.txt')
        else:
            map_file = None

        if os.path.isfile(os.path.join(args.data_path, dataset_name + '.tsv')):
            tsv_path = os.path.join(args.data_path, dataset_name + '.tsv')
        elif os.path.isdir(os.path.join(args.data_path, dataset_name)):
            tsv_list = []
            if len(tsv_list) > 0:
                tsv_path = [
                    os.path.join(args.data_path, dataset_name, f)
                    for f in tsv_list
                ]
            else:
                data_path = os.path.join(args.data_path, dataset_name)
                tsv_path = [
                    str(path)
                    for path in Path(data_path).glob('*.tsv')
                ]
            logging.info("Found %d tsv file(s) to load.", len(tsv_path))
        else:
            raise ValueError('Invalid TSVDataset format: {}'.format(args.dataset ))

        sas_token_file = [
            x for x in args.data_path.split('/') if x != ""
        ][-1] + '.txt'

        if not os.path.isfile(sas_token_file):
            sas_token_file = None
        logging.info("=> SAS token path: %s", sas_token_file)

        dataset = TSVDataset(
            tsv_path,
            transform=transforms,
            map_file=map_file,
            token_file=sas_token_file
        )
    else:
        dataset = datasets.ImageFolder(args.data_path, transform=transforms)
    print("%s set size: %d", 'train' if is_train else 'val', len(dataset))

    return dataset




class webvision_dataset(Dataset): 
    def __init__(self, args, transform, num_class=1000, is_train=True): 
        self.root = args.data_path
        self.transform = transform

        self.train_imgs = []
        self.train_labels = {}    
             
        with open(os.path.join(self.root, 'info/train_filelist_google.txt')) as f:
            lines=f.readlines()    
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    self.train_imgs.append(img)
                    self.train_labels[img]=target            
        
        with open(os.path.join(self.root, 'info/train_filelist_flickr.txt')) as f:
            lines=f.readlines()    
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    self.train_imgs.append(img)
                    self.train_labels[img]=target            

    def __getitem__(self, index):

        img_path = self.train_imgs[index]
        target = self.train_labels[img_path]
        file_path = os.path.join(self.root, img_path)

        image = Image.open(file_path).convert('RGB')   
        img = self.transform(image)        

        return img, target
                

    def __len__(self):
        return len(self.train_imgs)



def _build_openimage_dataset(args, transforms, is_train=True):

    files = 'train.tsv:train.balance_min1000.lineidx:train.label.verify_20191102.tsv:train.label.verify_20191102.6962.tag.labelmap'
    items = files.split(':')
    assert len(items) == 4, 'openimage dataset format: tsv_file:lineidx_file:label_file:map_file'

    root = args.data_path
    dataset = TSVOpenImageDataset(
        tsv_file=os.path.join(root, items[0]),
        lineidx_file=os.path.join(root, items[1]),
        label_file=os.path.join(root, items[2]),
        map_file=os.path.join(root, items[3]),
        transform=transforms
    )

    return dataset



class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, local_crops_size=96):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        if not isinstance(local_crops_size, tuple) or not isinstance(local_crops_size, list):
            local_crops_size = list(local_crops_size)

        
        if not isinstance(local_crops_number, tuple) or not isinstance(local_crops_number, list):
            local_crops_number = list(local_crops_number)

        self.local_crops_number = local_crops_number

        self.local_transfo = []
        for l_size in local_crops_size:
            self.local_transfo.append(transforms.Compose([
                transforms.RandomResizedCrop(l_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ]))

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        # print(f'self.local_crops_number {self.local_crops_number}')
        for i, n_crop in enumerate(self.local_crops_number):
            # print(n_crop)
            for _ in range(n_crop):
                crops.append(self.local_transfo[i](image))
        return crops



class DataAugmentationDEIT(object):
    def __init__(self, args):

        # first global crop
        self.global_transfo1 = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )

        # second global crop
        self.global_transfo2 = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        # transformation for the local small crops
        self.local_crops_number = args.local_crops_number
        self.local_transfo = create_transform(
            input_size=96,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops