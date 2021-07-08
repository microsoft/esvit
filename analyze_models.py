# Modified by Chunyuan Li (chunyl@microsoft.com)
#
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import utils
import models.vision_transformer as vits
from torchvision import datasets
from models import build_model
from config import config
from config import update_config
from config import save_config

import glob
from math import sqrt
import torch.nn.functional as F


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", query_position=[],figsize=(4, 4), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    if len(query_position)>0:
        plt.plot(query_position[0],query_position[1], '-', marker= 'o', color='skyblue', lw=1,  mec='k', mew=1 , markersize=20)


    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname, bbox_inches='tight')
    # print(f"{fname} saved.")

    plt.close('all')
    return

def compute_attn_entropy_sorted(attentions, layer_id, img, args, query=0):

    # we keep only the output patch attention
    nh = attentions.shape[1] # number of head
    attentions = attentions[0, :, query, :].reshape(nh, -1)

    attentions_entropy = ( -attentions * torch.log2(attentions)).sum(-1)
    attentions_entropy_val, idx = torch.sort(attentions_entropy, descending=True)

    return attentions_entropy_val

def compute_attn_entropy(attentions, layer_id, img, args, query=0):

    # we keep only the output patch attention
    nh = attentions.shape[1] # number of head
    attentions = attentions[0, :, query, :].reshape(nh, -1)

    attentions_entropy = ( -attentions * torch.log2(attentions)).sum(-1)

    return attentions_entropy



def visualize_attn(attentions, layer_id, img, args, query=0):
    # Input: attentions and input image

    height = width = img.shape[-1]
    nh = attentions.shape[1] # number of head
    window_size = w_featmap = h_featmap = int(sqrt(attentions.shape[2]))

    scale_factor = int(width/w_featmap)

    

    # we keep only the output patch attention
    attentions = attentions[0, :, query, :].reshape(nh, -1)

    query_position = []
    if query != 0:
        x = query // h_featmap * (224/w_featmap)
        y = (query % h_featmap) * (224/h_featmap) # (h_featmap - query % h_featmap) * args['patch_size']
        query_position = [x,y]

    attentions_entropy = ( -attentions * torch.log2(attentions)).sum(-1)
    attentions_entropy_val, idx = torch.sort(attentions_entropy, descending=True)
    attentions_sort = torch.zeros(attentions.shape)
    for j, v in enumerate(idx):
        attentions_sort[j,:] = attentions[v,:]
    attentions = attentions_sort

    print(f'attentions {attentions.shape} {attentions_entropy_val}')

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - args.threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()

    shift_size = 0 # if (layer_id % 2 == 0) else window_size // 2
    th_attn = torch.roll(th_attn, shifts=(shift_size, shift_size), dims=(1, 2))    
    # interpolate
    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor= scale_factor, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)

    attentions = torch.roll(attentions, shifts=(shift_size, shift_size), dims=(1, 2))
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor= scale_factor, mode="nearest")[0]
    attentions_cpu = attentions.cpu().numpy()



    # save attentions heatmaps
    os.makedirs(args.output_dir, exist_ok=True)



    save_attn_dir = os.path.join(args.output_dir, "layer" + str(layer_id) + "_query" + str(query) + "/attn")
    if not os.path.exists(save_attn_dir):
        os.makedirs(save_attn_dir)

    save_mask_dir = os.path.join(args.output_dir, "layer" + str(layer_id) + "_query" + str(query) +  "/mask_th" + str(args.threshold))
    if not os.path.exists(save_mask_dir):
        os.makedirs(save_mask_dir)


    attentions_all = torchvision.utils.make_grid(attentions.unsqueeze(1), nrow=4, padding=2, pad_value=0.5, normalize=True, scale_each=True).permute(1,2,0)[:,:,0]
    print(f'attentions {attentions.shape} attentions_all {attentions_all.shape} {attentions_all.max()}')
    fname = os.path.join(args.output_dir,  f"attn_all_{layer_id}_query{query}.png")
    plt.imsave(fname=fname, arr=attentions_all.cpu().numpy(), format='png', cmap=plt.cm.get_cmap('Blues'))


    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
    for j in range(nh):
        fname = os.path.join(save_attn_dir,  f"attn-head_{j:02}.png")
        plt.imsave(fname=fname, arr=attentions_cpu[j], format='png', cmap=plt.cm.get_cmap('Blues'))
        # print(f"{fname} saved.")

    image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))

    for j in range(nh):
        display_instances(image, th_attn[j], fname=os.path.join(save_mask_dir,  "mask_th" + str(args.threshold)  + f"_head_{j:02}.png"), query_position=query_position, blur=False)

    transform_compress = pth_transforms.Compose([
        pth_transforms.ToPILImage(),
        pth_transforms.Resize(400),
        pth_transforms.ToTensor()
    ])
        
    image_list = []
    for filename in sorted(glob.glob(os.path.join(save_mask_dir, "*.png"))): 
        im = torch.Tensor(skimage.io.imread(filename)).permute(2,0,1)
        image_list.append(im)
    
    image_list_all = torchvision.utils.make_grid(image_list, nrow=4, normalize=True, scale_each=True)

    # print(f'im {im.shape}')
    image_list_all_compressed = transform_compress(image_list_all)    
    print(f'image_list_all_compressed {image_list_all_compressed.shape} image_list_all {image_list_all.shape} {image_list_all.max()}')
    torchvision.utils.save_image(image_list_all, os.path.join(args.output_dir, f"attn_masked_all_{layer_id}_query{query}.png"))
    torchvision.utils.save_image(image_list_all_compressed, os.path.join(args.output_dir, f"attn_masked_all_{layer_id}_query{query}_compressed.png"))
    



def accuracy_correspondence(img0, args):

    img_ref_exist = False

    j = args.seed
    save_pair_dir = os.path.join(args.output_dir, "seed" + str(j) )
    if not os.path.exists(save_pair_dir):
        os.makedirs(save_pair_dir)


    transformb = pth_transforms.Compose([
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    flip_and_color_jitter = pth_transforms.Compose([
        pth_transforms.RandomHorizontalFlip(p=1.0),
        pth_transforms.RandomApply(
            [pth_transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            p=0.8
        ),
        pth_transforms.RandomGrayscale(p=1.0),
    ])

    transform1a = pth_transforms.Compose([
        pth_transforms.ToPILImage(),
        pth_transforms.ToTensor(),
    ])

    transform2a = pth_transforms.Compose([
        pth_transforms.ToPILImage(),
        flip_and_color_jitter,
        pth_transforms.ToTensor(),
    ])

    img1a = transform1a(img0.squeeze(0))
    img2a = transform2a(img0.squeeze(0))
        
    img1 = transformb(img1a).unsqueeze(0)
    img2 = transformb(img2a).unsqueeze(0)

    height, width = img1.shape[-2:]

    if 'deit' in args.arch:
        fea1 = model.forward_feature_maps(img1.cuda()).squeeze()
        fea1_g, fea1 = fea1[0].unsqueeze(0), fea1[1:]
        fea2 = model.forward_feature_maps(img2.cuda()).squeeze()
        fea2_g, fea2 = fea2[0].unsqueeze(0), fea2[1:]   
        window_size = args.patch_size

    else: 
        fea1_g, fea1 = model.forward_feature_maps(img1.cuda()) # .squeeze()
        fea1 = fea1.squeeze()
        fea2_g, fea2 = model.forward_feature_maps(img2.cuda()) # .squeeze()
        fea2 = fea2.squeeze()
        window_size = 32 # 16


   
    num_h, num_w = height/window_size, height/window_size
    coords_h = torch.arange(num_h)
    coords_w = torch.arange(num_w)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)* window_size + window_size/2.0  # 2 Wh*Ww

    # backbone_sim_matrix = torch.matmul(fea1 , fea2.permute(1, 0)) 
    backbone_sim_matrix = torch.matmul(F.normalize(fea1, p=2, dim=1) , F.normalize(fea2, p=2, dim=1).permute(1, 0))  # B x N_s x N_t
    dense_fea_sim, dense_fea_sim_ind = backbone_sim_matrix.max(dim=1) # B x N_s; collect the index in teacher for a given student feature



    pair_idx = {}
    pair_idx_sim = {}
    pair_idx_coords = {}
    for i in range(dense_fea_sim_ind.shape[0]):
        pair_idx[i] =  dense_fea_sim_ind[i].item()
        pair_idx_sim[i] = dense_fea_sim[i].item()
        pair_idx_coords[i] = coords_flatten[:,dense_fea_sim_ind[i]]
        
        

    # fig = plt.figure(frameon=False)
    # ax = plt.gca()


    margin= 5

    pair_idx_sim = {k: v for k, v in sorted(pair_idx_sim.items(), key=lambda item: item[1], reverse=True)}

    count = 0.0 
    correct = 0.0
    distance_sum = 0.0
    for i, v in pair_idx_sim.items(): # range(dense_fea_sim_ind.shape[0]):
        if count < 10:
            dots_x = [ coords_flatten[1,i],  margin + width + pair_idx_coords[i][1]]
            dots_y = [ coords_flatten[0,i],  pair_idx_coords[i][0]]

            distance = ( ((width - coords_flatten[1,i]) -  pair_idx_coords[i][1])**2 + (coords_flatten[0,i] -  pair_idx_coords[i][0])**2 )**(0.5)
            distance_sum += distance

            if distance == 0:
                correct += 1.0

            count +=1.0
        else:
            break

    distance_error = distance_sum / count
    accuracy = correct / count

    # print(f'break at count {count}, with r value {v}, accuracy {accuracy} distance_error {distance_error} ')
    return accuracy, distance_error, v



def visualize_correspondence(img0, args):


    img_ref_exist = False
    if os.path.isfile(args.image_path2):
        with open(args.image_path2, 'rb') as f:
            img_ref = Image.open(f)
            img_ref = img_ref.convert('RGB')
            img_ref_exist = True

        print(f'image_ref is chosen at {args.image_path2}')


    j = args.seed
    save_pair_dir = os.path.join(args.output_dir, "seed" + str(j) )
    if not os.path.exists(save_pair_dir):
        os.makedirs(save_pair_dir)

    transformb = pth_transforms.Compose([
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if not args.use_saved_aug:

        flip_and_color_jitter = pth_transforms.Compose([
            pth_transforms.RandomHorizontalFlip(p=0.5),
            pth_transforms.RandomApply(
                [pth_transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            pth_transforms.RandomGrayscale(p=0.2),
        ])

        transform1a = pth_transforms.Compose([
            pth_transforms.RandomResizedCrop(224, scale=(1.0, 1.0), interpolation=Image.BICUBIC),
            pth_transforms.ToTensor(),
        ])


        transform2a = pth_transforms.Compose([
            pth_transforms.RandomResizedCrop(224, scale=(0.4, 0.6), interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            pth_transforms.ToTensor(),
        ])

        img1a = transform1a(img0)

        if img_ref_exist:
            img2a = transform2a(img_ref)
        else:
            img2a = transform2a(img0)
            

        torchvision.utils.save_image(torchvision.utils.make_grid(img1a, normalize=True, scale_each=True), os.path.join(save_pair_dir, "img1.png"))
        torchvision.utils.save_image(torchvision.utils.make_grid(img2a, normalize=True, scale_each=True), os.path.join(save_pair_dir, "img2.png"))

    else:
        transform2t = pth_transforms.Compose([
            pth_transforms.ToTensor(),
        ])
        img1a = transform2t(skimage.io.imread(os.path.join(save_pair_dir, "img1.png")))
        img2a = transform2t(skimage.io.imread(os.path.join(save_pair_dir, "img2.png")))

    img1 = transformb(img1a).unsqueeze(0)
    img2 = transformb(img2a).unsqueeze(0)



    height, width = img1.shape[-2:]


    print(img1.shape)

    fea1_g, fea1 = model.forward_feature_maps(img1.cuda()) # .squeeze()
    # fea1_g, fea1 = fea1[0].unsqueeze(0), fea1[1:]
    fea1 = fea1.squeeze()
    print(fea1.shape)

    fea2_g, fea2 = model.forward_feature_maps(img2.cuda()) # .squeeze()
    # fea2_g, fea2 = fea2[0].unsqueeze(0), fea2[1:]
    fea2 = fea2.squeeze()
    print(fea2.shape)



    window_size = 32 # 16
    num_h, num_w = height/window_size, height/window_size
    coords_h = torch.arange(num_h)
    coords_w = torch.arange(num_w)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)* window_size + window_size/2.0  # 2 Wh*Ww

    print(coords_flatten)

    # print(coords_flatten)

    # backbone_sim_matrix = torch.matmul(fea1 , fea2.permute(1, 0)) 
    backbone_sim_matrix = torch.matmul(F.normalize(fea1, p=2, dim=1) , F.normalize(fea2, p=2, dim=1).permute(1, 0))  # B x N_s x N_t
    dense_fea_sim, dense_fea_sim_ind = backbone_sim_matrix.max(dim=1) # B x N_s; collect the index in teacher for a given student feature

    print(f'dense_fea_sim {dense_fea_sim.shape} dense_fea_sim_ind {dense_fea_sim_ind.shape}' )




    g_sim_matrix11 = torch.matmul(F.normalize(fea1_g, p=2, dim=1) , F.normalize(fea1, p=2, dim=1).permute(1, 0))  # B x N_s x N_t
    # g_sim_matrix11 = torch.matmul(fea1_g, fea1.permute(1, 0))  # B x N_s x N_t
    g_sim_sim11, g_sim_ind11 = g_sim_matrix11.max(dim=1) # B x N_s; collect the index in teacher for a given student feature

    g_sim_matrix1 = torch.matmul(F.normalize(fea1_g, p=2, dim=1) , F.normalize(fea2, p=2, dim=1).permute(1, 0))  # B x N_s x N_t
    g_sim_sim1, g_sim_ind1 = g_sim_matrix1.max(dim=1) # B x N_s; collect the index in teacher for a given student feature


    g_sim_matrix2 = torch.matmul(F.normalize(fea1, p=2, dim=1) , F.normalize(fea2_g, p=2, dim=1).permute(1, 0))  # B x N_s x N_t
    g_sim_sim2, g_sim_ind2 = g_sim_matrix2.max(dim=0) # B x N_s; collect the index in teacher for a given student feature

    g_sim_matrix22 = torch.matmul(F.normalize(fea2, p=2, dim=1) , F.normalize(fea2_g, p=2, dim=1).permute(1, 0))  # B x N_s x N_t
    g_sim_sim22, g_sim_ind22 = g_sim_matrix22.max(dim=0) # B x N_s; collect the index in teacher for a given student feature


    print(f'max fea1_g feature position {g_sim_ind11.item()} in image 1 with similarity {g_sim_sim11.item()}')
    print(f'max fea1_g feature position {g_sim_ind1.item()} in image 2 with similarity {g_sim_sim1.item()}')
    print(f'max fea2_g feature position {g_sim_ind2.item()} in image 1 with similarity {g_sim_sim2.item()}')
    print(f'max fea2_g feature position {g_sim_ind22.item()} in image 2 with similarity {g_sim_sim22.item()}')



    pair_idx = {}
    pair_idx_sim = {}
    pair_idx_coords = {}
    for i in range(dense_fea_sim_ind.shape[0]):
        pair_idx[i] =  dense_fea_sim_ind[i].item()
        pair_idx_sim[i] = dense_fea_sim[i].item()
        pair_idx_coords[i] = coords_flatten[:,dense_fea_sim_ind[i]]
        
        

    fig = plt.figure(frameon=False)
    ax = plt.gca()


    margin= 5

    pair_idx_sim = {k: v for k, v in sorted(pair_idx_sim.items(), key=lambda item: item[1], reverse=True)}

    count = 0 
    for i, v in pair_idx_sim.items(): # range(dense_fea_sim_ind.shape[0]):
        if count < 10: # 0.901:
            print(f'{i} corresponds to {pair_idx[i]} with similarity {pair_idx_sim[i]}')
            dots_x = [ coords_flatten[1,i],  margin + width + pair_idx_coords[i][1]]
            dots_y = [ coords_flatten[0,i],  pair_idx_coords[i][0]]
            plt.plot(dots_x, dots_y, '-', marker= 'o', color='yellow', lw=1,  mec='k', mew=1 , markersize=5)
            plt.text(dots_x[0], dots_y[0]-2, str(count), size=10, color='yellow')
            plt.text(dots_x[1], dots_y[1]-2, str(count), size=10, color='yellow')
            count +=1
        else:
            break

    print(f'break at count {count}, with r value {v} ')


    plt.plot(coords_flatten[1, g_sim_ind11], coords_flatten[0, g_sim_ind11], '-', marker= 'o', color='skyblue', lw=1,  mec='k', mew=1 , markersize=10)
    plt.plot(margin + width + coords_flatten[0,g_sim_ind1], coords_flatten[1, g_sim_ind1], '-', marker= 'v', color='red', lw=1,  mec='k', mew=1 , markersize=10)
    plt.plot(margin + width + coords_flatten[0, g_sim_ind22], coords_flatten[1, g_sim_ind22], '-', marker= 'o', color='skyblue', lw=1,  mec='k', mew=1 , markersize=10)
    plt.plot(coords_flatten[1,g_sim_ind2], coords_flatten[0,g_sim_ind2], '-', marker= 'v', color='red', lw=1,  mec='k', mew=1 , markersize=10)


    ax.axis('off')

    padding = torch.ones(3, height, margin)
    # print(img1a.shape)
    imgs = torch.cat([img1a,padding, img2a], -1 )
    imgsnp = np.array(imgs.squeeze(0).permute(1,2,0))
    # imgscv = cv2.cvtColor(imgsnp, cv2.COLOR_BGR2GRAY)



    fname=os.path.join(save_pair_dir,  "correspondence" + str(j) +".png")

    plt.imshow(imgsnp)

    fig.savefig(fname, bbox_inches='tight')




if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Correspondence and Self-Attention maps')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str)

    parser.add_argument('--arch', default='deit_small', type=str,
        choices=['swin_tiny','swin_small', 'swin', 'vil', 'vil_1281', 'vil_2262', 'deit_tiny', 'deit_small', 'vit_base'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using deit_tiny or deit_small.""")

    parser.add_argument('--use_saved_aug', default=False, type=utils.bool_flag,
        help="Whether to use saved the augmented images (Default: False)")
    parser.add_argument('--vis_correspondence', default=False, type=utils.bool_flag,
        help="Whether to visualize correspondence map (Default: False)")
    parser.add_argument('--measure_correspondence', default=False, type=utils.bool_flag,
        help="Whether to measure the quality correspondence map (Default: False)")        
    parser.add_argument('--vis_attention', default=False, type=utils.bool_flag,
        help="Whether to visualize self-attention map (Default: False)")
    parser.add_argument('--vis_entropy', default=False, type=utils.bool_flag,
        help="Whether to visualize self-attention entropy (Default: False)")        




    parser.add_argument('--seed', default=8, type=int, help='random seed.')

    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--learning", default="ssl", type=str,
        help='Key to use in the checkpoint (example: "teacher")')        
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_path2", default='.', type=str, help="Path of the 2nd image to load.")    
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=0.6, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")

    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')

    parser.add_argument("--rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)    

    args = parser.parse_args()


    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # build model
    # if the network is a 4-stage vision transformer (i.e. swin)
    if 'swin' in args.arch :
        update_config(config, args)
        model = build_model(config, is_teacher=True)

    else:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.cuda()

    if os.path.isfile(args.pretrained_weights):
        if args.learning == 'ssl':
            print(f"Take learning objective {args.learning} in provided checkpoint dict")
            state_dict = torch.load(args.pretrained_weights, map_location="cpu")
            if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
                print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[args.checkpoint_key]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)

        else:
            print(f"Take learning objective {args.learning} in provided checkpoint dict")
            state_dict = torch.load(args.pretrained_weights, map_location="cpu")
            if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
                print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[args.checkpoint_key]

            model_state_dict = state_dict['model']
            # for k, v in model_state_dict.items():
            #     print(k) 

            msg = model.load_state_dict(model_state_dict, strict=False)
                
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.arch == "deit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "deit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

    # open image
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        print(f'image_1 is chosen at {args.image_path}')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)



    
    ########################################################################
    #### Visualize correspondences between two views
    ######################################################################## 
    if args.vis_correspondence:
        args.output_dir = os.path.join(args.output_dir, os.path.basename(args.image_path).split('.')[0] )  
        visualize_correspondence(img, args)



    ########################################################################
    #### Measure correspondences on ImageNet val
    ######################################################################## 
    if args.measure_correspondence:

        args.output_dir = os.path.join(args.output_dir, os.path.basename(args.data_path) ) 
        os.makedirs(args.output_dir, exist_ok=True)

        val_transform = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
        ])

        dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=val_transform)

        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        print(f"Data loaded with {len(dataset_val)} val imgs.")

        accuracy_avg, distance_error_avg, sim_value_avg = .0, .0, .0
        count = .0
        metric_logger = utils.MetricLogger(delimiter="  ")
        for inp, target in metric_logger.log_every(val_loader, 50):
            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            attentions = model.forward_selfattention(inp, n=2)

            accuracy, distance_error, sim_value = accuracy_correspondence(inp, args)

            count += 1.0

            if count<2:
                accuracy_avg       = accuracy
                distance_error_avg = distance_error
                sim_value_avg      = sim_value
            else:
                accuracy_avg       = (accuracy + (count-1.0) * accuracy_avg ) / count
                distance_error_avg = (distance_error + (count-1.0) * distance_error_avg ) / count
                sim_value_avg      = (sim_value + (count-1.0) * sim_value_avg ) / count

            # if count > 1000: break

        print(f'break at count {count}, with sim_value_avg value {sim_value_avg}, accuracy {accuracy_avg} distance_error {distance_error_avg} ')

        dict_results = {'count': count, 'sim_value_avg': sim_value_avg, 'accuracy_avg': accuracy_avg, 'distance_error_avg':distance_error_avg }
        # save
        import pickle
        with open(os.path.join(args.output_dir, 'measure_correspondence.pickle'), 'wb') as handle:
            pickle.dump(dict_results, handle)

        

                

    ########################################################################
    #### Visualize attnetion map on individual images
    ######################################################################## 

    if args.vis_attention:
        args.output_dir = os.path.join(args.output_dir, os.path.basename(args.image_path).split('.')[0] ) 
        
        transform = pth_transforms.Compose([
            pth_transforms.Resize([224,224]),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        img = transform(img)
        # make the image divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
        img = img[:, :w, :h].unsqueeze(0)

        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size


        attentions = model.forward_selfattention(img.cuda(), n=2)

        for i, attn in enumerate(attentions):
            if i>=11:
                print(f'input img size {img.shape}; attn map size {attn.shape} at layer {i} ')
                if len(attn.shape) == 3: attn = attn.unsqueeze(0)
                visualize_attn(attn, i, img, args, query=9) # query=24  query=9


    ########################################################################
    #### Measure attention entropy on ImageNet val
    ######################################################################## 

    if args.vis_entropy:
        args.output_dir = os.path.join(args.output_dir, os.path.basename(args.data_path) ) 
        os.makedirs(args.output_dir, exist_ok=True)

        val_transform = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=val_transform)

        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        print(f"Data loaded with {len(dataset_val)} val imgs.")

        attn_entropy_avg = [None] * 12 
        count = .0
        metric_logger = utils.MetricLogger(delimiter="  ")
        for inp, target in metric_logger.log_every(val_loader, 50):
            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            attentions = model.forward_selfattention(inp, n=2)
            count += 1.0

            for i, attn in enumerate(attentions):
                if len(attn.shape) == 3: attn = attn.unsqueeze(0)

                attn_entropy_avg_query = 0
                for q in range(49):
                    attn_entropy = compute_attn_entropy(attn, i, img, args, query=q)
                    attn_entropy_avg_query += attn_entropy
                attn_entropy_avg_query = attn_entropy_avg_query/49.0


                if count<2:
                    attn_entropy_avg[i] = attn_entropy_avg_query
                else:
                    attn_entropy_avg[i] = (attn_entropy_avg_query + (count-1.0) * attn_entropy_avg[i]) / count

            # if count > 10: break
        # save
        import pickle
        with open(os.path.join(args.output_dir, 'attn_entropy_avg_all_queries_no_sort.pickle'), 'wb') as handle:
            pickle.dump(attn_entropy_avg, handle)

        print(attn_entropy_avg)

        



