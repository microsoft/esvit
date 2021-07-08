from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from io import BytesIO
import json
import logging
import base64
import random
from typing import Callable, List, Tuple, Union
from PIL import Image
from PIL import ImageFile
import torch.utils.data as data
from .tsv_file import TSVFile, CompositeTSVFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TSVDataset(data.Dataset):

    def __init__(self,
                 tsv_file: Union[str, List[str]],
                 transform: Callable = None,
                 map_file: str = None,
                 token_file: str = None):
        self.transform = transform
        self._chunk_sizes = None
        self.label2idx = self._load_map(map_file)
        self.class_selector = list(self.label2idx.keys()) if self.label2idx else None

        if isinstance(tsv_file, str):
            if os.path.splitext(tsv_file)[1] == '.tsv':
                self.tsv_file = TSVFile(
                    tsv_file, class_selector=self.class_selector
                )
            else:
                self.tsv_file = CompositeTSVFile(
                    tsv_file, class_selector=self.class_selector,
                    sas_token_path=token_file
                )
                self._chunk_sizes = self.tsv_file.get_chunk_size()
        elif isinstance(tsv_file, list):
            self.tsv_file = CompositeTSVFile(
                tsv_file, class_selector=self.class_selector,
                sas_token_path=token_file
            )
            self._chunk_sizes = self.tsv_file.get_chunk_size()
        else:
            raise ValueError("Invalid input! Please check the tsv filenames")

        logging.debug('=> {}\titems: {}'.format(tsv_file, len(self.tsv_file)))

    def fetch_blob(self, idx):
        image_tsv = self.tsv_file.file_list[idx]
        self.tsv_file.blob_storage.fetch_blob(image_tsv)

    def num_classes(self):
        return len(self.class_selector)

    def get_chunk_sizes(self):
        return self._chunk_sizes

    def get_class_boundaries(self):
        # The samples of each class are organized class-by-class.
        # _class_boundaries stores the lower- and upper-bound of each class.
        return self.tsv_file.get_class_boundaries()

    def get_filenames(self):
        filenames = [
            self.tsv_file.get_key(i)
            for i in range(self.tsv_file.num_rows())
        ]

        return filenames

    def _load_map(self, map_file: str):
        if not map_file:
            return None

        label2idx = {}
        with open(map_file) as f:
            for line in f:
                items = line.strip().split('\t')
                label2idx[items[0]] = int(items[1])

        return label2idx

    def __getitem__(self, index: Union[int, Tuple[int, int]]):
        items = self.tsv_file[index]
        _, target, img = self._decode_data(items)

        if self.transform:
            img = self.transform(img)

        return img, target

    def _decode_data(self, items: Tuple[str, str, str]):
        key = items[0]
        label = self._get_label(items[1])
        image = Image.open(BytesIO(base64.b64decode(items[2]))).convert('RGB')

        return key, label, image

    def _get_label(self, item: str):
        if not self.label2idx:
            return int(item)

        js = json.loads(item)
        return self.label2idx[js[0]['class']]

    def __len__(self):
        return len(self.tsv_file)


class TSVImageTextDataset(data.Dataset):
    """
        This class is intended for encapsulating Image/Text pair data for contrastive learning described in
        the following paper,
        "Learning Transferable Visual Models From Natural Language Supervision" (a.k.a CLIP)
    """
    def __init__(self,
                 image_tsv_file: Union[str, List[str]],
                 text_tsv_file: Union[str, List[str]],
                 transform: Callable = None,
                 tokenize: Callable = None,
                 context_length: int = 77,
                 num_captions: int = 1,
                 text_format: str = 'txt',
                 is_train: bool = True,
                 sas_token_path: str = None):
        self.transform = transform
        self.tokenize = tokenize
        self._chunk_sizes = None
        self.context_length = context_length
        self.num_captions = num_captions
        self.text_format = text_format
        self.tsv_file_list = []

        if isinstance(image_tsv_file, str) and isinstance(text_tsv_file, str):
            # single tsv file
            if (
                os.path.splitext(image_tsv_file)[1].lower() == '.tsv'
                and os.path.splitext(text_tsv_file)[1].lower() == '.tsv'
            ):
                self.tsv_file_list.append((image_tsv_file, text_tsv_file))
                self.image_tsv_file = TSVFile(
                    image_tsv_file, if_generate_lineidx=True
                )
                self.text_tsv_file = TSVFile(
                    text_tsv_file, if_generate_lineidx=True
                )
            else:
                raise ValueError("Invalid input! Please check the tsv filenames.")
        # multiple tsv files specified in a list
        elif (
            isinstance(image_tsv_file, list)
            and isinstance(text_tsv_file, list)
        ):
            assert len(image_tsv_file) == len(text_tsv_file), \
                "Inconsistent number of Image/Text tsv files!"
            self.tsv_file_list = [
                (txt, img)
                for img, txt in zip(image_tsv_file, text_tsv_file)
            ]
            self.image_tsv_file = CompositeTSVFile(
                image_tsv_file,
                is_train=is_train,
                sas_token_path=sas_token_path
            )
            self.text_tsv_file = CompositeTSVFile(
                text_tsv_file,
                is_train=is_train,
                sas_token_path=sas_token_path
            )
            self._chunk_sizes = self.image_tsv_file.get_chunk_size()
        else:
            raise ValueError("Invalid input! Please check the tsv filenames.")

        assert len(self.image_tsv_file) == len(self.text_tsv_file), \
            "Inconsistent size of Image/Text ({}/{}) data!".format(
                len(self.image_tsv_file), len(self.text_tsv_file)
            )

    def fetch_blob(self, idx):
        # image_tsv, text_tsv = self.tsv_file_list[idx]
        image_tsv = self.image_tsv_file.file_list[idx]
        text_tsv = self.text_tsv_file.file_list[idx]
        self.image_tsv_file.blob_storage.fetch_blob(image_tsv)
        self.image_tsv_file.blob_storage.fetch_blob(text_tsv)

    def get_chunk_sizes(self):
        return self._chunk_sizes

    def __getitem__(self, index: Union[int, Tuple[int, int]]):
        items_image = self.image_tsv_file[index]
        items_text = self.text_tsv_file[index]

        assert items_text[0] == items_image[0], 'keys do not match for image and text'

        _, img = self._decode_image(items_image)
        _, txt = self._decode_text(items_text)

        if self.transform:
            img = self.transform(img)

        tokens = self.tokenize(txt, context_length=self.context_length) if self.tokenize else txt
        tokens.squeeze_()

        return img, tokens

    def _decode_image(self, items: Tuple[str, str]):
        key = items[0]
        image = Image.open(BytesIO(base64.b64decode(items[1]))).convert('RGB')

        return key, image

    def _decode_text(self, items: Tuple[str, Union[str, dict]]):
        key = items[0]
        text = ''

        if self.text_format == 'json':
            js = json.loads(items[1])
            assert 'captions' in js or 'tags' in js, '"captions" or "tags" does not exist in {}'.format(js)
            captions = js['captions'] if 'captions' in js else js['tags']
            if isinstance(captions, list):
                if self.num_captions == 1:
                    text = random.choice(captions)
                else:
                    text = captions
                    if len(captions) > self.num_captions:
                        text = captions[:self.num_captions]
            elif isinstance(captions, str):
                text = captions
            else:
                raise ValueError('captions should be str or list')
        else:
            text = items[1]

        return key, text

    def __len__(self):
        return len(self.image_tsv_file)
