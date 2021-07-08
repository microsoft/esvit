from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from io import BytesIO
import json
import logging
import base64

import numpy as np
from PIL import Image
import torch.utils.data as data

from .tsv_file import TSVFile


def is_json(string):
    try:
        json.loads(string)
    except ValueError as e:
        return False
    return True


class TSVOpenImageDataset(data.Dataset):

    def __init__(self,
                 tsv_file,
                 lineidx_file,
                 label_file,
                 map_file,
                 transform=None):
        self.transform = transform
        self.tsv_file = TSVFile(tsv_file, False, lineidx=lineidx_file)
        self.label2idx, self.num_classes = self._load_map(map_file)
        logging.info(
            '=> num of classes for training: {}'.format(self.num_classes)
        )

        self.classes_not_in_label_map = set()
        self.key2idx = self._load_label(label_file)
        logging.info(
            '=> num of unused classes: {}'.format(
                len(self.classes_not_in_label_map)
            )
        )

        logging.info('=> {}\titems: {}'.format(tsv_file, len(self.tsv_file)))

    def get_filenames(self):
        filenames = [
            self.tsv_file.get_key(i)
            for i in range(self.tsv_file.num_rows())
        ]

        return filenames

    def _load_map(self, map_file):
        if not map_file:
            return None

        label2idx = {}
        idx = 0
        with open(map_file) as f:
            for l in f:
                label = l.strip()
                label2idx[label] = idx
                idx += 1

        return label2idx, idx+1

    def _load_label(self, label_file):
        key2idx = {}
        with open(label_file, 'r') as f:
            for l in f:
                items = l.strip().split('\t')
                key2idx[items[0]] = self._decode_label(items[1])

        return key2idx

    def _decode_label(self, string):
        if is_json(string):
            return self._decode_label_json(string)

        idx = []
        labels = string.split(';')
        for label in labels:
            if label.startswith('-'):
                continue
            if label not in self.label2idx:
                continue
            idx.append(self.label2idx[label])
        return idx

    def _decode_label_json(self, json_str):
        js = json.loads(json_str)
        idx = []
        for i in js:
            label = i['class']
            if label.startswith('-'):
                continue
            if label not in self.label2idx:
                if label not in self.classes_not_in_label_map:
                    self.classes_not_in_label_map.add(label)
                continue
            idx.append(self.label2idx[label])

        return idx

    def __getitem__(self, index):
        items = self.tsv_file[index]
        _, target, img = self._decode_data(items)

        if self.transform:
            img = self.transform(img)

        return img, target

    def _decode_data(self, items):
        key = items[0]
        label_idx = self.key2idx[key]
        label_vector = np.zeros((self.num_classes), dtype=np.float)
        for idx in label_idx:
            label_vector[idx] = 1
        image = Image.open(BytesIO(base64.b64decode(items[2])))

        return key, label_vector, image.convert('RGB')

    def __len__(self):
        return self.tsv_file.num_rows()
