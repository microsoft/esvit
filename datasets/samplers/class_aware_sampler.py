"""
Inspired and partially adopted from
    https://github.com/zhmiao/OpenLongTailRecognition-OLTR/blob/master/data/ClassAwareSampler.py
"""
import math
import random
import logging
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class CycleIter:

    def __init__(self, data, shuffle=True):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.shuffle = shuffle

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1

        if self.i == self.length:
            self.i = 0
            if self.shuffle:
                random.shuffle(self.data_list)

        return self.data_list[self.i]


class ClassAwareTargetSizeSampler(Sampler):
    def __init__(self, dataset, target_size, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        assert len(dataset) >= target_size, "Dataset size %d is smaller than target size %d." % (
            len(dataset), target_size)

        if hasattr(self.dataset, "num_classes"):
            self.num_classes = self.dataset.num_classes()
        else:
            raise RuntimeError("Dataset does not provide the info of class number!")

        if hasattr(self.dataset, "get_class_boundaries"):
            self.class_boundaries = self.dataset.get_class_boundaries()
        else:
            raise RuntimeError("Dataset does not provide the info of class boundary!")
        assert len(self.class_boundaries) == self.num_classes, "Number of classes and boundaries not consistent!"

        if target_size > 0:
            # Try to evenly draw the same number of data samples from each class
            self.num_samples_cls = int(math.ceil(target_size * 1.0 / self.num_classes))
            class_sizes = [(high - low + 1) for low, high in self.class_boundaries]
            self.target_size = sum([min(self.num_samples_cls, s) for s in class_sizes])

            # Increase the number of data samples drawn from each class until we reach the target size
            while self.target_size < target_size:
                self.num_samples_cls += 1
                self.target_size = sum([min(self.num_samples_cls, s) for s in class_sizes])
        else:
            # Sample all available data
            self.target_size = len(dataset)
            self.num_samples_cls = max([(high - low + 1) for low, high in self.class_boundaries])

        self.epoch = 0
        self.rank = rank
        self.num_replicas = num_replicas
        self.num_samples = int(math.ceil(self.target_size * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

        random.seed(self.epoch)
        self.indices = []
        for j in range(self.num_classes):
            lower_bound, upper_bound = self.class_boundaries[j]
            indices_ = list(range(lower_bound, upper_bound))
            if self.shuffle:
                random.shuffle(indices_)
            self.indices += indices_[:min(self.num_samples_cls, len(indices_))]

        # add extra samples to make it evenly divisible
        self.indices += self.indices[:(self.total_size - len(self.indices))]
        assert len(self.indices) == self.total_size
        logging.info("[ClassAwareTargetSizeSampler] sampled data size: %d" % self.total_size)

    def __iter__(self):
        # deterministically shuffle based on epoch
        random.seed(self.epoch)
        random.shuffle(self.indices)

        # subsample
        indices = self.indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class ClassAwareDistributedSampler(Sampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, num_samples_cls=1000):
        super().__init__(dataset)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        if hasattr(self.dataset, "num_classes"):
            self.num_classes = self.dataset.num_classes()
        else:
            raise RuntimeError("Dataset does not provide the info of class number!")
        if hasattr(self.dataset, "get_class_boundaries"):
            self.class_boundaries = self.dataset.get_class_boundaries()
        else:
            raise RuntimeError("Dataset does not provide the info of class boundary!")

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples_cls = num_samples_cls
        self.num_samples = int(math.ceil(self.num_samples_cls * self.num_classes * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        random.seed(self.epoch)

        class_iter = CycleIter(range(self.num_classes), shuffle=self.shuffle)
        data_iter = [CycleIter(range(bound[0], bound[1]), shuffle=self.shuffle) for bound in self.class_boundaries]

        indices = []
        for i in range(self.num_samples_cls):
            for j in range(self.num_classes):
                indices.append(next(data_iter[next(class_iter)]))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class ClassAwareAverageSampler(ClassAwareDistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if hasattr(dataset, "get_class_boundaries"):
            class_boundaries = dataset.get_class_boundaries()
            class_sizes = [bound[1] - bound[0] for bound in class_boundaries]
            average_size = int(sum(class_sizes) * 1.0 / len(class_sizes))
            logging.info("[ClassAwareAverageSampler] Average class size: %d", average_size)
        else:
            raise RuntimeError("Dataset does not provide the info of class boundary!")

        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, num_samples_cls=average_size)


class ClassAwareMedianSampler(ClassAwareDistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if hasattr(dataset, "get_class_boundaries"):
            class_boundaries = dataset.get_class_boundaries()
            class_sizes = sorted([bound[1] - bound[0] for bound in class_boundaries])
            median_size = class_sizes[len(class_sizes) // 2]
            logging.info("[ClassAwareMedianSampler] Median class size: %d", median_size)
        else:
            raise RuntimeError("Dataset does not provide the info of class boundary!")

        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, num_samples_cls=median_size)

