import math
import torch
from torch.utils.data import Sampler
import torch.distributed as dist
import random
import logging
import threading


def pre_fetch(fn_fetch, index):
    logging.debug(f'Pre-loading file index: {index} ...')
    fn_fetch(index)
    logging.debug(f'Pre-loading ended file index: {index} ...')


class DistributedChunkSampler(Sampler):
    def __init__(self,
                 dataset,
                 chunk_sizes=None,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 shuffle_chunk=False):
        if num_replicas is None:
            try:
                num_replicas = dist.get_world_size()
            except:
                num_replicas = torch.cuda.device_count()
        if rank is None:
            try:
                rank = dist.get_rank()
            except:
                rank = torch.cuda.current_device()
        if chunk_sizes is None:
            logging.info("[DistributedChunkSampler] No chunk size specified. Reduce to normal distributed sampler.")
            chunk_sizes = [len(dataset)]
        if torch.cuda.is_available():
            self.gpus_per_node = torch.cuda.device_count()
        self.dataset = dataset
        self.num_replicas = num_replicas  # num of GPUs
        self.rank = rank  # GPU id
        self.chunk_sizes = chunk_sizes
        self.min_chunk_size = min(self.chunk_sizes) - (min(self.chunk_sizes) % self.gpus_per_node)
        # logging.info("[DistributedChunkSampler] min chunk size: %s", self.min_chunk_size)
        self.epoch = 0
        self.num_samples = int(
            math.ceil(
                (len(self.chunk_sizes) * self.min_chunk_size) * 1.0 / self.num_replicas
            )
        )  # num of samples per GPU
        self.total_size = self.num_samples * self.num_replicas
        logging.info(
            "\n[DistributedChunkSampler]"
            "\n\t rank: {}"
            "\n\t num_replicas: {}"
            "\n\t gpus_per_node: {}"
            "\n\t chunk_sizes: {}"
            "\n\t num_samples per gpu: {}"
            "\n\t total size: {}"
            .format(
                rank,
                self.num_replicas,
                self.gpus_per_node,
                self.chunk_sizes,
                self.num_samples,
                self.total_size
            )
        )
        self.shuffle = shuffle
        self.shuffle_chunk = shuffle_chunk
        self.indices = None

    def _shuffle_chunk_elements(self, chunk_indices):
        """
        Generate randomly shuffled indices chunk-by-chunk.
        The generated indices are randomized in both chunk- and instance-level.

        Example::
        Input:
            chunk_size: [100, 100, 100, 100, 100]
            accum_chunk_sizes: [0, 100, 200, 300, 400, 500]
            chunk_indices: [1, 3, 2, 5, 4]
        Output:
            [12, 47, 29, ...
            283, 247, 212, ...
            192, 148, 183, ...
            482, 457, 431, ...
            314, 367, 352, ...]
        """
        accum_chunk_sizes = [0]
        for size in self.chunk_sizes:
            accum_chunk_sizes += [accum_chunk_sizes[-1] + size]

        # In case that the data size is greater than local cache (e.g., blobfuse),
        # reverse the order of consuming data between epochs to reduce the impact of cache miss.
        num_nodes = int(self.num_replicas / self.gpus_per_node)
        num_tsvs = int(len(chunk_indices) / num_nodes)
        if self.epoch % 2:
            for i in range(num_nodes):
                chunk_indices[i*num_tsvs:(i+1)*num_tsvs] = chunk_indices[
                    i*num_tsvs:(i+1)*num_tsvs][::-1]

        logging.info(
            "\n[DistributedChunkSampler]"
            "\n\t epoch: {}"
            "\n\t chunk indices: {}"
            .format(
                self.epoch,
                chunk_indices
            )
        )

        indices = []
        for idx in range(len(chunk_indices)):
            shuffled_chunk_elements = list(
                range(
                    accum_chunk_sizes[chunk_indices[idx] - 1],
                    accum_chunk_sizes[chunk_indices[idx]]
                )
            )
            random.shuffle(shuffled_chunk_elements)
            shuffled_chunk_elements = shuffled_chunk_elements[
                :self.min_chunk_size
            ]
            # insert tsv file index for pre-loading, skip the last tsv file
            if (idx+1) % num_tsvs:
                print('idx: {}'.format(idx))
                print('chunk indices: {}'.format(chunk_indices[idx]))
                print('min_chun_size: {}'.format(self.min_chunk_size))
                print('insert prefetch flag')
                print('len of chunk elements: {}'.format(
                    len(shuffled_chunk_elements))
                )
                shuffled_chunk_elements[0] = (
                    shuffled_chunk_elements[0],
                    chunk_indices[min(idx + 1, len(chunk_indices) - 1)] - 1,
                    False
                )
            if idx % num_tsvs == 0:
                print('insert prefetch flag')
                shuffled_chunk_elements[1] = (
                    shuffled_chunk_elements[1],
                    chunk_indices[idx] - 1,
                    True
                )
            indices += shuffled_chunk_elements

        return indices

    def __iter__(self):
        for item in self.indices:
            if isinstance(item, tuple):
                index = item[0]
                index_chunk = item[1]
                if item[2]:
                    pre_fetch(
                        self.dataset.fetch_blob,
                        index_chunk
                    )
                else:
                    x = threading.Thread(
                        target=pre_fetch,
                        args=(
                            self.dataset.fetch_blob,
                            index_chunk
                        ),
                        daemon=True
                    )
                    x.start()
            else:
                index = item
            yield index

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        # Deterministically shuffle based on epoch
        self.epoch = epoch
        random.seed(self.epoch)

        if self.shuffle:
            chunk_indices = list(range(1, len(self.chunk_sizes) + 1))
            if self.shuffle_chunk:
                random.shuffle(chunk_indices)
            self.indices = self._shuffle_chunk_elements(chunk_indices)
        else:
            self.indices = list(range(len(self.dataset)))
            self.indices = self.indices[:self.total_size]

        assert len(self.indices) == self.total_size, \
            'indices: {} vs total_size: {}'.format(
                len(self.indices), self.total_size
            )

        # Subsample
        rank = self.rank % self.gpus_per_node
        node_idx = int(self.rank / self.gpus_per_node)
        logging.info(
            "[DistributedChunkSampler] global rank/local rank/node_idx: %d/%d/%d",
            self.rank, rank, node_idx
        )
        idx_start = self.gpus_per_node * node_idx * self.num_samples
        idx_end = self.gpus_per_node * (node_idx + 1) * self.num_samples
        self.indices = self.indices[idx_start:idx_end]
        idx_start = rank
        idx_end = self.num_samples * self.gpus_per_node
        idx_step = self.gpus_per_node
        self.indices = self.indices[idx_start:idx_end:idx_step]

        assert len(self.indices) == self.num_samples, \
            'indices: {} vs num_samples: {}'.format(
                len(self.indices), self.num_samples
            )
