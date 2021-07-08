import logging
import os
import os.path as op
import json
from typing import List
from .blob_storage import BlobStorage, disk_usage


def generate_lineidx(filein: str, idxout: str) -> None:
    idxout_tmp = idxout + '.tmp'
    with open(filein, 'r') as tsvin, open(idxout_tmp, 'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        while fpos != fsize:
            tsvout.write(str(fpos) + "\n")
            tsvin.readline()
            fpos = tsvin.tell()
    os.rename(idxout_tmp, idxout)


def read_to_character(fp, c):
    result = []
    while True:
        s = fp.read(32)
        assert s != ''
        if c in s:
            result.append(s[: s.index(c)])
            break
        else:
            result.append(s)
    return ''.join(result)


class TSVFile(object):
    def __init__(self,
                 tsv_file: str,
                 if_generate_lineidx: bool = False,
                 lineidx: str = None,
                 class_selector: List[str] = None,
                 blob_storage: BlobStorage = None):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx' \
            if not lineidx else lineidx
        self.linelist = op.splitext(tsv_file)[0] + '.linelist'
        self.chunks = op.splitext(tsv_file)[0] + '.chunks'
        self._fp = None
        self._lineidx = None
        self._sample_indices = None
        self._class_boundaries = None
        self._class_selector = class_selector
        self._blob_storage = blob_storage
        # the process always keeps the process which opens the file.
        # If the pid is not equal to the currrent pid, we will re-open the file.
        self.pid = None
        # generate lineidx if not exist
        if not op.isfile(self.lineidx) and if_generate_lineidx:
            generate_lineidx(self.tsv_file, self.lineidx)

    def __del__(self):
        if self._fp:
            self._fp.close()
            # physically remove the tsv file if it is retrieved by BlobStorage
            if self._blob_storage and 'azcopy' in self.tsv_file and os.path.exists(self.tsv_file):
                try:
                    original_usage = disk_usage('/')
                    os.remove(self.tsv_file)
                    logging.info("Purged %s (disk usage: %.2f%% => %.2f%%)" %
                                 (self.tsv_file, original_usage, disk_usage('/') * 100))
                except:
                    # Known issue: multiple threads attempting to delete the file will raise a FileNotFound error.
                    # TODO: try Threadling.Lock to better handle the race condition
                    pass

    def __str__(self):
        return "TSVFile(tsv_file='{}')".format(self.tsv_file)

    def __repr__(self):
        return str(self)

    def get_class_boundaries(self):
        return self._class_boundaries

    def num_rows(self):
        self._ensure_lineidx_loaded()
        return len(self._sample_indices)

    def seek(self, idx: int):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        try:
            pos = self._lineidx[self._sample_indices[idx]]
        except:
            logging.info('=> {}-{}'.format(self.tsv_file, idx))
            raise
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def seek_first_column(self, idx: int):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return read_to_character(self._fp, '\t')

    def get_key(self, idx: int):
        return self.seek_first_column(idx)

    def __getitem__(self, index: int):
        return self.seek(index)

    def __len__(self):
        return self.num_rows()

    def _ensure_lineidx_loaded(self):
        if self._lineidx is None:
            logging.debug('=> loading lineidx: {}'.format(self.lineidx))
            with open(self.lineidx, 'r') as fp:
                lines = fp.readlines()
                lines = [line.strip() for line in lines]
                self._lineidx = [int(line) for line in lines]

            # read the line list if exists
            linelist = None
            if op.isfile(self.linelist):
                with open(self.linelist, 'r') as fp:
                    linelist = sorted(
                        [
                            int(line.strip())
                            for line in fp.readlines()
                        ]
                    )

            if op.isfile(self.chunks):
                self._sample_indices = []
                self._class_boundaries = []
                class_boundaries = json.load(open(self.chunks, 'r'))
                for class_name, boundary in class_boundaries.items():
                    start = len(self._sample_indices)
                    if class_name in self._class_selector:
                        for idx in range(boundary[0], boundary[1] + 1):
                            # NOTE: potentially slow when linelist is long, try to speed it up
                            if linelist and idx not in linelist:
                                continue
                            self._sample_indices.append(idx)
                    end = len(self._sample_indices)
                    self._class_boundaries.append((start, end))
            else:
                if linelist:
                    self._sample_indices = linelist
                else:
                    self._sample_indices = list(range(len(self._lineidx)))

    def _ensure_tsv_opened(self):
        if self._fp is None:
            if self._blob_storage:
                self._fp = self._blob_storage.open(self.tsv_file)
            else:
                self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

        if self.pid != os.getpid():
            logging.debug('=> re-open {} because the process id changed'.format(self.tsv_file))
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()


class CompositeTSVFile:
    def __init__(self,
                 file_list: List[str],
                 root: str = '.',
                 class_selector: List[str] = None,
                 is_train: bool = True,
                 sas_token_path: str = None):
        self.root = root
        self.tsvs = None
        self.chunk_sizes = None
        self.accum_chunk_sizes = None
        self._class_selector = class_selector
        self._class_boundaries = None
        self.initialized = False
        assert isinstance(file_list, list)
        self.blob_storage = BlobStorage(is_train, sas_token_path)
        self.file_list = self.blob_storage.register_local_tsv_paths(file_list)
        self.initialize()

    def get_key(self, index: int):
        idx_source, idx_row = self._calc_chunk_idx_row(index)
        k = self.tsvs[idx_source].get_key(idx_row)
        return '_'.join([self.file_list[idx_source], k])

    def get_class_boundaries(self):
        return self._class_boundaries

    def get_chunk_size(self):
        return self.chunk_sizes

    def num_rows(self):
        return sum(self.chunk_sizes)

    def _calc_chunk_idx_row(self, index: int):
        idx_chunk = 0
        idx_row = index
        while index >= self.accum_chunk_sizes[idx_chunk]:
            idx_chunk += 1
            idx_row = index - self.accum_chunk_sizes[idx_chunk-1]
        return idx_chunk, idx_row

    def __getitem__(self, index: int):
        idx_source, idx_row = self._calc_chunk_idx_row(index)
        if idx_source not in self.blob_storage:
            self.blob_storage[idx_source] = TSVFile(
                op.join(self.root, self.file_list[idx_source]),
                class_selector=self._class_selector,
                blob_storage=self.blob_storage
            )
        return self.blob_storage[idx_source].seek(idx_row)

    def __len__(self):
        return sum(self.chunk_sizes)

    def initialize(self):
        """
        this function has to be called in init function if cache_policy is
        enabled. Thus, let's always call it in init funciton to make it simple.
        """
        if self.initialized:
            return
        self.tsvs = [
            TSVFile(
                op.join(self.root, f),
                class_selector=self._class_selector
            ) for f in self.file_list
        ]
        logging.debug("=> Calculating chunk sizes ...")
        self.chunk_sizes = [len(tsv) for tsv in self.tsvs]

        self.accum_chunk_sizes = [0]
        for size in self.chunk_sizes:
            self.accum_chunk_sizes += [self.accum_chunk_sizes[-1] + size]
        self.accum_chunk_sizes = self.accum_chunk_sizes[1:]

        if (
            self._class_selector
            and all([tsv.get_class_boundaries() for tsv in self.tsvs])
        ):
            """
            Note: When using CompositeTSVFile, make sure that the classes contained in each
            tsv file do not overlap. Otherwise, the class boundaries won't be correct.
            """
            self._class_boundaries = []
            offset = 0
            for tsv in self.tsvs:
                boundaries = tsv.get_class_boundaries()
                for bound in boundaries:
                    self._class_boundaries.append((bound[0] + offset, bound[1] + offset))
                offset += len(tsv)
        self.initialized = True


def load_list_file(fname: str) -> List[str]:
    with open(fname, 'r') as fp:
        lines = fp.readlines()
    result = [line.strip() for line in lines]
    if len(result) > 0 and result[-1] == '':
        result = result[:-1]
    return result
