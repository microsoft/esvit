from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as op
import yaml
from yacs.config import CfgNode as CN

from .models import get_model_name


_C = CN()

_C.BASE = ['']
_C.NAME = ''
_C.DATA_DIR = ''
_C.DIST_BACKEND = 'nccl'
_C.GPUS = (0,)
# _C.LOG_DIR = ''
_C.MULTIPROCESSING_DISTRIBUTED = True
_C.OUTPUT_DIR = ''
_C.PIN_MEMORY = True
_C.PRINT_FREQ = 20
_C.RANK = 0
_C.VERBOSE = True
_C.WORKERS = 4
_C.MODEL_SUMMARY = False

_C.AMP = CN()
_C.AMP.ENABLED = False
_C.AMP.MEMORY_FORMAT = 'nchw'

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'cls_hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.PRETRAINED_LAYERS = ['*']
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.SPEC = CN(new_allowed=True)

_C.LOSS = CN(new_allowed=True)
_C.LOSS.LABEL_SMOOTHING = 0.0
_C.LOSS.LOSS = 'softmax'
_C.LOSS.FOCAL = CN()
_C.LOSS.FOCAL.NORMALIZE = True
_C.LOSS.FOCAL.ALPHA = 1.0
_C.LOSS.FOCAL.GAMMA = 0.5


# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'imagenet'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.LABELMAP = ''
_C.DATASET.TRAIN_TSV_LIST = []
_C.DATASET.TEST_TSV_LIST = []
_C.DATASET.COCO = CN(new_allowed=True)
_C.DATASET.COCO.SCALES = ['m', 'l']
_C.DATASET.COCO.BALANCE_DATA = True
_C.DATASET.LOADER = 'blobfuse'  # available options: "blobfuse" and "azcopy"
_C.DATASET.SAMPLER = 'default'    # available options: 'default', 'class_aware', 'class_aware_target_size', 'chunk'
_C.DATASET.NUM_SAMPLES_CLASS = 'average'  # 'average', 'median' or any integer

# Used by ClassAwareTargetSizeSampler. Set to the desired dataset size
# Or by default, sample all available data.
_C.DATASET.TARGET_SIZE = -1

# training data augmentation
_C.INPUT = CN()
_C.INPUT.MEAN = [0.485, 0.456, 0.406]
_C.INPUT.STD = [0.229, 0.224, 0.225]

# data augmentation
_C.AUG = CN()
_C.AUG.SCALE = (0.08, 1.0)
_C.AUG.RATIO = (3.0/4.0, 4.0/3.0)
_C.AUG.COLOR_JITTER = [0.4, 0.4, 0.4, 0.1, 0.0]
_C.AUG.GRAY_SCALE = 0.0
_C.AUG.GAUSSIAN_BLUR = 0.0
_C.AUG.DROPBLOCK_LAYERS = [3, 4]
_C.AUG.DROPBLOCK_KEEP_PROB = 1.0
_C.AUG.DROPBLOCK_BLOCK_SIZE = 7
_C.AUG.MIXUP_PROB = 0.0
_C.AUG.MIXUP = 0.0
_C.AUG.MIXCUT = 0.0
_C.AUG.MIXCUT_MINMAX = []
_C.AUG.MIXUP_SWITCH_PROB = 0.5
_C.AUG.MIXUP_MODE = 'batch'
_C.AUG.MIXCUT_AND_MIXUP = False
_C.AUG.INTERPOLATION = 2
_C.AUG.TIMM_AUG = CN(new_allowed=True)
_C.AUG.TIMM_AUG.USE_LOADER = False
_C.AUG.TIMM_AUG.USE_TRANSFORM = False

_C.SWA = CN()
_C.SWA.ENABLED = False
_C.SWA.DEVICE = 'cpu'
_C.SWA.BEGIN_EPOCH = -1
_C.SWA.LR_RATIO = 0.5
_C.SWA.ANNEAL_EPOCHS = 10
_C.SWA.ANNEAL_STRATEGY = 'cos'
_C.SWA.FROZEN_BN = False

# train
_C.TRAIN = CN()

_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.CHECKPOINT = ''
_C.TRAIN.LR_SCHEDULER = CN(new_allowed=True)
# _C.TRAIN.LR_FACTOR = 0.1
# _C.TRAIN.LR_STEP = [30, 60, 90]
_C.TRAIN.LR = 0.001

_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.OPTIMIZER_ARGS = CN(new_allowed=True)
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.WITHOUT_WD_LIST = []
_C.TRAIN.NESTEROV = True
# for adam
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 100

_C.TRAIN.IMAGE_SIZE = [224, 224]  # width * height, ex: 192 * 256
_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

_C.TRAIN.EMA_DECAY = 0.0
_C.TRAIN.EVAL_BEGIN_EPOCH = 0

_C.TRAIN.LARC = False

_C.TRAIN.DETECT_ANOMALY = False

_C.TRAIN.CLIP_GRAD_NORM = 0.0

_C.TRAIN.SAVE_ALL_MODELS = False

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
_C.TEST.CENTER_CROP = True
_C.TEST.IMAGE_SIZE = [224, 224]  # width * height, ex: 192 * 256
_C.TEST.INTERPOLATION = 2
_C.TEST.MODEL_FILE = ''
_C.TEST.REAL_LABELS = False
_C.TEST.VALID_LABELS = ''

_C.FINETUNE = CN()
_C.FINETUNE.FINETUNE = False
_C.FINETUNE.USE_TRAIN_AUG = False
_C.FINETUNE.BASE_LR = 0.003
_C.FINETUNE.BATCH_SIZE = 512
_C.FINETUNE.EVAL_EVERY = 3000
_C.FINETUNE.TRAIN_MODE = True
# _C.FINETUNE.MODEL_FILE = ''
_C.FINETUNE.FROZEN_LAYERS = []
_C.FINETUNE.LR_SCHEDULER = CN(new_allowed=True)
_C.FINETUNE.LR_SCHEDULER.DECAY_TYPE = 'step'

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False


def get_lr_scheduler_name(config):
    template = '{method}{setting}Ep{ep}'
    lr_scheduler = config.TRAIN.LR_SCHEDULER
    if lr_scheduler.METHOD == 'MultiStep':
        setting = '{}factor{}'.format(
            '_'.join([str(i) for i in lr_scheduler.MILESTONES]),
            lr_scheduler.GAMMA
        )
    elif lr_scheduler.METHOD == 'CosineAnnealing':
        setting = 'eta{}'.format(
            lr_scheduler.ETA_MIN,

        )
    elif lr_scheduler.METHOD == 'BitHyperRule':
        setting = ''
    elif lr_scheduler.METHOD == 'WarmupCosine':
        setting = '{}'.format(
            lr_scheduler.WARMUP_EPOCH
        )
    else:
        raise ValueError('Unknown lr scheduler: {}'.format(
            lr_scheduler.METHOD))

    return template.format(
        method=lr_scheduler.METHOD,
        setting=setting,
        ep=config.TRAIN.END_EPOCH
    )


def get_train_name(config, args):
    mixup = config.AUG.MIXUP
    mixcut = config.AUG.MIXCUT

    aug = ''
    if mixup > 0.0:
        aug = f'mixup{mixup}'
    elif mixcut > 0.0:
        aug = f'mixcut{mixcut}'
    template = 'labelsmooth{ls}{aug}{optim}-lr{lr:.5g}wd{wd}bs{bs}X{world_size}-{lr_scheduler}'

    return template.format(
        ls=config.LOSS.LABEL_SMOOTHING,
        optim=config.TRAIN.OPTIMIZER,
        lr=config.TRAIN.LR,
        wd=config.TRAIN.WD,
        aug=aug,
        bs=config.TRAIN.BATCH_SIZE_PER_GPU,
        world_size=args.world_size,
        lr_scheduler=get_lr_scheduler_name(config)
    )


def get_config_name(config):
    template = '{model_name}-{input_size}-{aug}-{train_cfg}'
    model_name = get_model_name(config)
    w, h = config.TRAIN.IMAGE_SIZE
    input_size = '{}x{}'.format(w, h)
    aug = ''
    cj = config.AUG.COLOR_JITTER
    if cj[-1] > 0.0:
        aug += 'cj{}'.format('_'.join([str(i) for i in cj]))
    gs = config.AUG.GRAY_SCALE
    if gs > 0.0:
        aug += 'gs{}'.format(gs)
    gb = config.AUG.GAUSSIAN_BLUR
    if gs > 0.0:
        aug += 'gb{}'.format(gb)
    db = config.AUG.DROPBLOCK_KEEP_PROB
    if db < 1.0:
        aug += 'db{}'.format(db)

    train_cfg = get_train_name(config)

    ema_decay = config.TRAIN.EMA_DECAY
    if ema_decay > 0.0:
        template = template + 'ema{}'.format(ema_decay)

    return template.format(
        model_name=model_name,
        input_size=input_size,
        aug=aug,
        train_cfg=train_cfg)


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, op.join(op.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    config.merge_from_list(args.opts)
    file_name, _ = op.splitext(op.basename(args.cfg))
    config.NAME = file_name + config.NAME
    config.RANK = args.rank

    if 'timm' == config.TRAIN.LR_SCHEDULER.METHOD:
        config.TRAIN.LR_SCHEDULER.ARGS.epochs = config.TRAIN.END_EPOCH

    if 'timm' == config.TRAIN.OPTIMIZER:
        config.TRAIN.OPTIMIZER_ARGS.lr = config.TRAIN.LR

    aug = config.AUG
    if aug.MIXUP > 0.0 or aug.MIXCUT > 0.0 or aug.MIXCUT_MINMAX:
        aug.MIXUP_PROB = 1.0
    config.freeze()


def save_config(cfg, path, args):
    if args.is_main_process():
        with open(path, 'w') as f:
            f.write(cfg.dump())


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

