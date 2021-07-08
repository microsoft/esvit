from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

# high_resoluton_net related params for classification
HIGH_RESOLUTION_NET = CN()
HIGH_RESOLUTION_NET.PRETRAINED_LAYERS = ['*']
HIGH_RESOLUTION_NET.STEM_INPLANES = 64
HIGH_RESOLUTION_NET.FINAL_CONV_KERNEL = 1
HIGH_RESOLUTION_NET.WITH_HEAD = True

HIGH_RESOLUTION_NET.STAGE2 = CN()
HIGH_RESOLUTION_NET.STAGE2.NUM_MODULES = 1
HIGH_RESOLUTION_NET.STAGE2.NUM_BRANCHES = 2
HIGH_RESOLUTION_NET.STAGE2.NUM_BLOCKS = [4, 4]
HIGH_RESOLUTION_NET.STAGE2.NUM_CHANNELS = [32, 64]
HIGH_RESOLUTION_NET.STAGE2.BLOCK = 'BASIC'
HIGH_RESOLUTION_NET.STAGE2.FUSE_METHOD = 'CAT'

HIGH_RESOLUTION_NET.STAGE3 = CN()
HIGH_RESOLUTION_NET.STAGE3.NUM_MODULES = 1
HIGH_RESOLUTION_NET.STAGE3.NUM_BRANCHES = 3
HIGH_RESOLUTION_NET.STAGE3.NUM_BLOCKS = [4, 4, 4]
HIGH_RESOLUTION_NET.STAGE3.NUM_CHANNELS = [32, 64, 128]
HIGH_RESOLUTION_NET.STAGE3.BLOCK = 'BASIC'
HIGH_RESOLUTION_NET.STAGE3.FUSE_METHOD = 'CAT'

HIGH_RESOLUTION_NET.STAGE4 = CN()
HIGH_RESOLUTION_NET.STAGE4.NUM_MODULES = 1
HIGH_RESOLUTION_NET.STAGE4.NUM_BRANCHES = 4
HIGH_RESOLUTION_NET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
HIGH_RESOLUTION_NET.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
HIGH_RESOLUTION_NET.STAGE4.BLOCK = 'BASIC'
HIGH_RESOLUTION_NET.STAGE4.FUSE_METHOD = 'CAT'

RESNEXT = CN()
RESNEXT.NUM_LAYERS = 50
RESNEXT.BASE_WIDTH = 4
RESNEXT.CARDINALITY = 32
RESNEXT.KERNEL_SIZE_STEM = 7

RESNET = CN()
RESNET.NUM_LAYERS = 50
RESNET.KERNEL_SIZE_STEM = 7


MODEL_SPECS = {
    'cls_hrnet': HIGH_RESOLUTION_NET,
}


def get_model_name(config):
    name = ''
    spec = config.MODEL.SPEC
    if config.MODEL.NAME in ['cls_resnet', 'cls_resnet_d2']:
        num_groups = spec.NUM_GROUPS
        depth = spec.NUM_LAYERS
        if num_groups == 1:
            model_type = 'r{}'.format(depth)
        else:
            model_type = 'x{}-{}x{}d'.format(
                depth, num_groups, spec.WIDTH_PER_GROUP
            )
        if 'DEEP_STEM' in spec and spec.DEEP_STEM:
            name = '{}-deepstemAvgdown{}'.format(
                model_type,
                int(spec.AVG_DOWN)
            )
        else:
            name = '{}-s{}a{}'.format(
                model_type,
                spec.KERNEL_SIZE_STEM,
                int(spec.AVG_DOWN)
            )
        if 'WITH_SE' in spec:
            name = 'se-' + name
    elif 'cls_hrnet' in config.MODEL.NAME:
        name = 'h{}'.format(spec.STAGES_SPEC.NUM_CHANNELS[0][0])
    elif config.MODEL.NAME == 'cls_bit_resnet':
        name = '{}'.format(spec.SPEC)
    else:
        raise ValueError('Known MODEL.NAME: {}'.format(config.MODEL.NAME))

    return name
