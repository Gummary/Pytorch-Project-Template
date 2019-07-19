from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN

_C = CN() 





_C.EXP_DIR = 'experiments'
_C.EXPNAME = ''
_C.LOG_DIR = 'logs'
_C.SUMMARY_DIR = 'summary'
_C.CHECKPOINT_DIR = 'checkpoint'
_C.OUTPUT_DIR = 'output'
_C.AGENT = 'RsSegNetAgent'
_C.WORKERS = 4

_C.GPUS=(0,)

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'unet'
_C.MODEL.PRETRAINED = ''
_C.MODEL.OS=16


_C.DATASET = CN()
_C.DATASET.ROOT = 'data/'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.VAL_SET = 'val'
_C.DATASET.TEST_SET = 'test'
_C.DATASET.NUM_CLASSES = 16

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE_PER_GPU = 8
_C.TRAIN.SHUFFLE = True
_C.TRAIN.IMAGE_SIZE = (256, 256)
_C.TRAIN.AUTO_RESUME = True

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.01
_C.TRAIN.EXTRA_LR = 0.001

_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False

_C.TRAIN.EPOCH = 200


_C.TEST = CN()
_C.TEST.BATCH_SIZE_PER_GPU = 8


_C.LOSS = CN()
_C.LOSS.USE_OHEM = True
_C.LOSS.OHEMTHRES = 0.9
_C.LOSS.OHEMKEEP = 100000
_C.LOSS.CLASS_BALANCE = False



config = _C

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)

    cfg.LOG_DIR = os.path.join(
        cfg.EXP_DIR, cfg.EXPNAME, cfg.LOG_DIR
    )

    cfg.SUMMARY_DIR = os.path.join(
        cfg.EXP_DIR, cfg.EXPNAME, cfg.SUMMARY_DIR
    )

    cfg.CHECKPOINT_DIR = os.path.join(
        cfg.EXP_DIR, cfg.EXPNAME, cfg.CHECKPOINT_DIR
    )

    cfg.OUTPUT_DIR = os.path.join(
        cfg.EXP_DIR, cfg.EXPNAME, cfg.OUTPUT_DIR
    )

    # if args.modelDir:
    #     cfg.OUTPUT_DIR = args.modelDir

    # if args.dataDir:
    #     cfg.DATA_DIR = args.dataDir

    # cfg.DATASET.ROOT = os.path.join(
    #     cfg.DATA_DIR, cfg.DATASET.ROOT
    # )

    cfg.freeze()
    return cfg


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

