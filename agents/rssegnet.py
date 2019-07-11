import numpy as np

from tqdm import tqdm
import shutil
import random

import logging

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

from agents.base import BaseAgent

# import your classes here
from datasets.rssrai import RssraiDataset
from utils.modelsummary import get_model_summary
from graphs.losses.criterion import OhemCrossEntropy, CrossEntropy
from graphs.models.unet import UNet

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics

logger = logging.getLogger(__name__)


class RsSegNetAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        logger.info(config)

        # cudnn related setting
        cudnn.benchmark = config.CUDNN.BENCHMARK
        cudnn.deterministic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED
        gpus = list(config.GPUS)

        writer_dict = {
            'writer': SummaryWriter(config.SUMMARY_DIR),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }

        self.checkpoint_dir = config.CHECKPOINT_DIR
        self.output_dir = config.OUTPUT_DIR

        self.model = UNet(3, config.DATASET.NUM_CLASSES)

        dump_input = torch.rand(
            (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
        )
        logger.info(get_model_summary(self.model.cuda(), dump_input.cuda()))

        self.train_dataset = RssraiDataset('train', config.DATASET)
        self.trainloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=True,
            drop_last=True)

        self.test_dataset = RssraiDataset('test', config.DATASET)
        self.testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=True)

        if config.LOSS.USE_OHEM:
            self.criterion = OhemCrossEntropy(thres=config.LOSS.OHEMTHRES,
                                                min_kept=config.LOSS.OHEMKEEP)
        else:
            self.criterion = CrossEntropy()

        if len(gpus) > 1:
            self.model = nn.DataParallel(self.model, device_ids=gpus).cuda()
        
        # optimizer
        if config.TRAIN.OPTIMIZER == 'sgd':
            self.optimizer = torch.optim.SGD([{'params':
                                    filter(lambda p: p.requires_grad,
                                            self.model.parameters()),
                                    'lr': config.TRAIN.LR}],
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV,
                                    )
        else:
            raise ValueError('Only Support SGD optimizer')


        self.start_epoch = 0
        self.end_epoch = config.TRAIN.EPOCH

        self.best_mIoU = -1



    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        pass

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=False):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        pass

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(self.start_epoch, self.end_epoch):
            self.train_one_epoch()
            self.save_checkpoint()
            valid_loss, mean_IoU, IoU_array = self.validate()
            if mean_IoU > self.best_mIoU:
                self.best_mIoU = mean_IoU
                self.save_checkpoint(file_name="best.pth.tar", is_best=True)
            msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                    valid_loss, mean_IoU, self.best_mIoU)
            logger.info(msg)
        self.save_checkpoint(file_name="final_state.pth.tar")

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.model.train()
        epoch_loss = AverageMeter()
        for i_iter, batch in enumerate(self.testloader):
            images, labels = batch[0].cuda(), batch[1].long().cuda()
            pred = self.model(images)
            loss = self.criterion(pred, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss.update(loss.item())
            logger.info(epoch_loss.val)
        

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        pass

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass
