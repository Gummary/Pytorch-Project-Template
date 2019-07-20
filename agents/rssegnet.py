import numpy as np

from tqdm import tqdm
import shutil
import random
import time
import os

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
from graphs.models.deeplab_xception import DeepLabv3_plus as DeepLabv3_plus_xce
from graphs.models.deeplab_resnet import DeepLabv3_plus as DeepLabv3_plus_res
from graphs.weights_initializer import init_model_weights

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList, IOUMetric
from utils.misc import print_cuda_statistics
from utils.train_utils import visualize_images

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

        self.writer_dict = {
            'writer': SummaryWriter(config.SUMMARY_DIR),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }

        self.checkpoint_dir = config.CHECKPOINT_DIR
        self.output_dir = config.OUTPUT_DIR

        self.num_classes = config.DATASET.NUM_CLASSES

        if config.MODEL.NAME == 'unet':
            self.model = UNet(3, config.DATASET.NUM_CLASSES)
        elif config.MODEL.NAME == 'dl_resnet':
            self.model = DeepLabv3_plus_res(3, 
            n_classes=config.DATASET.NUM_CLASSES, 
            os=config.MODEL.OS,
            pretrained=True)
        elif config.MODEL.NAME == 'dl_xception':
            self.model = DeepLabv3_plus_xce(3, 
            n_classes=config.DATASET.NUM_CLASSES, 
            os=config.MODEL.OS,
            pretrained=True)

        dump_input = torch.rand(
            (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
        )
        logger.info(get_model_summary(self.model.cuda(), dump_input.cuda()))

        self.train_dataset = RssraiDataset(
            'train', 
            config.DATASET, 
            mean=[0.2797, 0.3397, 0.3250], 
            std=[0.1271, 0.1446, 0.1320])
        self.trainloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=True,
            drop_last=True)

        self.test_dataset = RssraiDataset('val', config.DATASET,
            mean=[0.2797, 0.3397, 0.3250], 
            std=[0.1271, 0.1446, 0.1320])
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
        else:
            self.model.cuda()
        
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
        elif config.TRAIN.OPTIMIZER == 'adam':
            self.optimizer = torch.optim.Adam([{'params':
                                                filter(lambda p: p.requires_grad,
                                                        self.model.parameters()),
                                                'lr': config.TRAIN.LR}],
                                                lr = config.TRAIN.LR, 
                                                weight_decay=config.TRAIN.WD)
        else:
            raise ValueError('Only Support SGD optimizer')

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
        )


        self.start_epoch = 0
        self.current_epoch = self.start_epoch
        self.end_epoch = config.TRAIN.EPOCH

        init_model_weights(self.model) 
        if config.TRAIN.AUTO_RESUME:
            self.load_checkpoint(os.path.join(config.CHECKPOINT_DIR, 'checkpoint.pth.tar'))

        self.best_mIoU = -1



    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        if os.path.exists(file_name):
            logger.info("=> Loading weights from :"+file_name)
            checkpoint = torch.load(file_name)
            self.begin_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])

            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.current_epoch = checkpoint['epoch']
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                file_name, checkpoint['epoch']))
            

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=False):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        states = {
            'epoch': self.current_epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(states, os.path.join(self.checkpoint_dir, file_name))
        if is_best:
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "best.pth.tar"))

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
            self.lr_scheduler.step()
            self.train_one_epoch()
            self.current_epoch += 1
            self.save_checkpoint()
            valid_loss, mean_IoU = self.validate()
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
        data_time = AverageMeter()
        batch_time = AverageMeter()
        end = time.time()
        metrics = IOUMetric(self.num_classes)
        for i_iter, batch in enumerate(self.trainloader):

            data_time.update(time.time() - end)
            images, labels = batch[0].cuda(), batch[1].long().cuda()
            pred = self.model(images)
            loss = self.criterion(pred, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss.update(loss.item())
            batch_time.update(time.time() - end)

            _, pred_max = torch.max(pred, 1)
            metrics.add_batch(pred_max.cpu().numpy(), labels.cpu().numpy())
            if i_iter % 20 == 0:
                epoch_acc, _, epoch_iou_class, epoch_mean_iou, _ = metrics.evaluate()
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                    'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                    'MIOU {epoch_mean_iou}'.format(
                        self.current_epoch, i_iter, len(self.trainloader), batch_time=batch_time,
                        data_time=data_time, loss=epoch_loss, epoch_mean_iou=epoch_mean_iou)

                writer = self.writer_dict['writer']
                global_steps = self.writer_dict['train_global_steps']
                writer.add_scalar('train_loss', epoch_loss.val, global_steps)
                writer.add_scalar('train_acc', epoch_acc, global_steps)
                writer.add_scalar('train_mean_iou', epoch_mean_iou, global_steps)
                self.writer_dict['train_global_steps'] = global_steps + 1
                
                logger.info(msg)
        

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        ave_loss = AverageMeter()
        metrics = IOUMetric(self.num_classes)
        with torch.no_grad():
            for _, batch in enumerate(self.testloader):
                images, labels = batch
                size = labels.size()
                images = images.cuda()
                labels = labels.long().cuda()

                pred = self.model(images)
                losses = self.criterion(pred, labels)
                # pred = F.upsample(input=pred, size=(
                #             size[-2], size[-1]), mode='bilinear')
                loss = losses.mean()
                ave_loss.update(loss.item())

                _, pred_max = torch.max(pred, 1)
                metrics.add_batch(pred_max.cpu().numpy(), labels.cpu().numpy())

        epoch_acc, _, epoch_iou_class, epoch_mean_iou, _ = metrics.evaluate()

        writer = self.writer_dict['writer']
        global_steps = self.writer_dict['valid_global_steps']
        writer.add_scalar('val_loss', ave_loss.val, global_steps)
        writer.add_scalar('val_acc', epoch_acc, global_steps)
        writer.add_scalar('val_mean_iou', epoch_mean_iou, global_steps)
        self.writer_dict['valid_global_steps'] = global_steps + 1

        return ave_loss.val, epoch_mean_iou

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass
