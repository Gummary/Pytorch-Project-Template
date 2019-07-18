import os

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

import argparse
from utils.logger import setup_logging
from configs.default import update_config
from configs import config
from datasets.rssrai import RssraiDataset


from agents import *

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        '--cfg',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    return parser.parse_args()

def decode_segmap(image, l2p_mapping, nc=16):

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in np.unique(image):
        idx = image == l
        r[idx] = l2p_mapping[l][0]
        g[idx] = l2p_mapping[l][1]
        b[idx] = l2p_mapping[l][2]
        
    rgb = np.stack([r, g, b], axis=2)
    return rgb
    

def main():
    global config
    args = parse_args()
    config = update_config(config, args)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    test_dataset = RssraiDataset('val', config.DATASET)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        drop_last=True)

    model = UNet(3, config.DATASET.NUM_CLASSES)
    model.load_state_dict(torch.load("experiments\\unet_test1\\checkpoint\\best.pth.tar"))
    model.cuda()
    model.eval()

    for batch in testloader:
        images, labels = batch[0].cuda(), batch[1].long().cuda()

        pred = model(images)
        print(pred.shape)
        print(images.shape)
        om = torch.argmax(pred.squeeze(), dim=0).detach().cpu().numpy()
        rgb = decode_segmap(om, test_dataset.lbl2pixel, 16)
        fig = plt.figure()
        ax1 = fig.add_subplot(1,3,1)
        ax1.imshow(np.transpose(images[0].detach().cpu().numpy(), (1,2,0)))
        ax2 = fig.add_subplot(1,3,2)
        ax2.imshow(labels[0].detach().cpu().numpy(),cmap='gray', vmin=0, vmax=16)
        ax3 = fig.add_subplot(1,3,3)
        ax3.imshow(rgb)
        plt.show()
        break

    


if __name__ == '__main__':
    main()
