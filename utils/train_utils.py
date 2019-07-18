import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import math

import matplotlib.pyplot as plt

import numpy as np

"""
Learning rate adjustment used for CondenseNet model training
"""
def adjust_learning_rate(optimizer, epoch, config, batch=None, nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = config.max_epoch * nBatch
        T_cur = (epoch % config.max_epoch) * nBatch + batch
        lr = 0.5 * config.learning_rate * (1 + math.cos(math.pi * T_cur / T_total))
    else:
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = config.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def visualize_images(images):
    num_images = len(images)
    num_cols = 3
    num_rows = num_images // num_cols + 1
    figure = plt.figure()    
    for i in range(num_images):
        ax = figure.add_subplot(num_rows, num_cols, i+1)
        ax.imshow(images[i])
    plt.show()



