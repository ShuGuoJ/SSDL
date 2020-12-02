import torch
from torch.nn import init
from torch import nn
import os
from scipy.io import loadmat

def weight_init(m):
    if isinstance(m, nn.Linear):
        init.uniform_(m.weight, -1, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        init.uniform_(m.weight, -1, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

def loadLabel(path):
    '''
    :param path:
    :return: 训练样本标签， 测试样本标签
    '''
    assert os.path.exists(path), '{},路径不存在'.format(path)
    # keys:{train_gt, test_gt}
    gt = loadmat(path)
    return gt['train_gt'], gt['test_gt']