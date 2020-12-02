import torch
from torch import nn
from torch.nn import functional as F
import os
# 空间金字塔池化卷积网络
class SPCnn(nn.Module):
    def __init__(self):
        super(SPCnn, self).__init__()
        # 特征提取器
        # 1
        self.feature = nn.Sequential()
        self.feature.add_module('c1', nn.Conv2d(1, 1, 3, 1, 1))
        self.feature.add_module('b1', nn.BatchNorm2d(1))
        self.feature.add_module('r1', nn.ReLU())
        self.feature.add_module('s1', nn.MaxPool2d(2, 2))
        # 2
        self.feature.add_module('c2', nn.Conv2d(1, 2, 3, 1, 1))
        self.feature.add_module('b2', nn.BatchNorm2d(2))
        self.feature.add_module('r2', nn.ReLU())
        self.feature.add_module('s2', nn.MaxPool2d(2, 2))
        # 3
        self.feature.add_module('c3', nn.Conv2d(2, 2, 3, 1, 1))
        self.feature.add_module('b3', nn.BatchNorm2d(2))
        self.feature.add_module('r3', nn.ReLU())

    def forward(self, x):
        '''
        :param x:[batchsz, h, w, 1]
        :return:
        '''
        x = self.feature(x) # [bachsz, 2, 9, 9]
        pooling_4x4 = F.adaptive_max_pool2d(x, (4,4)) # [batchsz, 4, 4, 2]
        pooling_3x3 = F.adaptive_max_pool2d(x, (3,3)) # [batchsz, 3, 3, 2]
        pooling_2x2 = F.adaptive_max_pool2d(x, (2,2)) # [batchsz, 2, 2, 2]
        pooling_1x1 = F.adaptive_max_pool2d(x, (1,1)) # [batchsz, 1, 1, 2]
        pooling = [pooling_4x4.view((-1, 4*4*2)), pooling_3x3.view((-1, 3*3*2)), \
                   pooling_2x2.view((-1, 2*2*2)), pooling_1x1.view((-1, 1*1*2))]
        o = torch.cat(pooling, dim=-1) # [batchsz, 60]
        return o
# 编码器
class Encoder(nn.Module):
    def __init__(self, units):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential()
        last = 0
        for i, u in enumerate(units):
            if i != 0:
                self.encoder.add_module('e_{}'.format(i - 1), nn.Linear(last, u))
                self.encoder.add_module('r_{}'.format(i - 1), nn.ReLU())
            last = u
    def forward(self, x):
        o = self.encoder(x)
        return o

class GreedyTrainEncoder(Encoder):
    # first_layers:[1,...]
    def forward(self, x, first_layers):
        '''
        :param x:
        :param first_layers: 选择前n层
        :return: n-1层的输出和n层的输出
        '''
        input = x if first_layers == 1 else self.encoder[:2 * (first_layers - 1)](x)
        code = self.encoder[2 * (first_layers - 1):2 * first_layers](input)
        return input, code

    def unfreeze(self, layer):
        index = 2 * (layer - 1)
        for p in self.encoder[index].parameters():
            p.requires_grad = True

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def get_InAndOUt_Of_NLayer(self, layer):
        index = 2 * (layer - 1)
        shape = tuple(self.encoder[index].weight.shape)
        return shape[::-1]

# 解码器
class Decoder(nn.Module):
    def __init__(self, units):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential()
        last = 0
        for i, u in enumerate(units):
            if i != 0:
                self.decoder.add_module('d_{}'.format(i - 1), nn.Linear(last, u))
                if (i - 1) != len(units):
                    self.decoder.add_module('r_{}'.format(i - 1), nn.ReLU())
            last = u
    def forward(self, x):
        o = self.decoder(x)
        return torch.sigmoid(o)
# 总体框架
class SSDL(nn.Module):
    def __init__(self, units, nc, alpha=0.7):
        super(SSDL, self).__init__()
        self.alpha = alpha
        self.encoder = Encoder(units)
        self.spatial_extract = SPCnn()
        self.classifier = nn.Linear(120, nc)

    def forward(self, spectra, neighbor_region):
        spectra_code = self.encoder(spectra)
        spatial_code = self.spatial_extract(neighbor_region)
        fusion = torch.cat([self.alpha * spectra_code, (1 - self.alpha) * spatial_code], dim=-1)
        logits = self.classifier(fusion)
        return logits

    def loadEncoder(self, path):
        assert os.path.exists(path), '路径不存在'
        self.encoder.load_state_dict(torch.load(path))

# SSDL，整体架构
# net = SSDL((103, 60, 60, 60, 60), 16)
# print(net)
#
# spectra = torch.rand((2,103))
# neighbor_region = torch.rand((2, 1, 42, 42))
# out = net(spectra, neighbor_region)
# print(out.shape)
# 编码器
# net = Encoder((103, 60, 60, 60, 60))
# print(net)
# input = torch.rand((2,103))
# out = net(input)
# print(out.shape)
# 贪婪训练编码器
# net = GreedyTrainEncoder((103, 60, 120, 60, 60))
# input = torch.rand((1,103))
# _, out = net(input, 1)
# print(out.shape)
# _, out = net(input, 2)
# print(out.shape)
# print(net.get_InAndOUt_Of_NLayer(2))
# 解码器
# net = Decoder((60, 60, 60, 60, 103))
# print(net)
# input = torch.rand((2,60))
# out = net(input)
# print(out.shape)
# 金字塔池化CNN
# net = SPCnn()
# print(net)
# input = torch.rand((2, 1, 42, 42))
# out = net(input)
# print(out.shape)
