import torch
from Model.module import GreedyTrainEncoder
from torch import nn, optim
from torch.utils.data import DataLoader
from Loss import CrossEntropy
import numpy as np
from scipy.io import loadmat
import os
from HSIDataset import HSIDataset, DatasetInfo
from torch.utils.data import random_split
from visdom import Visdom
import argparse
from utils import weight_init
# 参数
UNITS = (103, 60, 60, 60, 60)
EPOCH_PER_LAYER = 50
LR = 1e-1
BATCHSZ = 128
NUM_WORKERS = 1
RATIO = 0.8
SEED = 971104
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(SEED)
datasetInfo = DatasetInfo()


# 模型训练
def train(model, train_loader, test_loader):
    model.freeze()
    model.to(DEVICE)
    # viz.line([[0.] * (len(UNITS) - 1)], [0], win='train', opts=dict(title='train_loss',
    #                                                                 legend=['layer_{}'.format(i) for i in range(1,len(UNITS))]))
    # viz.line([[0.] * (len(UNITS) - 1)], [0], win='eval', opts=dict(title='eval_loss',
    #                                                                legend=['layer_{}'.format(i) for i in range(1,len(UNITS))]))
    losses = torch.zeros((2, EPOCH_PER_LAYER, len(UNITS) - 1))
    for i in range(1, len(UNITS)):
        model.unfreeze(i)
        # note: 交叉熵并无法最小化误差，其误差最小化为x = 0.5
        # criterion = CrossEntropy() if  i == 1 else nn.MSELoss()
        criterion = nn.MSELoss()
        decoder = nn.Linear(*model.get_InAndOUt_Of_NLayer(i)[::-1])
        decoder.apply(weight_init)
        # 可训练参数 = encoder + decoder
        trainable_parameters = list(filter(lambda p: p.requires_grad, model.parameters())) + list(decoder.parameters())
        optimizer = optim.Adam(iter(trainable_parameters), lr=LR, weight_decay=1e-5)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=1, verbose=True)
        # PaviaU: step_size=2
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        criterion.to(DEVICE)
        decoder.to(DEVICE)
        for epoch in range(EPOCH_PER_LAYER):
            train_loss = []
            for step, ((x, _), _) in enumerate(train_loader):
                x = x.to(DEVICE)
                input, code = model(x, i)
                input, code = input.to(DEVICE), code.to(DEVICE)
                out = decoder(code)
                out = torch.sigmoid(out) if i==1 else torch.relu(out)
                # 计算Loss
                loss = criterion(out, input)
                train_loss.append(loss.item())
                #反向传播
                optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪
                # l2_norm = nn.utils.clip_grad_norm_(trainable_parameters, 10)
                optimizer.step()

                if step%50==0:
                    lr = optimizer.state_dict()['param_groups'][0]['lr']
                    # print('Layer-{} epoch:{} batch:{} loss:{:.6f} lr:{} l2-norm:{}'.format(i, epoch, step, loss.item(), lr, l2_norm))
                    print('Layer-{} epoch:{} batch:{} loss:{:.6f} lr:{}'.format(i, epoch, step, loss.item(),
                                                                                           lr))
            # 测试
            eval_loss = []
            for (x, _), _ in test_loader:
                x = x.to(DEVICE)
                input, code = model(x, i)
                out = decoder(code)
                out = torch.sigmoid(out) if i==1 else torch.relu(out)
                loss = criterion(out, input)
                eval_loss.append(loss.item())
            train_mean_loss = float(np.mean(train_loss))
            eval_mean_loss = float(np.mean(eval_loss))
            losses[0, epoch, i - 1] = train_mean_loss
            losses[1, epoch, i - 1] = eval_mean_loss
            print('Layer-{} epoch:{} eval_loss:{:.6f}'.format(i, epoch, eval_mean_loss))

            scheduler.step()
        model.freeze()
    viz.line(losses[0], list(range(EPOCH_PER_LAYER)), win='train', opts=dict(title='train_loss',
                                                                             legend=['layer_{}'.format(i) for i in range(1,len(UNITS))]))
    viz.line(losses[1], list(range(EPOCH_PER_LAYER)), win='eval', opts=dict(title='eval_loss',
                                                                            legend=['layer_{}'.format(i) for i in range(1,len(UNITS))]))


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Greedy layer-wise train')
    parse.add_argument('--epl', type=int, default=50,
                       help='Train epoch for per layer')
    parse.add_argument('--name', type=str, default='PaviaU',
                       help='The name of dataset')
    parse.add_argument('--lr', type=float, default=1e-1,
                       help='Learning rate')
    parse.add_argument('--workers', type=int, default=8,
                       help='The num of workers')
    args = parse.parse_args()

    EPOCH_PER_LAYER = args.epl
    datasetName = args.name
    LR = args.lr
    NUM_WORKERS = args.workers
    viz = Visdom()
    # 模型、数据预处理
    info = DatasetInfo.info[datasetName]
    root = './data/{}'.format(datasetName)
    data = loadmat(os.path.join(root, '{}.mat'.format(datasetName)))[info['data_key']]
    label = loadmat(os.path.join(root, '{}_gt.mat'.format(datasetName)))[info['label_key']]
    data, label = data.astype(np.float32), label.astype(np.int)
    dataset = HSIDataset(data, label)
    UNITS = (dataset.bands, 60, 60, 60, 60)
    train_size = int(len(dataset) * RATIO)
    train_dataset, eval_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCHSZ, num_workers=NUM_WORKERS, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCHSZ, num_workers=NUM_WORKERS)

    model = GreedyTrainEncoder(UNITS)
    model.apply(weight_init)
    train(model, train_loader, eval_loader)
    # 模型保存
    save_root = './encoder/{}'.format(datasetName)
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    # 文件格式名 数据集名称+encoder+每层训练的epoch
    torch.save(model.encoder.state_dict(), os.path.join(save_root, '{}_encoder_{}.ckpt'.format(datasetName, EPOCH_PER_LAYER)))
    print('finish')

