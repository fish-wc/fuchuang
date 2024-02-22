from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, nc, ndf, nz):
        super(Classifier, self).__init__()

        self.nc = nc
        self.ndf = ndf
        self.nz = nz

        # 调整卷积层以适应32x32的输入
        self.encoder = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 3, 1, 1),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Output: (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Output: (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Output: (ndf*4) x 4 x 4
        )

        # 调整全连接层以匹配卷积层的输出
        self.fc = nn.Sequential(
            nn.Linear(ndf * 4 * 4 * 4, nz * 5),
            nn.Dropout(0.5),
            nn.Linear(nz * 5, nz),
        )

    def forward(self, x, release=False):
        x = self.encoder(x)
        x = x.view(-1, self.ndf * 4 * 4 * 4)  # 调整视图以匹配全连接层的期望输入
        x = self.fc(x)

        if release:
            return F.softmax(x, dim=1)
        else:
            return F.log_softmax(x, dim=1)


class Inversion(nn.Module):
    def __init__(self, nc, ngf, nz, truncation, c):
        super(Inversion, self).__init__()

        self.nc = nc  # 输出通道数，对应CIFAR-10的通道数，例如RGB图像为3
        self.ngf = ngf  # 生成器特征图深度
        self.nz = nz  # 输入特征向量Z的维度
        self.truncation = truncation  # 截断数，用于前向传播中的top-k操作
        self.c = c  # 用于调整特征向量值的常数

        # 调整解码器以生成32x32的输出图像
        self.decoder = nn.Sequential(
            # 输入Z的维度是 nz
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 输出尺寸: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 输出尺寸: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 输出尺寸: (ngf*2) x 16 x 16
            # 移除了一层转置卷积以调整输出尺寸至32x32
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1),
            nn.Tanh()
            # 输出尺寸: (nc) x 32 x 32
        )

    def forward(self, x):
        topk, indices = torch.topk(x, self.truncation)
        topk = torch.clamp(torch.log(topk), min=-1000) + self.c
        topk_min = topk.min(1, keepdim=True)[0]
        topk = topk + F.relu(-topk_min)
        x = torch.zeros(len(x), self.nz, device=x.device).scatter_(1, indices, topk)

        x = x.view(-1, self.nz, 1, 1)
        x = self.decoder(x)
        # 输出调整为单通道32x32图像，如果CIFAR-10是多通道，需根据情况调整
        return x