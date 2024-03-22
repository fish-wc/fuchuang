import os
from datetime import datetime
import torch
import torch
from PIL import Image
from torch import optim
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
import numpy as np
import csv
from datetime import datetime
from PIL import Image
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
import random

'''
This class(Mydataset_numpy) load the training and testing data;
the data format requires numpy 3_32_32 (xxx.npy);
the training data will be preproceeded:pad+crop+flip;
the testing  data will not be preproceeded;

the return data format is torch.tensor(torch.FloatTensor)
'''


class Mydataset_numpy_client_UDK(Dataset):
    def __init__(self, mode, dataset, conf, ID):
        self.ID = ID

        # mode:"train" or "test"
        self.mode = mode
        # dataset:"your dataset folder name "
        self.dataset = dataset + "/" + "user" + str(self.ID)
        # print("Notice : this is  Mydataset_numpy\n "+self.dataset)
        # print()

        self.png_label_list_train = []
        self.png_label_list_test = []

        self.root_train = self.dataset + '/train/'
        self.root_test = self.dataset + '/test/'
        # 打开并读取训练和测试CSV文件
        csv_file_train = list(csv.reader(open(self.dataset + '/train.csv', 'r')))

        # 计算每个数据集中应该选取的行数
        num_samples_train = int(len(csv_file_train) / conf['no_models'])

        # 从csv_file_train中随机选择num_samples_train行
        random_samples_train = random.sample(csv_file_train, num_samples_train)

        # 处理随机选取的训练数据
        for line in random_samples_train:
            self.png_label_list_train.append([line[0], int(line[1])])

    def __len__(self):
        if (self.mode == "train"):
            return len(self.png_label_list_train)
        if (self.mode == "test"):
            return len(self.png_label_list_test)

    def __getitem__(self, index):
        if (self.mode == "train"):
            # load training data : xxx.npy
            featuremap = np.load(os.path.join(self.dataset + "/train", self.png_label_list_train[index][0]))
            # padding + crop  : 3*32*32 -> 3*40*40   ->3*32*32
            featuremap = np.pad(featuremap, ((0, 0), (4, 4), (4, 4)), 'constant', constant_values=0)
            # a random number in range[0,8]
            x = np.random.randint(0, 9)
            y = np.random.randint(0, 9)
            w = 32
            h = 32
            featuremap = featuremap[:, x:x + w, y:y + w]
            # flip by chance of 0.5
            if torch.rand(1) < 0.5:
                featuremap = np.flip(featuremap, 2)

            new_featuremap = featuremap.copy()
            featuremap_tensor = torch.from_numpy(new_featuremap)
            featuremap_tensor = featuremap_tensor.type(torch.FloatTensor)
            return featuremap_tensor, torch.tensor(self.png_label_list_train[index][1])

        if (self.mode == "test"):
            # load testing data : xxx.npy   no:padding+crop+flip
            featuremap = np.load(os.path.join(self.dataset + "/test", self.png_label_list_test[index][0]))
            new_featuremap = featuremap.copy()
            featuremap_tensor = torch.from_numpy(new_featuremap)
            featuremap_tensor = featuremap_tensor.type(torch.FloatTensor)
            return featuremap_tensor, torch.tensor(self.png_label_list_test[index][1])