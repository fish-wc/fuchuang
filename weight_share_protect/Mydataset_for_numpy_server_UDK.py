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

'''
This class(Mydataset_numpy) load the training and testing data;
the data format requires numpy 3_32_32 (xxx.npy);
the training data will be preproceeded:pad+crop+flip;
the testing  data will not be preproceeded;

the return data format is torch.tensor(torch.FloatTensor)
'''


class Mydataset_numpy_server_UDK(Dataset):
    def __init__(self, mode, dataset):
        self.png_label_list_train = []
        self.png_label_list_test = []
        # mode:"train" or "test"
        self.mode = mode
        # dataset:"your dataset folder name "
        self.dataset = dataset
        User_number = os.listdir(dataset).__len__()
        # print("Notice : this is  Mydataset_numpy\n " + self.dataset)
        # print()

        # require data format： numpy 3_32_32  (.numpy)
        self.root_train = self.dataset
        self.root_test = self.dataset

        for i in range(User_number):

            csv_file_train = csv.reader(open(self.dataset + "/user" + str(i + 1) + '/train.csv', 'r'))
            csv_file_test = csv.reader(open(self.dataset + "/user" + str(i + 1) + '/test.csv', 'r'))
            for line in csv_file_train:
                self.png_label_list_train.append([ "user" + str(i + 1)+"/train/" +line[0], int(line[1])])

            for line in csv_file_test:
                self.png_label_list_test.append([ "user" + str(i + 1)+"/test/" +line[0], int(line[1])])

    def __len__(self):
        if (self.mode == "train"):
            return len(self.png_label_list_train)
        if (self.mode == "test"):
            return len(self.png_label_list_test)

    def __getitem__(self, index):
        if (self.mode == "train"):
            # load training data : xxx.npy
            featuremap = np.load(os.path.join(self.root_train, self.png_label_list_train[index][0]))
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

            featuremap = np.load(os.path.join(self.root_test, self.png_label_list_test[index][0]))
            new_featuremap = featuremap.copy()
            featuremap_tensor = torch.from_numpy(new_featuremap)
            featuremap_tensor = featuremap_tensor.type(torch.FloatTensor)
            return featuremap_tensor, torch.tensor(self.png_label_list_test[index][1])
