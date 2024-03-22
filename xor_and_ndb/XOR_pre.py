"""
@Description : 
@File        : XOR_pre.py
@Project     : Python_Project
@Time        : 2021/9/18 10:40
@Author      : Dexter Lien
@Software    : PyCharm
"""



import os
import re
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data



def sort_key(name):
    # 使用正则表达式从文件名中提取数字
    numbers = re.findall(r'\d+', name)
    return [int(num) for num in numbers]
class cifar10(data.Dataset):
    """Ndb Dataset.

    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    def __init__(self, root, train=True, transform = None):
        self.train = train
        self.transform = transform

        # train_list=[
        #     'image_0.txt',
        #     'image_1.txt',
        #     'image_2.txt',
        #     'image_3.txt',
        #     'image_4.txt',
        #     'image_5.txt',
        #     'train_label.txt'
        # ]
        # test_list=['image_6.txt','image_7.txt','image_8.txt','image_9.txt','test_label.txt']



        # # 打乱文件列表
        # random.shuffle(files)

        # # 计算训练集的大小
        # train_size = int(len(files) * 0.8)
        #
        # # 切分训练集和测试集
        # train_list = files[:train_size]
        # train_list.append('train_label.txt')
        # test_list = files[train_size:]
        # test_list.append('test_label.txt')


        if self.train :
            files = [f for f in os.listdir('data/ndb_data/train')]
            files.sort(key=sort_key)
            train_list = files[1:]
            train_list.append('train_label.txt')
            list = train_list
            target_list=train_list[-1]
            self.root = root + "ndb_data/train/"
        else:
            files = [f for f in os.listdir('data/ndb_data/eval')]
            files.sort(key=sort_key)
            test_list = files[1:]
            test_list.append('eval_label.txt')
            list = test_list
            target_list = test_list[-1]
            self.root = root + "ndb_data/eval/"

        self.data=[]
        self.targets=[]
        for file_name in list[:-1]:
             file_path = os.path.join(self.root, file_name)
             with open(file_path,'r') as f:
                 datas=f.readlines()
                 i=0
                 while i <len(datas):
                     tmp=[]
                     for j in range(3):
                         for j in datas[i].split(' ')[:-1]:
                             tmp.append(float(j))
                         i+=1
                     self.data.append(tmp)

         #当RGB图像以一行进行存储时的读取规范
#        for file_name in list[:-1]:
#           file_path = os.path.join(root, file_name)
#           with open(file_path,'r') as f:
#                datas=f.readlines()
#                for i in range(len(datas)):
#                    tmp=[]
#                    for j in datas[i].split(' ')[:-1]:
#                        tmp.append(float(j))
#                    self.data.append(tmp)

        file_path_test = os.path.join(self.root, target_list)
        with open(file_path_test,'r') as f:
            # datas=f.readlines()
            # for i in datas:
            #     i=i.strip()
            #     i = float(i)
            for line in f:
                stripped_line = line.strip()
                self.targets.append(float(stripped_line))
        self.data =np.vstack(self.data).reshape(-1, 3, 32, 32)
        # self.data =np.vstack(self.data).reshape(-1, 1, 28, 28)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(np.uint8(np.transpose(img,(1,2,0))))   #将矩阵转换为图片形式 以便进行数据增强
        # img=self.__normalize(img)  对数据进行标准化处理
        if self.transform is not None:
#             print("对输入进行预处理")
            img = self.transform(img)
#         if self.train:
#             enchance_pad = np.pad(img, ((0, 0), (4, 4), (4, 4)), 'constant', constant_values=0)
#             x = np.random.randint(0, 9)
#             y = np.random.randint(0, 9)
#             w = 32
#             h = 32
#             enchance_flip = enchance_pad[:, x:x + w, y:y + h]
#             if torch.rand(1) < 0.5:
#                 enchance_flip = np.flip(enchance_flip, 2)

#             enchance = enchance_flip.copy()
#             data_tensor = torch.from_numpy(enchance)

#         else:
#             data_tensor = torch.from_numpy(img)
        target=torch.LongTensor([np.int64(target).item()])
        return img, target


    def __normalize__(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def __len__(self):
        return len(self.data)



# def get_cifar10(train, transform=None):
#
#     # dataset and data loader
#     cifar10_dataset = cifar10(root=param.root,
#                         train=train,
#                         transform=transform,)
#
#     return cifar10_dataset
# #
# if __name__ == '__main__':
#     src_data = get_cifar10(True)
#     cifar10_data_loader = torch.utils.data.DataLoader(
#         dataset=src_data,
#         batch_size=param.batch_size,
#         shuffle=True)



    # for (images, labels) in cifar10_data_loader:
    #     images = images.cpu()
    #     labels = labels.cpu()
    #     # labels=labels.cpu().squeeze_()
    #     # images = make_variable(images, volatile=True)
    #     # labels = make_variable(labels).squeeze_()
    #     print(images[0])
    #     print(labels[0])
    #     print("+++++++++++++++++++++")
    #     print(labels)
