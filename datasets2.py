from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import Subset
import random
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F
from xor_and_ndb import XOR_pre



class custmResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = img.unsqueeze(0)
        img = F.interpolate(img, (self.size,self.size))
        img = img.squeeze()
        return img

class CustomDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform

        # 假设所有文件都存储在一个文件夹中
        self.file_list = os.listdir(data_root)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_root, self.file_list[idx])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image

# # 使用自定义数据集
# transform = transforms.Compose([transforms.ToTensor()])
# data_root = 'path/to/your/dataset'  # 指定数据集路径
# custom_dataset = CustomDataset(data_root, transform=transform)
#
# # 现在可以使用这个数据集了，例如在数据加载器中
# from torch.utils.data import DataLoader
#
# data_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

# from torch.utils.data import random_split
#
# # 假设 custom_dataset 是您已经加载的完整数据集
# dataset_size = len(custom_dataset)
# train_size = int(0.8 * dataset_size)  # 例如，80% 作为训练集
# test_size = dataset_size - train_size
#
# train_dataset, test_dataset = random_split(custom_dataset, [train_size, test_size])

# from torch.utils.data import DataLoader
#
# # 为训练数据集创建 DataLoader
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
#
# # 为测试数据集创建 DataLoader
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)


def get_dataset(dir, name ,choice=None,subset_size=None):
	if name == 'mnist':
		train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
		eval_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())
		
	if name == 'cifar':
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		transform_2 = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			custmResize(64),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
		if choice == 3:
			train_dataset = XOR_pre.cifar10(root=dir, train=True, transform=transform_train)
			eval_dataset = XOR_pre.cifar10(root=dir, train=False, transform=transform_test)
		elif choice == 4:
			train_dataset = datasets.CIFAR10(dir, train=True, download=True, transform=transform_2)
			eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_2)
		else:
			train_dataset = datasets.CIFAR10(dir, train=True, download=True ,transform=transform_train)
			eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)



	if subset_size is not None:
		# 假设 total_size 是原始数据集的大小
		train_size = len(train_dataset)  # 或 eval_dataset
		eval_size = len(eval_dataset)

		# 生成一个从0到 total_size-1 的索引列表
		indices_train = list(range(train_size))
		indices_eval  = list(range(eval_size))

		# # 随机打乱索引
		# random.shuffle(indices_train)
		# random.shuffle(indices_eval)

		# 从打乱的索引中取出前 subset_size 个创建子数据集
		train_subset_indices = indices_train[:subset_size]
		eval_subset_indices = indices_eval[:subset_size]

		# 创建子数据集
		train_dataset = Subset(train_dataset, train_subset_indices)
		eval_dataset = Subset(eval_dataset, eval_subset_indices)
		# train_dataset = Subset(train_dataset, list(range(subset_size)))
		# eval_dataset = Subset(eval_dataset, list(range(subset_size)))

	return train_dataset, eval_dataset



# transform_train
# RandomCrop(32, padding=4): 这个转换首先对图像周围添加4个像素的填充（默认填充模式是零填充），然后随机裁剪出一个32x32像素的区域。这种类型的随机裁剪是一种数据增强技术，有助于减少模型对图像位置的依赖。
#
# RandomHorizontalFlip(): 随机水平翻转图像。这也是一种数据增强技术，有助于模型学习不依赖于图像的水平对称性。
#
# ToTensor(): 将PIL图像或NumPy ndarray转换为torch.Tensor。这个转换也将图像的像素值从[0, 255]缩放到[0.0, 1.0]的范围。
#
# Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)): 标准化图像，使其具有特定的均值和标准差。在这里，(0.4914, 0.4822, 0.4465) 是通道的均值，(0.2023, 0.1994, 0.2010) 是通道的标准差。这有助于模型训练，因为它确保了输入数据具有一致的分布。
#
# transform_test
# 对于测试数据集，通常不进行随机转换或数据增强：
#
# ToTensor(): 同上，将图像转换为张量并缩放像素值。
#
# Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)): 同上，标准化图像。


# import torch
# import torchvision
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
#
# def image_to_binary(image):
#     # 将图像转换为 NumPy 数组
#     img_array = np.array(image)
#     # 将像素值转换为二进制字符串
#     binary_str = ''.join(format(pixel, '08b') for pixel in img_array.flatten())
#     return binary_str
#
# # 加载 CIFAR-10 数据集
# transform = transforms.Compose([transforms.ToTensor()])
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
#
# # 遍历数据集并将图像转换为二进制字符串
# for i, data in enumerate(trainloader, 0):
#     inputs, labels = data
#     image = transforms.ToPILImage()(inputs[0])
#     binary_string = image_to_binary(image)
#     # 在这里应用您的处理逻辑（例如异或操作等）
#     # ...
#
#     # 为了演示，我们只处理几个图像
#     if i >= 10:
#         break
