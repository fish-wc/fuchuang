import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import sys

import all_models.resnet_total

sys.path.append('all_models')

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义CIFAR-10类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = all_models.resnet50.ResNet50()
# GPU 模式
model = model.to(device) # 扔到GPU中
# 假设 model_ft 是你的模型实例
save_path="./mymodel/0.pth"

# model = torch.load(save_path)
model = torch.load(save_path, map_location=torch.device('cpu'))


model.eval()  # 设置为评估模式

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50需要的输入尺寸为224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载CIFAR-10测试集
test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=4, shuffle=True)

# 获取一批图像
dataiter = iter(test_loader)
images, labels = next(dataiter)


# 预测
outputs = model(images.to(device))
_, predicted = torch.max(outputs, 1)

# 显示图片和预测结果
def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
