import torch
import torchvision
from PIL import Image
import os
import numpy as np
import random

import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt

# 定义类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def eval(model_path,eval_loader):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(model_path, map_location=torch.device('cpu'))
    # 将模型设置为评估模式，这对于推理很重要
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    for batch_id, batch in enumerate(eval_loader):
        data, target = batch
        dataset_size += data.size()[0]

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        output = model(data)

        total_loss += torch.nn.functional.cross_entropy(output, target,
                                                        reduction='sum').item()  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))
    total_l = total_loss / dataset_size

    return acc, total_l


# 保存CIFAR-10的图片到指定目录，提前保存好了,所以不封装到其他地方
def save_cifar10_images(image_label_pairs, root_dir):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for i, (image_array, label) in enumerate(image_label_pairs):
        # 创建标签对应的目录
        label_dir = os.path.join(root_dir, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        # 转换 NumPy 数组到 PIL.Image 对象
        image = Image.fromarray(image_array)
        # 构造图像保存路径
        image_path = os.path.join(label_dir, f'{i}.png')
        # 保存图像
        image.save(image_path)

# 假设模型已经加载


def load_images_from_folder(folder):
    """从指定文件夹加载图像，并返回图像列表和对应的标签列表"""
    images = []
    labels = []
    images_name=[]
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                try:
                    # 确保每次都打开一个新的图像对象
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                    labels.append(int(label))
                    images_name.append(filename)
                except IOError:
                    print(f"无法打开文件：{img_path}")
                    pass

    return images, labels,images_name

# 注意: 你需要确保 load_images_from_folder 函数能够正确运行并返回images和true_labels
def predict_and_evaluate(model_path, dataset_folder):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()  # 将模型设置为评估模式

    images, true_labels,images_name = load_images_from_folder(dataset_folder)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    # 初始化损失函数
    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    total_loss = 0.0
    predicted_labels = []  # 用于存储预测标签

    with torch.no_grad():  # 关闭梯度计算
        for img, label in zip(images, true_labels):
            img_t = transform(img).unsqueeze(0)
            output = model(img_t)
            _, predicted = torch.max(output, 1)
            correct += (predicted.item() == label)
            predicted_labels.append(predicted.item())

            # 计算损失
            loss = criterion(output, torch.tensor([label]))
            total_loss += loss.item()

    accuracy = (correct / len(images)) * 100
    average_loss = total_loss / len(images)

    return accuracy, average_loss, images, predicted_labels,true_labels,images_name


def show_images(myimages, true_labels, predicted_labels,num_images=25):

    # 随机选择 25 张图片
    indices = list(range(len(myimages)))
    selected_indices = random.sample(indices, 25)
    images = [myimages[i] for i in selected_indices]
    predict_labels = [predicted_labels[i] for i in selected_indices]
    true_labels = [true_labels[i] for i in selected_indices]
    # 显示图像、标签和路径

    """显示一个 5x5 的图像矩阵，展示共 25 张图片及其对应的标签和类别"""
    if num_images > len(images):
        raise ValueError("num_images is greater than the total number of images")

    plt.figure(figsize=(22, 22))  # 调整图像大小
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)  # 创建 5x5 的网格
        plt.imshow(images[i])
        plt.xticks([])  # 去除x轴标记
        plt.yticks([])  # 去除y轴标记
        # 使用标签索引从classes获取对应的类别名称
        true_class_name = classes[true_labels[i]]
        predict_class_name=classes[predict_labels[i]]
        plt.title(f"True:{true_class_name} ({true_labels[i]}),Predict:{predict_class_name}", fontsize=10)
    plt.tight_layout()
    plt.savefig('./data/figures/show_pchoice2.png',dpi=300)
    plt.show()


def show_image(image,predict_label,true_label):
    plt.figure(figsize=(5,5))  # 调整图像大小
    plt.imshow(image)
    plt.xticks([])  # 去除x轴标记
    plt.yticks([])  # 去除y轴标记
    # 使用标签索引从classes获取对应的类别名称
    predict_class_name=classes[predict_label]
    true_label_name=classes[true_label]
    plt.title(f"Predict:{predict_class_name},label({predict_label}),True:{true_label_name},label({true_label})", fontsize=10)
    plt.savefig("./data/figures/show_oneimage.png",dpi=300)
    plt.show()

def predict_image(image_path, model_path):
    parts = image_path.split('/')
    # 倒数第二个斜杠后面的部分
    true_label = parts[-2]
    # return int(number_part)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 调整图像大小以匹配模型输入
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
    ])

    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()  # 将模型设置为评估模式
    """读取图像，进行预测，并显示结果"""
    # 加载图像
    img = Image.open(image_path).convert('RGB')

    # 图像预处理
    img_t = transform(img)
    img_t = img_t.unsqueeze(0)  # 添加批次维度

    # 预测
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
        predicted_label = predicted.item()


    return int(predicted_label),img,int(true_label)

import json
def load_json(save_path):

    # 从 JSON 文件读取
    with open(save_path, 'r') as f:
        data_dict = json.load(f)

    # 从字典中提取 keys 和 values
    values = list(data_dict.values())
    return values

def plot_category(values):
    # 统计每个类别的数量
    category_counts = [0] * 10  # 由于有10个类别（0-9），初始化一个长度为10的列表
    for category in values:
        category_counts[category] += 1
    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(classes, category_counts, color='skyblue')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Count of Each Category in CIFAR-10 Prediction')
    plt.xticks(classes, ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    plt.savefig('.\\data\\figures\\category.png',dpi=300)
    plt.show()

if __name__=='__main__':
    pass
