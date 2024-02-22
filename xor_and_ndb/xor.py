import torch
import torchvision
import torchvision.transforms as transforms
import random
import os
import datasets2

import numpy as np
# 设置随机种子
random.seed(42)

# 转换为二进制字符串
def convert_to_binary_string(data):
    binary_data = []
    for item in data:
        # 确保像素值在 [0, 1] 范围内
        normalized_item = np.clip(item.numpy(), 0, 1)
        # 分别处理 R, G, B 三个通道，并将每个通道的值转换为二进制字符串
        binary_item_r = ''.join(format(int(byte * 255), '08b') for byte in normalized_item[0].flatten())
        binary_item_g = ''.join(format(int(byte * 255), '08b') for byte in normalized_item[1].flatten())
        binary_item_b = ''.join(format(int(byte * 255), '08b') for byte in normalized_item[2].flatten())
        # 将三个通道的二进制字符串分别存储
        # print(len(binary_item_r))
        # print(len(binary_item_g))
        # print(len(binary_item_b))
        # print()
        binary_data.append((binary_item_r, binary_item_g, binary_item_b))
    return binary_data




# 随机生成与图像大小相同的二进制密钥
# def generate_xor_key(data):
#     key = ''.join([str(random.randint(0, 1)) for _ in range(len(data[0]))])
#     return key
#
# # 对数据集进行异或操作
# def xor_dataset(dataset, xor_key):
#     xored_data = []
#     for item in dataset:
#         xored_item = ''.join([str(int(a) ^ int(b)) for a, b in zip(item, xor_key)])
#         xored_data.append(xored_item)
#     return xored_data
def generate_xor_key(length=128):
    key = ''.join([str(random.randint(0, 1)) for _ in range(length)])
    return key

def xor_dataset(dataset, xor_key):
    xored_data = []
    key_length = len(xor_key)
    for item in dataset:
        xored_item_r, xored_item_g, xored_item_b = '', '', ''
        for channel in item:
            # 分割每个通道的二进制字符串为 128 位的块
            xored_channel = ''
            for i in range(0, len(channel), key_length):
                block = channel[i:i + key_length]
                # 确保块和密钥长度相同（最后一个块可能需要处理）
                if len(block) < key_length:
                    block = block.ljust(key_length, '0')  # 或者其他处理方式
                # 对每个块应用 XOR 操作
                xored_block = ''.join([str(int(a) ^ int(b)) for a, b in zip(block, xor_key)])
                xored_channel += xored_block
            if xored_item_r == '':
                xored_item_r = xored_channel
            elif xored_item_g == '':
                xored_item_g = xored_channel
            else:
                xored_item_b = xored_channel
        xored_data.append((xored_item_r, xored_item_g, xored_item_b))
    return xored_data


def XOR():
    # 加载CIFAR-10数据集
    trainset, eval_datasets = datasets2.get_dataset("data/", 'cifar',subset_size= 10000)#这里
    # 获取 trainset 的索引
    subset_indices = trainset.indices

    # 从原始数据集中获取目标标签
    original_targets = trainset.dataset.targets

    # 使用索引从原始目标标签中提取对应的子集标签
    target = [original_targets[i] for i in subset_indices]


    # 获取数据集中的图像数据
    images = [data[0] for data in trainset]

    # 转换图像数据为二进制字符串
    binary_images = convert_to_binary_string(images)

    # # 生成随机密钥
    # xor_key = generate_xor_key(binary_images[0])
    #
    # # 对图像数据进行异或操作
    # xored_images = xor_dataset(binary_images, xor_key)

    xor_key = generate_xor_key(128)
    xored_images = xor_dataset(binary_images, xor_key)
    # 保存异或后的数据到文件
    output_directory = 'data/xored_data'
    os.makedirs(output_directory, exist_ok=True)

    for i, xored_item in enumerate(xored_images):
        filename = os.path.join(output_directory, f'xored_image_{i}.txt')
        with open(filename, 'w') as f:
            # 分别写入红色、绿色和蓝色通道的二进制字符串
            for channel_data in xored_item:
                f.write(channel_data + "\n")
    output_directory_label = 'data/ndb_data/'
    os.makedirs(output_directory_label, exist_ok=True)
    trainfile_label = os.path.join(output_directory_label,f'train_label.txt')
    train_num = int(len(target) * 0.8)

    with open(trainfile_label,'w') as f:
        subset1 = target[:train_num]
        for i in range(train_num):
            f.write(str(subset1[i])+'\n')
    testfile_label = os.path.join(output_directory_label,f'test_label.txt')
    with open(testfile_label,'w') as f:
        subset2 = target[train_num:]
        for item in subset2:
            f.write(str(item) + '\n')


