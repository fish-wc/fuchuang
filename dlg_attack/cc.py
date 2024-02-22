import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(4 * 4 * 128, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # 使用Xavier初始化权重
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # 使用Xavier初始化权重
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

max_norm = 0.5  # 设置梯度缩放因子
desired_max_norm = 0.5
lr = 3e-4
desired_max_norm = 0.5
R_lowerbound = 0.000000157 # lower bound of the LRP
batch_size = 1
scale = 1  # 缩放因子
#32*3 + 64*32 + 128*64 + 256*32 + 160
block = 43269
#delta = 2 * scale / batch_size
#delta = 2 * scale * block * lr / batch_size
delta0 = 2 * scale * 32 * lr / batch_size
delta2 = 2 * scale * 64 * lr / batch_size
delta4 = 2 * scale * 128 * lr / batch_size
delta6 = 2 * scale * 256 * lr / batch_size
delta8 = 2 * scale * 10 * lr / batch_size
deltab = 2 * scale * lr / batch_size  # * sh
#delta = 0.0002
balance = 100
#delta = 2 * scale / batch_size
scale_factor = 0.01
stable = 1e-10

# 定义添加Laplace噪声的函数
# def add_laplace_noise(tensor, epsilon):
#     #print(tensor.size())
#     noise = np.random.laplace(0, delta/epsilon, size=tensor.size()).astype(np.float32)
#     return tensor + torch.from_numpy(noise).to(tensor.device)

def grade_class8(gw):
    min_gw = torch.min(gw)
    max_gw = torch.max(gw)
    grade_class = (max_gw - min_gw) / 8
    # 将不同范围的值映射到对应的等级
    gw_mapped = torch.zeros_like(gw)
    gw_mapped[(gw >= min_gw) & (gw < min_gw + grade_class)] = 4
    gw_mapped[(gw >= min_gw + grade_class) & (gw < min_gw + 2 * grade_class)] = 3
    gw_mapped[(gw >= min_gw + 2 * grade_class) & (gw < min_gw + 3 * grade_class)] = 2
    gw_mapped[(gw >= min_gw + 3 * grade_class) & (gw < min_gw + 4 * grade_class)] = 1
    gw_mapped[(gw >= min_gw + 4 * grade_class) & (gw < min_gw + 5 * grade_class)] = 1
    gw_mapped[(gw >= min_gw + 5 * grade_class) & (gw < min_gw + 6 * grade_class)] = 2
    gw_mapped[(gw >= min_gw + 6 * grade_class) & (gw < min_gw + 7 * grade_class)] = 3
    gw_mapped[(gw >= min_gw + 7 * grade_class) & (gw <= min_gw + 8 * grade_class)] = 4
    return gw_mapped

# 32数量 3通道 5行 5列       梯度裁剪
def scale_grad(grad):
    # grad /= batch_size
    norms = torch.norm(grad, p=1, dim=(1, 2, 3)) + stable  # 计算每个通道上的2D梯度向量的1范数
    grad = grad / norms.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * scale  # 缩放梯度
    return grad


def scale_grad_b(grad, param):
    #grad += stable
    # grad /= batch_size
    norms = torch.norm(grad, p=1) + stable # 计算每个通道上的2D梯度向量的1范数
    grad = grad / norms * scale
    return grad


#根据grad_diff 加入噪声
def add_laplace_noise_b(grad, epsilon, grad_diff, sh):

    grad_diff = grade_class8(grad_diff)

    #print("---------------")
    #print(grad_diff)
    sum = torch.abs(grad_diff).sum() + stable
    #print(sum)
    result = grad_diff / sum
    #print(result)
    result_toNumpy = result.cpu().numpy()

    result_toNumpy = np.clip(result_toNumpy, 0.8 / sh, 1.2 / sh)

    laplace_noise = np.random.laplace(0, deltab * sh / (epsilon * result_toNumpy), sh)
    #laplace_noise = np.random.laplace(0, delta / epsilon, sh)
    grad += torch.from_numpy(laplace_noise).to(torch.float32).to(grad.device)
    return grad / batch_size

def scale_grad0(grad):
    # print('ids = 0, max:{}, min:{}, mean:{}'.format(grad.max().item(), grad.min().item(),grad.mean().item()))
    # 形状为 [32, 3, 5, 5]
    scale = 0.1  # 缩放因子
    # 计算每个子张量的绝对值和
    abs_sum = torch.abs(grad).reshape(32, 3, -1).sum(dim=2, keepdim=True) + 1e-6
    # 计算缩放因子
    scale_factor = scale / abs_sum
    # 对每个子张量进行归一化和缩放
    grad *= scale_factor.reshape(32, 3, 1, 1)
    # print('ids = 0, max:{}, min:{}, mean:{}'.format(scaled_grad.max().item(), scaled_grad.min().item(),scaled_grad.mean().item()))
    return grad


#根据grad_diff 加入噪声
def add_laplace_noise0(grad, epsilon, grad_diff):
    # # 找到为 0 的元素并加上 1e-5
    # grad[grad == 0] += 1e-5
    grad_diff = grade_class8(grad_diff)

    #5*5的行列和   32*3组   求第二和第三维的绝对值求和
    sum_123d = torch.abs(grad_diff).sum(dim=(1, 2, 3), keepdim=True) + stable
    sum_123d = sum_123d.squeeze(dim=1).squeeze(dim=1).squeeze(dim=1)
    # 在第一维上求和
    sum_1d = torch.sum(sum_123d)  # 形状为[32]
    # 对sum_23d每个元素除以对应的sum_1d的值    行列在通道上的占比
    result = sum_123d / sum_1d # 形状为[32, 3]
    #占比为负数  则置位取反， 为0则分配为R_lowerbound
    # result[result == 0] = R_lowerbound
    # result[result < 0] *= -1
    result_toNumpy = result.cpu().numpy()

    #result_toNumpy = np.clip(result_toNumpy, 0.8 / 3, 1.2 / 3)

    laplace_noise = np.random.laplace(0, delta0 / (epsilon * result_toNumpy[:, np.newaxis, np.newaxis, np.newaxis]),
                                      size=(32, 3, 5, 5))
    # print(3 * epsilon * result_toNumpy[:, :, np.newaxis, np.newaxis])
    # print("=============")
    # laplace_noise = np.random.laplace(0, delta / epsilon,
    #                                   size=(32, 3, 5, 5))
    grad += torch.from_numpy(laplace_noise).to(torch.float32).to(grad.device)
    return grad / batch_size


# 64 32 5 5
def scale_grad2(grad):
    scale = 0.1
    abs_sum = torch.abs(grad).reshape(64, 32, -1).sum(dim=2, keepdim=True) + 1e-6
    # abs_sum += 1e-8
    scale_factor = scale / abs_sum
    grad *= scale_factor.reshape(64, 32, 1, 1)
    return grad

#根据grad_diff 加入噪声
def add_laplace_noise2(grad, epsilon, grad_diff):
    # # 找到为 0 的元素并加上 1e-5
    # grad[grad == 0] += 1e-5
    grad_diff = grade_class8(grad_diff)
    #print(torch.abs(grad).mean())
    sum_123d = torch.abs(grad_diff).sum(dim=(1, 2, 3), keepdim=True) + stable
    sum_123d = sum_123d.squeeze(dim=1).squeeze(dim=1).squeeze(dim=1)
    # 在第一维上求和
    sum_1d = torch.sum(sum_123d)  # 形状为[32]
    # 对sum_23d每个元素除以对应的sum_1d的值    行列在通道上的占比
    result = sum_123d / sum_1d  # 形状为[32, 3]
    # 占比为负数  则置位取反， 为0则分配为R_lowerbound
    # result[result == 0] = R_lowerbound
    # result[result < 0] *= -1
    result_toNumpy = result.cpu().numpy()

    #result_toNumpy = np.clip(result_toNumpy, 0.8 / 32, 1.2 / 32)

    laplace_noise = np.random.laplace(0, delta2 / (epsilon * result_toNumpy[:, np.newaxis, np.newaxis, np.newaxis]),
                                      size=(64, 32, 5, 5))
    # print(32 * epsilon * result_toNumpy[:, :, np.newaxis, np.newaxis])
    # print("=============")
    # laplace_noise = np.random.laplace(0, delta / epsilon,
    #                                   size=(64, 32, 5, 5))
    grad += torch.from_numpy(laplace_noise).to(torch.float32).to(grad.device)
    # print(np.abs(laplace_noise).mean())
    # print(torch.abs(grad).mean())
    # print("=====================")
    return grad / batch_size

# 128  64  5  5
def scale_grad4(grad):
    scale = 0.1
    abs_sum = torch.abs(grad).reshape(128, 64, -1).sum(dim=2, keepdim=True) + 1e-6
    scale_factor = scale / abs_sum
    grad *= scale_factor.reshape(128, 64, 1, 1)
    return grad

#根据grad_diff 加入噪声
# def add_laplace_noise4(grad, epsilon, grad_diff):
#     # # 找到为 0 的元素并加上 1e-5
#     # grad[grad == 0] += 1e-5
#     grad_diff = grade_class8(grad_diff)
#
#     sum_123d = torch.abs(grad_diff).sum(dim=(1, 2, 3), keepdim=True) + stable
#     sum_123d = sum_123d.squeeze(dim=1).squeeze(dim=1).squeeze(dim=1)
#     # 在第一维上求和
#     sum_1d = torch.sum(sum_123d)  # 形状为[32]
#     # 对sum_23d每个元素除以对应的sum_1d的值    行列在通道上的占比
#     result = sum_123d / sum_1d  # 形状为[32, 3]
#     # 占比为负数  则置位取反， 为0则分配为R_lowerbound
#     # result[result == 0] = R_lowerbound
#     # result[result < 0] *= -1
#     result_toNumpy = result.cpu().numpy()
#
#     #result_toNumpy = np.clip(result_toNumpy, 0.8 / 64, 1.2 / 64)
#
#     laplace_noise = np.random.laplace(0, delta4 / (epsilon * result_toNumpy[:, np.newaxis, np.newaxis, np.newaxis]),
#                                       size=(128, 64, 5, 5))
#     # print(64 * epsilon * result_toNumpy[:, :, np.newaxis, np.newaxis])
#     # print("=============")
#     # laplace_noise = np.random.laplace(0, delta / epsilon,
#     #                                   size=(128, 64, 5, 5))
#     grad += torch.from_numpy(laplace_noise).to(torch.float32).to(grad.device)
#     return grad / batch_size

def add_laplace_noise4(grad, epsilon, grad_diff):
    # 假设 stable 是一个先前定义的常量，用来避免除以零的情况
    stable = 1e-5

    # grad_diff 的维度检查
    dim = len(grad_diff.shape)

    if dim == 4:  # 四维张量（例如卷积层）
        sum_123d = torch.abs(grad_diff).sum(dim=(1, 2, 3), keepdim=True) + stable
        sum_123d = sum_123d.squeeze()
        sum_1d = torch.sum(sum_123d)  # 形状为[32]
        # 对sum_123d每个元素除以对应的sum_1d的值，行列在通道上的占比
        result = sum_123d / sum_1d  # 形状为[32]
    elif dim == 2:  # 二维张量（例如全连接层）
        sum_1d = torch.abs(grad_diff).sum(dim=1, keepdim=True) + stable
        sum_1d = sum_1d.squeeze()
        # 对于全连接层，不需要处理多余的维度
        result = torch.ones_like(sum_1d)  # 形状为[256] 或其他全连接层的大小
    else:
        raise ValueError(f"Unexpected number of dimensions: {dim}")

    # 将 result 转换为 NumPy 数组，并在必要时扩展其维度
    result_toNumpy = result.cpu().numpy()
    # 对 result_toNumpy 进行处理以确保它符合 laplace 噪声函数的需要
    if dim == 4:
        result_toNumpy = result_toNumpy[:, np.newaxis, np.newaxis, np.newaxis]
    elif dim == 2:
        result_toNumpy = result_toNumpy[:, np.newaxis]

    # 根据 result_toNumpy 计算 laplace 噪声
    laplace_noise = np.random.laplace(0, delta4 / (epsilon * result_toNumpy), size=grad_diff.shape)

    # 将 NumPy 中的 laplace 噪声转换为 PyTorch 张量
    laplace_noise_torch = torch.from_numpy(laplace_noise).float().to(grad.device)

    # 将噪声添加到梯度
    grad += laplace_noise_torch
    return grad / batch_size

# 256  2048
def scale_grad6(grad):
    #grad /= batch_size
    grad = grad.reshape(256, 8, 16, 16)
    norms = torch.norm(grad, p=1, dim=(1, 2, 3)) + stable  # 计算每个通道上的2D梯度向量的1范数
    # print(norms)
    grad = grad / norms.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * scale  # 缩放梯度
    grad = grad.reshape(256, 2048)
    return grad

#根据grad_diff 加入噪声
def add_laplace_noise6(grad, epsilon, grad_diff):
    # # 找到为 0 的元素并加上 1e-5
    # grad[grad == 0] += 1e-5
    grad_diff = grade_class8(grad_diff)

    grad_diff = grad_diff.reshape(256, 8, 16, 16)
    sum_123d = torch.abs(grad_diff).sum(dim=(1, 2, 3), keepdim=True) + stable
    sum_123d = sum_123d.squeeze(dim=1).squeeze(dim=1).squeeze(dim=1)
    # 在第一维上求和
    sum_1d = torch.sum(sum_123d)  # 形状为[32]
    # 对sum_23d每个元素除以对应的sum_1d的值    行列在通道上的占比
    result = sum_123d / sum_1d  # 形状为[32, 3]
    # 占比为负数  则置位取反， 为0则分配为R_lowerbound
    # result[result == 0] = R_lowerbound
    # result[result < 0] *= -1
    result_toNumpy = result.cpu().numpy()

    #result_toNumpy = np.clip(result_toNumpy, 0.8 / 8, 1.2 / 8)

    laplace_noise = np.random.laplace(0, delta6 / (epsilon * result_toNumpy[:, np.newaxis, np.newaxis, np.newaxis]),
                                      size=(256, 8, 16, 16))
    # print(128 * epsilon * result_toNumpy[:, :, np.newaxis, np.newaxis])
    # print("=============")
    # laplace_noise = np.random.laplace(0, delta / epsilon,
    #                                   size=(256, 128, 4, 4))
    laplace_noise = laplace_noise.reshape(256, 2048)
    grad += torch.from_numpy(laplace_noise).to(torch.float32).to(grad.device)
    return grad / batch_size

# 10 256
def scale_grad8(grad):
    #grad = torch.clamp(grad, -5, 5)
    #grad /= batch_size
    grad = grad.reshape(10, 4, 8, 8)
    norms = torch.norm(grad, p=1, dim=(1, 2, 3)) + stable  # 计算每个通道上的2D梯度向量的1范数
    # print(norms)
    grad = grad / norms.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * scale  # 缩放梯度
    grad = grad.reshape(10, 256)
    return grad

#根据grad_diff 加入噪声
def add_laplace_noise8(grad, epsilon, grad_diff):
    # # 找到为 0 的元素并加上 1e-5
    # grad[grad == 0] += 1e-5
    grad_diff = grade_class8(grad_diff)

    grad_diff = grad_diff.reshape(10, 4, 8, 8)
    sum_123d = torch.abs(grad_diff).sum(dim=(1, 2, 3), keepdim=True) + stable
    sum_123d = sum_123d.squeeze(dim=1).squeeze(dim=1).squeeze(dim=1)
    # 在第一维上求和
    sum_1d = torch.sum(sum_123d)  # 形状为[32]
    # 对sum_23d每个元素除以对应的sum_1d的值    行列在通道上的占比
    result = sum_123d / sum_1d  # 形状为[32, 3]
    # 占比为负数  则置位取反， 为0则分配为R_lowerbound
    # result[result == 0] = R_lowerbound
    # result[result < 0] *= -1
    result_toNumpy = result.cpu().numpy()

    #result_toNumpy = np.clip(result_toNumpy, 0.8 / 4, 1.2 / 4)

    laplace_noise = np.random.laplace(0, delta8 / (epsilon * result_toNumpy[:, np.newaxis, np.newaxis, np.newaxis]),
                                      size=(10, 4, 8, 8))
    # laplace_noise1 = np.random(10, 16, 4, 4)
    # for i in range(10):
    #     for j in range(16):
    #         laplace_noise1[i, j, :, :] = np.random.laplace(0, delta / (16 * epsilon * result_toNumpy[i ,j]), size=(4, 4))
    #
    # print(result_toNumpy)
    # print(result_toNumpy[:, :, np.newaxis, np.newaxis])
    # print("=======================")
    # print(16 * epsilon * result_toNumpy[:, :, np.newaxis, np.newaxis])
    # print("=============")
    laplace_noise = laplace_noise.reshape(10, 256)
    grad += torch.from_numpy(laplace_noise).to(torch.float32).to(grad.device)
    return grad / batch_size


