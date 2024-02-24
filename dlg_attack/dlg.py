import time
import os
import math
import itertools

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim
from xor_and_ndb import XOR_pre
from dlg_attack.cc import *
from weight_share_protect.User_UDK import *
from weight_share_protect.Mydataset_for_numpy_client_UDK import *


def dlg_attack_no_noisy():
    batchsz = 1  # 128
    # 无隐私     acc   83.65   150
    dataset = 'cifar10'
    data_path = 'data/'
    save_path = 'dlg_attack/no_noisy_dlg_attack'

    learn_rate = 1.0
    num_dummy = 1
    Iteration = 500
    num_exp = 1  # 500

    cifar_train = datasets.CIFAR10(data_path, True, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.Resize((32, 32)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        # transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ]), download=True)

    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10(data_path, False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        #                           R      G      B
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    device = torch.device('cuda')
    model = ConvNet().to(device)
    net = ConvNet().to(device)
    # model = ResNet18().to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=3e-4)
    # print(model)

    for epoch in range(num_exp):
        model.train()
        # ========================================================================================================================
        for batchidx, (x, label) in enumerate(cifar_train):


            x, label = x.to(device), label.to(device)

            # image_array = x.squeeze().cpu().numpy()
            # image_array = np.transpose(image_array, (1, 2, 0))  # 转置为(32, 32, 3)
            # # #显示图像
            # plt.imshow(image_array, cmap='gray')
            # plt.title('Image')
            # plt.show()

            logits = model(x)
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()

            # 获取原模型的参数
            original_params = list(model.parameters())

            # 创建参数的深层副本
            copied_params = [param.detach().clone() for param in original_params]

            # 构建新的模型，使用深层副本的参数
            net = ConvNet().to(device)

            # 获取新模型的参数
            new_params = list(net.parameters())

            # 将深层副本的参数复制到新模型
            for new_param, copied_param in zip(new_params, copied_params):
                new_param.data.copy_(copied_param.data)

            tt = transforms.Compose([transforms.ToTensor()])
            tp = transforms.Compose([transforms.ToPILImage()])

            # print(dataset, 'root_path:', root_path)
            # print(dataset, 'data_path:', data_path)
            # print(dataset, 'save_path:', save_path)

            if not os.path.exists(save_path):
                os.mkdir(save_path)

            if dataset == 'cifar10':
                shape_img = (32, 32)
                num_classes = 10
                channel = 3
                hidden = 256
                dst = datasets.CIFAR10(data_path, download=True)

            print('running %d|%d experiment' % (epoch + 1, num_exp))
            idx_shuffle = np.random.permutation(len(dst))

            for method in ['DLG', 'iDLG']:
                print('%s, Try to generate %d images' % (method, num_dummy))

                imidx_list = []

                for imidx in range(num_dummy):
                    idx = idx_shuffle[imidx]
                    imidx_list.append(idx)
                    if imidx == 0:
                        gt_data = x
                        gt_label = label

                dy_dx = [param.grad for param in model.parameters()]
                original_dy_dx = list((_.detach().clone() for _ in dy_dx))

                # generate dummy data and label
                dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
                dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

                if method == 'DLG':
                    opt = torch.optim.LBFGS([dummy_data, dummy_label], lr=learn_rate)
                elif method == 'iDLG':
                    opt = torch.optim.LBFGS([dummy_data, ], lr=learn_rate)
                    # predict the ground-truth label
                    label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
                        (1,)).requires_grad_(False)

                history = []
                history_iters = []
                losses = []
                mses = []
                train_iters = []

                # print('lr =', learn_rate)
                for iters in range(Iteration):

                    def closure():
                        opt.zero_grad()
                        pred = net(dummy_data)
                        if method == 'DLG':
                            dummy_loss = - torch.mean(
                                torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)),
                                          dim=-1))
                            # dummy_loss = criterion(pred, gt_label)
                        elif method == 'iDLG':
                            dummy_loss = criteon(pred, label_pred)

                        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                        grad_diff = 0
                        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                            grad_diff += ((gx - gy) ** 2).sum()
                        grad_diff.backward()
                        return grad_diff

                    opt.step(closure)
                    current_loss = closure().item()
                    train_iters.append(iters)
                    losses.append(current_loss)
                    mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

                    if iters % int(Iteration / 10) == 0:
                        current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                        print(current_time, iters, 'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))
                        history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                        history_iters.append(iters)

                        for imidx in range(num_dummy):
                            plt.figure(figsize=(12, 8))
                            plt.subplot(3, 10, 1)
                            plt.imshow(tp(gt_data[imidx].cpu()))
                            for i in range(min(len(history), 29)):
                                plt.subplot(3, 10, i + 2)
                                plt.imshow(history[i][imidx])
                                plt.title('iter=%d' % (history_iters[i]))
                                plt.axis('off')
                            if method == 'DLG':
                                plt.savefig(
                                    '%s/DLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                                plt.close()
                            elif method == 'iDLG':
                                plt.savefig(
                                    '%s/iDLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                                plt.close()
                        # for imidx in range(num_dummy):
                        #     for i in range(min(len(history), 29)):
                        #         plt.figure(figsize=(4, 4))  # 设置一个正方形的图像框架
                        #         plt.subplot(1, 1, 1)
                        #         plt.imshow(tp(gt_data[imidx].cpu()))  # 使用插值方法显示图像
                        #         plt.axis('off')  # 关闭坐标轴显示
                        #         # 确保保存的图像不包括额外的空白区域
                        #         plt.savefig('%s/original_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]),
                        #                     dpi=300, bbox_inches='tight', pad_inches=0)
                        #         plt.close()
                        #
                        #         plt.figure(figsize=(4, 4))  # 为history中的每个图像创建新的图像
                        #         plt.subplot(1, 1, 1)
                        #         plt.imshow(history[i][imidx])  # 显示当前迭代的图像
                        #         plt.title('iter=%d' % (history_iters[i]))  # 设置图像标题为当前迭代次数
                        #         plt.axis('off')  # 关闭坐标轴显示
                        #
                        #         # 根据方法名称选择保存路径和文件名
                        #         if method == 'DLG':
                        #             file_name = '%s/DLG_iter%d_on_%s_%05d.png' % (
                        #             save_path, history_iters[i], imidx_list, imidx_list[imidx])
                        #         elif method == 'iDLG':
                        #             file_name = '%s/iDLG_iter%d_on_%s_%05d.png' % (
                        #             save_path, history_iters[i], imidx_list, imidx_list[imidx])
                        #
                        #         plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)  # 保存当前迭代的图像
                        #         plt.close()  # 关闭图像，释放资源

                        if current_loss < 0.000001:  # converge
                            break

                if method == 'DLG':
                    loss_DLG = losses
                    label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
                    mse_DLG = mses
                elif method == 'iDLG':
                    loss_iDLG = losses
                    label_iDLG = label_pred.item()
                    mse_iDLG = mses

            print('imidx_list:', imidx_list)
            print('loss_DLG:', loss_DLG[-1], 'loss_iDLG:', loss_iDLG[-1])
            print('mse_DLG:', mse_DLG[-1], 'mse_iDLG:', mse_iDLG[-1])
            print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_DLG:', label_DLG, 'lab_iDLG:',
                  label_iDLG)

            print('----------------------\n\n')

            optimizer.step()


def dlg_attack_add_noisy():
    batchsz = 1  # 128
    # 无隐私     acc   83.65   150
    epsilon = 8  # 82%
    # epsilon = 8  #
    # epsilon0 = 8  #
    # epsilon2 = 8  #
    # epsilon4 = 8  #
    # epsilon6 = 8  #
    # epsilon8 = 8  #
    epsilonb = 8  #
    # 保存前一层参数的梯度
    prev_grad_list = []
    # 获取当前梯度
    curr_grad_list = []
    # 保存梯度差
    grad_diff_list = []
    # 初始化存储前一个迭代梯度的字典
    prev_grads = {}
    dataset = 'cifar10'
    data_path = 'data/'
    save_path = 'dlg_attack/add_noisy_dlg_attack'

    learn_rate = 1.0
    num_dummy = 1
    Iteration = 500
    num_exp = 1  # 500

    cifar_train = datasets.CIFAR10(data_path, True, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.Resize((32, 32)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        # transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ]), download=True)

    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10(data_path, False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        #                           R      G      B
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    device = torch.device('cuda')
    model = ConvNet().to(device)
    net = ConvNet().to(device)
    # model = ResNet18().to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=3e-4)
    # print(model)

    for epoch in range(num_exp):
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            x, label = x.to(device), label.to(device)

            # image_array = x.squeeze().cpu().numpy()
            # image_array = np.transpose(image_array, (1, 2, 0))  # 转置为(32, 32, 3)
            # # #显示图像
            # plt.imshow(image_array, cmap='gray')
            # plt.title('Image')
            # plt.show()

            logits = model(x)
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()

            # 获取原模型的参数
            original_params = list(model.parameters())

            # 创建参数的深层副本
            copied_params = [param.detach().clone() for param in original_params]

            # 构建新的模型，使用深层副本的参数
            net = ConvNet().to(device)

            # 获取新模型的参数
            new_params = list(net.parameters())

            # 将深层副本的参数复制到新模型
            for new_param, copied_param in zip(new_params, copied_params):
                new_param.data.copy_(copied_param.data)

            # 清空当前梯度列表
            curr_grad_list = []

            # 遍历所有参数，计算梯度差异并添加噪声
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # 获取当前梯度
                    curr_grad = param.grad.clone()
                    curr_grad_list.append(curr_grad)

                    # 计算梯度差异
                    if name in prev_grads:
                        grad_diff = curr_grad - prev_grads[name]
                    else:
                        grad_diff = curr_grad.clone()

                    # 更新保存的梯度
                    prev_grads[name] = curr_grad.clone()

                    # 检查参数类型并应用相应的噪声函数
                    if 'bias' in name:
                        # 计算sh（偏置的数量）
                        sh = param.numel()
                        param.grad = add_laplace_noise_b(param.grad, epsilonb, grad_diff, sh)
                    else:
                        # 从参数名称中提取层索引
                        layer_name = name.split('.')[0]
                        layer_index = None
                        for i, layer in enumerate(['conv1', 'conv2', 'conv3', 'fc1', 'fc2']):
                            if layer in layer_name:
                                layer_index = i + 1
                                break

                        if layer_index is not None:
                            noise_function_name = f'add_laplace_noise{layer_index}'
                            if noise_function_name in globals():
                                noise_function = globals()[noise_function_name]
                                param.grad = noise_function(param.grad, epsilon, grad_diff)
                        else:
                            print(f"无法从参数名称中提取层索引：{name}")

            tt = transforms.Compose([transforms.ToTensor()])
            tp = transforms.Compose([transforms.ToPILImage()])

            # print(dataset, 'root_path:', root_path)
            # print(dataset, 'data_path:', data_path)
            # print(dataset, 'save_path:', save_path)

            if not os.path.exists(save_path):
                os.mkdir(save_path)

            if dataset == 'cifar10':
                shape_img = (32, 32)
                num_classes = 10
                channel = 3
                hidden = 256
                dst = datasets.CIFAR10(data_path, download=True)

            print('running %d|%d experiment' % (epoch + 1, num_exp))
            idx_shuffle = np.random.permutation(len(dst))

            for method in ['DLG', 'iDLG']:
                print('%s, Try to generate %d images' % (method, num_dummy))

                imidx_list = []

                for imidx in range(num_dummy):
                    idx = idx_shuffle[imidx]
                    imidx_list.append(idx)
                    if imidx == 0:
                        gt_data = x
                        gt_label = label

                dy_dx = [param.grad for param in model.parameters()]
                original_dy_dx = list((_.detach().clone() for _ in dy_dx))

                # generate dummy data and label
                dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
                dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

                if method == 'DLG':
                    opt = torch.optim.LBFGS([dummy_data, dummy_label], lr=learn_rate)
                elif method == 'iDLG':
                    opt = torch.optim.LBFGS([dummy_data, ], lr=learn_rate)
                    # predict the ground-truth label
                    label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
                        (1,)).requires_grad_(False)

                history = []
                history_iters = []
                losses = []
                mses = []
                train_iters = []

                # print('lr =', learn_rate)
                for iters in range(Iteration):

                    def closure():
                        opt.zero_grad()
                        pred = net(dummy_data)
                        if method == 'DLG':
                            dummy_loss = - torch.mean(
                                torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)),
                                          dim=-1))
                            # dummy_loss = criterion(pred, gt_label)
                        elif method == 'iDLG':
                            dummy_loss = criteon(pred, label_pred)

                        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                        grad_diff = 0
                        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                            grad_diff += ((gx - gy) ** 2).sum()
                        grad_diff.backward()
                        return grad_diff

                    opt.step(closure)
                    current_loss = closure().item()
                    train_iters.append(iters)
                    losses.append(current_loss)
                    mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

                    if iters % int(Iteration / 10) == 0:
                        current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                        print(current_time, iters, 'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))
                        history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                        history_iters.append(iters)

                        for imidx in range(num_dummy):
                            plt.figure(figsize=(12, 8))
                            plt.subplot(3, 10, 1)
                            plt.imshow(tp(gt_data[imidx].cpu()))
                            for i in range(min(len(history), 29)):
                                plt.subplot(3, 10, i + 2)
                                plt.imshow(history[i][imidx])
                                plt.title('iter=%d' % (history_iters[i]))
                                plt.axis('off')
                            if method == 'DLG':
                                plt.savefig(
                                    '%s/DLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                                plt.close()
                            elif method == 'iDLG':
                                plt.savefig(
                                    '%s/iDLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                                plt.close()

                        if current_loss < 0.000001:  # converge
                            break

                if method == 'DLG':
                    loss_DLG = losses
                    label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
                    mse_DLG = mses
                elif method == 'iDLG':
                    loss_iDLG = losses
                    label_iDLG = label_pred.item()
                    mse_iDLG = mses

            print('imidx_list:', imidx_list)
            print('loss_DLG:', loss_DLG[-1], 'loss_iDLG:', loss_iDLG[-1])
            print('mse_DLG:', mse_DLG[-1], 'mse_iDLG:', mse_iDLG[-1])
            print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_DLG:', label_DLG, 'lab_iDLG:',
                  label_iDLG)

            print('----------------------\n\n')

            optimizer.step()
            if prev_grad_list:
                grad_diff_list = [curr - prev for curr, prev in zip(curr_grad_list, prev_grad_list)]
            # 保存当前梯度作为下一次迭代的上一个梯度
            prev_grad_list = curr_grad_list.copy()

        # print(epoch, 'loss:', loss.item())

        # model.eval()
        # with torch.no_grad():
        #     # test
        #     total_correct = 0
        #     total_num = 0
        #     for x, label in cifar_test:
        #         # [b, 3, 32, 32]
        #         # [b]
        #         x, label = x.to(device), label.to(device)
        #
        #         # [b, 10]
        #         logits = model(x)
        #         # [b]
        #         pred = logits.argmax(dim=1)
        #         # [b] vs [b] => scalar tensor
        #         correct = torch.eq(pred, label).float().sum().item()
        #         total_correct += correct
        #         total_num += x.size(0)
        #         # print(correct)
        #
        #     acc = total_correct / total_num
        #     print(epoch, 'test acc:', acc)


def dlg_attack_xor_and_ndb():
    batchsz = 1  # 128
    # 无隐私     acc   83.65   150
    dataset = 'cifar10'
    data_path = 'data/'
    save_path = 'dlg_attack/xor_and_ndb_dlg_attack'

    learn_rate = 1.0
    num_dummy = 1
    Iteration = 500
    num_exp = 1  # 500
    transform_train = transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.Resize((32, 32)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        # transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        #                           R      G      B
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    cifar_train = XOR_pre.cifar10(root=data_path, train=True, transform=transform_train)
    cifar_test = XOR_pre.cifar10(root=data_path, train=False, transform=transform_test)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    device = torch.device('cuda')
    model = ConvNet().to(device)
    net = ConvNet().to(device)
    # model = ResNet18().to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=3e-4)
    # print(model)

    for epoch in range(num_exp):
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            x, label = x.to(device), label.to(device).squeeze(1)

            # image_array = x.squeeze().cpu().numpy()
            # image_array = np.transpose(image_array, (1, 2, 0))  # 转置为(32, 32, 3)
            # # #显示图像
            # plt.imshow(image_array, cmap='gray')
            # plt.title('Image')
            # plt.show()

            logits = model(x)
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()

            # 获取原模型的参数
            original_params = list(model.parameters())

            # 创建参数的深层副本
            copied_params = [param.detach().clone() for param in original_params]

            # 构建新的模型，使用深层副本的参数
            net = ConvNet().to(device)

            # 获取新模型的参数
            new_params = list(net.parameters())

            # 将深层副本的参数复制到新模型
            for new_param, copied_param in zip(new_params, copied_params):
                new_param.data.copy_(copied_param.data)

            tt = transforms.Compose([transforms.ToTensor()])
            tp = transforms.Compose([transforms.ToPILImage()])

            # print(dataset, 'root_path:', root_path)
            # print(dataset, 'data_path:', data_path)
            # print(dataset, 'save_path:', save_path)

            if not os.path.exists(save_path):
                os.mkdir(save_path)

            if dataset == 'cifar10':
                shape_img = (32, 32)
                num_classes = 10
                channel = 3
                hidden = 256
                dst = XOR_pre.cifar10(data_path, train=True)

            print('running %d|%d experiment' % (epoch + 1, num_exp))
            idx_shuffle = np.random.permutation(len(dst))

            for method in ['DLG', 'iDLG']:
                print('%s, Try to generate %d images' % (method, num_dummy))

                imidx_list = []

                for imidx in range(num_dummy):
                    idx = idx_shuffle[imidx]
                    imidx_list.append(idx)
                    if imidx == 0:
                        gt_data = x
                        gt_label = label

                dy_dx = [param.grad for param in model.parameters()]
                original_dy_dx = list((_.detach().clone() for _ in dy_dx))

                # generate dummy data and label
                dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
                dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

                if method == 'DLG':
                    opt = torch.optim.LBFGS([dummy_data, dummy_label], lr=learn_rate)
                elif method == 'iDLG':
                    opt = torch.optim.LBFGS([dummy_data, ], lr=learn_rate)
                    # predict the ground-truth label
                    label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
                        (1,)).requires_grad_(False)

                history = []
                history_iters = []
                losses = []
                mses = []
                train_iters = []

                # print('lr =', learn_rate)
                for iters in range(Iteration):

                    def closure():
                        opt.zero_grad()
                        pred = net(dummy_data)
                        if method == 'DLG':
                            dummy_loss = - torch.mean(
                                torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)),
                                          dim=-1))
                            # dummy_loss = criterion(pred, gt_label)
                        elif method == 'iDLG':
                            dummy_loss = criteon(pred, label_pred)

                        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                        grad_diff = 0
                        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                            grad_diff += ((gx - gy) ** 2).sum()
                        grad_diff.backward()
                        return grad_diff

                    opt.step(closure)
                    current_loss = closure().item()
                    train_iters.append(iters)
                    losses.append(current_loss)
                    mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

                    if iters % int(Iteration / 10) == 0:
                        current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                        print(current_time, iters, 'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))
                        history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                        history_iters.append(iters)

                        for imidx in range(num_dummy):
                            plt.figure(figsize=(12, 8))
                            plt.subplot(3, 10, 1)
                            plt.imshow(tp(gt_data[imidx].cpu()))
                            for i in range(min(len(history), 29)):
                                plt.subplot(3, 10, i + 2)
                                plt.imshow(history[i][imidx])
                                plt.title('iter=%d' % (history_iters[i]))
                                plt.axis('off')
                            if method == 'DLG':
                                plt.savefig(
                                    '%s/DLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                                plt.close()
                            elif method == 'iDLG':
                                plt.savefig(
                                    '%s/iDLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                                plt.close()

                        if current_loss < 0.000001:  # converge
                            break

                if method == 'DLG':
                    loss_DLG = losses
                    label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
                    mse_DLG = mses
                elif method == 'iDLG':
                    loss_iDLG = losses
                    label_iDLG = label_pred.item()
                    mse_iDLG = mses

            print('imidx_list:', imidx_list)
            print('loss_DLG:', loss_DLG[-1], 'loss_iDLG:', loss_iDLG[-1])
            print('mse_DLG:', mse_DLG[-1], 'mse_iDLG:', mse_iDLG[-1])
            print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_DLG:', label_DLG, 'lab_iDLG:',
                  label_iDLG)

            print('----------------------\n\n')

            optimizer.step()


def dlg_attack_pputl(G):
    G.eval()
    batchsz = 1  # 128
    # 无隐私     acc   83.65   150
    dataset = 'cifar10'
    data_path = 'data/'
    save_path = 'dlg_attack/pputl_dlg_attack'

    learn_rate = 1.0
    num_dummy = 1
    Iteration = 500
    num_exp = 1  # 500

    cifar_train = datasets.CIFAR10(data_path, True, download=True)
    num_batches = math.ceil(len(cifar_train) / batchsz)

    device = torch.device('cuda')
    model = ConvNet2().to(device)
    net = ConvNet2().to(device)
    # model = ResNet18().to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=3e-4)
    # print(model)

    for epoch in range(num_exp):
        model.train()
        for i in range(num_batches):
            z_ = torch.randn(batchsz, 128).to(device)
            y_d = (torch.rand(batchsz, 1) * 10).type(torch.LongTensor).to(device)
            y_label_ = torch.zeros(batchsz, 10).to(device)
            y_label_.scatter_(1, y_d.view(batchsz, 1), 1)
            y_label_c_ = y_d.view(batchsz).to(device)
            G_result = G(z_, y_label_)
            C_result = model(G_result)
            loss = criteon(C_result, y_label_c_)
            average_intensity = torch.mean(G_result).item()
            # 将平均强度转换为一个随机数字（例如，通过取整）
            random_number = int(average_intensity * 100) % 10
            # image_array = x.squeeze().cpu().numpy()
            # image_array = np.transpose(image_array, (1, 2, 0))  # 转置为(32, 32, 3)
            # # #显示图像
            # plt.imshow(image_array, cmap='gray')
            # plt.title('Image')
            # plt.show()

            # backprop
            optimizer.zero_grad()
            loss.backward()

            # 获取原模型的参数
            original_params = list(model.parameters())

            # 创建参数的深层副本
            copied_params = [param.detach().clone() for param in original_params]

            # 构建新的模型，使用深层副本的参数
            net = ConvNet2().to(device)

            # 获取新模型的参数
            new_params = list(net.parameters())

            # 将深层副本的参数复制到新模型
            for new_param, copied_param in zip(new_params, copied_params):
                new_param.data.copy_(copied_param.data)

            tt = transforms.Compose([transforms.ToTensor()])
            tp = transforms.Compose([transforms.ToPILImage()])

            # print(dataset, 'root_path:', root_path)
            # print(dataset, 'data_path:', data_path)
            # print(dataset, 'save_path:', save_path)

            if not os.path.exists(save_path):
                os.mkdir(save_path)

            print('running %d|%d experiment' % (epoch + 1, num_exp))

            for method in ['DLG', 'iDLG']:
                print('%s, Try to generate %d images' % (method, num_dummy))

                gt_data = G_result
                gt_label = y_label_c_

                # 获取模型参数的梯度
                dy_dx = [param.grad for param in model.parameters()]
                original_dy_dx = [_.detach().clone() for _ in dy_dx]

                # 生成虚拟数据和标签
                dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
                dummy_label = torch.randn((gt_data.shape[0], 10)).to(device).requires_grad_(True)

                if method == 'DLG':
                    opt = torch.optim.LBFGS([dummy_data, dummy_label], lr=learn_rate)
                elif method == 'iDLG':
                    opt = torch.optim.LBFGS([dummy_data, ], lr=learn_rate)
                    # predict the ground-truth label
                    label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
                        (1,)).requires_grad_(False)

                history = []
                history_iters = []
                losses = []
                mses = []
                train_iters = []

                # print('lr =', learn_rate)
                for iters in range(Iteration):

                    def closure():
                        opt.zero_grad()
                        pred = net(dummy_data)
                        if method == 'DLG':
                            dummy_loss = - torch.mean(
                                torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)),
                                          dim=-1))
                            # dummy_loss = criterion(pred, gt_label)
                        elif method == 'iDLG':
                            dummy_loss = criteon(pred, label_pred)

                        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                        grad_diff = 0
                        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                            grad_diff += ((gx - gy) ** 2).sum()
                        grad_diff.backward()
                        return grad_diff

                    opt.step(closure)
                    current_loss = closure().item()
                    train_iters.append(iters)
                    losses.append(current_loss)
                    mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

                    if iters % int(Iteration / 10) == 0:
                        current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                        print(current_time, iters, 'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))
                        history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                        history_iters.append(iters)

                        for imidx in range(num_dummy):
                            plt.figure(figsize=(12, 8))
                            plt.subplot(3, 10, 1)
                            img = gt_data[imidx].cpu().data.permute(1, 2, 0).numpy()
                            plt.imshow((img + 1) / 2)
                            for i in range(min(len(history), 29)):
                                plt.subplot(3, 10, i + 2)
                                plt.imshow(history[i][imidx])
                                plt.title('iter=%d' % (history_iters[i]))
                                plt.axis('off')

                            if method == 'DLG':
                                plt.savefig('%s/DLG_%d.png' % (save_path, random_number))
                            elif method == 'iDLG':
                                plt.savefig('%s/iDLG_%d.png' % (save_path, random_number))
                            plt.close()

                        if current_loss < 0.000001:  # converge
                            break

                if method == 'DLG':
                    loss_DLG = losses
                    label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
                    mse_DLG = mses
                elif method == 'iDLG':
                    loss_iDLG = losses
                    label_iDLG = label_pred.item()
                    mse_iDLG = mses

            print('loss_DLG:', loss_DLG[-1], 'loss_iDLG:', loss_iDLG[-1])
            print('mse_DLG:', mse_DLG[-1], 'mse_iDLG:', mse_iDLG[-1])
            print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_DLG:', label_DLG, 'lab_iDLG:',
                  label_iDLG)

            print('----------------------\n\n')

            optimizer.step()


def dlg_attack_weight_share_protect():
    batchsz = 1  # 128
    # 无隐私     acc   83.65   150
    dataset = 'cifar10'
    data_path = 'weight_share_protect/UDK_fl_add_mul_sort'
    save_path = 'dlg_attack/weight_share_protect_dlg_attack'

    learn_rate = 1.0  # 1.0
    num_dummy = 1
    Iteration = 500
    num_exp = 1  # 500

    cifar_train = torch.utils.data.DataLoader(
        Mydataset_numpy_client_UDK(mode='train', dataset=data_path, ID=4),
        batch_size=batchsz, shuffle=True, num_workers=8, pin_memory=True)
    cifar_test = torch.utils.data.DataLoader(
        Mydataset_numpy_client_UDK(mode='test', dataset=data_path, ID=4),
        batch_size=batchsz, shuffle=True, num_workers=8, pin_memory=True)

    device = torch.device('cuda')
    model = ConvNet().to(device)
    net = ConvNet().to(device)
    # model = ResNet18().to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=3e-4)
    # print(model)

    for epoch in range(num_exp):
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            x, label = x.to(device), label.to(device)

            # image_array = x.squeeze().cpu().numpy()
            # image_array = np.transpose(image_array, (1, 2, 0))  # 转置为(32, 32, 3)
            # # #显示图像
            # plt.imshow(image_array, cmap='gray')
            # plt.title('Image')
            # plt.show()

            logits = model(x)
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()

            # 获取原模型的参数
            original_params = list(model.parameters())

            # 创建参数的深层副本
            copied_params = [param.detach().clone() for param in original_params]

            # 构建新的模型，使用深层副本的参数
            net = ConvNet().to(device)

            # 获取新模型的参数
            new_params = list(net.parameters())

            # 将深层副本的参数复制到新模型
            for new_param, copied_param in zip(new_params, copied_params):
                new_param.data.copy_(copied_param.data)

            tt = transforms.Compose([transforms.ToTensor()])
            tp = transforms.Compose([transforms.ToPILImage()])

            # print(dataset, 'root_path:', root_path)
            # print(dataset, 'data_path:', data_path)
            # print(dataset, 'save_path:', save_path)

            if not os.path.exists(save_path):
                os.mkdir(save_path)

            if dataset == 'cifar10':
                shape_img = (32, 32)
                num_classes = 10
                channel = 3
                hidden = 256
                dst = Mydataset_numpy_client_UDK(mode='train', dataset=data_path, ID=1)

            print('running %d|%d experiment' % (epoch + 1, num_exp))
            idx_shuffle = np.random.permutation(len(dst))

            for method in ['DLG', 'iDLG']:
                print('%s, Try to generate %d images' % (method, num_dummy))

                imidx_list = []

                for imidx in range(num_dummy):
                    idx = idx_shuffle[imidx]
                    imidx_list.append(idx)
                    if imidx == 0:
                        gt_data = x
                        gt_label = label

                dy_dx = [param.grad for param in model.parameters()]
                original_dy_dx = list((_.detach().clone() for _ in dy_dx))

                # generate dummy data and label
                dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
                dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

                if method == 'DLG':
                    opt = torch.optim.LBFGS([dummy_data, dummy_label], lr=learn_rate)
                elif method == 'iDLG':
                    opt = torch.optim.LBFGS([dummy_data, ], lr=learn_rate)
                    # predict the ground-truth label
                    label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
                        (1,)).requires_grad_(False)

                history = []
                history_iters = []
                losses = []
                mses = []
                train_iters = []

                # print('lr =', learn_rate)
                for iters in range(Iteration):

                    def closure():
                        opt.zero_grad()
                        pred = net(dummy_data)
                        if method == 'DLG':
                            dummy_loss = - torch.mean(
                                torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)),
                                          dim=-1))
                            # dummy_loss = criterion(pred, gt_label)
                        elif method == 'iDLG':
                            dummy_loss = criteon(pred, label_pred)

                        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                        grad_diff = 0
                        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                            grad_diff += ((gx - gy) ** 2).sum()
                        grad_diff.backward()
                        return grad_diff

                    opt.step(closure)
                    current_loss = closure().item()
                    train_iters.append(iters)
                    losses.append(current_loss)
                    mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

                    if iters % int(Iteration / 10) == 0:
                        current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                        print(current_time, iters, 'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))
                        history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                        history_iters.append(iters)

                        for imidx in range(num_dummy):
                            plt.figure(figsize=(12, 8))
                            plt.subplot(3, 10, 1)
                            plt.imshow(tp(gt_data[imidx].cpu()))
                            for i in range(min(len(history), 29)):
                                plt.subplot(3, 10, i + 2)
                                plt.imshow(history[i][imidx])
                                plt.title('iter=%d' % (history_iters[i]))
                                plt.axis('off')
                            if method == 'DLG':
                                plt.savefig(
                                    '%s/DLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                                plt.close()
                            elif method == 'iDLG':
                                plt.savefig(
                                    '%s/iDLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                                plt.close()

                        if current_loss < 0.000001:  # converge
                            break

                if method == 'DLG':
                    loss_DLG = losses
                    label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
                    mse_DLG = mses
                elif method == 'iDLG':
                    loss_iDLG = losses
                    label_iDLG = label_pred.item()
                    mse_iDLG = mses

            print('imidx_list:', imidx_list)
            print('loss_DLG:', loss_DLG[-1], 'loss_iDLG:', loss_iDLG[-1])
            print('mse_DLG:', mse_DLG[-1], 'mse_iDLG:', mse_iDLG[-1])
            print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_DLG:', label_DLG, 'lab_iDLG:',
                  label_iDLG)

            print('----------------------\n\n')

            optimizer.step()
