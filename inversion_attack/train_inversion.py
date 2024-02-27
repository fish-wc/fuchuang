from __future__ import print_function
import torch.nn as nn
import os, shutil
from inversion_attack.inversion_attack_model import Classifier, Inversion
import torchvision.utils as vutils
from xor_and_ndb import XOR_pre
from weight_share_protect.Mydataset_for_numpy_client_UDK import *

# Training settings
batch_size = 256
test_batch_size = 1000
epochs = 1  # 10
lr = 0.1  # Learning rate
momentum = 0.5
no_cuda = False
seed = 1
log_interval = 10
nc = 3  # Number of channels
ndf = 256  # Number of discriminator filters
ngf = 512  # Number of generator filters
nz = 10  # Size of the latent Z vector
truncation = 5
c = 50.0  # A constant for adjustment, could be used for any purpose in the model
num_workers = 8


def train(classifier, inversion, log_interval, device, data_loader, optimizer, epoch, choice):
    classifier.eval()
    inversion.train()

    for batch_idx, (data, target) in enumerate(data_loader):
        if choice == 2:
            data, target = data.to(device), target.to(device).squeeze(1)
        else:
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            prediction = classifier(data, release=True)
        reconstruction = inversion(prediction)
        loss = F.mse_loss(reconstruction, data)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                 len(data_loader.dataset), loss.item()))


def test(classifier, inversion, device, data_loader, epoch, msg, result_path, choice):
    classifier.eval()
    inversion.eval()
    mse_loss = 0
    plot = True
    with torch.no_grad():
        for data, target in data_loader:
            if choice == 2:
                data, target = data.to(device), target.to(device).squeeze(1)
            else:
                data, target = data.to(device), target.to(device)
            prediction = classifier(data, release=True)
            reconstruction = inversion(prediction)
            mse_loss += F.mse_loss(reconstruction, data, reduction='sum').item()

            if plot:
                # 选择要展示的图像数量，确保它们能在一张图中合理展示
                num_images = 16  # 选择较小的数量以适应32x32的尺寸
                truth = data[:num_images]
                inverse = reconstruction[:num_images]
                # 交替排列真实图像和重构图像
                out = torch.cat((inverse, truth), dim=0)
                # # 保存对比图像，不需要对i进行迭代
                # 路径
                vutils.save_image(out, '{}out/recon_{}_{}.png'.format(result_path, msg.replace(" ", ""), epoch),
                                  nrow=num_images,
                                  normalize=True)
                plot = False

    mse_loss /= len(data_loader.dataset) * 32 * 32
    print('\nTest inversion model on {} set: Average MSE loss: {:.6f}\n'.format(msg, mse_loss))
    return mse_loss


def inversion_model_train(result_path, choice=None):
    # 路径
    os.makedirs(result_path + 'out', exist_ok=True)

    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

    torch.manual_seed(seed)

    # 定义预处理变换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化处理
    ])
    if choice == 0 or choice == 1:
        # 加载CIFAR-10训练集
        train_set = datasets.CIFAR10(root='data/', train=True,
                                     download=True, transform=transform)

        # 加载CIFAR-10测试集
        test_set = datasets.CIFAR10(root='data/', train=False,
                                    download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
        test1_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, **kwargs)

    elif choice == 2:
        train_dataset = XOR_pre.cifar10(root='data/', train=True, transform=transform)
        eval_dataset = XOR_pre.cifar10(root='data/', train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        test1_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)

    elif choice == 3:

        data_path = 'weight_share_protect/UDK_fl_add_mul_sort'

        eval_dataset = Mydataset_numpy_client_UDK(mode='test', dataset=data_path, ID=4)
        train_dataset = Mydataset_numpy_client_UDK(mode='train', dataset=data_path, ID=4)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        test1_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)

    classifier = nn.DataParallel(Classifier(nc=nc, ndf=ndf, nz=nz)).to(device)
    inversion = nn.DataParallel(Inversion(nc=nc, ngf=ngf, nz=nz, truncation=truncation, c=c)).to(device)
    optimizer = optim.Adam(inversion.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True)

    # 路径Load classifier
    path = result_path + 'out/classifier.pth'
    try:
        checkpoint = torch.load(path)
        classifier.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        best_cl_acc = checkpoint['best_cl_acc']
        print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch, best_cl_acc))
    except:
        print("=> load classifier checkpoint '{}' failed".format(path))
        return

    # Train inversion model
    best_recon_loss = 99999999
    for epoch in range(1, epochs + 1):
        train(classifier, inversion, log_interval, device, train_loader, optimizer, epoch, choice)
        recon_loss = test(classifier, inversion, device, test1_loader, epoch, 'test', result_path, choice)

        if recon_loss < best_recon_loss:
            best_recon_loss = recon_loss
            state = {
                'epoch': epoch,
                'model': inversion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_recon_loss': best_recon_loss
            }
            # 路径
            torch.save(state, result_path + 'out/inversion.pth')
            shutil.copyfile(result_path + 'out/recon_test_{}.png'.format(epoch), result_path + 'out/best_test.png')

    for i in range(1, epochs + 1):
        # 路径
        os.remove(f'{result_path}out/recon_test_{i}.png')
