from __future__ import print_function
import torch.nn as nn
from inversion_attack.inversion_attack_model import Classifier
from xor_and_ndb import XOR_pre
from weight_share_protect.Mydataset_for_numpy_client_UDK import *
# Training settings
batch_size = 256
test_batch_size = 1000
epochs = 1#10
lr = 0.1  # Learning rate
momentum = 0.5
no_cuda = False
seed = 1
log_interval = 10
nc = 3  # Number of channels
ndf = 256  # Number of discriminator filters
nz = 10  # Size of the latent Z vector
num_workers = 8

def add_laplace_noise(grad, epsilon=8.0, sensitivity=2.0):
    # 计算噪声的尺度参数
    scale = sensitivity / epsilon
    # 生成拉普拉斯噪声
    noise = torch.tensor(np.random.laplace(0, scale, grad.shape), dtype=torch.float32).to('cuda')
    # 添加噪声到梯度
    noisy_grad = grad + noise
    return noisy_grad

def train(classifier, log_interval, device, data_loader, optimizer, epoch,choice):
    classifier.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        if choice==2:
            data, target = data.to(device), target.to(device).squeeze(1)
        else:
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = classifier(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if choice == 1:
            # 对每个参数的梯度添加噪声
            epsilon = 5.0
            for param in classifier.parameters():
                if param.grad is not None:
                    param.grad.data = add_laplace_noise(param.grad.data, epsilon=epsilon)
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format( epoch, batch_idx * len(data),
                                                                  len(data_loader.dataset), loss.item()))

def test(classifier, device, data_loader,choice):
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            if choice == 2:
                data, target = data.to(device), target.to(device).squeeze(1)
            else:
                data, target = data.to(device), target.to(device)
            output = classifier(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest classifier: Average loss: {:.6f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
    return correct / len(data_loader.dataset)

def inversion_attack_classifer(result_path,choice=None):
    #路径
    os.makedirs(result_path+'out', exist_ok=True)

    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

    torch.manual_seed(seed)

    # 定义预处理变换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化处理
    ])
    if choice==0 or choice==1:
        # 加载CIFAR-10训练集
        train_set = datasets.CIFAR10(root='data/', train=True,
                                     download=True, transform=transform)

        # 加载CIFAR-10测试集
        test_set = datasets.CIFAR10(root='data/', train=False,
                                    download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, **kwargs)

    elif choice==2:
        train_dataset = XOR_pre.cifar10(root='data/', train=True, transform=transform)
        eval_dataset = XOR_pre.cifar10(root='data/', train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)

    elif choice == 3:

        data_path = 'weight_share_protect/UDK_fl_add_mul_sort'

        eval_dataset = Mydataset_numpy_client_UDK(mode='test', dataset=data_path, ID=4)
        train_dataset = Mydataset_numpy_client_UDK(mode='train', dataset=data_path, ID=4)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)

    classifier = nn.DataParallel(Classifier(nc=nc, ndf=ndf, nz=nz)).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True)

    best_cl_acc = 0
    best_cl_epoch = 0

    # Train classifier
    for epoch in range(1, epochs + 1):
        train(classifier, log_interval, device, train_loader, optimizer, epoch,choice)
        cl_acc = test(classifier, device, test_loader,choice)

        if cl_acc > best_cl_acc:
            best_cl_acc = cl_acc
            best_cl_epoch = epoch
            state = {
                'epoch': epoch,
                'model': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_cl_acc': best_cl_acc,
            }
            torch.save(state, result_path+'out/classifier.pth')

    print("Best classifier: epoch {}, acc {:.4f}".format(best_cl_epoch, best_cl_acc))


