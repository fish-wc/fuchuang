from ml_attack.classifier import train as train_model, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from xor_and_ndb import XOR_pre
from weight_share_protect.Mydataset_for_numpy_client_UDK import *
import torch.optim as optim
np.random.seed(21312)
test_feat = None  # 测试特征文件路径，默认为 None
test_label = None  # 测试标签文件路径，默认为 None
save_model = False  # 是否保存模型，默认为 0 (不保存)
MODEL_PATH = 'ml_attack/ml_model/'
DATA_PATH = 'data/'
# 基本数据配置
train_feat = 'ml_attack/cifar10_train_features.txt'  # 训练特征文件路径
train_label = 'ml_attack/cifar10_train_labels.txt'  # 训练标签文件路径
ml_data = 'ml_attack/ml_data/'
target_datapath = 'target_data/'
shadow_datapath = 'shadow_data/'
if not os.path.exists(ml_data):
    os.makedirs(ml_data)

if not os.path.exists(ml_data + target_datapath):
    os.makedirs(ml_data + target_datapath)

if not os.path.exists(ml_data + shadow_datapath):
    os.makedirs(ml_data + shadow_datapath)

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
# 如果未提供测试数据，则进行训练测试分割的配置
test_ratio = 0.3  # 训练/测试分割比例，默认为 0.3

# 目标模型和影子模型配置
n_shadow = 10  # 影子模型数量，默认为 10
target_data_size = int(1e4)  # 目标模型使用的数据点数量，默认为 10000
target_model = 'nn'  # 目标模型类型，默认为 'nn'
target_learning_rate = 0.01  # 目标模型学习率，默认为 0.01
target_batch_size = 100  # 目标模型批大小，默认为 100
target_n_hidden = 50  # 目标模型隐藏层节点数，默认为 50
target_epochs = 50  # 目标模型训练轮数，默认为 50
target_l2_ratio = 1e-6  # 目标模型L2正则化比率，默认为 1e-6

# 攻击模型配置
attack_model = 'softmax'  # 攻击模型类型，默认为 'softmax'
attack_learning_rate = 0.01  # 攻击模型学习率，默认为 0.01
attack_batch_size = 100  # 攻击模型批大小，默认为 100
attack_n_hidden = 50  # 攻击模型隐藏层节点数，默认为 50
attack_epochs = 50  # 攻击模型训练轮数，默认为 50
attack_l2_ratio = 1e-6  # 攻击模型L2正则化比率，默认为 1e-6



class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def add_laplace_noise(grad, epsilon=8.0, sensitivity=2.0):
    # 计算噪声的尺度参数
    scale = sensitivity / epsilon
    # 生成拉普拉斯噪声
    noise = torch.tensor(np.random.laplace(0, scale, grad.shape), dtype=torch.float32)
    # 添加噪声到梯度
    noisy_grad = grad + noise
    return noisy_grad

def load_trained_indices():
    fname = MODEL_PATH + 'data_indices.npz'
    with np.load(fname) as f:
        indices = [f['arr_%d' % i] for i in range(len(f.files))]
    return indices


def get_data_indices(data_size, target_train_size=int(1e4), sample_target_data=True):
    train_indices = np.arange(data_size)
    if sample_target_data:
        target_data_indices = np.random.choice(train_indices, target_train_size, replace=False)
        shadow_indices = np.setdiff1d(train_indices, target_data_indices)
    else:
        target_data_indices = train_indices[:target_train_size]
        shadow_indices = train_indices[target_train_size:]
    return target_data_indices, shadow_indices


def load_attack_data():
    fname = MODEL_PATH + 'attack_train_data.npz'
    with np.load(fname) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    fname = MODEL_PATH + 'attack_test_data.npz'
    with np.load(fname) as f:
        test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x.astype('float32'), train_y.astype('int32'), test_x.astype('float32'), test_y.astype('int32')


def train_target_model(dataset, epochs=100, batch_size=100, learning_rate=0.01, l2_ratio=1e-7,
                       n_hidden=512, model='nn',input_size=3072, output_size=10,save=True,choice=None):
    if model == 'nn':
        # 初始化神经网络模型
       model = SimpleNN(input_size, n_hidden, output_size)
    train_x, train_y, test_x, test_y = dataset
    train_dataset = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y).long())
    test_dataset = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y).long())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_ratio)

    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if choice==1:
                # 对每个参数的梯度添加噪声
                epsilon =5.0
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data = add_laplace_noise(param.grad.data, epsilon=epsilon)
            optimizer.step()

    # Collecting data for attack model
    model.eval()
    attack_x, attack_y = [], []
    with torch.no_grad():
        for inputs, labels in train_loader:
            outputs = torch.softmax(model(inputs), dim=1)
            attack_x.append(outputs.numpy())
            attack_y.append(np.ones(labels.size(0)))
        for inputs, labels in test_loader:
            outputs = torch.softmax(model(inputs), dim=1)
            attack_x.append(outputs.numpy())
            attack_y.append(np.zeros(labels.size(0)))

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')

    if save:
        np.savez(MODEL_PATH + 'attack_test_data.npz', attack_x, attack_y)
        # 保存PyTorch模型的推荐方法是保存整个模型或者模型的状态字典
        torch.save(model.state_dict(), MODEL_PATH + 'target_model.pth')

    classes = np.concatenate([train_y, test_y])
    return attack_x, attack_y, classes




def train_shadow_models(n_hidden=512, epochs=100, n_shadow=20, learning_rate=0.05, batch_size=100, l2_ratio=1e-7,model='nn',
                        input_size= 3072, output_size=10, save=True):
    attack_x, attack_y = [], []
    classes = []

    for i in range(n_shadow):
        print(f'Training shadow model {i}')
        #路径
        data = np.load(f'{ml_data}{shadow_datapath}shadow{i}_data.npz')
        train_x, train_y, test_x, test_y = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

        # Convert numpy arrays to torch tensors
        train_dataset = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y).long())
        test_dataset = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y).long())

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if model == 'nn':
            # 初始化神经网络模型
            model = SimpleNN(input_size, n_hidden, output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_ratio)

        # Training the model
        model.train()
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Collecting data for attack model
        model.eval()
        with torch.no_grad():
            for inputs, labels in train_loader:
                outputs = torch.softmax(model(inputs), dim=1)
                attack_x.append(outputs.numpy())
                attack_y.append(np.ones(labels.size(0)))

            for inputs, labels in test_loader:
                outputs = torch.softmax(model(inputs), dim=1)
                attack_x.append(outputs.numpy())
                attack_y.append(np.zeros(labels.size(0)))

        classes.append(np.concatenate([train_y, test_y]))

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y).astype('int32')
    classes = np.concatenate(classes)

    if save:
        np.savez('attack_train_data.npz', attack_x, attack_y)

    return attack_x, attack_y, classes


def train_attack_model(classes, dataset=None, n_hidden=512, learning_rate=0.01, batch_size=200, epochs=50,
                       model='nn', l2_ratio=1e-7):
    if dataset is None:
        dataset = load_attack_data()

    train_x, train_y, test_x, test_y = dataset

    train_classes, test_classes = classes
    train_indices = np.arange(len(train_x))
    test_indices = np.arange(len(test_x))
    unique_classes = np.unique(train_classes)

    true_y = []
    pred_y = []
    for c in unique_classes:
        print('Training attack model for class {}...'.format(c))
        c_train_indices = train_indices[train_classes == c]
        c_train_x, c_train_y = train_x[c_train_indices], train_y[c_train_indices]
        c_test_indices = test_indices[test_classes == c]
        c_test_x, c_test_y = test_x[c_test_indices], test_y[c_test_indices]
        c_dataset = (c_train_x, c_train_y, c_test_x, c_test_y)
        c_pred_y = train_model(c_dataset, n_hidden=n_hidden, epochs=epochs, learning_rate=learning_rate,
                               batch_size=batch_size, model=model, rtn_layer=False, l2_ratio=l2_ratio)
        true_y.append(c_test_y)
        pred_y.append(c_pred_y)

    print ('-' * 10 + 'FINAL EVALUATION' + '-' * 10 + '\n')
    true_y = np.concatenate(true_y)
    pred_y = np.concatenate(pred_y)
    print('Testing Accuracy: {}'.format(accuracy_score(true_y, pred_y)))
    return classification_report(true_y, pred_y)


def save_data():
    print('-' * 10 + 'SAVING DATA TO DISK' + '-' * 10 + '\n')

    x, y, test_x, test_y = load_dataset(train_feat, train_label, test_feat, train_label)
    if test_x is None:
        print ('Splitting train/test data with ratio {}/{}'.format(1 - test_ratio, test_ratio))
        x, test_x, y, test_y = train_test_split(x, y, test_size=test_ratio, stratify=y)

    # need to partition target and shadow model data
    assert len(x) > 2 * target_data_size

    target_data_indices, shadow_indices = get_data_indices(len(x), target_train_size=target_data_size)
    np.savez(MODEL_PATH + 'data_indices.npz', target_data_indices, shadow_indices)

    # target model's data
    print('Saving data for target model')
    train_x, train_y = x[target_data_indices], y[target_data_indices]
    size = len(target_data_indices)
    if size < len(test_x):
        test_x = test_x[:size]
        test_y = test_y[:size]
    # save target data
    np.savez(ml_data+target_datapath + 'target_data.npz', train_x, train_y, test_x, test_y)

    # shadow model's data
    target_size = len(target_data_indices)
    shadow_x, shadow_y = x[shadow_indices], y[shadow_indices]
    shadow_indices = np.arange(len(shadow_indices))

    for i in range(n_shadow):
        print('Saving data for shadow model {}'.format(i))
        shadow_i_indices = np.random.choice(shadow_indices, 2 * target_size, replace=False)
        shadow_i_x, shadow_i_y = shadow_x[shadow_i_indices], shadow_y[shadow_i_indices]
        train_x, train_y = shadow_i_x[:target_size], shadow_i_y[:target_size]
        test_x, test_y = shadow_i_x[target_size:], shadow_i_y[target_size:]
        np.savez(ml_data+shadow_datapath + 'shadow{}_data.npz'.format(i), train_x, train_y, test_x, test_y)


def load_data(data_name):
    #路径
    with np.load(ml_data + data_name) as f:
        train_x, train_y, test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y, test_x, test_y


def attack_experiment(choice):
    print('-' * 10 + 'TRAIN TARGET' + '-' * 10 + '\n')
    #路径
    dataset = load_data(target_datapath+'target_data.npz')
    attack_test_x, attack_test_y, test_classes = train_target_model(
        dataset=dataset,
        epochs=target_epochs,
        batch_size=target_batch_size,
        learning_rate=target_learning_rate,
        n_hidden=target_n_hidden,
        l2_ratio=target_l2_ratio,
        model=target_model,
        save=save_model,choice=choice)

    print('-' * 10 + 'TRAIN SHADOW' + '-' * 10 + '\n')
    attack_train_x, attack_train_y, train_classes = train_shadow_models(
        epochs=target_epochs,
        batch_size=target_batch_size,
        learning_rate=target_learning_rate,
        n_shadow=n_shadow,
        n_hidden=target_n_hidden,
        l2_ratio=target_l2_ratio,
        model=target_model,
        save=save_model)

    print('-' * 10 + 'TRAIN ATTACK' + '-' * 10 + '\n')
    dataset = (attack_train_x, attack_train_y, attack_test_x, attack_test_y)
    report = train_attack_model(
        dataset=dataset,
        epochs=attack_epochs,
        batch_size=attack_batch_size,
        learning_rate=attack_learning_rate,
        n_hidden=attack_n_hidden,
        l2_ratio=attack_l2_ratio,
        model=attack_model,
        classes=(train_classes, test_classes))

    return report

def save_to_text(features, labels, feature_file, label_file):
    # Flatten features and convert to 1D array for saving
    features_flat = features.reshape(features.shape[0], -1)

    # Save features to text file
    with open(feature_file, 'w') as f_feat:
        for feature in features_flat:
            feature_str = ','.join(map(str, feature))
            f_feat.write(f"{feature_str}\n")

    # Save labels to text file
    with open(label_file, 'w') as f_label:
        for label in labels:
            f_label.write(f"{label}\n")

def mla_attack(choice=None):
    # 检查文件是否存在

    # 文件不存在，执行数据加载和保存
    # 加载 CIFAR-10 数据集
    if choice==0 or choice==1:
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)

        # testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

    elif choice==2:
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = XOR_pre.cifar10(root=DATA_PATH, train=True, transform=transform)
        # eval_dataset = XOR_pre.cifar10(root=DATA_PATH, train=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        # testloader = torch.utils.data.DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=False)

    elif choice ==3:
        data_path = 'weight_share_protect/UDK_fl_add_mul_sort'
        # eval_dataset = Mydataset_numpy_client_UDK(mode='test', dataset=data_path, ID=1)
        train_dataset = Mydataset_numpy_client_UDK(mode='train', dataset=data_path,ID=4)
        trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=len(train_dataset), shuffle=True, num_workers=8, pin_memory=True)


    # 提取训练数据和标签
    train_features, train_labels = next(iter(trainloader))
    train_features = train_features.numpy()
    train_labels = train_labels.numpy()

    save_to_text(train_features, train_labels, train_feat, train_label)
    save_data()

    return attack_experiment(choice)

