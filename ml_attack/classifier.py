import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F

class SoftmaxRegression(nn.Module):
    def __init__(self, n_in, n_out):
        super(SoftmaxRegression, self).__init__()
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x):
        x = F.softmax(self.fc(x), dim=1)
        return x

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(dataset, n_hidden=50, batch_size=100, epochs=100, learning_rate=0.01, model='nn', l2_ratio=1e-7, rtn_layer=True):
    train_x, train_y, test_x, test_y = dataset
    n_in = train_x.shape[1]
    n_out = len(np.unique(train_y))

    train_dataset = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y).long())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if model == 'nn':
        print('Using neural network...')
        net = NeuralNetwork(n_in, n_hidden, n_out)
    else:  # 使用Softmax回归
        print('Using softmax regression...')
        net = SoftmaxRegression(n_in, n_out)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=l2_ratio)

    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch}, train loss {running_loss:.3f}')

    # 评估训练准确率
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in train_loader:
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Training Accuracy: {:.2f}%'.format(100 * correct / total))

    predictions = []
    if test_x is not None and test_y is not None:
        test_dataset = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y).long())
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.numpy())  # 收集预测结果
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Testing Accuracy: {:.2f}%'.format(100 * correct / total))

    if rtn_layer:
        return net
    else:
        return np.array(predictions)  # 确保返回的是一维数组



def load_dataset(train_feat, train_label, test_feat=None, test_label=None):
    train_x = np.genfromtxt(train_feat, delimiter=',', dtype='float32')
    train_y = np.genfromtxt(train_label, dtype='int32')
    min_y = np.min(train_y)
    train_y -= min_y
    if test_feat is not None and test_label is not None:
        test_x = np.genfromtxt(train_feat, delimiter=',', dtype='float32')
        test_y = np.genfromtxt(train_label, dtype='int32')
        test_y -= min_y
    else:
        test_x = None
        test_y = None
    return train_x, train_y, test_x, test_y



