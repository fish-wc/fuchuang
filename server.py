import math
from resnet50 import ResNet50
import models, torch
import copy
from homomorphic_encryption import paillier
import numpy as np
import torch.nn as nn
from weight_share_protect.Mydataset_for_numpy_server_UDK import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Server(object):
    public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)

    def __init__(self, conf, eval_dataset, choice):

        self.conf = conf

        if choice == 2:
            self.global_model = models.LR_Model(public_key=Server.public_key, w_size=self.conf["feature_num"] + 1)
            self.eval_x = eval_dataset[0]
            self.eval_y = eval_dataset[1]
        elif choice == 5:
            self.global_model = ResNet50()
            self.dataset_path = "weight_share_protect/UDK_fl_add_mul_sort"
            self.global_testloader = eval_dataset
        else:
            self.global_model = ResNet50()
            self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"])

        self.eval_dataset_size = len(eval_dataset)

    def model_aggregate_norm(self, weight_accumulator):
        for name, data in self.global_model.state_dict().items():

            update_per_layer = weight_accumulator[name] * self.conf["lambda_"]

            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    def model_aggregate_differential_privacy(self, weight_accumulator):
        for name, data in self.global_model.state_dict().items():

            update_per_layer = weight_accumulator[name] * self.conf["lambda_"]

            if self.conf['dp']:
                sigma = self.conf['sigma']
                if torch.cuda.is_available():
                    noise = torch.cuda.FloatTensor(update_per_layer.shape).normal_(0, sigma)
                else:
                    noise = torch.FloatTensor(update_per_layer.shape).normal_(0, sigma)

                update_per_layer.add_(noise)

            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    def average_weights(self, w):
        """
        Returns the average of the weights.
        """
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg

    def weight_share_protect_eval(self):
        self.global_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.global_testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.global_model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        loss = test_loss / len(self.global_testloader.dataset)
        acc = 100. * correct / len(self.global_testloader.dataset)
        return acc, loss

    def model_eval_xndb(self):
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        loss = 0
        acc = 0.0
        for (images, labels) in self.eval_loader:
            images = images.to(torch.float32).to(device)
            labels = labels.to(device).squeeze(1)

            preds = self.global_model(images)
            loss += criterion(preds, labels).item()

            pred_cls = preds.data.max(1)[1]
            acc += pred_cls.eq(labels.data).sum().item()

        loss /= len(self.eval_loader)
        acc /= len(self.eval_loader.dataset)

        return acc * 100, loss

    def model_eval(self):
        self.global_model.eval()
        # print("\n\nstart to model evaluation......")
        # for name, layer in self.global_model.named_parameters():
        #	print(name, "->", torch.mean(layer.data))

        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = self.global_model(data)

            # print(output)

            total_loss += torch.nn.functional.cross_entropy(output, target,
                                                            reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l

    def model_eval_pputl(self, G):
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        # print("\n\nstart to model evaluation......")
        # for name, layer in self.global_model.named_parameters():
        #	print(name, "->", torch.mean(layer.data))
        batch_size = self.conf["batch_size"]
        num_batches = math.ceil(self.eval_dataset_size / batch_size)
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for i, _ in enumerate(range(num_batches)):
            z_ = torch.randn(batch_size, 128).to(device)
            y_d = (torch.rand(batch_size, 1) * 10).type(torch.LongTensor).to(device)
            y_label_ = torch.zeros(batch_size, 10).to(device)
            y_label_.scatter_(1, y_d.view(batch_size, 1), 1)
            y_label_c_ = y_d.view(batch_size).to(device)
            G_result = G(z_, y_label_)
            output = self.global_model(G_result)

            loss = criterion(output, y_label_c_)
            total_loss += loss.item() * output.size(0)  # 累积批次损失
            pred = output.argmax(dim=1)  # 获取最大对数概率的索引
            correct += (pred == y_label_c_).sum().item()
            dataset_size += output.size(0)

        # 计算平均损失和准确率
        total_l = total_loss / dataset_size
        acc = 100.0 * correct / dataset_size

        return acc, total_l

    def model_aggregate_homomorphic_encryption(self, weight_accumulator):

        for id, data in enumerate(self.global_model.encrypt_weights):
            update_per_layer = weight_accumulator[id] * self.conf["lambda_"]

            self.global_model.encrypt_weights[id] = self.global_model.encrypt_weights[id] + update_per_layer

    def model_eval_homomorphic_encryption(self):
        total_loss = 0.0
        correct = 0
        dataset_size = 0

        batch_num = int(self.eval_x.shape[0] / self.conf["batch_size"])

        self.global_model.weights = models.decrypt_vector(Server.private_key, self.global_model.encrypt_weights)

        for batch_id in range(batch_num):
            x = self.eval_x[batch_id * self.conf["batch_size"]: (batch_id + 1) * self.conf["batch_size"]]
            x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            y = self.eval_y[batch_id * self.conf["batch_size"]: (batch_id + 1) * self.conf["batch_size"]].reshape(
                (-1, 1))

            dataset_size += x.shape[0]

            wxs = x.dot(self.global_model.weights)

            # 使用更小的修正值
            epsilon = 1e-20
            pred_y = np.clip([1.0 / (1 + np.exp(-np.clip(wx, -500, 500))) for wx in wxs], epsilon, 1 - epsilon)
            pred_y = np.array([1 if pred > 0.5 else 0 for pred in pred_y]).reshape((-1, 1))

            # 使用更小的修正值
            loss = -np.mean(y * np.log(pred_y + epsilon) + (1 - y) * np.log(1 - pred_y + epsilon))
            total_loss += loss

            correct += np.sum(y == pred_y)

        acc = 100.0 * (float(correct) / float(dataset_size))
        average_loss = total_loss / batch_num  # 平均损失

        return acc, average_loss

    @staticmethod
    def re_encrypt(w):
        return models.encrypt_vector(Server.public_key, models.decrypt_vector(Server.private_key, w))

