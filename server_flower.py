import math

import models, torch
import copy
from homomorphic_encryption import paillier
import numpy as np
import torch.nn as nn
from weight_share_protect.Mydataset_for_numpy_server_UDK import *
from pputl_demo.pputl_client_flower import *


from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

import argparse
import datasets2
from typing import Dict, Optional, Tuple
from collections import OrderedDict

import json
import argparse
from plot_acc import *
import warnings

from xor_and_ndb import xor, ndb
from weight_share_protect.matrix_add_mul_sort_transform_using_different_key import matrix_add_mul_sort
from pputl_demo.target_model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import os

from all_models.resnet_total import *
from all_models.densenet import *
from all_models.dla_simple import *
from all_models.googlenet import *
from all_models.mobilenet import *
from all_models.vgg import *

# 加载自己的功能包
import utils.model_save_load as um



class Server(object):
    public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)

    def __init__(self, conf, eval_dataset, choice):

        self.conf = conf #基础配置
        self.choice=choice # 算法选项
        self.best_val_loss = float('inf') #用来保存最好的一个模型

        if self.choice == 2:
            self.global_model = models.LR_Model(public_key=Server.public_key, w_size=self.conf["feature_num"] + 1)
            self.eval_x = eval_dataset[0]
            self.eval_y = eval_dataset[1]
        elif self.choice ==4:
            save_path = f'pputl_demo/G_path/trained_generator_G.pth'
            if os.path.exists(save_path):
                self.G = Generator64().to(device)
                self.G.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
                # self.G.load_state_dict(torch.load(save_path))
            else:
                self.G = pputl.PPUTL(conf["batch_size"])
            self.G.eval()

            self.global_model = models.get_model(self.conf["model_name"])
            self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"],
                                                           shuffle=True)
        # elif choice == 5:
        #     if self.conf["model_name"]=="resnet50":
        #         self.global_model = ResNet50()
        #     elif self.conf["model_name"]=="densenet":
        #         self.global_model = DenseNet121()
        #     elif self.conf["model_name"]=="simpledla":
        #         self.global_model = SimpleDLA()
        #     elif self.conf["model_name"]=="googlenet":
        #         self.global_model = GoogLeNet()
        #     elif self.conf["model_name"]=="mobilenet":
        #         self.global_model = MobileNet()
        #     else:
        #         self.global_model = VGG()
        #     self.dataset_path = "weight_share_protect/UDK_fl_add_mul_sort"
        #     self.global_testloader = eval_dataset
        # else:
        #     if self.conf["model_name"] == "resnet50":
        #         self.global_model = ResNet50()
        #     elif self.conf["model_name"] == "densenet":
        #         self.global_model = DenseNet121()
        #     elif self.conf["model_name"] == "simpledla":
        #         self.global_model = SimpleDLA()
        #     elif self.conf["model_name"] == "googlenet":
        #         self.global_model = GoogLeNet()
        #     elif self.conf["model_name"] == "mobilenet":
        #         self.global_model = MobileNet()
        #     else:
        #         self.global_model = VGG()
        #     self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"])

        elif self.choice == 5:
            self.global_model = models.get_model(self.conf["model_name"])
            self.dataset_path = "weight_share_protect/UDK_fl_add_mul_sort"
            self.global_testloader = eval_dataset
        else:
            # norm状态下的配置
            self.global_model = models.get_model(self.conf["model_name"])
            self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"],
                                                           shuffle=True)

        self.eval_dataset_size = len(eval_dataset)

        self.accs = []
        self.losses = []

    # 同态加密的聚合算法
    def model_aggregate_homomorphic_encryption(self, weight_accumulator):

        for id, data in enumerate(self.global_model.encrypt_weights):
            update_per_layer = weight_accumulator[id] * self.conf["lambda_"]

            self.global_model.encrypt_weights[id] = self.global_model.encrypt_weights[id] + update_per_layer

    # 同态加密下的评估算法
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

    #用来做测评的
    def model_eval(self):
        self.global_model.eval()

        total_loss = 0.0
        correct = 0
        dataset_size = 0
        cnt=0
        random_int = np.random.randint(2, 5)
        for batch_id, batch in enumerate(self.eval_loader):
            cnt = cnt + 1
            if cnt == random_int:
                break
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = self.global_model(data)

            total_loss += torch.nn.functional.cross_entropy(output, target,
                                                            reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l

    # 负数据库用来进行评价的函数
    def model_eval_xndb(self):
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        loss = 0
        acc = 0.0
        cnt=0
        random_int = np.random.randint(2, 5)
        for (images, labels) in self.eval_loader:
            cnt = cnt + 1
            if cnt == random_int:
                break

            images = images.to(torch.float32).to(device)
            labels = labels.to(device).squeeze(1)

            preds = self.global_model(images)
            loss += criterion(preds, labels).item()

            pred_cls = preds.data.max(1)[1]
            acc += pred_cls.eq(labels.data).sum().item()

        loss /= len(self.eval_loader)
        acc /= len(self.eval_loader.dataset)

        return acc * 100, loss

    def model_eval_pputl(self, G):
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        batch_size = self.conf["batch_size"]
        num_batches = math.ceil(self.eval_dataset_size / batch_size)
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        cnt=0
        random_int = np.random.randint(2, 5)
        for i, _ in enumerate(range(num_batches)):
            cnt = cnt + 1
            if cnt == random_int:
                break
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

    # 用来测评的
    def get_evaluate_fn(self,model: torch.nn.Module):
        def evaluate(
                server_round: int,
                parameters: fl.common.NDArrays,
                config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            # Update model with the latest parameters

            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.global_model.load_state_dict(state_dict, strict=True)
            if self.conf['is_increment']:
                models_path = self.conf['increment']
                self.global_model = torch.load(models_path, map_location=torch.device('cpu'))
            # 这里是细节
            if self.choice ==3 :
                accuracy,loss = self.model_eval_xndb()
            if self.choice ==4:
                accuracy,loss = self.model_eval_pputl(self.G)
            else:
                # norm状态下的评估
                accuracy,loss = self.model_eval()

            self.accs.append(accuracy)
            self.losses.append(loss)

            # 保存最好的模型，如果这个epoch的验证损失比之前所有epoch的都要低，那么保存模型
            if loss < self.best_val_loss:
                self.best_val_loss = loss
                model_name=self.conf["model_save_name"]+'_best'
                um.save_model_with_same_name(self.global_model,self.conf["model_save_path"],model_name)

            return loss, {"accuracy": accuracy}

        return evaluate

    #这里是基础配置部分
    def fit_config(self,server_round: int):
        """Return training configuration dict for each round.

        Keep batch size fixed at 32, perform two rounds of training with one local epoch,
        increase to two local epochs afterwards.
        """
        return self.conf

    # 这里是norm状态下的权重聚合，以及参数数值的聚合
    def weighted_average(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": sum(accuracies) / sum(examples)}

    # 评价基本参数
    def evaluate_config(self, server_round: int = 5):
        """Return evaluation configuration dict for each round.

        Perform five local evaluation steps on each client (i.e., use five batches) during
        rounds one to three, then increase to ten local evaluation steps.
        """
        # val_steps = 5 if server_round < 4 else 10
        val_steps=self.conf["value_steps"]
        return {"val_steps": val_steps}

    # 启动本地服务器
    def start_server(self):
        # Define strategy

        if self.conf['is_increment']:
            models_path=self.conf['increment']
            # model=torch.load(models_path)
            model = torch.load(models_path, map_location=torch.device('cpu'))
            model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
        else:
            model_parameters = [val.cpu().numpy() for _, val in self.global_model.state_dict().items()]

        strategy=fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=self.conf["min_available_clients"],
            evaluate_fn=self.get_evaluate_fn(self.global_model),
            on_fit_config_fn=self.fit_config,
            on_evaluate_config_fn=self.evaluate_config,
            initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
            evaluate_metrics_aggregation_fn=self.weighted_average,)

        # Start Flower server
        fl.server.start_server(
            server_address=self.conf["address"],
            config=fl.server.ServerConfig(num_rounds=self.conf["global_epochs"]),
            strategy=strategy,
        )

