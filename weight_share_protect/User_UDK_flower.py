import csv


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import models

import sys
import os
# 添加上层目录到 sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from collections import OrderedDict
from flwr.common import Scalar, NDArrays
from typing import Dict, List, Tuple
import torch, copy

import torch.nn as nn
import torch.optim as optim
import flwr as fl
import numpy as np
import argparse
import datasets2
import models


device = 'cuda' if torch.cuda.is_available() else 'cpu'
class User_UDK(fl.client.NumPyClient):
    def __init__(self, conf, model, train_loader,eval_loader, id = -1):
        self.conf = conf

        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.local_model = model
        # self.local_model = models.get_model(self.conf["model_name"])
        self.local_model = self.local_model.to(device)

        # Check GPU availability and move the model to GPU if available
        # if torch.cuda.is_available():
        # 	self.local_model = self.local_model.cuda()
        self.mask = {}
        for name, param in self.local_model.state_dict().items():
            p = torch.ones_like(param) * self.conf["prop"]
            if torch.is_floating_point(param):
                self.mask[name] = torch.bernoulli(p)
            else:
                self.mask[name] = torch.bernoulli(p).long()

        self.client_id = id


    def local_train(self,model):

        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],momentum=self.conf['momentum'])
        self.local_model.train()

        for e in range(self.conf["local_epochs"]):

            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()

                optimizer.step()
            print("Epoch %d done." % e)


    def get_local_model(self):
        return self.local_model.state_dict()

    def model_eval(self):
        self.local_model.eval()

        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = self.local_model(data)

            # print(output)

            total_loss += torch.nn.functional.cross_entropy(output, target,
                                                            reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        self.local_model.train()  # 设置为train格式
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.local_model.state_dict().items()]


    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.local_model.train()

        params_dict = zip(self.local_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.local_model.load_state_dict(state_dict, strict=True)


    def fit(
            self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        self.local_train(self.local_model)
        return self.get_parameters(config={}), len(self.train_loader), {}


    def evaluate(
            self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        loss, accuracy = self.model_eval()
        return float(loss), len(self.eval_loader.dataset), {"accuracy": float(accuracy)}

