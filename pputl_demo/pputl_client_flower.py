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


import json
import argparse

import warnings
from pputl_demo import pputl
from xor_and_ndb import xor, ndb
from weight_share_protect.matrix_add_mul_sort_transform_using_different_key import matrix_add_mul_sort
from pputl_demo.target_model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PPUTL_Client(fl.client.NumPyClient):

    def __init__(self, conf, model, G,train_dataset_size,train_dataset,eval_dataset, id=-1):

        self.conf = conf
        self.G = G
        self.local_model = models.get_model(self.conf["model_name"])
        self.local_model = self.local_model.to(device)

        self.mask = {}
        for name, param in self.local_model.state_dict().items():
            p = torch.ones_like(param) * self.conf["prop"]
            if torch.is_floating_point(param):
                self.mask[name] = torch.bernoulli(p)
            else:
                self.mask[name] = torch.bernoulli(p).long()

        self.client_id = id
        self.train_dataset_size = train_dataset_size

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['no_models'])
        train_indices = all_range[id * data_len: (id + 1) * data_len]

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                            train_indices))

        self.eval_loader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=self.conf["batch_size"],
                                                       shuffle=True)

    def local_train(self, model):

        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        # print(id(model))

        optimizer_T = optim.Adam(self.local_model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()
        self.local_model.train()
        batch_size = self.conf["batch_size"]
        num_batches = math.ceil(self.train_dataset_size / batch_size)
        for e in range(self.conf["local_epochs"]):
            for i in range(num_batches):
                z_ = torch.randn(batch_size, 128).to(device)
                y_d = (torch.rand(batch_size, 1) * 10).type(torch.LongTensor).to(device)
                y_label_ = torch.zeros(batch_size, 10).to(device)
                y_label_.scatter_(1, y_d.view(batch_size, 1), 1)
                y_label_c_ = y_d.view(batch_size).to(device)
                G_result = self.G(z_, y_label_)
                C_result = self.local_model(G_result)
                loss = criterion(C_result, y_label_c_)
                optimizer_T.zero_grad()
                loss.backward()
                optimizer_T.step()
            print("Epoch %d done." % e)
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - model.state_dict()[name])
            diff[name] = diff[name] * self.mask[name]
        # print(diff[name])

        return diff

    def model_eval(self):
        self.local_model.eval()
        criterion = nn.CrossEntropyLoss()
        loss = 0
        acc = 0.0
        for (images, labels) in self.eval_loader:
            images = images.to(torch.float32).to(device)
            if labels.dim() > 1:
                labels = labels.squeeze(1)

            # labels = labels.to(device).squeeze(1)

            preds = self.local_model(images)
            loss += criterion(preds, labels).item()

            pred_cls = preds.data.max(1)[1]
            acc += pred_cls.eq(labels.data).sum().item()

        loss /= len(self.eval_loader)
        acc /= len(self.eval_loader.dataset)

        return loss,acc * 100

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


def main() -> None:
    ''' 如下是client的编写测试'''
    """Load data, start CifarClient."""
    conf = {"model_name": "resnet50", "no_models": 3, "type": "cifar", "global_epochs": 3, "local_epochs": 3,
            "k": 3,
            "batch_size": 8, "client_batchsize": 100, "global_batchsize": 500, "lr": 0.1, "momentum": 0.9,
            "lambda": 0.1,
            "dp": True, "C": 1000, "sigma": 0.01, "q": 0.2, "W": 2, "feature_num": 30, "eta": 2, "alpha": 1.0,
            "poison_label": 2, "poisoning_per_batch": 4, "prop": 0.6, "root": "ndb_cifar10_data/"
            }
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--node-id", type=int, default=1, choices=range(0, 10))
    args = parser.parse_args()

    train_datasets, eval_datasets = datasets2.get_dataset("../data/", conf["type"], choice=0, subset_size=1000)

    global_model = models.get_model(conf["model_name"])

    train_dataset_size = len(train_datasets)

    save_path = f'pputl_demo/G_path/trained_generator_G.pth'
    if os.path.exists(save_path):
        G = Generator64().to(device)
        G.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
        # G.load_state_dict(torch.load(save_path))
    else:
        G = pputl.PPUTL(conf["batch_size"])
    G.eval()

    # Start client
    client = PPUTL_Client(conf, global_model, G,train_dataset_size,train_datasets,eval_datasets, id=args.node_id).to_client()
    # client = PPUTL_Client(conf, global_model, train_datasets, eval_datasets, id=args.node_id).to_client()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
    print('正常运行到这里')




