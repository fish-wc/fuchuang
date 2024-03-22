from collections import OrderedDict
from flwr.common import Scalar, NDArrays
from typing import Dict, List, Tuple
import models, torch, copy
import torch.nn as nn
import torch.optim as optim
import flwr as fl
import numpy as np
import argparse
import datasets2

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Norm_Client(fl.client.NumPyClient):
    '''1.和原来的初始化部分，不一样的地方包括了__init__()函数里面多加了一个eval_dataset，加了关于eval_dataset的初始化'''


    def __init__(self, conf, model, train_dataset,eval_dataset, id=-1):

        self.conf = conf

        self.local_model = model
        self.local_model = self.local_model.to(device)

        # self.local_model = models.get_model(self.conf["model_name"])
        # self.local_model = self.local_model.to(device)

        self.mask = {}
        for name, param in self.local_model.state_dict().items():
            p = torch.ones_like(param) * self.conf["prop"]
            if torch.is_floating_point(param):
                self.mask[name] = torch.bernoulli(p)
            else:
                self.mask[name] = torch.bernoulli(p).long()

        self.client_id = id

        # 数据的加载部分
        self.train_dataset = train_dataset
        self.eval_dataset=eval_dataset

        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['no_models'])
        train_indices = all_range[id * data_len: (id + 1) * data_len]

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                            train_indices))

        self.eval_loader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)

    def local_train(self, model):
        '''
        进行模型训练
        '''
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        # print(id(model))
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])
        # print(id(self.local_model))
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
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - model.state_dict()[name])
            diff[name] = diff[name] * self.mask[name]
        # print(diff[name])
        return diff

    def model_eval(self):
        self.local_model.eval()
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
        self.local_model.train() # 设置为train格式
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
        return self.get_parameters(config={}),len(self.train_loader),{}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        loss, accuracy = self.model_eval()
        return float(loss), len(self.eval_loader.dataset), {"accuracy": float(accuracy)}


def main() -> None:
    ''' 如下是client的编写测试'''
    """Load data, start CifarClient."""
    conf = {"model_name": "resnet50", "no_models": 3, "type": "cifar", "global_epochs": 3, "local_epochs": 3, "k": 3,
            "batch_size": 8, "client_batchsize": 100, "global_batchsize": 500, "lr": 0.1, "momentum": 0.9,
            "lambda": 0.1, "dp": True, "C": 1000, "sigma": 0.01, "q": 0.2, "W": 2, "feature_num": 30, "eta": 2,
            "alpha": 1.0,
            "poison_label": 2, "poisoning_per_batch": 4, "prop": 0.6, "root": "ndb_cifar10_data/",
            "address": "127.0.0.1:8080",
            "min_available_clients": 2,
            }
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--node-id", type=int, default=1,choices=range(0, 10))
    args = parser.parse_args()

    train_datasets, eval_datasets = datasets2.get_dataset("data/", conf["type"], choice=0,subset_size=1000)

    global_model = models.get_model(conf["model_name"])

    # Start client
    client = Norm_Client(conf,global_model, train_datasets, eval_datasets,id=args.node_id).to_client()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)



if __name__ == "__main__":
    main()
    print('okkk')