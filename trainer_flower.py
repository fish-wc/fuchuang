from server_flower import *
from norm_client_flower import *
import datasets2
from differential_privacy.differential_privacy_client_flower import *
import random
from homomorphic_encryption.homomorphic_encryption_client_flower import *

from xor_and_ndb.xndb_client_flower import *
from pputl_demo.pputl_client_flower import *
from weight_share_protect.User_UDK import *
from weight_share_protect.Mydataset_for_numpy_client_UDK import *
from weight_share_protect.Mydataset_for_numpy_server_UDK import *
import argparse
import subprocess
import threading

model_path = "./models/global_model"  # 假设你想将模型保存在这里
directory = os.path.dirname(model_path)
# 如果目录不存在，创建它
if not os.path.exists(directory):
    os.makedirs(directory)

class Train(object):
    '''
    如下是没有采用多线程技术设计的，最终应该是要设计成这种形式
    '''
    def __init__(self, conf, choice,node_id):
        self.conf = conf
        self.node_id=node_id
        if choice == 2:
            pca = PCA(n_components=30)

            self.train_datasets, self.eval_datasets = datasets2.get_dataset("data/", self.conf["type"])
            self.train_datasets = (
            pca.fit_transform(self.train_datasets.data.reshape(self.train_datasets.data.shape[0], -1)),
            np.array(self.train_datasets.targets))
            self.eval_datasets = (
            pca.fit_transform(self.eval_datasets.data.reshape(self.eval_datasets.data.shape[0], -1)),
            np.array(self.eval_datasets.targets))
            self.server = Server(self.conf, self.eval_datasets, choice)
        elif choice == 5:
            self.dataset_path = "weight_share_protect/UDK_fl_add_mul_sort"
            self.global_testloader = torch.utils.data.DataLoader(
                Mydataset_numpy_server_UDK(mode='test', dataset=self.dataset_path),
                batch_size=self.conf["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
            self.server = Server(self.conf, self.global_testloader, choice)

        else:
            self.train_datasets, self.eval_datasets = datasets2.get_dataset("data/", self.conf["type"], choice,subset_size=1000)
            self.server = Server(self.conf, self.eval_datasets, choice)

            self.eval_loader = torch.utils.data.DataLoader(self.eval_datasets, batch_size=self.conf["batch_size"],
                                                           shuffle=True)
        self.clients = []
        self.accs = []
        self.losses = []

    def start_server(self):
        '''细节：这里一定是要创建一个函数'''
        self.server.start_server()
        #细节：打印训练过程的数据在这里打印
        self.accs = self.server.accs
        self.losses = self.server.losses

    def norm_train_start_client(self):
        '''
        直接实例化client端口，结合flower框架自动实现模型本地训练，中央服务器聚合
        :return:
        '''
        # 实例化客户端
        client=Norm_Client(self.conf, self.server.global_model, self.train_datasets, self.eval_datasets, self.node_id)
        self.clients.append(client)

        # 启动客户端口服务器
        to_client = client.to_client()
        fl.client.start_client(server_address=self.conf["address"], client=to_client)

        self.accs = self.server.accs
        self.losses = self.server.losses

        print("results:",self.eval(model_path))

    def start_differential_privacy_train(self):
        # 实例化客户端
        client = Differential_Privacy_Client(self.conf, self.server.global_model, self.train_datasets, self.eval_datasets, self.node_id)
        self.clients.append(client)

        # 启动客户端口服务器
        to_client = client.to_client()
        fl.client.start_client(server_address=self.conf["address"], client=to_client)

        self.accs = self.server.accs
        self.losses = self.server.losses

        print("results:", self.eval(model_path))

    def start_homomorphic_encryption_train(self):
        #这里需要注意一下，跟原来是一样的
        train_size = self.train_datasets[0].shape[0]

        per_client_size = int(train_size / (self.conf["no_models"]))
        for c in range(self.conf["no_models"]):
            self.clients.append(
                Homomorphic_Encryption_Client(self.conf, Server.public_key,
                                              self.server.global_model.encrypt_weights,
                                              self.train_datasets[0][
                                              c * per_client_size: (c + 1) * per_client_size],
                                              self.train_datasets[1][
                                              c * per_client_size: (c + 1) * per_client_size]))

        # print(server.global_model.weights)

        for e in range(self.conf["global_epochs"]):
            self.server.global_model.encrypt_weights = models.encrypt_vector(Server.public_key,
                                                                             models.decrypt_vector(
                                                                                 Server.private_key,
                                                                                 self.server.global_model.encrypt_weights))

            candidates = random.sample(self.clients, self.conf["k"])

            weight_accumulator = [Server.public_key.encrypt(0.0)] * (self.conf["feature_num"] + 1)

            for c in candidates:
                # print(models.decrypt_vector(Server.private_key, server.global_model.encrypt_weights))
                diff = c.local_train(self.server.global_model.encrypt_weights)

                for i in range(len(weight_accumulator)):
                    weight_accumulator[i] = weight_accumulator[i] + diff[i]

            self.server.model_aggregate_homomorphic_encryption(weight_accumulator)

            acc, loss = self.server.model_eval_homomorphic_encryption()
            self.accs.append(acc)
            self.losses.append(loss)

            print("Epoch %d, acc: %f\n" % (e, acc))

    def start_xndb_train(self):
        # 实例化客户端
        client = XNDB_Client(self.conf, self.server.global_model, self.train_datasets,
                                             self.eval_datasets, self.node_id)
        self.clients.append(client)

        # 启动客户端口服务器
        to_client = client.to_client()
        fl.client.start_client(server_address=self.conf["address"], client=to_client)

        self.accs = self.server.accs
        self.losses = self.server.losses

        print("results:", self.eval(model_path))

    def start_pputl_client_train(self,G):
        train_dataset_size = len(self.train_datasets)
        client=PPUTL_Client(self.conf, self.server.global_model, G, train_dataset_size, self.train_datasets,self.eval_datasets, self.node_id)

        self.clients.append(client)

        # 启动客户端口服务器
        to_client = client.to_client()
        fl.client.start_client(server_address=self.conf["address"], client=to_client)

        self.accs = self.server.accs
        self.losses = self.server.losses

        print("results:", self.eval(model_path))

    def eval(self,model_path):
        # self.model=models.get_model(self.conf["model_name"])
        global_epochs=self.conf["global_epochs"]-1
        save_model_path=model_path+f"{global_epochs}.pth"
        self.model=torch.load(save_model_path)
        # 将模型设置为评估模式，这对于推理很重要
        self.model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = self.model(data)

            # print(output)

            total_loss += torch.nn.functional.cross_entropy(output, target,
                                                            reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l


def main():
    """Load data, start CifarClient."""
    conf = {"model_name": "resnet50", "no_models": 3, "type": "cifar", "global_epochs": 3, "local_epochs": 3, "k": 3,
            "batch_size": 8, "client_batchsize": 100, "global_batchsize": 500, "lr": 0.1, "momentum": 0.9,
            "lambda": 0.1,"dp": True, "C": 1000, "sigma": 0.01, "q": 0.2, "W": 2, "feature_num": 30, "eta": 2, "alpha": 1.0,
            "poison_label": 2, "poisoning_per_batch": 4, "prop": 0.6, "root": "ndb_cifar10_data/","address":"127.0.0.1:8080",
            "min_available_clients":2,
            }
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--node-id", type=int, default=0, choices=range(0, 10))
    parser.add_argument("--choice",type=int,default=-1,choices=range(-1,5))
    args = parser.parse_args()
    trainers=Train(conf,choice=args.choice)
    if args.choice==-1:
        trainers.start_server()
    elif args.choice==0:
        trainers.norm_train_start_client()
    else:
        pass


if __name__ == "__main__":
    main()

    print('正常运行')


