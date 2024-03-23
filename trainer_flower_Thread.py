from server_flower import *
from norm_client_flower import *
import datasets2
from differential_privacy.differential_privacy_client_flower import *
import random
from homomorphic_encryption.homomorphic_encryption_client import *

from xor_and_ndb.xndb_client_flower import *
from pputl_demo.pputl_client_flower import *
from weight_share_protect.User_UDK_flower import *
from weight_share_protect.Mydataset_for_numpy_client_UDK import *
from weight_share_protect.Mydataset_for_numpy_server_UDK import *
import argparse
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor



class Train(object):
    def __init__(self, conf, choice):
        self.conf = conf

        if choice == 2:
            pca = PCA(n_components=30)

            self.train_datasets, self.eval_datasets = datasets2.get_dataset("data/", self.conf["type"],subset_size=1000)
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

        self.clients = []
        self.accs = []
        self.losses = []

    # 首先定义通用函数部分
    # 启动服务器
    def start_server(self,server):
        '''细节：这里一定是要创建一个函数'''
        server.start_server()

    def start_client(self,client):
        # 启动客户端连接到服务器
        fl.client.start_client(server_address="127.0.0.1:8080", client=client)


    def norm_train(self):
        '''
        直接实例化client端口，结合flower框架自动实现模型本地训练，中央服务器聚合
        :return:
        '''
        # 启动服务器的线程
        server_thread = threading.Thread(target=self.start_server, args=(self.server,))
        server_thread.start()

        client_threads = []

        for c in range(self.conf["no_models"]):
            # 实例化客户端
            client = Norm_Client(self.conf, self.server.global_model, self.train_datasets, self.eval_datasets, c)
            self.clients.append(client)
            to_client = client.to_client()

            # 启动客户端的线程
            client_thread = threading.Thread(target=self.start_client, args=(to_client,))
            client_thread.start()
            client_threads.append(client_thread)

        # 等待服务器线程完成
        server_thread.join()

        # 等待所有客户端线程完成
        for thread in client_threads:
            thread.join()

        self.accs=self.server.accs
        self.losses=self.server.losses


    def differential_privacy_train(self):
        '''
        直接实例化client端口，结合flower框架自动实现模型本地训练，中央服务器聚合
        :return:
        '''
        # 启动服务器的线程
        server_thread = threading.Thread(target=self.start_server, args=(self.server,))
        server_thread.start()

        client_threads = []

        for c in range(self.conf["no_models"]):
            # 实例化客户端
            client = Differential_Privacy_Client(self.conf, self.server.global_model, self.train_datasets, self.eval_datasets, c)
            self.clients.append(client)
            to_client = client.to_client()

            # 启动客户端的线程
            client_thread = threading.Thread(target=self.start_client, args=(to_client,))
            client_thread.start()
            client_threads.append(client_thread)

        # 等待服务器线程完成
        server_thread.join()

        # 等待所有客户端线程完成
        for thread in client_threads:
            thread.join()

        self.accs=self.server.accs
        self.losses=self.server.losses

    def homomorphic_encryption_train(self):
        '''如下还没改，有点难'''
        pass
        train_size = self.train_datasets[0].shape[0]

        per_client_size = int(train_size / (self.conf["no_models"]))
        for c in range(self.conf["no_models"]):
            self.clients.append(
                Homomorphic_Encryption_Client(self.conf, Server.public_key, self.server.global_model.encrypt_weights,
                                              self.train_datasets[0][c * per_client_size: (c + 1) * per_client_size],
                                              self.train_datasets[1][c * per_client_size: (c + 1) * per_client_size]))

        # print(server.global_model.weights)

        for e in range(self.conf["global_epochs"]):
            self.server.global_model.encrypt_weights = models.encrypt_vector(Server.public_key,
                                                                             models.decrypt_vector(Server.private_key,
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

    def xndb_train(self):
        '''
        直接实例化client端口，结合flower框架自动实现模型本地训练，中央服务器聚合
        :return:
        '''
        # 启动服务器的线程
        server_thread = threading.Thread(target=self.start_server, args=(self.server,))
        server_thread.start()

        client_threads = []

        for c in range(self.conf["no_models"]):
            # 实例化客户端
            client = XNDB_Client(self.conf, self.server.global_model, self.train_datasets, self.eval_datasets, c)
            self.clients.append(client)
            to_client = client.to_client()

            # 启动客户端的线程
            client_thread = threading.Thread(target=self.start_client, args=(to_client,))
            client_thread.start()
            client_threads.append(client_thread)

        # 等待服务器线程完成
        server_thread.join()

        # 等待所有客户端线程完成
        for thread in client_threads:
            thread.join()

        self.accs=self.server.accs
        self.losses=self.server.losses


    def pputl_client_train(self, G):
        train_dataset_size = len(self.train_datasets)

        # 启动服务器的线程
        server_thread = threading.Thread(target=self.start_server, args=(self.server,))
        server_thread.start()

        client_threads = []

        for c in range(self.conf["no_models"]):
            client=PPUTL_Client(self.conf, self.server.global_model, G, train_dataset_size,self.train_datasets,self.eval_datasets, c)
            self.clients.append(client)

            to_client = client.to_client()

            # 启动客户端的线程
            client_thread = threading.Thread(target=self.start_client, args=(to_client,))
            client_thread.start()
            client_threads.append(client_thread)

            # 等待服务器线程完成
        server_thread.join()

        # 等待所有客户端线程完成
        for thread in client_threads:
            thread.join()

        self.accs = self.server.accs
        self.losses = self.server.losses


    def weight_share_protect_train(self, G):
        # train_dataset_size = len(self.train_datasets)
        #
        # # 启动服务器的线程
        # server_thread = threading.Thread(target=self.start_server, args=(self.server,))
        # server_thread.start()
        #
        # client_threads = []
        #
        # for c in range(self.conf["no_models"]):
        #     client=PPUTL_Client(self.conf, self.server.global_model, G, train_dataset_size,self.train_datasets,self.eval_datasets, c)
        #     self.clients.append(client)
        #
        #     to_client = client.to_client()
        #
        #     # 启动客户端的线程
        #     client_thread = threading.Thread(target=self.start_client, args=(to_client,))
        #     client_thread.start()
        #     client_threads.append(client_thread)
        #
        #     # 等待服务器线程完成
        # server_thread.join()
        #
        # # 等待所有客户端线程完成
        # for thread in client_threads:
        #     thread.join()
        #
        # self.accs = self.server.accs
        # self.losses = self.server.losses
        pass


# 如下定义的是测试函数
def main():
    """Load data, start CifarClient."""
    conf = {"model_name": "resnet50", "no_models": 3, "type": "cifar", "global_epochs": 3, "local_epochs": 3, "k": 3,
            "batch_size": 8, "client_batchsize": 100, "global_batchsize": 500, "lr": 0.1, "momentum": 0.9,
            "lambda": 0.1,"dp": True, "C": 1000, "sigma": 0.01, "q": 0.2, "W": 2, "feature_num": 30, "eta": 2, "alpha": 1.0,
            "poison_label": 2, "poisoning_per_batch": 4, "prop": 0.6, "root": "ndb_cifar10_data/"
            }

    trainers=Train(conf,choice=4)
    # trainers.norm_train() 测试norm_train
    dataset_path = "weight_share_protect/UDK_fl_add_mul_sort"

    save_path = f'pputl_demo/G_path/trained_generator_G.pth'
    if os.path.exists(save_path):
        G = Generator64().to(device)
        G.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
        # G.load_state_dict(torch.load(save_path))
    else:
        G = pputl.PPUTL(conf["batch_size"])
    G.eval()

    trainers.pputl_client_train(G=G)


if __name__ == "__main__":
    main()

    print('正常运行')


