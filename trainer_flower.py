# 导入各个模块集成的部分
from server_flower import *

from norm_client_flower import *
from differential_privacy.differential_privacy_client_flower import *
from homomorphic_encryption.homomorphic_encryption_client_flower import *
from xor_and_ndb.xndb_client_flower import *
from pputl_demo.pputl_client_flower import *
from weight_share_protect.User_UDK_flower import *
from weight_share_protect.Mydataset_for_numpy_client_UDK import *
from weight_share_protect.Mydataset_for_numpy_server_UDK import *

# 导入自己定义的功能包
import datasets2
import utils.utils as uu
import utils.model_save_load as um

# 导入常用包
import torch
import random

class Train(object):
    # 如下是没有采用多线程技术设计的，最终应该是要设计成这种形式
    # 集成了多个功能，包括数据的初始化和集成，把数据初始化放到这里，这样来集成各处的代码，这样便于汇总
    # 现在先把normal状态下的参数调通再说
    def __init__(self, conf, choice,node_id):
        self.conf = conf # 配置文档
        self.node_id=node_id # 区别用的id

        if choice == 2:
            # 对于同态加密这一块目前还没有实现多机通信
            pca = PCA(n_components=30)
            self.train_datasets, self.eval_datasets = datasets2.get_dataset(self.conf["data_path"], self.conf["type"])
            self.train_datasets = (
                pca.fit_transform(self.train_datasets.data.reshape(self.train_datasets.data.shape[0], -1)),
                np.array(self.train_datasets.targets))
            self.eval_datasets = (
                pca.fit_transform(self.eval_datasets.data.reshape(self.eval_datasets.data.shape[0], -1)),
                np.array(self.eval_datasets.targets))
            self.server = Server(self.conf, self.eval_datasets, choice)
        elif choice == 5:
            # 贡献权重部分
            self.dataset_path = "weight_share_protect/UDK_fl_add_mul_sort2/mul10_user_number_2"
            self.global_testloader = torch.utils.data.DataLoader(
                Mydataset_numpy_server_UDK(mode='test', dataset=self.dataset_path),
                batch_size=self.conf["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
            self.server = Server(self.conf, self.global_testloader, choice)

        else:
            self.train_datasets, self.eval_datasets = datasets2.get_dataset(self.conf["data_path"], self.conf["type"], choice,subsize_rate=self.conf["subsize_rate"])
            self.server = Server(self.conf, self.eval_datasets, choice)
            # 获取eval_dataloader
            self.eval_loader = torch.utils.data.DataLoader(self.eval_datasets, batch_size=self.conf["batch_size"],shuffle=True)

        self.clients = []
        self.accs = []
        self.losses = []

    # 基础功能部分
    # 启动服务器
    def start_server(self):
        '''细节：这里一定是要创建一个函数'''
        self.server.start_server()
        #细节：打印训练过程的数据在这里打印
        self.accs = self.server.accs
        self.losses = self.server.losses

    # 启动正常状态下的norm_train
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

    # 启动差分隐私状态下的train
    def start_differential_privacy_train(self):
        # 实例化客户端
        client = Differential_Privacy_Client(self.conf, self.server.global_model, self.train_datasets, self.eval_datasets, self.node_id)
        self.clients.append(client)

        # 启动客户端口服务器
        to_client = client.to_client()
        fl.client.start_client(server_address=self.conf["address"], client=to_client)

        self.accs = self.server.accs
        self.losses = self.server.losses

    # 启动同态加密，不过同态加密由于太难了，这里实现的是本地运行的版本
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

            # print("Epoch %d, acc: %f\n" % (e, acc))

    # 实现负数据库下的train
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

    # 实现改进gan下的train
    def start_pputl_client_train(self,G):
        train_dataset_size = len(self.train_datasets)
        client=PPUTL_Client(self.conf, self.server.global_model, G, train_dataset_size, self.train_datasets,self.eval_datasets, self.node_id)

        self.clients.append(client)

        # 启动客户端口服务器
        to_client = client.to_client()
        fl.client.start_client(server_address=self.conf["address"], client=to_client)

        self.accs = self.server.accs
        self.losses = self.server.losses

    # 实现共享权重下的train
    def start_weight_share_protect_train(self):
        trainloader = torch.utils.data.DataLoader(
            Mydataset_numpy_client_UDK(mode='train', dataset=self.dataset_path, conf=self.conf, ID=4),
            batch_size=self.conf["batch_size"], shuffle=True)
        client=User_UDK(self.conf, self.server.global_model, trainloader,self.global_testloader, self.node_id)

        self.clients.append(client)

        # 启动客户端口服务器
        to_client = client.to_client()
        fl.client.start_client(server_address=self.conf["address"], client=to_client)

        self.accs = self.server.accs
        self.losses = self.server.losses


    # 测试模型accuracy和loss , 模型预测部分，有不同的预测方法
    def model_eval(self,model_path):
        if self.conf["type"] in uu.diseases:
            acc,loss=uu.eval_diseases(model_path,self.eval_loader)
            return acc, loss
        acc, loss = uu.eval(model_path, self.eval_loader)
        return acc, loss




