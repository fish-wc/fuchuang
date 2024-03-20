from server import *
from norm_client import *
import datasets2
from differential_privacy.differential_privacy_client import *
import random
from homomorphic_encryption.homomorphic_encryption_client import *

from xor_and_ndb.xndb_client import *
from pputl_demo.pputl_client import *
from weight_share_protect.User_UDK import *
from weight_share_protect.Mydataset_for_numpy_client_UDK import *
from weight_share_protect.Mydataset_for_numpy_server_UDK import *


class Train(object):
    def __init__(self, conf, choice):
        self.conf = conf
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
                Mydataset_numpy_server_UDK(mode='test', dataset=self.dataset_path, conf=self.conf),
                batch_size=self.conf["batch_size"])
            self.server = Server(self.conf, self.global_testloader, choice)

        else:
            self.train_datasets, self.eval_datasets = datasets2.get_dataset("data/", self.conf["type"], choice)
            self.server = Server(self.conf, self.eval_datasets, choice)

        self.clients = []
        self.accs = []
        self.losses = []

    def norm_train(self):

        for c in range(self.conf["no_models"]):
            self.clients.append(Norm_Client(self.conf, self.server.global_model, self.train_datasets, c))

        for e in range(self.conf["global_epochs"]):

            candidates = random.sample(self.clients, self.conf["k"])

            weight_accumulator = {}

            for name, params in self.server.global_model.state_dict().items():
                weight_accumulator[name] = torch.zeros_like(params)

            for c in candidates:
                diff = c.local_train(self.server.global_model)

                for name, params in self.server.global_model.state_dict().items():
                    weight_accumulator[name].add_(diff[name])

            self.server.model_aggregate_norm(weight_accumulator)

            acc, loss = self.server.model_eval()
            self.accs.append(acc)
            self.losses.append(loss)
            print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

    def weight_share_protect_train(self):

        for c in range(self.conf["no_models"]):
            trainloader = torch.utils.data.DataLoader(
                Mydataset_numpy_client_UDK(mode='train', dataset=self.dataset_path, conf=self.conf, ID=4),
                batch_size=self.conf["batch_size"], shuffle=True)

            u = User_UDK(self.conf, self.server.global_model, trainloader, c)
            self.clients.append(u)

        for e in range(self.conf["global_epochs"]):
            local_weights = []

            candidates = random.sample(self.clients, self.conf["k"])

            for c in candidates:
                c.local_train(self.server.global_model)
                w = c.get_local_model()
                local_weights.append(copy.deepcopy(w))

            global_weights = self.server.average_weights(local_weights)
            self.server.global_model.load_state_dict(global_weights)

            acc, loss = self.server.weight_share_protect_eval()
            self.accs.append(acc)
            self.losses.append(loss)
            print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

    def pputl_client_train(self, G):
        train_dataset_size = len(self.train_datasets)
        for c in range(self.conf["no_models"]):
            self.clients.append(PPUTL_Client(self.conf, self.server.global_model, G,train_dataset_size/self.conf["no_models"], c))

        for e in range(self.conf["global_epochs"]):

            candidates = random.sample(self.clients, self.conf["k"])

            weight_accumulator = {}

            for name, params in self.server.global_model.state_dict().items():
                weight_accumulator[name] = torch.zeros_like(params)

            for c in candidates:
                diff = c.local_train(self.server.global_model)
                for name, params in self.server.global_model.state_dict().items():
                    weight_accumulator[name].add_(diff[name])

            self.server.model_aggregate_norm(weight_accumulator)

            acc, loss = self.server.model_eval_pputl(G)
            self.accs.append(acc)
            self.losses.append(loss)
            print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

    def differential_privacy_train(self):
        for c in range(self.conf["no_models"]):
            self.clients.append(
                Differential_Privacy_Client(self.conf, self.server.global_model, self.train_datasets, c))

        for e in range(self.conf["global_epochs"]):

            candidates = random.sample(self.clients, self.conf["k"])

            weight_accumulator = {}

            for name, params in self.server.global_model.state_dict().items():
                weight_accumulator[name] = torch.zeros_like(params)

            for c in candidates:
                diff = c.local_train(self.server.global_model)

                for name, params in self.server.global_model.state_dict().items():
                    weight_accumulator[name].add_(diff[name])

            self.server.model_aggregate_differential_privacy(weight_accumulator)

            acc, loss = self.server.model_eval()
            self.accs.append(acc)
            self.losses.append(loss)
            print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

    def homomorphic_encryption_train(self):

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
        for c in range(self.conf["no_models"]):
            self.clients.append(XNDB_Client(self.conf, self.server.global_model, self.train_datasets, c))

        for e in range(self.conf["global_epochs"]):

            candidates = random.sample(self.clients, self.conf["k"])

            weight_accumulator = {}

            for name, params in self.server.global_model.state_dict().items():
                weight_accumulator[name] = torch.zeros_like(params)

            for c in candidates:
                diff = c.local_train(self.server.global_model)
                for name, params in self.server.global_model.state_dict().items():
                    weight_accumulator[name].add_(diff[name])

            self.server.model_aggregate_norm(weight_accumulator)

            acc, loss = self.server.model_eval_xndb()
            self.accs.append(acc)
            self.losses.append(loss)
            print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
