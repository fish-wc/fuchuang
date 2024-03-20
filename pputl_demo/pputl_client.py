import models, torch, copy
import torch.nn as nn
import torch.optim as optim
import math
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PPUTL_Client(object):

    def __init__(self, conf, model, G,train_dataset_size, id=-1):

        self.conf = conf
        self.G = G
        self.local_model = model
        self.local_model = self.local_model.to(device)

        # Check GPU availability and move the model to GPU if available
        # if torch.cuda.is_available():
        # 	self.local_model = self.local_model.cuda()
        # self.mask = {}
        # for name, param in self.local_model.state_dict().items():
        #     p = torch.ones_like(param) * self.conf["prop"]
        #     if torch.is_floating_point(param):
        #         self.mask[name] = torch.bernoulli(p)
        #     else:
        #         self.mask[name] = torch.bernoulli(p).long()

        self.client_id = id
        self.train_dataset_size = train_dataset_size

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
        # print(diff[name])

        return diff



