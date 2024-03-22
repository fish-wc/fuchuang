import models, torch, copy
import torch.nn as nn
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class XNDB_Client(object):

    def __init__(self, conf, model, train_dataset, id=-1):

        self.conf = conf

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

        self.train_dataset = train_dataset

        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['no_models'])
        train_indices = all_range[id * data_len: (id + 1) * data_len]

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices))



    def local_train(self, model):
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'],
                              weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.conf['local_epochs'])
        self.local_model.train()
        for e in range(self.conf["local_epochs"]):

            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch

                if torch.cuda.is_available():
                    data = data.to(torch.float32).to(device)
                    target = target.to(device).squeeze(1)

                optimizer.zero_grad()
                output = self.local_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            print("Epoch %d done." % e)
            scheduler.step()
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - model.state_dict()[name])
        return diff
