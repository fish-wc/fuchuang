import csv


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import models


device = 'cuda' if torch.cuda.is_available() else 'cpu'
class User_UDK:
    def __init__(self, conf, model, trainloader,testloader, id = -1):
        self.conf = conf

        self.local_model = models.get_model(self.conf["model_name"])
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

        self.train_loader =trainloader
        self.test_loader =testloader
        self.client_id = id

        # self.client_id = id
        # self.ID = 0
        # self.model = None
        # self.best_accuracy = 0
        # self.loss = 0
        #
        # self.epoch = 10
        # self.save = True
        #
        # self.learning_rate = 0.2
        #
        # self.model_save_path = ""
        #
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # def setLearning_rate(self, lr):
    #     self.learning_rate = lr
    #
    # def setEpoch(self, e):
    #     self.epoch = e
    #
    # def setattr(self, id, model, train_loader, test_loader):
    #     self.ID = id
    #     self.model = model
    #     self.train_loader = train_loader
    #     self.test_loader = test_loader
        # self.date = date
        # self.model_save_path = self.date + "/User" + str(self.ID) + "/"
        # # if not os.path.exists(self.model_save_path ): os.makedirs(self.model_save_path )
        #
        # if not os.path.exists('records/' + self.date + '/user_' + str(self.ID)): os.makedirs(
        #     'records/' + self.date + '/user_' + str(self.ID))

    def local_train(self,model):

        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        # self.model.train()
        # self.model = self.model.to(self.device)
        # if self.device == 'cuda':
        #     net = torch.nn.DataParallel(self.model)
        #     cudnn.benchmark = True
        # criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],momentum=self.conf['momentum'])
        self.local_model.train()
        # optimizer = optim.SGD(net.parameters(), lr=learningrate)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100, gamma=0.65)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epoch)
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

    # def update_local_model(self, model):
    #     self.model = model
        # for epo in range(self.epoch):
        #     train_loss = 0
        #     correct = 0
        #     total = 0
        #     for batch_idx, (inputs, targets) in enumerate(self.train_loader):
        #         inputs, targets = inputs.to(self.device), targets.to(self.device)
        #         optimizer.zero_grad()
        #         outputs = net(inputs)
        #         loss = criterion(outputs, targets)
        #         loss.backward()
        #         optimizer.step()
        #
        #         train_loss += loss.item()
        #         _, predicted = outputs.max(1)
        #         total += targets.size(0)
        #         correct += predicted.eq(targets).sum().item()
        #
        #         if (batch_idx + 1) % 10 == 0:
        #             print('User: {} Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(self.ID,
        #                                                                                    epo, (batch_idx + 1) * len(
        #                     inputs), len(self.train_loader.dataset),
        #                                                                                    100. * (batch_idx + 1) / len(
        #                                                                                        self.train_loader),
        #                                                                                    train_loss / (
        #                                                                                                batch_idx + 1)))
        #         # scheduler.step()
        #     if not os.path.exists("records"): os.makedirs("records")
        #     with open('records/' + self.date + '/user_' + str(self.ID) + '/accuracy_train.csv', 'a+',
        #               newline='') as csvfile:
        #         # 创建字段名
        #         fieldnames = ['user', 'epoch', 'loss', 'matched', 'total', 'accuracy']
        #         # 创建字段写入对象
        #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #         # writer.writeheader()
        #         # 写入表格内容
        #         writer.writerow(
        #             {'user': str(self.ID), 'epoch': epo, 'loss': str(train_loss / len(self.train_loader.dataset)),
        #              'matched': str(correct), 'total': str(total),
        #              'accuracy': str(100. * correct / len(self.train_loader.dataset))})

    def test(self):

        self.local_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.local_model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    #
    #     print('\nUser:{} Test  Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(self.ID,
    #                                                                                                test_loss / len(
    #                                                                                                    self.test_loader.dataset),
    #                                                                                                correct,
    #                                                                                                len(self.test_loader.dataset),
    #                                                                                                100. * correct / len(
    #                                                                                                    self.test_loader.dataset)))
    #
    #     if not os.path.exists("records"): os.makedirs("records")
    #     with open('records/' + self.date + '/user_' + str(self.ID) + '/accuracy_test.csv', 'a+', newline='') as csvfile:
    #         # 创建字段名
    #         fieldnames = ['user', 'loss', 'matched', 'total', 'accuracy']
    #         # 创建字段写入对象
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #         # writer.writeheader()
    #         # 写入表格内容
    #         writer.writerow(
    #             {'user': str(self.ID), 'loss': test_loss / len(self.test_loader.dataset),
    #              'matched': str(correct), 'total': str(total),
    #              'accuracy': str(100. * correct / len(self.test_loader.dataset))})
    #
    #     temp_accuracy = 100. * correct / len(self.test_loader.dataset)
    #     if temp_accuracy > self.best_accuracy:
    #         best_accuracy = temp_accuracy
    #         # torch.save(self.model.state_dict(),  self.model_save_path+'net-params-best-accuracy.pkl')


    #

