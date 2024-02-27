import os, time
import matplotlib.pyplot as plt
import pickle
# import imageio
import ssl
import numpy as np
import torch
import torch.optim as optim
from pputl_demo.source_model import *
from pputl_demo.target_model import *
from torchvision import datasets, transforms
from pputl_demo.component import show_result, show_train_hist
import models
import datasets2
ssl._create_default_https_context = ssl._create_unverified_context


# training parameters


# train_hist['per_epoch_ptimes'] = []
# train_hist['total_ptime'] = []
def PPUTL(batchsz=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # start_time = time.time()
    channel = 3
    batch_size = batchsz
    lr_D = 0.4#0.0004
    lr_G = 0.1#0.0001
    train_epoch = 100#100
    train_step = 100000#100000
    alpha = 10
    k_gen = 0.1
    k_re = 1
    k_sem = 0.1
    sensitivety = 1
    epsilon = 1
    start_adaptation = 0
    start_train_C = 0

    # fixed noise & label
    temp_z_ = torch.randn(10, 128)
    fixed_z_ = temp_z_
    fixed_y_ = torch.zeros(10, 1)
    for i in range(9):
        fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
        temp = torch.ones(10, 1) + i
        fixed_y_ = torch.cat([fixed_y_, temp], 0)
    fixed_y_label_ = torch.zeros(100, 10)
    fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
    fixed_z_, fixed_y_label_ = fixed_z_.to(device), fixed_y_label_.to(device)

    # data_loader
    # transform_1 = transforms.Compose([
    #     transforms.ToTensor(),
    #     custmResize(64),
    #     transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    # transform_2 = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     custmResize(64),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    # transform_2 = transforms.Compose([
    #     transforms.ToTensor(),
    #     custmResize(64),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    sourceset, eval_datasets = datasets2.get_dataset("data/", 'cifar', 4)
    trainset, eval_datasets = datasets2.get_dataset("data/", 'cifar', 4)
    source_loader = torch.utils.data.DataLoader(sourceset,batch_size=batch_size, shuffle=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle=True, drop_last=True)

    # network
    G = Generator64().to(device)
    D = Discriminator64().to(device)
    C_extractor = Extractor().to(device)
    C_classifier = Classifier().to(device)

    loss_fn = Wasserstein()
    criterion = nn.CrossEntropyLoss()
    selfEntropy = self_entropy()
    feature_distance = feature_dis()
    l2Norm = l2_norm()
    vat = VATLoss()

    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=lr_G, betas=(0.0, 0.9))
    D_optimizer = optim.Adam(D.parameters(), lr=lr_D, betas=(0.0, 0.9))
    C_optimizer = optim.Adam([{'params': C_extractor.parameters()}, {'params': C_classifier.parameters()}])

    sched_G = optim.lr_scheduler.LambdaLR(
        G_optimizer, lambda step: 1 - step / train_step)
    sched_D = optim.lr_scheduler.LambdaLR(
        D_optimizer, lambda step: 1 - step / train_step)

    # results save folder
    root = './pputl_demo/PPUTL_G_CIFAR10_results/'
    if not os.path.isdir(root):
        os.mkdir(root)
    if not os.path.isdir(root + 'Fixed_results_' + str(channel)):
        os.mkdir(root + 'Fixed_results_' + str(channel))

    train_hist = {}

    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['C_tar_losses'] = []
    train_hist['C_losses'] = []
    train_hist['C_src_losses'] = []
    G.train()
    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []
        C_losses = []
        C_src_losses = []
        C_tar_losses = []

        if (epoch + 1) == 2:
            start_adaptation = 1

        if (epoch + 1) == 41:
            start_train_C = 1

        # epoch_start_time = time.time()
        target_looper = infiniteloop(train_loader)
        source_looper = infiniteloop(source_loader)
        for batch_idx in range(len(train_loader)):
            if batch_idx % 100 == 0:
                print("\nBatch index: ", batch_idx)
            for i in range(1):
                D_optimizer.zero_grad()

                with torch.no_grad():
                    z_ = torch.randn(batch_size, 128).to(device)
                    y_d = (torch.rand(batch_size, 1) * 10).type(torch.LongTensor).to(device)
                    y_label_ = torch.zeros(batch_size, 10).to(device)
                    y_label_.scatter_(1, y_d.view(batch_size, 1), 1)
                    G_result = G(z_, y_label_).detach()  # batch_size * 3 * 32 * 32

                x_, y_ = next(target_looper)
                x_, y_ = x_.to(device), y_.to(device)

                D_real = D(x_)
                D_fake = D(G_result)
                loss = loss_fn(D_real, D_fake)
                loss_gp = loss_fn.cacl_gradient_penalty(D, x_, G_result)
                D_train_loss = loss + 10 * loss_gp

                D_train_loss.backward()
                D_optimizer.step()
                D_losses.append(D_train_loss.item())
            if batch_idx % 100 == 0:
                print("Discriminator Loss: ", D_train_loss.item())

            if start_train_C:
                for i in range(1):
                    x_s, y_s = next(source_looper)
                    x_s, y_s = x_s.to(device), y_s.to(device)
                    C_optimizer.zero_grad()

                    y_label_c_ = y_d.view(batch_size).to(device)
                    C_src_result = C_classifier(C_extractor(x_s))
                    C_src_loss = criterion(C_src_result, y_s)
                    C_src_losses.append(C_src_loss.item())
                    if batch_idx % 100 == 0:
                        print("C_source Loss: ", C_src_loss.item())
                    G_result = G(z_, y_label_)
                    C_result = C_classifier(C_extractor(G_result))
                    C_result_softmax = F.softmax(C_result, dim=1)
                    C_selfEntropy = selfEntropy(C_result_softmax)
                    C_loss = criterion(C_result, y_label_c_)  # (except_x,except_y)

                    C_train_loss = C_src_loss + k_gen * C_loss + k_re * C_selfEntropy
                    C_train_loss.backward()
                    C_optimizer.step()
                    C_losses.append(C_train_loss.item())
                if batch_idx % 100 == 0:
                    print("C_train Loss: ", C_train_loss.item())

            for i in range(1):
                G_optimizer.zero_grad()

                x_s, y_s = next(source_looper)
                x_s, y_s = x_s.to(device), y_s.to(device)

                z_ = torch.randn(batch_size, 128).to(device)
                # y_d = (torch.rand(batch_size, 1) * 10).type(torch.LongTensor).to(device)
                y_d = y_s.view(batch_size, 1)
                y_label_ = torch.zeros(batch_size, 10).to(device)
                y_label_.scatter_(1, y_d.view(batch_size, 1), 1)
                y_label_c_ = y_d.view(batch_size).to(device)
                G_result = G(z_, y_label_)
                D_result = D(G_result)
                G_loss = loss_fn(D_result)

                if start_adaptation:
                    G_result = G(z_, y_label_)  # channel*32*32
                    C_tar_feat = C_extractor(G_result)
                    C_result = C_classifier(C_tar_feat)
                    C_src_feat = C_extractor(x_s)
                    feat_dis = feature_distance(C_tar_feat, C_src_feat)
                    # mmd_loss = mmd(C_tar_feat,C_src_feat)
                    C_train_loss = criterion(C_result, y_label_c_)
                    C_tar_losses.append(C_train_loss.item())
                    # G_train_loss = G_loss + k_sem * C_train_loss
                    # G_train_loss = G_loss + k_sem * (C_train_loss + feat_dis + mmd_loss)
                    G_train_loss = G_loss + k_sem * (C_train_loss + feat_dis)
                else:
                    G_train_loss = G_loss

                G_train_loss.backward()
                G_optimizer.step()
                G_losses.append(G_train_loss.item())
            if batch_idx % 100 == 0:
                if start_adaptation:
                    print("C_target Loss: ", C_train_loss.item())
                print("Generator Loss: ", G_train_loss.item())

            sched_G.step()
            sched_D.step()

        # epoch_end_time = time.time()
        # per_epoch_ptime = epoch_end_time - epoch_start_time

        print('\n[%d/%d]' % ((epoch + 1), train_epoch) )
        # per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
        #     # torch.mean(torch.FloatTensor(G_losses)), torch.mean(torch.FloatTensor(C_losses)),
        #     # torch.mean(torch.FloatTensor(C_src_losses)), torch.mean(torch.FloatTensor(C_tar_losses))))
        # fixed_p = root + 'Fixed_results_' + str(channel) + '/' + str(epoch + 1) + '.png'
        # show_result((epoch + 1), fixed_z_, fixed_y_label_, G, save=True, path=fixed_p)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['C_losses'].append(torch.mean(torch.FloatTensor(C_losses)))
        train_hist['C_src_losses'].append(torch.mean(torch.FloatTensor(C_src_losses)))
        train_hist['C_tar_losses'].append(torch.mean(torch.FloatTensor(C_tar_losses)))
        # train_hist['per_epoch_ptimes'].append(per_epoch_ptime)


    # end_time = time.time()
    # total_ptime = end_time - start_time
    # train_hist['total_ptime'].append(total_ptime)
    # print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
    #     torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
    # show_train_hist(train_hist, save=True, path=root + 'train_hist_' + str(channel) + '.png')
    if batch_size != 64:
        save_path = f'pputl_demo/G_path/trained_generator_G_{batchsz}.pth'  # 替换为您希望保存的路径
        torch.save(G.state_dict(), save_path)
    return G


