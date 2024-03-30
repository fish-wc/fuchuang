
import json
import argparse
from plot_acc import *
import warnings
from pputl_demo import pputl
from xor_and_ndb import xor, ndb
from weight_share_protect.matrix_add_mul_sort_transform_using_different_key import matrix_add_mul_sort
from pputl_demo.target_model import *


from trainer_flower import *
import json
import argparse
from plot_acc import *
import warnings
from pputl_demo import pputl
from xor_and_ndb import xor,ndb
from weight_share_protect.matrix_add_mul_sort_transform_using_different_key import matrix_add_mul_sort
from pputl_demo.target_model import *
warnings.filterwarnings("ignore", category=UserWarning)
import utils.utils as uu

warnings.filterwarnings("ignore", category=UserWarning)

import os
import myopenai as mo



# 修改了
if __name__ == '__main__':

    # 预置超参数
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('--choice', type=int, required=True, help='choice,-1 means server')
    parser.add_argument('--no_models', type=int, default=3, help='clients_num')
    parser.add_argument('--model_name', type=str, default="resnet50", help='model')
    parser.add_argument('--global_epochs', type=int, default=5, help='global_epochs')
    parser.add_argument('--local_epochs', type=int, default=3, help='local_epochs')
    parser.add_argument('--k', type=int, default=3, help='train_clients_num')
    parser.add_argument('--batch_size', type=int, default=100, help='batch_size')
    # parser.add_argument('--client_batchsize', type=int, default=100, help='weight_share_protect_client_batchsize')
    # parser.add_argument('--global_batchsize', type=int, default=500, help='weight_share_protect_global_batchsize')
    parser.add_argument('--lr', type=float, default=0.1, help='learning_rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--lambda_', type=float, default=0.1, help='lambda_')
    parser.add_argument('--dp', type=bool, default=True, help='dp')
    parser.add_argument('--C', type=int, default=1000, help='c')
    parser.add_argument('--sigma', type=float, default=0.01, help='sigma')
    parser.add_argument('--q', type=float, default=0.2, help='q')
    parser.add_argument('--w', type=int, default=2, help='w')
    parser.add_argument('--feature_num', type=int, default=30, help='feature_num')
    parser.add_argument('--eta', type=int, default=2, help='eta')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha')
    parser.add_argument('--poison_label', type=int, default=2, help='poison_label')
    parser.add_argument('--poisoning_per_batch', type=int, default=4, help='poisoning_per_batch')
    parser.add_argument('--prop', type=float, default=0.6, help='prop')
    parser.add_argument('--root', type=str, default="ndb_cifar10_data/", help='root')
    # parser.add_argument('-c', '--conf', dest='conf', default='./utils/conf.json')
    parser.add_argument('--address',type=str,default="127.0.0.1:8080",help="address")
    parser.add_argument('--min_available_clients',type=int,default=2,help="min available clients")

    parser.add_argument('--node_id',type=int,default=1,help="client's id")
    parser.add_argument('--value_steps',type=int,default=5,help="number of clients to evaluate")

    parser.add_argument('--pchoice',type=int,default=1,help="Which predict to use")

    parser.add_argument('--save_json',type=str,default='./data/answer.json',help="Json to save")

    #关于模型的
    parser.add_argument('--model_save_path',type=str,default='./mymodel',help='where to save my trained model')
    parser.add_argument('--model_save_name',type=str,default='test',help='save model name')
    # 增量训练部分，模型一定要与要训练的模型相匹配才行
    parser.add_argument('--increment', type=str, default="./mymodel/resnet50.pth", help="the model you want to train")
    parser.add_argument('--is_increment', type=bool, default=True, help="do you need increment?")

    # 关于数据的参数
    parser.add_argument('--type', type=str, default="cifar", help='dataset_name') # 这个是数据种类，有12个数据集
    parser.add_argument('--data_path', type=str, default='./data', help='data to choose')
    parser.add_argument('--subsize_rate',type=float,default=0.05,help='subsize data rate')
    args = parser.parse_args()
    '''
    数据集type选择种类：'cifar'，'mnist'，"PathMNIST"，"DermaMNIST"，"OCTMNIST"，"PneumoniaMNIST"，"RetinaMNIST"，"BreastMNIST"
    "BloodMNIST"，"TissueMNIST"，"OrganAMNIST"，"ChestMNIST" 欧阳说有两个数据集还没弄，忘记是哪两个了
    '''#数据集type的可选项有整整12个

    # 选择保护算法
    # 改进的生成
    print("正常模式输入0；差分隐私输入1；同态加密输入2:负数据库输入3;改进的生成对抗网络输入4;共享权重模式协作学习输入5;其他功能6:")
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    choice = args.choice
    pchoice=args.pchoice
    node_id=args.node_id
    conf = {"choice": args.choice, "no_models": args.no_models, "model_name": args.model_name, "type": args.type,
            "global_epochs": args.global_epochs,
            "local_epochs": args.local_epochs, "k": args.k, "batch_size": args.batch_size, "lr": args.lr,
            "momentum": args.momentum, "lambda_": args.lambda_, "dp": args.dp, "C": args.C, "sigma": args.sigma,
            "q": args.q, "w": args.w, "feature_num": args.feature_num,
            "eta": args.eta, "alpha": args.alpha, "poison_label": args.poison_label,
            "poisoning_per_batch": args.poisoning_per_batch, "prop": args.prop, "root": args.root,

            "address":args.address ,"min_available_clients": args.min_available_clients,"node_id":args.node_id #这里又加了两排
            ,"value_steps":args.value_steps,"increment":args.increment,"is_increment":args.is_increment,
            "pchoice":args.pchoice,"save_json":args.save_json,"data_path":args.data_path,
            "subsize_rate":args.subsize_rate,"model_save_path":args.model_save_path,"model_save_name":args.model_save_name,
            }
    # model_path="./mymodel/resnet50.pth" # 这个部分需要自己传
    conf["type"]="OrganAMNIST"
    conf["increment"] = "./model_jishe/model_result_OrganAMNIST.pth"

    trainer = Train(conf, choice,node_id)
    print("start training!")
    if choice == -1:
        #启动服务器
        print("正在为您启动服务器...")
        trainer.start_server()
        print("服务器训练结束")

        # 在这里获取训练的评价指标
        acc_loss(trainer.accs, trainer.losses)

    elif choice == 0:
        # 启动正常训练情况下的train
        print("正常情况下的本地训练开始")
        print(f"正在启动编号为{node_id}的client...")
        trainer.norm_train_start_client()

    elif choice == 1:
        print("差分隐私加密算法保护情况下开始训练")
        print(f"正在启动编号为{node_id}的client...")
        trainer.start_differential_privacy_train()

    elif choice ==2 :  ### 先不管
        #  同态机密算法现在还不能实现多机分布式训练，实现有难度
        print("同态加密算法保护情况下开始训练")
        print(f"正在启动编号为{node_id}的client...")
        trainer.start_homomorphic_encryption_train()
        acc_loss(trainer.accs, trainer.losses)

    elif choice == 3:
        # 以下两步数据转换
        # xor.XOR()
        # ndb.NDB_f()
        print("负数据库加密情况下的本地训练开始")
        print(f"正在启动编号为{node_id}的client...")
        trainer.start_xndb_train()

    elif choice == 4:
        save_path = f'pputl_demo/G_path/trained_generator_G.pth'
        if os.path.exists(save_path):
            G = Generator64().to(device)
            G.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
            # G.load_state_dict(torch.load(save_path))
        else:
            G = pputl.PPUTL(conf["batch_size"])
        G.eval()
        print("pputl情况下的本地训练开始")
        print(f"正在启动编号为{node_id}的client...")
        trainer.start_pputl_client_train(G)

    elif choice == 5:
     #   matrix_add_mul_sort(user_num=conf["no_models"])
        trainer.start_weight_share_protect_train()

    elif choice==6:

        model_path=conf["increment"]

        if pchoice==1:
            # 这里只是用来测试的，只是评估
            acc,loss=trainer.model_eval(model_path)
            print("accuracy:",acc,"loss:",loss)

        elif pchoice==2:
            save_path=conf["save_json"]
            # 可以下载json文件，还可以绘制图。
            # 这里是预测结果
            accuracy, average_loss, images, predicted_labels, true_labels, images_name = uu.predict_and_evaluate(model_path, dataset_folder='./data/cifar10_png')
            # 这里是展示结果
            uu.show_images(images, true_labels, predicted_labels, 25,save_name='show_images')
            # 这里将返回的结果保存为json
            uu.save_json(save_path,images_name,predicted_labels)
            # 现在是将保存的结果读出来
            values=uu.load_json(save_path)
            # 现在是基于json文件绘制类别的柱状图
            uu.plot_category('classPicture',values)
            # 打印评价指标
            print("accuracy:",accuracy,"loss:",average_loss)

        elif pchoice==3:
            # 这里是只放一个图片的情况,放了一张
            one_image_path='./data/cifar10_png/0/52.png'
            # 首先来做预测
            predict_label,image,true_label=uu.predict_image(image_path=one_image_path,model_path=model_path)
            # 打印一张图的那种
            uu.show_image(image,predict_label,true_label,"showOneImage")
            name=classes[predict_label]
            print(f"该图片的预测结果是{name},label({predict_label})")

        elif pchoice==4:
            content="你觉得一个人一顿饭能吃多少"
            response=mo.myChatGPT(content)
            print(response)






