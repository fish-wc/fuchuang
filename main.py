from trainer import *
import json
import argparse
from plot_acc import *
import warnings
from pputl_demo import pputl
from xor_and_ndb import xor, ndb
from weight_share_protect.matrix_add_mul_sort_transform_using_different_key import matrix_add_mul_sort
from pputl_demo.target_model import *

warnings.filterwarnings("ignore", category=UserWarning)

# 修改了
if __name__ == '__main__':

    # 预置超参数
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('--choice', type=int, required=True, help='choice')
    parser.add_argument('--no_models', type=int, default=10, help='clients_num')
    parser.add_argument('--model_name', type=str, default="resnet50", help='model')
    parser.add_argument('--type', type=str, default="cifar", help='dataset_name')
    parser.add_argument('--global_epochs', type=int, default=100, help='global_epochs')
    parser.add_argument('--local_epochs', type=int, default=3, help='local_epochs')
    parser.add_argument('--k', type=int, default=7, help='train_clients_num')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    # parser.add_argument('--client_batchsize', type=int, default=100, help='weight_share_protect_client_batchsize')
    # parser.add_argument('--global_batchsize', type=int, default=500, help='weight_share_protect_global_batchsize')
    parser.add_argument('--lr', type=float, default=0.1, help='learning_rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='momentum')
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
    args = parser.parse_args()

    # with open(args.conf, 'r') as f:
    # 	conf = json.load(f)
    # 选择保护算法
    print("正常模式输入0；差分隐私输入1；同态加密输入2:负数据库输入3;改进的生成对抗网络输入4;共享权重模式协作学习输入5:")
    choice = args.choice
    conf = {"choice": args.choice, "no_models": args.no_models, "model_name": args.model_name, "type": args.type,
            "global_epochs": args.global_epochs,
            "local_epochs": args.local_epochs, "k": args.k, "batch_size": args.batch_size, "lr": args.lr,
            "momentum": args.momentum, "lambda_": args.lambda_, "dp": args.dp, "C": args.C, "sigma": args.sigma,
            "q": args.q, "w": args.w, "feature_num": args.feature_num,
            "eta": args.eta, "alpha": args.alpha, "poison_label": args.poison_label,
            "poisoning_per_batch": args.poisoning_per_batch, "prop": args.prop, "root": args.root}

    trainer = Train(conf, choice)
    print("start training!")
    if choice == 0:
        trainer.norm_train()
    elif choice == 1:
        trainer.differential_privacy_train()
    elif choice == 3:
        # 以下两步数据转换
        xor.XOR()
        ndb.NDB_f()
        trainer.xndb_train()
    elif choice == 4:
        G = pputl.PPUTL(conf["batch_size"])
        G.eval()
        trainer.pputl_client_train(G)
    elif choice == 5:
        matrix_add_mul_sort(user_num=conf["no_models"])
        trainer.weight_share_protect_train()
    else:
        trainer.homomorphic_encryption_train()
    acc_loss(trainer.accs, trainer.losses)












