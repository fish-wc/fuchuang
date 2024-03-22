# import warnings
# import os
# from pputl_demo.target_model import *
# from ml_attack.ml import mla_attack
# import argparse
# from inversion_attack.train_classifier import inversion_attack_classifer
# from  inversion_attack.train_inversion import inversion_model_train
# from pputl_demo import pputl
# warnings.filterwarnings("ignore", category=UserWarning)
# from dlg_attack.dlg import dlg_attack_add_noisy,dlg_attack_no_noisy, dlg_attack_xor_and_ndb,dlg_attack_pputl,dlg_attack_weight_share_protect
# from ml_attack.save_result import savere
# from xor_and_ndb import xor,ndb
# from weight_share_protect.matrix_add_mul_sort_transform_using_different_key import matrix_add_mul_sort
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Attack_and_Defense')
#     parser.add_argument('--attack_type', type=int, default=2, help='attack_type')
#     parser.add_argument('--choice', type=int, default=0, help='choice')
#     args = parser.parse_args()
#     print("attack_type为0时-->梯度泄露攻击------attack_type为1时-->成员推理攻击------attack_type为2时-->模型逆向攻击")
#     print("------------------------------------------------------------------------------------------------------------")
#     print("当attack_type为0且choice为0时模拟梯度泄露攻击-----当choice为（1-4）分别对应以下不同隐私保护方案：")
#     print("1-->差分隐私\n2-->负数据库\n3-->改进的生成对抗网络\n4-->共享权重模式协作学习")
#     print("------------------------------------------------------------------------------------------------------------")
#     print("当attack_type为1且choice为0时模拟成员推理攻击-----当choice为（1-3）分别对应以下不同隐私保护方案：")
#     print("1-->差分隐私\n2-->负数据库\n3-->共享权重模式协作学习")
#     print("------------------------------------------------------------------------------------------------------------")
#     print("当attack_type为2且choice为0时模拟模型逆向攻击-----当choice为（1-3）分别对应以下不同隐私保护方案：")
#     print("1-->差分隐私\n2-->负数据库\n3-->共享权重模式协作学习")
#     print("------------------------------------------------------------------------------------------------------------")
#     choice = args.choice
#     mode = args.attack_type
#     if mode == 0:
#         if choice==0:
#             dlg_attack_no_noisy()
#         elif choice==1:
#             dlg_attack_add_noisy()
#         elif choice==2:
#             #转换数据
#             xor.XOR()
#             ndb.NDB_f()
#             dlg_attack_xor_and_ndb()
#         elif choice==3:
#             # load_path = f'pputl_demo/G_path/trained_generator_G.pth'
#             # # 加载模型权重
#             # G = Generator64().to(device)
#             # G.load_state_dict(torch.load(load_path))
#             #正式运行程序应该将上述注释掉，使用下面方法生成G
#             G = pputl.PPUTL()
#             dlg_attack_pputl(G)
#         elif choice==4:
#             matrix_add_mul_sort()
#             dlg_attack_weight_share_protect()
#
#     elif mode == 1:
#         if choice == 0:
#             result_path = "ml_attack/mla_result/"
#             report = mla_attack(choice)
#             savere(result_path,report)
#         elif choice==1:
#             result_path = "ml_attack/mla_add_noisy_result/"
#             report = mla_attack(choice)
#             savere(result_path, report)
#         elif choice==2:
#             # 转换数据
#             xor.XOR()
#             ndb.NDB_f()
#             #该程序需要样本量，目前样本量不足直接运行会报错
#             result_path = "ml_attack/mla_xor_and_ndb_result/"
#             report=mla_attack(choice)
#             savere(result_path, report)
#
#         elif choice==3:
#             #这个也是
#             matrix_add_mul_sort()
#             result_path = "ml_attack/mla_weight_share_protect_result/"
#             report = mla_attack(choice)
#             savere(result_path, report)
#
#     elif mode == 2:
#         if choice == 0:
#             result_path = 'inversion_attack/norm_attack_'
#             inversion_attack_classifer(result_path,choice)
#             inversion_model_train(result_path,choice)
#
#         elif choice == 1:
#             result_path = 'inversion_attack/add_noisy_'
#             inversion_attack_classifer(result_path, choice)
#             inversion_model_train(result_path,choice)
#
#         elif choice == 2:
#             # 转换数据
#             xor.XOR()
#             ndb.NDB_f()
#             result_path = 'inversion_attack/xor_and_ndb_'
#             inversion_attack_classifer(result_path, choice)
#             inversion_model_train(result_path, choice)
#
#         elif choice == 3:
#             # 这个也是
#             matrix_add_mul_sort()
#             result_path = 'inversion_attack/weight_share_protect_'
#             inversion_attack_classifer(result_path, choice)
#             inversion_model_train(result_path, choice)

import warnings
import os
from pputl_demo.target_model import *
from ml_attack.ml import mla_attack
import argparse
from inversion_attack.train_classifier import inversion_attack_classifer
from  inversion_attack.train_inversion import inversion_model_train
from pputl_demo import pputl
warnings.filterwarnings("ignore", category=UserWarning)
from dlg_attack.dlg import dlg_attack_add_noisy,dlg_attack_no_noisy, dlg_attack_xor_and_ndb,dlg_attack_pputl,dlg_attack_weight_share_protect
from ml_attack.save_result import savere
from xor_and_ndb import xor,ndb
from weight_share_protect.matrix_add_mul_sort_transform_using_different_key import matrix_add_mul_sort
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attack_and_Defense')
    parser.add_argument('--attack_type', type=int, default=2, help='attack_type')
    parser.add_argument('--choice', type=int, default=0, help='choice')
    args = parser.parse_args()
    print("attack_type为0时-->梯度泄露攻击------attack_type为1时-->成员推理攻击------attack_type为2时-->模型逆向攻击")
    print("------------------------------------------------------------------------------------------------------------")
    print("当attack_type为0且choice为0时模拟梯度泄露攻击-----当choice为（1-4）分别对应以下不同隐私保护方案：")
    print("1-->差分隐私\n2-->负数据库\n3-->改进的生成对抗网络\n4-->共享权重模式协作学习")
    print("------------------------------------------------------------------------------------------------------------")
    print("当attack_type为1且choice为0时模拟成员推理攻击-----当choice为（1-3）分别对应以下不同隐私保护方案：")
    print("1-->差分隐私\n2-->负数据库\n3-->共享权重模式协作学习")
    print("------------------------------------------------------------------------------------------------------------")
    print("当attack_type为2且choice为0时模拟模型逆向攻击-----当choice为（1-3）分别对应以下不同隐私保护方案：")
    print("1-->差分隐私\n2-->负数据库\n3-->共享权重模式协作学习")
    print("------------------------------------------------------------------------------------------------------------")
    choice = args.choice
    mode = args.attack_type
    if mode == 0:
        if choice==0:
            dlg_attack_no_noisy()
        elif choice==1:
            dlg_attack_add_noisy()
        elif choice==2:
            #转换数据
            # xor.XOR()
            # ndb.NDB_f()
            dlg_attack_xor_and_ndb()
        elif choice==3:
            load_path = f'pputl_demo/G_path/trained_generator_G.pth'
            # 加载模型权重
            G = Generator64().to(device)
            G.load_state_dict(torch.load(load_path))
            #正式运行程序应该将上述注释掉，使用下面方法生成G
            # G = pputl.PPUTL()
            dlg_attack_pputl(G)
        elif choice==4:
            # matrix_add_mul_sort()
            dlg_attack_weight_share_protect()

    elif mode == 1:
        if choice == 0:
            result_path = "ml_attack/mla_result/"
            report = mla_attack(choice)
            savere(result_path,report)
        elif choice==1:
            result_path = "ml_attack/mla_add_noisy_result/"
            report = mla_attack(choice)
            savere(result_path, report)
        elif choice==2:
            # 转换数据
            # xor.XOR()
            # ndb.NDB_f()
            #该程序需要样本量，目前样本量不足直接运行会报错
            result_path = "ml_attack/mla_xor_and_ndb_result/"
            report=mla_attack(choice)
            savere(result_path, report)

        elif choice==3:
            #这个也是
            # matrix_add_mul_sort()
            result_path = "ml_attack/mla_weight_share_protect_result/"
            report = mla_attack(choice)
            savere(result_path, report)

    elif mode == 2:
        if choice == 0:
            result_path = 'inversion_attack/norm_attack_'
            inversion_attack_classifer(result_path,choice)
            inversion_model_train(result_path,choice)

        elif choice == 1:
            result_path = 'inversion_attack/add_noisy_'
            inversion_attack_classifer(result_path, choice)
            inversion_model_train(result_path,choice)

        elif choice == 2:
            # 转换数据
            # xor.XOR()
            # ndb.NDB_f()
            result_path = 'inversion_attack/xor_and_ndb_'
            inversion_attack_classifer(result_path, choice)
            inversion_model_train(result_path, choice)

        elif choice == 3:
            # 这个也是
            # matrix_add_mul_sort()
            result_path = 'inversion_attack/weight_share_protect_'
            inversion_attack_classifer(result_path, choice)
            inversion_model_train(result_path, choice)

