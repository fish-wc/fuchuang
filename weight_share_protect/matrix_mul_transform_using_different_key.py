import random
import math
import os
import random
import cv2
import numpy as np
import time
# from PIL import Image
import csv
def matrix_sort():
    # 原始 cifar10 数据集路径
    img_origin = "weight_share_protect/cifar10_png_version9-28/data/"
    img_train_origin = img_origin + "train/"
    img_test_origin = img_origin + "test/"
    csv_train_origin = img_origin + "cifar10_train.csv"
    csv_test_origin = img_origin + "cifar10_test.csv"



    dataset_path = "weight_share_protect/UDK_fl_add_mul_sort2/"
    #if not os.path.exists(dataset_path): os.makedirs(dataset_path)

    keep = 1  # 1  保存生成的图像     0   不 保存

    png = 0

    if png == 1:
        img_path_train_png = "sort_png" + "/train/"
        img_path_test_png = "sort_png" + "/test/"
        csv_train_png = "sort_png" + "/train.csv"
        csv_test_png = "sort_png" + "/test.csv"
        if not os.path.exists(img_path_train_png): os.makedirs(img_path_train_png)
        if not os.path.exists(img_path_test_png): os.makedirs(img_path_test_png)

    list_mul_value = [10]
    # list_add_value = [256,512,1024,2048]
    list_user_number = [1,2,3, 4,5, 6,7, 8, 9,10]



    for i_mul_value in range(len(list_mul_value)):
        for i_user_number in range(len(list_user_number)):
            mul_value = list_mul_value[i_mul_value]
            user_number = list_user_number[i_user_number]
            for many_times in range(1):


                if keep == 1:
                    # numpy格式的数据保存位置
                    img_path = dataset_path + "mul" + str(mul_value) + "_user_number_" + str(
                        user_number) + "/"
                    # img_path_train = img_path + "train/"
                    # img_path_test = img_path + "test/"
                    # csv_train = img_path + "cifar10_train.csv"
                    # csv_test = img_path + "cifar10_test.csv"

                    if not os.path.exists(img_path): os.makedirs(img_path)
                    # if not os.path.exists(img_path_train): os.makedirs(img_path_train)
                    # if not os.path.exists(img_path_test): os.makedirs(img_path_test)

                    for i in range(user_number):
                        if not os.path.exists(img_path+"user"+str(i+1)): os.makedirs(img_path+"user"+str(i+1))
                        if not os.path.exists(img_path + "user" + str(i + 1)+"/train"): os.makedirs(img_path + "user" + str(i + 1)+"/train")
                        if not os.path.exists(img_path + "user" + str(i + 1) + "/test"): os.makedirs(
                            img_path + "user" + str(i + 1) + "/test")

                user_matrix_key = []
                for i in range(user_number):
                    random_matrix = np.random.randint(1, mul_value + 1, (32, 32, 3))
                    user_matrix_key.append(random_matrix)
                user_matrix_key.append(random_matrix)

                csv_file_train = csv.reader(open(csv_train_origin, 'r'))
                each_user_data_number = int(50000 / user_number)

                number_counter = 0
                content_train = []  # 用来存储整个文件的数据，存成一个列表，列表的每一个元素又是一个列表，表示的是文件的某一行
                START_TIME = time.time()
                for line in csv_file_train:
                    # 将图片以灰度图的形式读取，生成ndarray，第二个参数表示灰度图形式
                    img_data = cv2.imread(img_train_origin + line[0], 1)  # 32 *32 *3

                    key_number=int(number_counter / each_user_data_number)
                    if key_number>=user_number:
                        key_number=user_number-1
                    print("train:", line[0])
                    label = line[1]
                    img_data_int64 = img_data.astype("int64")
                    new_img_data = img_data_int64 * user_matrix_key[key_number]
                    # new_img_data_sort = np.zeros([32, 32, 3], dtype='float32')
                    # for i_rgb in range(3):  #3通道
                    #
                    #     list_temp=[]
                    #     for i_1 in range(32):
                    #         for i_2 in range(32):
                    #             list_temp.append(img_data_int64[i_1,i_2,i_rgb])
                    #     #去重
                    #     list_temp=list(set(list_temp))
                    #     list_numpy=np.array(list_temp)
                    #     sorted_index=np.argsort(list_numpy)
                    #     key_value={}
                    #     for kkk in range(sorted_index.size):
                    #             key_value[list_numpy[sorted_index[kkk]]] = kkk
                    #     for i_1 in range(32):
                    #          for i_2 in range(32):
                    #             new_img_data_sort[i_1,i_2,i_rgb]=key_value[img_data_int64[i_1,i_2,i_rgb]]

                    a = new_img_data.transpose(2, 0, 1)
                    number_counter = number_counter + 1
                    if keep == 1:
                        # cv2.imwrite(img_path_train + line[0], new_img_data_sort) #keep in png format
                        # 3_32_32 格式的 numpy文件
                        np.save(img_path + "user" + str(key_number + 1)+"/train/" + line[0].split('.')[0], a)
                        with open(img_path + "user" + str(key_number + 1)+"/train.csv", 'a+', newline='') as csvfile:
                            fieldnames = ['filename', 'label']
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writerow({'filename': str(line[0].split('.')[0]) + '.npy', 'label': int(line[1])})

                    # if png == 1:
                    #     new_img_data_png = np.zeros([32, 32, 3], dtype='uint8')
                    #     for i_rgb in range(3):
                    #         max_value = np.amax(new_img_data[:, :, i_rgb])
                    #         min_value = np.amin(new_img_data[:, :, i_rgb])
                    #         for i_row in range(32):
                    #             for i_col in range(32):
                    #                 new_img_data_png[i_row, i_col, i_rgb] = int(
                    #                     255 * ((new_img_data[i_row, i_col, i_rgb] - min_value) / (max_value - min_value)))
                    #
                    #         cv2.imwrite(img_path_train_png + line[0], new_img_data_png)  # keep in png format
                    #         with open(csv_train_png, 'a+', newline='') as csvfile:
                    #             fieldnames = ['filename', 'label']
                    #             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    #             writer.writerow({'filename': str(line[0].split('.')[0]) + '.png', 'label': int(line[1])})

                # 生成测试集排序图像
                csv_file_test = csv.reader(open(csv_test_origin, 'r'))
                each_user_data_number = int(10000 / user_number)

                number_counter = 0
                for line in csv_file_test:
                    img_data = cv2.imread(img_test_origin + line[0], 1)
                    label = line[1]
                    print("test:", line[0])

                    key_number=int(number_counter / each_user_data_number)
                    if key_number>=user_number:
                        key_number=user_number-1
                    img_data_int64 = img_data.astype("int64")
                    new_img_data = img_data_int64 * user_matrix_key[key_number]

                    # new_img_data_sort = np.zeros([32, 32, 3], dtype='float32')
                    #
                    # for i_rgb in range(3):  #3通道
                    #     list_temp=[]
                    #     for i_1 in range(32):
                    #         for i_2 in range(32):
                    #             list_temp.append(img_data_int64[i_1,i_2,i_rgb])
                    #     #去重
                    #     list_temp=list(set(list_temp))
                    #     list_numpy=np.array(list_temp)
                    #     sorted_index=np.argsort(list_numpy)
                    #     key_value={}
                    #     for kkk in range(sorted_index.size):
                    #             key_value[list_numpy[sorted_index[kkk]]] = kkk
                    #     for i_1 in range(32):
                    #          for i_2 in range(32):
                    #             new_img_data_sort[i_1,i_2,i_rgb]=key_value[img_data_int64[i_1,i_2,i_rgb]]
                    number_counter = number_counter + 1

                    a = new_img_data.transpose(2, 0, 1)
                    if keep == 1:
                        # cv2.imwrite(img_path_test + line[0], new_img_data_int)
                        # np.save(img_test_origin + line[0], new_img_data)
                        np.save(img_path + "user" + str(key_number + 1) + "/test/" + line[0].split('.')[0], a)
                        with open(img_path + "user" + str(key_number + 1) + "/test.csv", 'a+', newline='') as csvfile:
                            fieldnames = ['filename', 'label']
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writerow({'filename': str(line[0].split('.')[0]) + '.npy', 'label': int(line[1])})
                    if png == 1:
                        new_img_data_png = np.zeros([32, 32, 3], dtype='uint8')
                        for i_rgb in range(3):
                            max_value = np.amax(new_img_data[:, :, i_rgb])
                            min_value = np.amin(new_img_data[:, :, i_rgb])
                            for i_row in range(32):
                                for i_col in range(32):
                                    new_img_data_png[i_row, i_col, i_rgb] = int(
                                        255 * ((new_img_data[i_row, i_col, i_rgb] - min_value) / (max_value - min_value)))

                            cv2.imwrite(img_path_test_png + line[0], new_img_data_png)  # keep in png format
                            with open(csv_test_png, 'a+', newline='') as csvfile:
                                fieldnames = ['filename', 'label']
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                writer.writerow({'filename': str(line[0].split('.')[0]) + '.png', 'label': int(line[1])})

                END_TIME = time.time()
                print(END_TIME - START_TIME)



