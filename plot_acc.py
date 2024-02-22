import matplotlib.pyplot as plt
import random


def acc_loss(acc, loss):
    x = range(len(acc))

    # 中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 创建画布和子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), dpi=80)

    # 绘制准确率图像
    ax1.plot(x, acc, color="r", linestyle='-.', label="准确率")
    ax1.legend()
    ax1.set_xticks(x)
    ax1.set_xticklabels(["epoch{}".format(i) for i in x])
    ax1.grid(linestyle='--', alpha=0.5)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("准确率")

    # 绘制损失值图像
    ax2.plot(x, loss, color="b", label="损失值")
    ax2.legend()
    ax2.set_xticks(x)
    ax2.set_xticklabels(["epoch{}".format(i) for i in x])
    ax2.grid(linestyle='--', alpha=0.5)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("损失值")

    # 调整布局
    plt.tight_layout()

    # 显示图像
    plt.show()