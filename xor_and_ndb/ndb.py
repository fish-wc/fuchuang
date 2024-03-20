import math
import random
import time
import os
import re
# Constants equivalent to the #define in C++
N = 600000
M = 20000
RN = 500
RM = 20
L = 8

# The equivalent of the C++ struct Ent
class Ent:
    def __init__(self):
        self.p = [0, 0, 0]  # 三个位
        self.c = ['', '', '']

# Global variables
NDb = [['' for _ in range(8192)] for _ in range(M)]
s = ['' for _ in range(M)]
m = 0
cn = 0
r = 0.0
len = 0
p = [0.0 for _ in range(4)]
q = [0.0 for _ in range(8)]
NDB = [Ent() for _ in range(N)]
posbility = [[0.0, 0.0] for _ in range(8)]

xq = [[0.0 for _ in range(1024)] for _ in range(30000)]
N0 = [0 for _ in range(8192)]  # 第i个负数据库中第j个位为0的个数
N1 = [0 for _ in range(8192)]
pow_b = [1, 2, 4, 8, 16, 32, 64, 128, 256]
Q = [[0.0 for _ in range(256)] for _ in range(1024)]
q0 = [0.0 for _ in range(10)]
q1 = [0.0 for _ in range(10)]

def sort_key(name):
    # 使用正则表达式从文件名中提取数字
    numbers = re.findall(r'\d+', name)
    return [int(num) for num in numbers]
def diff(i):
    Ndiff = 0.0
    Nsame = 0.0
    for j in range(1, 4):
        Ndiff += j * p[j] * q[i]

    for j in range(1, 4):
        Nsame += ((3 - j) * p[j]) / 8

    Pdiff = Ndiff / (Ndiff + Nsame)
    return Pdiff

def init(cr):
    global m, r, cn, p, q, posbility
    m = 8192  #表示二进制串位数
    r = cr  #用于控制 NDB 的大小
    p[0] = 0  #用于控制 K 种不同类型记录的概率参数
    p[1] = 0.70
    p[2] = 0.24
    p[3] = 1 - p[1] - p[2]
    cn = int(m * r + 0.5)

    q[0] = 0.95
    q[1] = 0.0
    q[2] = 0.0
    q[3] = 0.0
    q[4] = 0.0
    q[5] = 0.0
    q[6] = 0.0
    q[7] = 0.05

    for i in range(8):
        posbility[i][0] = diff(i)
        posbility[i][1] = 1 - posbility[i][0]


def rand1(n=None):
    if n is not None:
        return random.randint(0, n-1)  # 返回 0 到 n-1 之间的随机整数
    else:
        return random.random()  # 返回 0 到 1 之间的随机浮点数

def generateRandomNumbers(l):
    if l < q[0]:
        return 0
    elif l < q[0] + q[1]:
        return 1
    elif l < q[0] + q[1] + q[2]:
        return 2
    elif l < q[0] + q[1] + q[2] + q[3]:
        return 3
    elif l < q[0] + q[1] + q[2] + q[3] + q[4]:
        return 4
    elif l < q[0] + q[1] + q[2] + q[3] + q[4] + q[5]:
        return 5
    elif l < q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6]:
        return 6
    else:
        return 7


def addToNDB(x):
    global cn, NDB
    for i in range(3):
        NDB[cn].p[i] = x.p[i]
        NDB[cn].c[i] = x.c[i]
    cn += 1


def f1(s):
    global cn, NDB, m, len, r

    m = 8192
    len = m // L
    n = int(m * r + 0.5)
    cn = 0

    while cn < n:
        v = Ent()
        t = rand1()  # 生成0到1之间的随机数
        if t < p[1]:  # 生成类型一
            u = rand1()
            v.p[0] = generateRandomNumbers(u) + rand1(len) * L

            v.c[0] = '1' if s[v.p[0]] == '0' else '0'

            # 生成第二个随机位，不同于第一个位
            bit2 = rand1(L)
            attr2 = rand1(len)
            while (bit2 + attr2 * L) == v.p[0]:
                bit2 = rand1(L)
                attr2 = rand1(len)
            v.p[1] = bit2 + attr2 * L

            v.c[1] = s[v.p[1]]

            # 生成第三个随机位，不同于前两个位
            bit3 = rand1(L)
            attr3 = rand1(len)
            while (bit3 + attr3 * L) == v.p[0] or (bit3 + attr3 * L) == v.p[1]:
                bit3 = rand1(L)
                attr3 = rand1(len)
            v.p[2] = bit3 + attr3 * L
            v.c[2] = s[v.p[2]]

        elif t < p[1] + p[2]:  # 生成类型二
            v.p[0] = generateRandomNumbers(rand1()) + rand1(len) * L
            v.c[0] = '1' if s[v.p[0]] == '0' else '0'

            bit2 = generateRandomNumbers(rand1())
            attr2 = rand1(len)
            while (bit2 + attr2 * L) == v.p[0]:
                bit2 = generateRandomNumbers(rand1())
                attr2 = rand1(len)
            v.p[1] = bit2 + attr2 * L
            v.c[1] = '1' if s[v.p[1]] == '0' else '0'

            bit3 = generateRandomNumbers(rand1())
            attr3 = rand1(len)
            while (bit3 + attr3 * L) == v.p[1] or (bit3 + attr3 * L) == v.p[0]:
                bit3 = generateRandomNumbers(rand1())
                attr3 = rand1(len)
            v.p[2] = bit3 + attr3 * L

            v.c[2] = '1' if s[v.p[2]] == '0' else '0'

        else:  # 其他类型
            v.p[0] = generateRandomNumbers(rand1()) + rand1(len) * L
            v.c[0] = '1' if s[v.p[0]] == '0' else '0'

            bit2 = generateRandomNumbers(rand1())
            attr2 = rand1(len)
            while (bit2 + attr2 * L) == v.p[0]:
                bit2 = generateRandomNumbers(rand1())
                attr2 = rand1(len)
            v.p[1] = bit2 + attr2 * L
            v.c[1] = '1' if s[v.p[1]] == '0' else '0'

            bit3 = generateRandomNumbers(rand1())
            attr3 = rand1(len)
            while (bit3 + attr3 * L) == v.p[1] or (bit3 + attr3 * L) == v.p[0]:
                bit3 = generateRandomNumbers(rand1())
                attr3 = rand1(len)
            v.p[2] = bit3 + attr3 * L
            v.c[2] = '1' if s[v.p[2]] == '0' else '0'

        addToNDB(v)


def calQ(index):
    global N0, N1, Q, xq, NDB, cn, len

    maxval = 255

    # 初始化N0和N1
    for id in range(m):
        N0[id] = 0
        N1[id] = 0

    # 更新N0和N1
    for num in range(cn):
        for j in range(3):
            if NDB[num].c[j] == '1':
                N1[NDB[num].p[j]] += 1
            else:
                N0[NDB[num].p[j]] += 1

    # 计算第i个负数据库第j个属性取值为k的概率 Q[M][N][k]
    for j in range(len):
        for k in range(8):
            id = L * j + k
            var1 = N0[id] * math.log(posbility[k][1] + 1e-7) + N1[id] * math.log(posbility[k][0] + 1e-7)
            var2 = N0[id] * math.log(posbility[k][0] + 1e-7) + N1[id] * math.log(posbility[k][1] + 1e-7)

            q0[k] = var1 - math.log(math.exp(var1) + math.exp(var2))
            q1[k] = var2 - math.log(math.exp(var1) + math.exp(var2))

        # 求和与标准化
        sumv = 0
        for a in range(maxval + 1):
            Q[j][a] = 0
            k = a
            for b in range(7, -1, -1):
                if k >= pow_b[b]:
                    Q[j][a] += q1[7 - b]
                    k -= pow_b[b]
                else:
                    Q[j][a] += q0[7 - b]
            sumv += math.exp(Q[j][a])

        for a in range(maxval + 1):
            Q[j][a] = math.exp(Q[j][a]) / sumv

    # 计算重构值
    for j in range(len):
        s2 = 0
        for x in range(maxval + 1):
            s2 += x * Q[j][x]
        xq[index][j] = s2


def printNDB():
    for i in range(cn):
        for j in range(3):
            print(f"{NDB[i].p[j]} {NDB[i].c[j]}", end="  ")
        print()


def NDB_f():
    random.seed(42)

    # 设置转换率和初始化
    change_r = 6.5
    init(change_r)

    # 设置文件路径，这些路径可能需要根据您的文件系统进行调整
    # filein = [
    #     "xored_cifar10_data/xored_image_0.txt",
    #     "xored_cifar10_data/xored_image_1.txt",
    #     "xored_cifar10_data/xored_image_2.txt",
    #     "xored_cifar10_data/xored_image_3.txt",
    #     "xored_cifar10_data/xored_image_4.txt",
    #     "xored_cifar10_data/xored_image_5.txt",
    #     "xored_cifar10_data/xored_image_6.txt",
    #     "xored_cifar10_data/xored_image_7.txt",
    #     "xored_cifar10_data/xored_image_8.txt",
    #     "xored_cifar10_data/xored_image_9.txt"
    #
    #     # ... 其他文件路径
    # ]
    # fileout = [
    #     "ndb_cifar10_data/image_0.txt",
    #     "ndb_cifar10_data/image_1.txt",
    #     "ndb_cifar10_data/image_2.txt",
    #     'ndb_cifar10_data/image_3.txt',
    #     "ndb_cifar10_data/image_4.txt",
    #     "ndb_cifar10_data/image_5.txt",
    #     "ndb_cifar10_data/image_6.txt",
    #     'ndb_cifar10_data/image_7.txt',
    #     "ndb_cifar10_data/image_8.txt",
    #     "ndb_cifar10_data/image_9.txt"
    #     # ... 其他文件路径
    # ]
    filein = [f for f in os.listdir('data/xored_data/train') if os.path.isfile(os.path.join('data/xored_data/train/', f))]
    filein.sort(key=sort_key)
    file_count = sum(1 for _ in filein)

    filein2 = [f for f in os.listdir('data/xored_data/eval') if os.path.isfile(os.path.join('data/xored_data/eval/', f))]
    filein2.sort(key=sort_key)
    file_count2 = sum(1 for _ in filein2)


    fileout = [ f'image_{i}.txt' for i in range(file_count)]#这里

    fileout2 = [f'image_{i}.txt' for i in range(file_count2)]  # 这里



    for each in range(file_count):  # 假设有1000个文件进行处理#这里
        # 读取文件
        with open('data/xored_data/train/'+filein[each], 'r') as file:
            lines = file.readlines()
            count = 0
            for line in lines:
                NDb[count] = line.strip()
                count += 1
        print(f"{count}条数据已读取完成")

        # 处理每条数据
        start_time = time.time()
        for i in range(count):
            f1(NDb[i])  # 生成每条数据的负数据库记录
            calQ(i)    # 计算重构值
            if i % 2500 < count :
                print(f"已处理 {i+1} 条数据")

        end_time = time.time()
        print(f"处理时间：{end_time - start_time}秒")
        print(f"完成第{each+1}次迭代")

        # 写入文件
        directory = "data/ndb_data/train/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory+fileout[each], 'w') as file:
            for i in range(count):
                for j in range(1024):
                    file.write(f"{xq[i][j]} ")
                file.write("\n")

    for each in range(file_count2):  # 假设有1000个文件进行处理#这里
        # 读取文件
        with open('data/xored_data/eval/'+filein2[each], 'r') as file:
            lines = file.readlines()
            count = 0
            for line in lines:
                NDb[count] = line.strip()
                count += 1
        print(f"{count}条数据已读取完成")

        # 处理每条数据
        start_time = time.time()
        for i in range(count):
            f1(NDb[i])  # 生成每条数据的负数据库记录
            calQ(i)    # 计算重构值
            if i % 2500 < count :
                print(f"已处理 {i+1} 条数据")

        end_time = time.time()
        print(f"处理时间：{end_time - start_time}秒")
        print(f"完成第{each+1}次迭代")

        # 写入文件
        directory2 = "data/ndb_data/eval/"
        if not os.path.exists(directory2):
            os.makedirs(directory2)
        with open(directory2+fileout2[each], 'w') as file2:
            for i in range(count):
                for j in range(1024):
                    file2.write(f"{xq[i][j]} ")
                file2.write("\n")

    print("所有数据处理完成")

