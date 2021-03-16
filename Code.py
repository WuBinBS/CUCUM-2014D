import xlrd
import numpy as np

# 读入附件一：药盒型号中的数据
data1 = xlrd.open_workbook('C:/Users/Administrator/Desktop/备选题目/2014D-可考虑/附件1-药盒型号.xls')
table = data1.sheet_by_name('ID_SIZE')

# 提取1920个药盒的长宽高数据
box_lwh = np.zeros([0, 3])
for i in range(1, table._dimnrows):
    box_lwh = np.append(box_lwh,
                        np.array([[table._cell_values[i][1], table._cell_values[i][3], table._cell_values[i][2]]]),
                        axis=0)

# 删除已经读入的excel表格
del data1, table

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

######################################################
# 第1小问
# 作1919个药盒长宽高的三维分布图像和三个切面的二维分布图像：xy平面，xz平面,yz平面
fig1 = plt.figure('length-width-height')
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(box_lwh[:, 0], box_lwh[:, 1], box_lwh[:, 2], c='c', marker='.')
plt.show()
fig2 = plt.figure('length-width')
ax2 = fig2.add_subplot(111)
ax2.scatter(box_lwh[:, 0], box_lwh[:, 1], c='r', marker='.')
plt.show()
fig3 = plt.figure('length-height')
ax3 = fig3.add_subplot(111)
ax3.scatter(box_lwh[:, 0], box_lwh[:, 2], c='b', marker='.')
plt.show()
fig4 = plt.figure('width-height')
ax4 = fig4.add_subplot(111)
ax4.scatter(box_lwh[:, 1], box_lwh[:, 2], c='g', marker='.')
plt.show()

# 各作出长宽高的频率分布直方图和概率密度曲线
import seaborn as sns

fig5 = plt.figure('length')
sns.set_style('darkgrid')
sns.distplot(box_lwh[:, 0], color='#4B4B4B')
plt.show()
fig6 = plt.figure('width')
sns.set_style('darkgrid')
sns.distplot(box_lwh[:, 1], color='#4B4B4B')
plt.show()
fig7 = plt.figure('height')
sns.set_style('darkgrid')
sns.distplot(box_lwh[:, 2], color='#4B4B4B')
plt.show()

# 查找规格不满足"长>=高>=宽"的药盒编号和对应的规格数据
for i in range(1919):
    t1, t2, t3 = box_lwh[i, 0], box_lwh[i, 1], box_lwh[i, 2]
    if not (t1 >= t3 >= t2):
        print('药盒编号是:', i + 1, '，长宽高数据分别为(mm)：', t1, t2, t3)

# 演示可行侧翻极限小值的取值范围
fig8 = plt.figure('theta2_doable')
x = np.linspace(0, np.pi / 2, 200)
plt.plot(x, 33 * np.sin(x) + 75 * np.cos(x), linewidth=2)
plt.show()
print('最高点:', np.arctan(33 / 75))

# 确定每个药盒侧翻角度极限值的取值范围
theta1 = np.zeros([1919, 2])
theta1[:, 1] = np.pi / 2
for i in range(1919):
    W, H = box_lwh[i, 1], box_lwh[i, 2]
    theta1[i, 0] = np.arctan(W / H)
    while 1:
        t = W * np.sin(theta1[i, 1]) + H * np.cos(theta1[i, 1])
        if t >= W + 4:
            break
        else:
            theta1[i, 1] -= 0.01

# 确定每个药盒水平翻转角度极限值的取值范围
theta2 = np.zeros([1919, 2])
theta2[:, 1] = np.pi / 2
for i in range(1919):
    L, W = box_lwh[i, 0], box_lwh[i, 1]
    theta2[i, 0] = max(np.arctan(W / L), np.arccos(W / L))
    while 1:
        t = W * np.sin(theta2[i, 1]) + L * np.cos(theta2[i, 1])
        if t >= W + 4:
            break
        else:
            theta2[i, 1] -= 0.01

# 画出最终侧翻极限值和水平旋转极限值可行的取值范围图
fig9 = plt.figure('theta1')
for i in range(1919):
    plt.plot(theta1[i, :], [i + 1, i + 1])
plt.show()
fig10 = plt.figure('theta2')
for i in range(1919):
    plt.plot(theta2[i, :], [i + 1, i + 1])
plt.show()

# 画出各种药盒规格的水平翻转角度极限值取值范围的重合部分
theta1_c, t = [], 0
fig11 = plt.figure('theta1_chonghe')
while t <= np.pi / 2:
    for i in range(1919):
        if not (theta1[i, 0] <= t <= theta1[i, 1]):
            break
    else:
        theta1_c.append(t)
        plt.plot([t, t], [0, 2000])
    t += 0.0001
plt.xlim(0, np.pi / 2)
plt.show()

# 画出各种药盒规格的侧翻角度极限值取值范围的重合部分
theta2_c, t = [], 0
fig12 = plt.figure('theta2_chonghe')
while t <= np.pi / 2:
    for i in range(1919):
        if not (theta2[i, 0] <= t <= theta2[i, 1]):
            break
    else:
        theta2_c.append(t)
        plt.plot([t, t], [0, 2000])
    t += 0.0001
plt.xlim(0, np.pi / 2)
plt.ylim(0, 2000)
plt.show()

# 确定最终的每个药盒的侧翻，水平翻转极限值
theta1_f = 7 * np.pi / 18
theta2_f = np.zeros([1, 1919])[0]
for i in range(1919):
    theta2_f[i] = theta2[i, 0] if theta2[i, 0] > 7 * np.pi / 18 else 7 * np.pi / 18

# 确定竖向隔板类型和种类数-直接策略一：从宽度最小的数据开始，按顺序创建竖向隔板间距和归类
index = np.argsort(box_lwh[:, 1]).astype('int')  # 得到排序后的索引
sort_box_lwh = box_lwh.copy()
sort_box_lwh = sort_box_lwh[index]  # 将长宽高数据按宽度从小到大排列
category = []  # 初始化分类结果
fig13 = plt.figure('strategy_1')
for k in range(1919):
    L, W, H = sort_box_lwh[k, 0], sort_box_lwh[k, 1], sort_box_lwh[k, 2]
    sub = W + 4  # 计算下界
    up = min(W * np.sin(theta1_f) + H * theta1_f,
             W * np.sin(theta2_f[index[k]]) + L * np.cos(theta2_f[index[k]]),
             2 * W)  # 计算上界
    if len(category) == 0:
        category.append([sub, up, [index[k]]])
    else:
        for item in category:
            if not (sub > item[1] or up < item[0]):
                item[0] = max(item[0], sub)
                item[1] = min(item[1], up)
                item[2].append(index[k])
                break
        else:
            category.append([sub, up, [index[k]]])
    # 画出此次迭代后目前得到的所有间距类型
    for item in category:
        plt.plot(item[0:2], [k + 1, k + 1])
plt.show()

for item in category:
    print(item[0:2], len(item[2]))

# 确定最终的每个药盒的侧翻，水平翻转极限值
theta1_f = 4 * np.pi / 9
theta2_f = np.zeros([1, 1919])[0]
for i in range(1919):
    theta2_f[i] = theta2[i, 0] if theta2[i, 0] > 4 * np.pi / 9 else 4 * np.pi / 9

# 确定竖向隔板类型和种类数-直接策略二：从宽度最大的数据开始，按顺序创建竖向隔板间距和归类
index = np.argsort(box_lwh[:, 1]).astype('int')
index = list(reversed(index))  # 得到排序后的索引
sort_box_lwh = box_lwh.copy()
sort_box_lwh = sort_box_lwh[index]  # 将长宽高数据按宽度从大到小排列
category = []  # 初始化分类结果
fig14 = plt.figure('strategy_2')
for k in range(1919):
    L, W, H = sort_box_lwh[k, 0], sort_box_lwh[k, 1], sort_box_lwh[k, 2]
    sub = W + 4  # 计算下界
    up = min(W * np.sin(theta1_f) + H * theta1_f,
             W * np.sin(theta2_f[index[k]]) + L * np.cos(theta2_f[index[k]]),
             2 * W)  # 计算上界
    if len(category) == 0:
        category.append([sub, up, [index[k]]])
    else:
        for item in category:
            if not (sub > item[1] or up < item[0]):
                item[0] = max(item[0], sub)
                item[1] = min(item[1], up)
                item[2].append(index[k])
                break
        else:
            category.append([sub, up, [index[k]]])
    # 画出此次迭代后目前得到的所有间距类型
    for item in category:
        plt.plot(item[0:2], [k + 1, k + 1])
plt.show()

for item in category:
    print(item[0:2], len(item[2]))

# 确定最终的每个药盒的侧翻，水平翻转极限值
theta1_f = np.pi / 3
theta2_f = np.zeros([1, 1919])[0]
for i in range(1919):
    theta2_f[i] = theta2[i, 0] if theta2[i, 0] > np.pi / 3 else np.pi / 3

import random

# 确定竖向隔板类型和种类数-直接策略三：随机取药盒并采用均摊的方式
index = random.sample(range(1919), 1919)  # 得到随机抽取的索引
sort_box_lwh = box_lwh.copy()
sort_box_lwh = sort_box_lwh[index]
category = []  # 初始化分类结果
fig15 = plt.figure('strategy_3')
for k in range(1919):
    L, W, H = sort_box_lwh[k, 0], sort_box_lwh[k, 1], sort_box_lwh[k, 2]
    sub = W + 4  # 计算下界
    up = min(W * np.sin(theta1_f) + H * theta1_f,
             W * np.sin(theta2_f[index[k]]) + L * np.cos(theta2_f[index[k]]),
             2 * W)  # 计算上界
    if len(category) == 0:
        category.append([sub, up, [index[k]]])
    else:
        mark = []  # 记录在已有的间距类型中与当前样品需求有重合部分的间距索引和长度
        for k2 in range(len(category)):
            item = category[k2]
            if not (sub > item[1] or up < item[0]):
                mark.append([k2, item[1] - item[0]])
        if len(mark) != 0:
            mark = sorted(mark, key=lambda t: t[1], reverse=True)
            index2 = mark[0][0]
            category[index2][0] = max(sub, category[index2][0])
            category[index2][1] = min(up, category[index2][1])
            category[index2][2].append(index[k])
        else:
            category.append([sub, up, [index[k]]])
    # 画出此次迭代后目前得到的所有间距类型
    for item in category:
        plt.plot(item[0:2], [k + 1, k + 1])
plt.show()

for item in category:
    print(item[0:2], len(item[2]))

# 画出策略1,2得到的各类隔板间距容纳的药盒数（频率分布直方图）
fig16 = plt.figure('fig16')
plt.bar(range(1, 5), [123, 1078, 577, 141], color='c')
plt.bar(range(6, 11), [123, 1078, 449, 230, 39], color='g')
plt.bar(range(12, 20), [123, 769, 430, 182, 70, 150, 169, 26], color='r')
plt.show()
fig17 = plt.figure('fig17')
plt.bar(range(1, 5), [20, 936, 411, 552], color='c')
plt.bar(range(6, 11), [9, 371, 602, 437, 500], color='g')
plt.bar(range(12, 20), [12, 43, 415, 513, 238, 399, 253, 146], color='r')
plt.show()

# 策略3在各种要求下各算1000次统计各个种类数出现的次数
record = []
for kk in range(1000):
    index = random.sample(range(1919), 1919)  # 得到随机抽取的索引
    sort_box_lwh = box_lwh.copy()
    sort_box_lwh = sort_box_lwh[index]
    category = []  # 初始化分类结果
    for k in range(1919):
        L, W, H = sort_box_lwh[k, 0], sort_box_lwh[k, 1], sort_box_lwh[k, 2]
        sub = W + 4  # 计算下界
        up = min(W * np.sin(theta1_f) + H * theta1_f,
                 W * np.sin(theta2_f[index[k]]) + L * np.cos(theta2_f[index[k]]),
                 2 * W)  # 计算上界
        if len(category) == 0:
            category.append([sub, up, [index[k]]])
        else:
            mark = []  # 记录在已有的间距类型中与当前样品需求有重合部分的间距索引和长度
            for k2 in range(len(category)):
                item = category[k2]
                if not (sub > item[1] or up < item[0]):
                    mark.append([k2, item[1] - item[0]])
            if len(mark) != 0:
                mark = sorted(mark, key=lambda t: t[1], reverse=True)
                index2 = mark[0][0]
                category[index2][0] = max(sub, category[index2][0])
                category[index2][1] = min(up, category[index2][1])
                category[index2][2].append(index[k])
            else:
                category.append([sub, up, [index[k]]])
    record.append(len(category))
    print('已完成第', kk, '次迭代')

# 求数学期望和画频率分布直方图
fig18 = plt.figure('fig18')
sns.set_style('darkgrid')
sns.distplot(record)
plt.show()
print(sum(record) / 1000)

#####################################################
# 确定最终的每个药盒的侧翻，水平翻转极限值
theta1_f = np.pi / 3
theta2_f = np.zeros([1, 1919])[0]
for i in range(1919):
    theta2_f[i] = theta2[i, 0] if theta2[i, 0] > np.pi / 3 else np.pi / 3

# 生成每个药盒在某一个theta1,theta2下的间距范围
fanwei = []
for k in range(1919):
    L, W, H = box_lwh[k, 0], box_lwh[k, 1], box_lwh[k, 2]
    sub = W + 4  # 计算下界
    up = min(W * np.sin(theta1_f) + H * theta1_f,
             W * np.sin(theta2_f[k]) + L * np.cos(theta2_f[k]),
             2 * W)  # 计算上界
    fanwei.append([sub, up])


# 用模拟退火算法解决第2小问
# 定义宽度冗余目标函数（这个函数作废不使用）
def aimfun(x):
    f = 0
    x2 = sorted(x)
    for k in range(1919):
        for item in x2:
            if item >= box_lwh[k, 1]:
                t = item - box_lwh[k, 1]
                if t > 4:
                    f += (t - 4)
                break
    return f


# 定义宽度冗余目标函数
def aimfun2(x):
    f = 0
    x2 = sorted(x)
    for k in range(1919):
        for item in x2:
            if item >= fanwei[k][0]:
                if item <= fanwei[k][1]:
                    t = item - box_lwh[k, 1]
                    if t > 4:
                        f += (t - 4)
                    break
                else:
                    f = 1e5
                    return f
    return f


# 执行模拟退火算法
fig19 = plt.figure('fig19')
f0_com, x0_com = 1e50, []  # 初始化运行模拟退火100次得到的最小总宽度冗余和对应的间距罗列
for k_iter in range(100):
    nc = 15  # 提前确定隔板间距种类数
    opt_minmax = -1
    sub = np.array([14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 60])
    up = np.array([60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60])
    delt = (up - sub) / 5
    yita = 0.99
    trace, x0_trace = [], []
    k_total = 1000
    T = 1e5
    x0 = np.linspace(14, 60, nc)
    f0 = aimfun2(x0)
    for k in range(k_total):

        # 此处编程将一个个分量进行邻域移动，整体移动出现的错误尚不明晰
        x1 = x0.copy()
        for i in range(nc):
            x1[i] = x0[i] + (2 * np.random.rand(1) - 1) * delt[i]
            x1[i] = min(x1[i], up[i])
            x1[i] = max(x1[i], sub[i])

        f1 = aimfun2(x1)
        if opt_minmax * f1 > opt_minmax * f0:
            x0, f0 = x1, f1
        elif np.random.rand(1) < np.exp(opt_minmax * (f1 - f0) / T):
            x0, f0 = x1, f1
        T = yita * T
        trace.append(f0)
        x0_trace.append(x0)
        if f0 < f0_com:
            f0_com, x0_com = f0, x0.copy()
        print('完成第', k + 1, '次迭代')
    if f0_com == min(trace):
        plt.clf()
        plt.plot(range(k_total), trace)
plt.show()

# 做出不同种类下总宽度冗余的折线图
fig20 = plt.figure('fig20')
plt.plot(np.linspace(1, 10, 10), [1e5, 1e5, 1e5, 12485.269345639841, 8176.02216273206, 6163.556801894988,
                                  5575.616371227198, 4794.8382319492475, 4221.517670889129, 3721.873022307072], c='r')
plt.ylim([3500, 13000])
plt.show()

fig21 = plt.figure('fig21')
plt.plot(np.linspace(1, 10, 10), [1e5, 1e5, 1e5, 1e5, 8707.170306652779, 6567.811168910131,
                                  5249.718312773785, 4627.979112988672, 4172.612109014825, 3736.0709518248223], c='r')
plt.ylim([3500, 9000])
plt.show()

fig22 = plt.figure('fig22')
plt.plot(np.linspace(6, 15, 10), [1e5, 1e5, 1e5, 1e5, 4455.426144318016, 3570.102940282154,
                                  3225.520156841734, 3159.8318806121783, 3072.7971403179376, 2767.177861907718], c='r')
plt.ylim([2500, 5000])
plt.show()

############################################
# 用模拟退火算法解决第3小问
sx = [19.20, 25.11, 31.82, 39.24, 49.95, 60.00]


# 定义总平面冗余函数
def aimfun3(x):
    f = 0
    x3 = sorted(x)
    for k in range(1919):
        for item in sx:
            if item >= box_lwh[k, 1] + 4:
                sx_c = item
                break

        for item in x3:
            if item >= box_lwh[k, 2] + 2:
                hx_c = item
                break

        f += (sx_c - box_lwh[k, 1] - 4) * (hx_c - box_lwh[k, 2] - 2)
    return f


# 执行模拟退火算法
fig23 = plt.figure('fig23')
f0_com, x0_com = 1e50, []  # 初始化运行模拟退火100次得到的最小总宽度冗余和对应的间距罗列
for k_iter in range(50):
    nc = 7  # 提前确定隔板间距种类数
    opt_minmax = -1
    sub = np.array([30, 30, 30, 30, 30, 30, 127])
    up = np.array([127, 127, 127, 127, 127, 127, 127])
    delt = (up - sub) / 5
    yita = 0.99
    trace, x0_trace = [], []
    k_total = 1000
    T = 50 * 100 * 2000
    x0 = np.linspace(30, 127, nc)
    f0 = aimfun3(x0)
    for k in range(k_total):

        # 此处编程将一个个分量进行邻域移动，整体移动出现的错误尚不明晰
        x1 = x0.copy()
        for i in range(nc):
            x1[i] = x0[i] + (2 * np.random.rand(1) - 1) * delt[i]
            x1[i] = min(x1[i], up[i])
            x1[i] = max(x1[i], sub[i])

        f1 = aimfun3(x1)
        if opt_minmax * f1 > opt_minmax * f0:
            x0, f0 = x1, f1
        elif np.random.rand(1) < np.exp(opt_minmax * (f1 - f0) / T):
            x0, f0 = x1, f1
        T = yita * T
        trace.append(f0)
        x0_trace.append(x0)
        if f0 < f0_com:
            f0_com, x0_com = f0, x0.copy()
        print('完成第', k + 1, '次迭代')
    if f0_com == min(trace):
        plt.clf()
        plt.plot(range(k_total), trace)
plt.show()

# 做出不同种类下总平面冗余的折线图
fig24 = plt.figure('fig24')
plt.plot([4, 5, 6, 7, 8, 9, 11],
         [56171.23829260804, 47382.85927597274, 38327.224203954946, 32382.596384022272, 28674.01469584825, 23870.53211588389,
          21739.036606897727], c='r')
plt.ylim([20000, 60000])
plt.show()