#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

print("第6章 图与图表")
print('\t6.1 matplotlib')
print('\t\t6.1.5 箱线图')
print()

"""
update20231121 
创建一个箱线图
"""

"""
该代码是一个使用Python进行数据可视化的例子，
它利用numpy库生成两种随机分布（正态分布和对数正态分布），
然后使用matplotlib库对生成的分布进行箱形图展示。以下是详细的逐行注释：

"""

# 设置matplotlib的绘图风格为'ggplot'，这是一种在Python中常用的数据可视化风格
plt.style.use('ggplot')

# 定义一个变量N并赋值为500，表示我们生成的数据点的数量
N = 500

# 使用numpy的random.normal函数生成N个服从以loc=0.0, scale=1.0为参数的正态分布的随机数
normal = np.random.normal(loc=0.0, scale=1.0, size=N)

# 使用numpy的random.lognormal函数生成N个服从以mean=0.0, sigma=1.0为参数的对数正态分布的随机数
lognormal = np.random.lognormal(mean=0.0, sigma=1.0, size=N)

# 这一行代码原本是生成一个包含N个随机整数的数组，其中每个整数都在[0, N-1]范围内。
# 但注释中提到，random_integers函数在某些版本的numpy中已经被弃用，因此这行代码可能不会工作。
# index_value = np.random.random_integers(low=0,high=N-1,size=N) # 书中函数random_integers已弃用
# 这一行是用来替换上述注释中被弃用的函数的。
# 使用numpy的random.randint函数生成一个包含N个随机整数的数组，每个整数都在[0, N-1]范围内。
index_value = np.random.randint(0, N - 1, size=N)

# 使用生成的index_value数组作为索引，从normal和lognormal数组中提取相应数量的随机数。这样做的效果是，每次从normal和lognormal中各取一个数。
normal_sample = normal[index_value]
lognormal_sample = lognormal[index_value]

# 将生成的四个数组组合成一个列表box_plot_data，用于后续的绘图操作
box_plot_data = [normal, normal_sample, lognormal, lognormal_sample]

# 创建一个新的图形对象fig，为后续的绘图操作做准备
fig = plt.figure()

# 在fig上添加一个子图对象ax1，设置布局为1行1列，从上到下第一个位置。
ax1 = fig.add_subplot(1, 1, 1)

# 为四个分布分别设置标签，用于后续的图形展示。标签将显示在箱形图的各个箱子上。
box_labels = ['normal', 'normal_sample', 'lognormal', 'lognormal_sample']

# 使用ax1对象的boxplot函数绘制箱形图。输入参数包括要绘制的数据（box_plot_data）和一些图形属性（如是否显示中位数、箱子的形状等）。
# labels参数设置每个箱子的标签。
'''
这行代码使用boxplot函数创建4个箱线图。notch=False表示箱体是矩形，而不是在中间收缩。
sym='.' 表示离群点（就是直线之外的点）使用圆点，而不是默认的 + 符号。
vert=True 表示箱体是垂直的，不是水平的。whis=1.5 设定了直线从第一四分位数和
第三四分位数延伸出的范围（例如：Q3+whis*IQR，IQR 就是四分位距，等于 Q3-Q1）。
showmeans=True 表示箱体在显示中位数的同时也显示均值。
labels=box_labels表示使用box_labels中的值来标记箱线图
'''
ax1.boxplot(box_plot_data, notch=False, sym='.', vert=True, whis=1.5, showmeans=True, labels=box_labels)

# 设置x轴标签在底部显示，y轴标签在左侧显示。这是通过设置ticks_position属性来实现的。
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

# 设置图形的标题和x轴、y轴的标签。这些信息将显示在图形的顶部和右侧。
ax1.set_title('Box Plots: Resamping of Two Distributions')
ax1.set_xlabel('Distribution')
ax1.set_ylabel('Value')

# 原本这行代码是用来将图形保存为png文件的，但因为此处注释掉了，所以没有实现这个功能。如果需要保存图形，可以取消注释这行代码。
# plt.savefig('box_plot.png',dpi=400,bbox_inches='tight')  # 保存图形为png文件，文件名为'box_plot.png'，分辨率为400，边框紧缩。
plt.show()  # 显示图形。如果之前有保存图形，这行代码将不会再次保存图形。
