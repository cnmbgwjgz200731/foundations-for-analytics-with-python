# 导入Python的NumPy库，这是一个用于处理数组和矩阵的强大的科学计算库
import numpy as np

# 导入matplotlib的pyplot模块，这是一个用于创建图形和图像的库
import matplotlib.pyplot as plt


# update20231113
print("第6章 图与图表")
print('\t6.1 matplotlib')
print('\t\t6.1.2 直方图')
print()

"""
update20231117
创建一个直方图
"""
# 设置matplotlib的样式为'ggplot'，这是一种流行的数据可视化样式
plt.style.use('ggplot')

# 设置直方图的一些参数：平均值，标准差
mu1, mu2, sigma = 100, 130, 25

# 使用上述参数生成两个随机数列，它们分别服从以mu1和mu2为均值，sigma为标准差的正态分布
# 使用随机数生成器创建两个正态分布变量x1和x2 , X1的均值是100，x2的均值是130.
x1 = mu1 + sigma * np.random.randn(10000)
x2 = mu2 + sigma * np.random.randn(10000)

# 创建一个新的图形对象
fig = plt.figure()

# 在该图形对象上添加一个新的子图，占满整个图形的空间
ax1 = fig.add_subplot(1, 1, 1)

# 在子图上创建一个直方图，x1数组中的数据被用于生成直方图的数据点
# bins参数定义了直方图的条形数量，表示每个变量的值应该被分成50份。
# normed参数定义了是否应用归一化，density=False表示直方图显示的是频率分布，而不是概率密度
# color定义了颜色，'darkgreen'表示深绿色
# alpha参数定义了颜色的透明度，0.5表示半透明
""" 
原书中normed 运行报错 修改为density 正常|| normed已弃用！
"""
# n,bins,patches = ax1.hist(x1,bins=50,normed=False,color='darkgreen')
# n,bins,patches = ax1.hist(x2,bins=50,normed=False,color='orange',alpha=0.5)
n, bins, patches = ax1.hist(x1, bins=50, density=False, color='darkgreen')
n, bins, patches = ax1.hist(x2, bins=50, density=False, color='orange', alpha=0.5)

# 设置x轴和y轴的标签位置，分别为底部和左侧
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

# 设置图形的x轴和y轴的标签，分别为'Bins'和'Number of Values in Bin'
plt.xlabel('Bins')
plt.ylabel('Number of Values in Bin')

# 设置图形的标题，为'Histograms'，字体大小为14，字体权重为粗体
fig.suptitle('Histograms', fontsize=14, fontweight='bold')

# 在子图上设置一个标题，为'Two Frequency Distributions'
ax1.set_title('Two Frequency Distributions')

# 将图形保存为名为'histogram.png'的PNG图像文件，分辨率为400dpi，图像边缘保持紧凑，无空白边距
plt.savefig('histogram.png', dpi=400, bbox_inches='tight')

# 显示生成的图形
plt.show()