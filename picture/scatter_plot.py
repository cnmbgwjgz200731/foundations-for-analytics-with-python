#!/usr/bin/env python3  # 声明该脚本使用的Python版本，并且确保在任何系统上都能运行。

print("第6章 图与图表")
print('\t6.1 matplotlib')
print('\t\t6.1.4 散点图')
print()

"""
update20231121
创建一个散点图
"""
import numpy as np  # 导入numpy库，并使用np作为别名。numpy是Python中常用的科学计算库。
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，并使用plt作为别名。matplotlib是用于绘制图表和图像的库。

plt.style.use('ggplot')  # 设置matplotlib的样式为'ggplot'，这是一种模仿R语言ggplot2的绘图风格。

x = np.arange(start=1., stop=15., step=1.)  # 使用numpy的arange函数生成一个从1.到15.（包含15）的等差数列。

y_linear = x + 5. * np.random.randn(14)  # 为线性数据生成随机噪声。这里的随机数是从标准正态分布（平均值为0，标准差为1）中抽取的。
y_quadratic = x ** 2 + 10. * np.random.randn(14)  # 为二次数据生成随机噪声。这里的随机数同样是从标准正态分布中抽取的。

# 使用numpy的polyfit函数来找到y_linear的最佳拟合多项式的系数，然后使用poly1d函数将这个系数转换为一个多项式函数，方便后续使用。
fn_linear = np.poly1d(np.polyfit(x, y_linear, deg=1))
# 对y_quadratic进行同样的操作，但这里使用的是二次多项式。
fn_quadratic = np.poly1d(np.polyfit(x, y_quadratic, deg=2))

fig = plt.figure()  # 创建一个新的图形。
ax1 = fig.add_subplot(1, 1, 1)  # 在图形中添加一个子图。这个子图是图形的第一个（1, 1, 1）。

# 在子图中绘制数据点和拟合线。'bo'表示蓝点（线性数据），'go'表示绿点（二次数据），'b-'和'g-'分别表示蓝线和绿线（对应于最佳拟合线）。
# linewidth=2.可以设置线的宽度
ax1.plot(x, y_linear, 'bo', x, y_quadratic, 'go', x, fn_linear(x), 'b-',\
         x, fn_quadratic(x), 'g-',linewidth=2.)

ax1.xaxis.set_ticks_position('bottom')  # 设置x轴的标签位于底部。
ax1.yaxis.set_ticks_position('left')  # 设置y轴的标签位于左侧。

ax1.set_title('Scatter Plots with Best Fit Lines')  # 设置图像的标题。
plt.xlabel('x')  # 设置x轴的标签。
plt.ylabel('f(x)')  # 设置y轴的标签，这里的'f(x)'表示函数f(x)的值。
plt.xlim((min(x) - 1., max(x) + 1.))  # 设置x轴的范围，从x的最小值减1到x的最大值加1。
plt.ylim((min(y_quadratic) - 10., max(y_quadratic) + 10.))  # 设置y轴的范围，从y_quadratic的最小值减10到y_quadratic的最大值加10。

# plt.savefig('scatter_plot.png', dpi=400, bbox_inches='tight')  # 原本这行代码是用来保存图像的，但现在注释掉了，所以图像不会被保存。如果你想保存图像，可以取消注释这行代码，并将图像保存为scatter_plot.png。其中dpi=400表示图像的分辨率是400，bbox_inches='tight'表示在保存图像时会尝试去掉多余的空白边距。
plt.show()  # 显示图像。这行代码会触发之前所有设置的效果，例如绘制数据点和拟合线，设置标题、标签和范围等。