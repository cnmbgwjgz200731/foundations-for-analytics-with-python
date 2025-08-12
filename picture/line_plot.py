# 导入Python 3的内置模块，该模块允许使用一种指定脚本的shebang行来运行Python解释器。在这里，指定的是/usr/bin/env python3，意味着这个脚本应该由位于/usr/bin/env的python3解释器来执行。
# ! /usr/bin/env python3

print("第6章 图与图表")
print('\t6.1 matplotlib')
print('\t\t6.1.3 折线图')
print()

"""
update20231120
创建一个折线图
"""
# 导入numpy库中的random模块，该模块提供了生成随机数的功能。randn函数是该模块中的一个函数，用于生成符合标准正态分布的随机数。
from numpy.random import randn

# 导入matplotlib库中的pyplot模块，该模块是用于数据可视化的强大工具。
import matplotlib.pyplot as plt

# 设置matplotlib的样式为'ggplot'，这是一种流行的数据可视化样式。
plt.style.use('ggplot')

# 使用randn函数生成50个符合标准正态分布的随机数，并计算它们的累加值，生成四个随机数列。
plot_data1 = randn(50).cumsum()
plot_data2 = randn(50).cumsum()
plot_data3 = randn(50).cumsum()
plot_data4 = randn(50).cumsum()

# 创建一个新的图形对象。
fig = plt.figure()

# 在该图形对象上添加一个新的子图，占满整个图形的空间。
ax1 = fig.add_subplot(1, 1, 1)

# 在子图上分别绘制四条曲线，每条曲线对应一个随机数列。使用不同的标记符号、颜色、线型以及标签进行区分。
ax1.plot(plot_data1, marker=r'o', color=u'blue', linestyle='-', label='Blue Solid')
ax1.plot(plot_data2, marker=r'+', color=u'red', linestyle='--', label='Red Dashed')
ax1.plot(plot_data3, marker=r'*', color=u'green', linestyle='-.', label='Green Dash Dot')
ax1.plot(plot_data4, marker=r's', color=u'orange', linestyle=':', label='Orange Dotted')

# 设置x轴和y轴的标签位置分别为底部和左侧。
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

# 设置整个图形的标题为“Line Plots:Markers,Colors,and Linestyles”。
ax1.set_title('Line Plots:Markers,Colors,and Linestyles')

# 设置x轴的标签为“Draw”。
plt.xlabel('Draw')

# 设置y轴的标签为“Random Number”。
plt.ylabel('Random Number')

# 显示图例，位置最佳。
plt.legend(loc='best')

# 将图形保存为名为“line_plot.png”的PNG图像文件，分辨率为400dpi，图像边缘保持紧凑，无空白边距。
# plt.savefig('line_plot.png', dpi=400, bbox_inches='tight')

# 显示图形。
plt.show()