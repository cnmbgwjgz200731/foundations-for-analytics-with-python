#! /usr/bin/env python3
# update20231113
print("第6章 图与图表")
print('\t6.1 matplotlib')
print('\t\t6.1.1 条形图')
print()

"""
update20231115 
创建一个垂直条形图
"""

# 导入 matplotlib 库的 pyplot 模块，并为其指定别名 plt
import matplotlib.pyplot as plt

# 设置 matplotlib 的样式为 ggplot，这会改变图形的视觉效果
# book:使用ggplot样式表来模拟ggplot2风格的图形，ggplot2是一个常用的R语言绘图包
plt.style.use('ggplot')

"""
book:
这3行代码为条形图准备数据。我创建了1个客户索引列表，
因为xticks函数在设置标签时要求索引位置和标签值。
"""
# 定义一个名为 customers 的列表，里面包含五个字符串，代表五个客户的名称
customers = ['ABC', 'DEF', 'GHI', 'JKL', 'MNO']
# 创建一个名为 customers_index 的列表，其元素是 customers 列表的索引，范围从0到4
customers_index = range(len(customers))
# 定义一个名为 sale_amounts 的列表，里面包含五个数字，代表五个客户对应的销售量
sale_amounts = [127, 90, 201, 111, 232]

"""
book:
这2行代码：
使用matplotlib绘图时，首先要创建一个基础图，然后在基础图中创建一个或多个子图。
第1行代码创建了一个基础图，第2行代码向基础图中添加一个子图。因为可以向基础图中添加多个子图,所以必须指定要创建
几行和几列子图，以及使用哪个子图。
1,1,1表示创建一行一列的子图，并使用第1个也是唯一的一个子图。

"""
# 创建一个figure对象，这是创建各种图形的基础
fig = plt.figure()
# 在 figure 中添加一个子图，布局是1行1列，位置是第1个
ax1 = fig.add_subplot(1, 1, 1)

"""
该行代码中align='center' 设置条形与标签中间对齐。color='darkblue' 设置条形的颜色
"""
# 在子图中创建一个条形图，x轴是 customers_index 列表，y轴是 sale_amounts 列表，条形图的位置与x轴对齐，颜色为深蓝色
ax1.bar(customers_index, sale_amounts, align='center', color='darkblue')

"""
这2行代码通过设置刻度线位置在 x 轴底部和 y 轴左侧，使图形的上部和右侧不显示刻度线。
"""
# 设置x轴的刻度标签位于图形的底部
ax1.xaxis.set_ticks_position('bottom')
# 设置y轴的刻度标签位于图形的左侧
ax1.yaxis.set_ticks_position('left')

"""
该行代码将条形的刻度线标签由客户索引值更改为实际的客户名称。
rotation=0 表示刻度标签应该是水平的，而不是倾斜一个角度。
fontsize='small' 将刻度标签的字体设为小字体。
"""
# 设置x轴的刻度标签为 customers 列表中的内容，旋转角度为0（不旋转），字体大小为 small
plt.xticks(customers_index, customers, rotation=0, fontsize='small')

# 设置图形的x轴标签为 "Customer Name"
plt.xlabel('Customer Name')

# 设置图形的y轴标签为 "Sale Amount"
plt.ylabel('Sale Amount')

# 设置图形的标题为 "Sale Amount per Customer"
plt.title('Sale Amount per Customer')

"""
bbox_inches='tight' 表示在保存图形时，将图形四周的空白部分去掉。
"""
# 保存图形为 'bar_plot.png' 文件，分辨率为400，边框紧贴图形（bbox_inches='tight'），无黑边
# plt.savefig('bar_plot.png', dpi=400, bbox_inches='tight')
# plt.savefig(dpi=400, bbox_inches='tight') # 删除参数1 执行会报错！
# 显示图形
plt.show()

