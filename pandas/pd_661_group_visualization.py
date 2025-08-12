import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
# import jinja2

from io import StringIO
from io import BytesIO


input_file = 'E:/bat/input_files/sales_2013.xlsx'
lx_test = 'E:/bat/input_files/split_mpos_less.xlsx'
xls_file = 'E:/bat/input_files/sales_2013.xls'
m_file = 'E:/bat/input_files/winequality-red.csv'
big_file = 'E:/bat/input_files/dq_split_file.xlsx'
team_file = 'E:/bat/input_files/team.xlsx'
csv_file = 'E:/bat/input_files/sales_january_2014.csv'

path = 'E:/bat/output_files/pandas_read_csv_20240118.csv'

# TODO 忽略特定的警告 | 当遇到 openpyxl 中的 UserWarning 类型的警告时，它们将不会被打印出来。
warnings.filterwarnings('ignore',category=UserWarning,module='openpyxl')


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.6 分组可视化')
print('\t6.6.1 绘图方法plot()')
print()
# update20240429
'''
Pandas为我们提供了几个简单的、与分组相关的可视化方法，
这些可视化方法能够提高我们对聚合数据的视觉直观认识，本节就来做一些介绍。

'''
# 小节注释
'''
在1.3节中我们介绍过plot()方法，它能够为我们绘制出我们想要的常见图形。
数据分组对象也支持plot()，不过它以分组对象中每个DataFrame或Series为对象，绘制出所有分组的图形。
默认情况下，它绘制的是折线图，示例代码如下。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 分组，设置索引为name')
grouped = df.set_index('name').groupby('team')
# 绘制图形
# 折线图
grouped.plot(kind='bar') # 默认 kind='line'
plt.show()
# print(plt.show())
'''
还可以通过plot.x()或者plot(kind='x')的形式调用其他形状的图形，
比如：
plot.line：折线图
plot.pie：饼图
plot.bar：柱状图
plot.hist：直方图
plot.box：箱形图
plot.area：面积图
plot.scatter：散点图
plot.hexbin：六边形分箱图
plot()可传入丰富的参数来控制图形的样式，可参阅第16章。

'''


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.6 分组可视化')
print('\t6.6.2 直方图hist()')
print()
# update20240429

# 小节注释
'''
分组对象的hist()可以绘制出每个分组的直方图矩阵，每个矩阵为一个分组：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 分组，设置索引为name')
grouped = df.set_index('name').groupby('team')
# 绘制图形
grouped.hist()
plt.show()
# print()


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.6 分组可视化')
print('\t6.6.3 箱线图boxplot()')
print()
# update20240430
'''
Pandas为我们提供了几个简单的、与分组相关的可视化方法，
这些可视化方法能够提高我们对聚合数据的视觉直观认识，本节就来做一些介绍。

'''
# 小节注释
'''
分组的boxplot()方法绘制出每个组的箱线图。
箱线图展示了各个字段的最大值、最小值、分位数等信息，为我们展示了数据的大体形象，代码如下。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 分组箱线图')
grouped = df.set_index('name').groupby('team')
# 绘制图形
# grouped.boxplot()
# grouped.boxplot(figsize=(15,12)) # 比默认的图片要大几倍
# plt.show()

print()
print('# Dataframe 分组箱线图')
df.boxplot(by='team',figsize=(15,10))
plt.show()
'''以上代码会按team分组并返回箱线图'''

'''
6.6.4 小结
本节介绍了关于分组的可视化方法，它们会将一个分组对象中的各组数据进行分别展示，
便于我们比较，从不同角度发现数据的变化规律，从而得出分析结果。

这些操作都是在分拆应用之后进行的，合并后数据的可视化并没有什么特殊的，
第16章将对数据可视化进行统一讲解

'''