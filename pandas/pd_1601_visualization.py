import datetime

import dateutil.tz
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import glob
import os
import jinja2

import matplotlib

from io import StringIO
from io import BytesIO

input_file = 'E:/bat/input_files/sales_2013.xlsx'
lx_test = 'E:/bat/input_files/split_mpos_less.xlsx'
xls_file = 'E:/bat/input_files/sales_2013.xls'
m_file = 'E:/bat/input_files/winequality-red.csv'
big_file = 'E:/bat/input_files/dq_split_file.xlsx'
team_file = 'E:/bat/input_files/team.xlsx'
csv_file = 'E:/bat/input_files/sales_january_2014.csv'
csv_feb_file = 'E:/bat/input_files/sales_february_2014.csv'
multindex_file = 'E:/bat/input_files/pandas_out_20240509042.xlsx'

path = 'E:/bat/output_files/pandas_read_csv_20240118.csv'
# df1.to_excel('E:/bat/output_files/pandas_out_20240510053.xlsx')

# TODO 忽略特定的警告 | 当遇到 openpyxl 中的 UserWarning 类型的警告时，它们将不会被打印出来。
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# 读取数据
df = pd.read_excel(team_file)

'''
                第六部分
                 可视化
        可视化是数据分析的终点也是起点。
得益于生动呈现的可视化数据效果，我们能够跨越对于数据的认知鸿沟。
本部分主要介绍Pandas的样式功能如何让数据表格更有表现力，
Pandas的绘图功能如何让数据自己说话，如何定义不同类型的数据图形，
以及如何对图形中的线条、颜色、字体、背景等进行细节处理。
'''

print()
print('------------------------------------------------------------')
print('第16章 Pandas可视化')
# update20240708
'''
一图胜千言，人类是视觉敏感的动物，大多数人无法在短时间内找到数据中所蕴含的规律和业务意义，
但可以通过图形快速了解数据的比例、分布、趋势等信息，因此，可视化势在必行。
Pandas的可视化图表（Chart）可以让数据直达我们的大脑，让数据自己说话。
'''

print()
print('------------------------------------------------------------')
print('\t16.1 plot()方法')
'''
Pandas提供的plot()方法可以快速方便地将Series和DataFrame中的数据进行可视化，
它是对matplotlib.axes.Axes.plot的封装。
代码执行后会生成一张可视化图形，并直接显示在Jupyter Notebook上。

除了plot()方法，本节还将介绍一些关于可视化及Python操作可视化
的背景知识。这些内容能帮助我们更好地理解和编写可视化逻辑代码。
'''

print()
print('------------------------------------------------------------')
print('\t16.1.1 plot()概述')
# update20240708
'''
plot默认是指折线图。折线图是最常用和最基础的可视化图形，足以满足我们日常80%的需求。示例如下：
'''

# DataFrame调用
df.plot()
# test_df.show()
print('# 显示图形')
# plt.show('E:/bat/output_files/pandas_out_20240708011_001.png') # error

# 保存图形
plt.savefig('E:/bat/output_files/pandas_out_20240708011_001.png')  # success
plt.savefig('E:/bat/output_files/pandas_out_20240708011_002.jpeg')  # success

# 显示图形
plt.show()  # success

'''
以上默认会生成折线图，x轴为索引，y轴为数据。
对于DataFrame，会将所有的数字列以多条折线的形式显示在图形中。

我们可以在plot后增加调用来使用其他图形（这些图形对数据结构有不同的要求，本章后面会逐个介绍）：
'''

df.plot.bar()  # 柱状图
plt.savefig('E:/bat/output_files/pandas_out_20240708011_003.jpeg')
plt.show()

print('其它函数')
# df.plot.line()  # 折线图
# df.plot.bar()  # 柱状图
# df.plot.barh()  # 横向柱状图（条形图）
# df.plot.hist()  # 直方图
# df.plot.box()  # 箱型图
# df.plot.kde()  # 核密度图估计图
# df.plot.density()  # 同df.plot.kde()
# df.plot.area()  # 面积图
'''
测试这3个函数报错，可能需要参数！
df.plot.pie() # 饼图
df.plot.scatter() # 散点图
df.plot.hexbin() # 六边形箱体图，或简称六边形图
'''

# plt.savefig('E:/bat/output_files/pandas_out_20240708011_003.jpeg')
plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.2 plot()基础方法')
# update20240708
'''
Series数据调用plot方法时，它的索引信息会排布在x轴上，y轴则是x轴上的索引对应的具体数据值。

示例代码如下，效果如图16-1所示。
'''

# DataFrame调用
# df.plot()
# test_df.show()
print('# series绘图')
# plt.show('E:/bat/output_files/pandas_out_20240708011_001.png') # error

# print(range(5))
# print(list(range(5)))
# print(list(range(5))+list(range(5)))
ts = pd.Series(list(range(5)) + list(range(5)),
               index=pd.date_range('1/1/2020', periods=10))
# print(ts)

ts.plot()
plt.show()

print('# DataFrame绘图')

'''
DataFrame调用plot时，x轴为DataFrame的索引，y轴上将显示其多列的多条折线数据。
'''
df = pd.DataFrame(np.random.randn(6, 4),
                  index=pd.date_range('1/1/2020', periods=6),
                  columns=list('ABCD')
                  )

df = abs(df)
print(df)

df.plot()

plt.show()

print('DataFrame在绘图时可以指定x轴和y轴的列')
df = pd.DataFrame(np.random.randn(50, 2),
                  columns=['B', 'C']).cumsum()

# print(df)
df['A'] = pd.Series(list(range(len(df))))
# print(df)

df.plot(x='A', y='B')  # 指定x和y轴的内容
plt.show()

# y轴指定两列
df.plot(x='A', y=['B', 'C'])
plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.3 图形类型')
# update20240709
'''
默认的plot()方法可以帮助我们快速绘制各种图形，接下来介绍在使用plot绘制不同图形时常用的一些参数。
df.plot()的kind参数，可以指定图形的类型：
'''

# df.plot(kind='bar')
# df.plot(kind='barh')
# df.plot(kind='hist')
# df.plot(kind='kde')
# df.plot(kind='density')
# df.plot(kind='box')
# df.plot(kind='area')
# df.plot(kind='scatter')  # error

# s = df['A']
# print(s)
# s.plot(kind='pie')
# plt.show()


print()
print('------------------------------------------------------------')
print('\t16.1.4 x轴和y轴')
# update20240709
'''
x、y参数可指定对应轴上的数据，常用在折线图、柱状图、面积图、散点图等图形上。
如果是Series，则索引是x轴，无须传入y轴的值。
'''

print('# 可以不写参数名，直接按位置传入')

# print(df)
# df.plot('name', 'Q1')
# df.plot.bar('name', ['Q1', 'Q2'])
# df.plot.bar(x='name', y='Q4')
# df.plot.area('name', ['Q1', 'Q2'])
# df.plot.scatter('name', 'Q3')
'''注意，散点图只允许有一个y值。'''

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.5 图形标题')
# update20240709
'''
用title参数来指定图形的标题，标题会显示在图形的顶部
'''

print('# 指定标题')

df.plot.bar(title='前五位学生成绩分布图')
'''对于中文出现乱码的情况，后文会给出解决方案。'''

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.6 字体大小')
# update20240709
'''
fontsize指定轴上的字体大小，单位是pt（磅）
'''

print('# 指定轴上的字体大小')

df.set_index('name')[:5].plot(fontsize=20)

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.7 线条样式')
# update20240709
'''
style可指定图的线条等样式，并组合使用
'''
df.set_index('name', inplace=True)

# df[0:5].plot(style=':')  # 虚线
# df.set_index('name')[0:5].plot(style=':')  # 虚线
# df.plot(style='-.')  # 虚实相间
# df.plot.line(style='-.')
# df.plot.bar(style='-.')  # 可以显示柱状图形，但是没有看出来啥效果
# df.plot(style='--')  # 长虚线
# df.plot(style='.')  # 点
# df.plot(style='-')  # 实线（默认）
# df.plot(style='*-')  # 实线，数值为星星
# df.plot(style='*--')  # 虚线，数值为星星
# df.plot(style='^-')  # 实线，数值为三角形

print('# 指定不同的线条样式')
# df.plot(style=[':', '--', '.-', '*-'])
df.plot(title='指定不同线条样式', style=[':', '--', '.-', '*-'])

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.8 背景辅助线')
# update20240709
'''
grid会给x方向和y方向增加背景辅助线
'''
df.set_index('name', inplace=True)

print('# 增加背景辅助线')
df.plot(style=[':', '-.'], grid=True)

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.9 图例')
# update20240709
'''
plot()默认会显示图例，传入参数legend=False可隐藏图例。
'''
df.set_index('name', inplace=True)

print('# 不显示图例')
# df.plot(style=[':', '-.'], grid=True, legend=False)

print('# 将图例倒排')
df.plot(legend='reverse')

# df.plot(color='#D0D0D0').legend(loc='best') # 自动选择最佳位置（默认）
# df.plot(color='#D0D0D0').legend(loc='lower left')  # 左下角
# df.plot(color='#D0D0D0').legend(loc='lower right')  # 右下角
df.plot(color='#D0D0D0').legend(loc='center right')  # 右下角

'''
'best'：自动选择最佳位置（默认）
'upper right'：右上角
'upper left'：左上角
'lower left'：左下角
'lower right'：右下角
'right'：右侧
'center left'：左侧中央
'center right'：右侧中央
'lower center'：底部中央
'upper center'：顶部中央
'center'：中央
'''

plt.show()

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.10 图形大小')
# update20240709
'''
figsize参数传入一个元组，可以指定图形的宽、高值（单位为英寸，1英寸约为0.0254米）：
'''
df.set_index('name', inplace=True)

print('# 定义图形大小')
# df.plot.bar(figsize=(10.5, 5))
# df.plot.bar(figsize=(10.5, 5), legend=False)

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.11 色系')
# update20240709
'''
colormap指定图形的配色，具体值可参考Matplotlib库的色系表。
'''
df.set_index('name', inplace=True)

print('# 指定色系')

df.plot.barh(colormap='rainbow')
plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.12 绘图引擎')
# update20240709
'''
backend 参数可以指定一个新的绘图引擎，默认使用的是Matplotlib。
'''
df.set_index('name', inplace=True)

# 使用bokeh
# import pandas_bokeh
#
# pandas_bokeh.output_notebook()  # Notebook展示
# df.head(10).plot.bar('name', ['Q1', 'Q2'], backend='pandas_bokeh')
'''搞了半天 总是出错，放弃！'''

df.plot()
plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.13 Matplotlib的其他参数')
# update20240710
'''
我们知道，Pandas的默认绘图引擎是Matplotlib。
在plot()中可以使用Matplotlib的一系列参数，示例代码如下：
'''
df.set_index('name', inplace=True)

# df.plot.line(color='k')  # 图的颜色
df.plot.bar(rot=45)  # 主轴上文字的方向度数

plt.show()

'''
更多参数和使用方法参见官网 https://matplotlib.org/stable/api/pyplot_summary.html 。
'''

print()
print('------------------------------------------------------------')
print('\t16.1.14 图形叠加')
# update20240710
'''
如果希望将两种类型的图形叠加在一起，可以将两个图形的绘图语
句组成一个元组或者列表，下例实现了将5位同学的第一季度成绩（柱
状图）与其4个季度的平均成绩叠加在一起，方便对比。
'''
df.set_index('name', inplace=True)

(
    df['Q1'].plot.bar(),
    df.select_dtypes('number').mean(1).plot(color='r')
)

plt.show()

'''
更多参数和使用方法参见官网 https://matplotlib.org/stable/api/pyplot_summary.html 。
'''

print()
print('------------------------------------------------------------')
print('\t16.1.15 颜色的表示')
# update20240710
'''
在可视化中颜色与CSS的表示方法相同，可以用CSS颜色名和CSS
合法颜色值表示。17种标准颜色的名称为：
aqua、black、blue、fuchsia、gray、green、lime、maroon、navy、olive、orange、purple、
red、silver、teal、white、yellow。
'''
df.set_index('name', inplace=True)

# df.plot(color='fuchsia') # 紫红色
# df.plot(color='olive') # 橄榄色
# df.plot(color='aqua') # 浅绿色
# df.plot(color='lime') # 石灰色 | 树叶绿？
# df.plot(color='gray')  # 灰色
# df.plot(color='teal')  # 蓝绿色
# df.plot(color='maroon')  # 栗色
# df.plot(color='navy')  # 海军蓝
# df.plot(color='purple')  # 紫色
# df.plot(color='silver')  # 银色
# df.plot(color='golden') # 非标准色
df.plot(color='#D0D0D0')

plt.show()

'''
更多16进制颜色参数和使用方法参见官网 https://gairuo.com/p/web-color 。
'''

print()
print('------------------------------------------------------------')
print('\t16.1.16 解决图形中的中文乱码问题')
# update20240710
'''
Pandas绘图依赖的Matplotlib库在安装初始化时会加载一个配置文
件，这个文件包含了将要用到的字体，而中文字体不在这个文件中，所
以会造成在绘图过程中图形中的中文显示为方框或乱码的情况。
'''
df.set_index('name', inplace=True)

print('# 临时方案')
# jupyter notebooks plt 图表配置
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (15.0, 8.0)  # 固定显示大小
plt.rcParams['font.family'] = ['sans-serif']  # 显示中文问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文问题
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

'''
16.1.17 小结
DataFrame和Series都支持用plot()方法快速生成各种常用图形，plot()的参数可以对图形完成精细化处理。
本节还介绍了可视化图形中颜色的表达方式和中文字符在图形中的兼容显示问题。
'''

print()
print('------------------------------------------------------------')
print('\t16.2 常用可视化图形')
'''
本节将介绍plot()方法适配的几个最为常用的图形绘制方法，
这些方法甚至不需要额外的参数就能快速将数据可视化，使用起来非常方便。
'''

print()
print('------------------------------------------------------------')
print('\t16.2.1 折线图plot.line')
# update20240711
'''
折线图（Line Chart）是用线条连接各数据点而成的图形，它能表达数据的走势，一般与时间相关。

plot的默认图形是折线图，因此对于折线图，可以省略df.plot.line()中的line()方法。
DataFrame可以直接调用plot来生成折线图，其中，x轴为索引，其他数字类型的列为y轴上的线条。
'''
# df.set_index('name', inplace=True)

print('# 折线图的使用')
# df.plot()
# df.plot.line()  # 全写方式

'''
基于以上逻辑，如果希望指定的列显示为x轴，可以先将其设为索引.
'''

'''
(
    df.head(10)
    .set_index('team')
    .sort_values(by='team')  # 排序有效
    .plot()
)
'''

# Series索引为x轴，值为y轴，有值为非数字时会报错：
(
    df.set_index('name')
    .head()
    .Q1  # Series
    .plot()
)

plt.show()

print('# Series索引为x轴，值为y轴，有值为非数字时会报错：')
# 测试非数字值

# df.iloc[3,3] = '文本'
df = pd.read_excel(team_file)
# df.iloc[3,2] = np.NaN # 运行显示一半

df.iloc[3, 2] = np.NaN  # 'text' | 文本直接报错！error
# df

(
    df.set_index('name')
    .head()
    .Q1  # Series
    .plot()
)

print('# 指定x轴和y轴')

# df.plot(x='name', y='Q1')
df.plot(x='name', y=['Q1', 'Q2'])  # 指定多条

plt.show()

print('# 显示子图')

# df.plot(x='name', y='Q1')
df.plot(x='name', y=['Q1', 'Q2'], subplots=True)  # 指定多条

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.2.2 饼图plot.pie')
# update20240716
'''
饼图（Pie Chart）可以表示不同分类的数据在总体中的占比情况，
将一个完整的圆形划分为若干个“小饼”，占比大小体现在弧度大小上，整个圆饼为数据的总量。

如果数据中有NaN值，会自动将其填充为0；
如果数据中有负值，则会引发ValueError错误。
'''
# df.set_index('name', inplace=True)

print('# 饼图的使用')

s = pd.Series(3 * np.random.rand(4),
              index=['a', 'b', 'c', 'd'],
              name='数列'
              )

print(s)

s.plot.pie(figsize=(6, 6))
plt.show()

# TODO random函数参考：
'''
np.random.rand和np.random.random生成均匀分布在[0, 1)的浮点数。
np.random.randn生成服从标准正态分布的浮点数。
np.random.randint生成指定范围内的整数。

np.random.rand(5) # 生成一个1维数组，包含5个[0, 1)之间的随机数
np.random.randn(2, 3) # 生成一个2x3的数组，包含标准正态分布的随机数
np.random.randint(low, high=None, size=None, dtype=int)
    low: 随机整数的下界（包括）。
    high: 随机整数的上界（不包括）。如果没有指定，则生成[0, low)之间的随机数。
    size: 输出随机数的形状。
    dtype: 输出数组的数据类型。
np.random.randint(0, 10, size=5) # 生成一个1维数组，包含5个[0, 10)之间的随机整数
np.random.randint(0, 10, size=(5,3)) # 生成一个2维数组，包含5行3列个[0, 10)之间的随机整数
np.random.randint(5, size=(5,3), dtype=int) # 生成一个2维数组，包含5行3列个[0, 5)之间的随机整数

np.random.random((2, 2)) # 生成一个2x2的数组，包含[0, 1)之间的随机数

np.random.choice(a, size=None, replace=True, p=None)
    参数：
    a: 生成随机数的数组或整数。
    size: 输出随机数的形状。
    replace: 是否允许重复抽样。
    p: 每个元素被抽中的概率。

# 从用户列表中随机抽取3个用户，允许重复
users = ['User1', 'User2', 'User3', 'User4', 'User5']
sample_users = np.random.choice(users, size=3, replace=True)
print(sample_users)

'''

print('# DataFrame需要指定y值')

df1 = pd.DataFrame(3 * np.random.rand(4, 2),
                   index=['a', 'b', 'c', 'd'],
                   columns=['x', 'y']
                   )

print(df1)

df1.plot.pie(y='x')

plt.show()

'''如果数据的总和小于1.0，则Matplotlib将绘制一个扇形.'''
# 实际Matplotlib会自动归一化处理数据 还是会显示饼图，扇形图显示不出来。
s2 = pd.Series([0.1] * 4,
               index=['a', 'b', 'c', 'd'],
               name='series2')

s2.plot.pie(figsize=(6, 6), autopct='%1.1f%%')

print('# 子图')
df1.plot.pie(subplots=True, figsize=(8, 4))

# 子图，不显示图例
df1.plot.pie(subplots=True, figsize=(8, 4), legend=False)

# 设定如下代码中的其他常用参数
s.plot.pie(labels=['AA', 'BB', 'CC', 'DD'],  # 标签，指定项目名称
           colors=['r', 'g', 'b', 'c'],  # 指定颜色
           autopct='%.2f',  # 数字格式
           fontsize=20,  # 字体大小
           figsize=(6, 6)  # 图大小
           )

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.2.3 柱状图plot.bar')
# update20240716
'''
柱状图（Bar Chart）又称条形图，使用与轴垂直的柱子，通过柱形的高低来表达数据的大小。
它适用于数据的对比，在整体中也能看到数据的变化趋势。
'''
# df.set_index('name', inplace=True)

df.plot.bar()
df.plot.barh()  # 横向
df[:5].plot.bar(x='name', y='Q4')  # 指定x、y轴 || name列不能设定为索引列
df[:5].plot.bar(x='name', y=['Q1', 'Q2'])  # 指定x、y轴
# 合并逻辑
(
    df.head()
    .set_index('name')  # 设为索引
    .plot
    .bar()
)

'''
Series索引为x轴，值为y轴，有值为非数字时会报错。与折线图一
样，如果数据中有负值，则0刻度会在x轴之上，不会在图形底边
'''
(
    df.assign(Q1=df.Q1 - 70)  # 让Q1产生部分负值
    .loc[:6]  # 取部分
    .set_index('name')  # 设为索引
    .plot
    .barh()
)

'''可以将同一索引的多个数据堆叠起来'''

(
    df.loc[:6]
    .set_index('name')
    .plot
    .bar(stacked=True)  # 柱状图，堆叠
)

# 柱状图，横向+堆叠
(
    df.loc[:6]
    .set_index('name')
    .plot
    .barh(stacked=True)  # 柱状图，横向+堆叠
)

# 柱状图 子图
(
    df.set_index('name')
    .plot
    .bar(subplots=True)
)

# 横向柱状图 子图
(
    df.set_index('name')
    .plot
    .barh(subplots=True)
)

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.2.4 直方图plot.hist')
# update20240716
'''
直方图（Histogram）又称质量分布图，由一系列高度不等的纵向条纹或线段表示数据分布的情况。
一般用横轴表示数据类型，纵轴表示分布情况。
'''
# df.set_index('name', inplace=True)

df2 = pd.DataFrame({'a': np.random.randn(1000) + 1,
                    'b': np.random.randn(1000),
                    'c': np.random.randn(1000) - 1,
                    },
                   columns=['a', 'b', 'c']
                   )

# print(df2)
df2.plot.hist(alpha=0.1)  # alpha参数调控图形透明度，值越大透明度越小。

# 单直方图
df2.a.plot.hist(alpha=0.5)

# 堆叠，指定分箱数量
df2.plot.hist(stacked=True, bins=20)

# 绘制子图
df.hist(alpha=0.5)

# 可以单独绘制子图，指定分箱数量：
df2.a.hist(bins=20, alpha=0.5)
df2.hist('a', bins=20, alpha=0.5)  # 同上

'''by参数可用来进行分组，生成分组后的子图'''
# 分组
df.Q1.hist(by=df.team)

'''还可以传递matplotlib hist支持的其他关键字，详情请参考Matplotlib官方文档。'''
plt.show()

print()
print('------------------------------------------------------------')
print('\t16.2.5 箱形图plot.box')
# update20240717
'''
箱形图（Box Chart）又称盒形图、盒式图或箱线图，是一种用来显示一组数据分布情况的统计图。

从箱形图中我们可以观察到：
    一组数据的关键值，如中位数、最大值、最小值等；
    数据集中是否存在异常值，以及异常值的具体数值；
    数据是不是对称的；
    这组数据的分布是否密集、集中；
    数据是否扭曲，即是否有偏向性。

'''
# df.set_index('name', inplace=True)

df.plot.box()  # 所有列

# 其他的一些方法如下：
df.Q1.plot.box()  # 单列
df.boxplot()  # 自带背景
df.boxplot('Q1')

print('为图形中的一些元素指定颜色，')

color = {'boxes': 'DarkGreen',  # 箱体颜色
         'whiskers': 'DarkOrange',  # 连线颜色
         'medians': 'DarkBlue',  # 中位数颜色
         'caps': 'Gray'  # 极值颜色
         }

df.plot.box(color=color, sym='r+')

print('# 横向+位置调整')
df.plot.box(vert=False, positions=[1, 2, 5, 6])
df.plot.box(vert=False)

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.2.6 面积图plot.area')
# update20240717
'''
面积图（Area Chart）又叫区域图。将折线图中折线与自变量坐标
轴之间的区域使用颜色或纹理填充，这样的填充区域就叫作面积。

默认情况下，面积图是堆叠的。
要生成堆积面积图，每列必须全部为正值或全部为负值。
当输入数据包含NaN时，它将被自动填充0。
如果要删除或填充不同的值，请在调用图之前使用DataFrame.dropna()或DataFrame.fillna()。

'''
# df.set_index('name', inplace=True)

df4 = pd.DataFrame(np.random.randn(10, 4), columns=['a', 'b', 'c', 'd'])

df4 = df4.abs()  # 默认情况下，面积图是堆叠的。要生成堆积面积图，每列必须全部为正值或全部为负值。
df4.a.plot.area()  # 单列
print(df4)
df4.plot.area(alpha=0.5)

print('# 未堆积')
# df4.plot.area(stacked=False)

'''
ValueError: When stacked is True, each column must be either all positive or all negative. 
Column 'a' contains both positive and negative values.

这个错误是由于 plot.area() 方法默认情况下会堆叠（stack）各列的数据，
而堆叠时，要求每列数据要么全为正值，要么全为负值。如果某列数据同时包含正值和负值，就会引发这个错误。
'''

print('指定x轴和y轴：')
# df4.plot.area(y='a')  # 同单列
# df4.plot.area(y=['b', 'c'])
df4.plot.area(x='a')

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.2.7 散点图plot.scatter')
# update20240717
'''
散点图（Scatter Graph）也叫x-y图，它将所有的数据以点的形式展现在直角坐标系上，
以显示变量之间的相互影响程度，点的位置由变量的数值决定。

散点图要求x轴和y轴为数字列，这些可以通过x和y关键字指定。

'''
# df.set_index('name', inplace=True)
print('Q1成绩与平均分的散点图')

(
    df.assign(avg=df.select_dtypes('number').mean(1))  # 增加一列平均分
    .plot
    # .scatter()  # TypeError: PlotAccessor.scatter() missing 2 required positional arguments: 'x' and 'y'
    .scatter(x='Q1', y='avg')  # success
)

'''通过散点图发现，学生Q1的成绩与平均成绩呈现一定的相关性。'''

print('# 指定颜色')

# df.plot.scatter(x='Q1', y='Q2', c='b', s=50)
'''
c可以取以下值：
    字符，RGB或RGBA码，如red、#a98d19；
    序列，颜色列表，对应每个点的颜色；
    列名称或位置，其值将用于根据颜色图为标记点着色。
'''

df.plot.scatter(x='Q1', y='Q2', c=['green', 'yellow'] * 3, s=50)
df.plot.scatter(x='Q1', y='Q2', c='DarkBlue')
df.plot.scatter(x='Q1', y='Q2', c='Q1', colormap='viridis')
df.plot.scatter(x='Q1', y='Q2', c='Q1', colormap='coolwarm')

'''
传入参数colorbar=True会在当前坐标区或图的右侧显示一个垂直颜色栏，
颜色栏显示当前颜色图并指示数据值到颜色图的映射。
'''
# 色阶栏
df.plot.scatter(x='Q1', y='Q2', c='DarkBlue', colorbar=True)
# df.plot.scatter(x='Q1', y='Q2', c='coolwarm', colorbar=True)  # error
# df.plot.scatter(x='Q1', y='Q2', c='DarkBlue', colormap='coolwarm')  # error

print('s用于指定点的大小：')
df.plot.scatter(x='Q1', y='Q2', s=df['Q1'] * 10)  # 值越大 点越大
df.plot.scatter(x='Q1', y='Q2', s=50)  # 每个点是同样大小


print()
print('------------------------------------------------------------')
print('\t16.2.8 六边形分箱图plot.hexbin')
# update20240717
'''
六边形分箱图（Hexagonal Binning）也称六边形箱体图，或简称六边形图，它是一种以六边形为主要元素的统计图表。
它比较特殊，既是散点图的延伸，又兼具直方图和热力图的特征。

使用DataFrame.plot.hexbin()创建六边形图。
如果你的数据过于密集而无法单独绘制每个点，则可以用六边形图替代散点图。
'''
# df.set_index('name', inplace=True)
# print('Q1成绩与平均分的散点图')
df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])
df['b'] = df['b'] + np.arange(1000)

print(df)

df.plot.hexbin(x='a', y='b')  # 默认散点尺寸，较小
df.plot.hexbin(x='a', y='b', gridsize=25)

plt.show()

'''
16.2.9 小结
本节介绍了数据分析中最为常见的可视化图形，基本涵盖了大多数数据可视化场景，需要熟练掌握本节内容。
这些功能与Matplotlib紧密相连，如果想了解更为高级的用法，可以到Matplotlib官网查看文档进行学习。
'''