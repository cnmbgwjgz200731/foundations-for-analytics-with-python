import numpy as np
import pandas as pd
import warnings
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
print('\t6.5 数据分箱')
print('\t6.5.1 定界分箱pd.cut()')
print()
# update20240429
'''
数据分箱（data binning，也称为离散组合或数据分桶）是一种数据预处理技术，
它将原始数据分成几个小区间，即bin（小箱子），是一种量子化的形式。数据分箱
可以最大限度减小观察误差的影响。落入给定区间的原始数据值被代表该区间的值（通常是中心值）替换。
然后将其替换为针对该区间计算的常规值。这具有平滑输入数据的作用，
并且在小数据集的情况下还可以减少过拟合。

Pandas主要基于以两个函数实现连续数据的离散化处理。
pandas.cut：根据指定分界点对连续数据进行分箱处理。
pandas.qcut：根据指定区间数量对连续数据进行等宽分箱处理。
所谓等宽，指的是每个区间中的数据量是相同的。

'''
# 小节注释
'''
pd.cut()可以指定区间将数字进行划分。
以下例子中的0、60、100三个值将数据划分成两个区间，从而将及格或者不及格分数进行划分

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 将Q1成绩换60分及以上、60分以下进行分类')

print(pd.cut(df.Q1,bins=[0,60,100]))
# 0    (60, 100]
# 1      (0, 60]
# 2      (0, 60]
# 3    (60, 100]
# 4    (60, 100]
# 5    (60, 100]
# Name: Q1, dtype: category
# Categories (2, interval[int64, right]): [(0, 60] < (60, 100]]

# 将分箱结果应用到groupby分组中：
# Series使用
print(df.Q1.groupby(pd.cut(df.Q1,bins=[0,60,100])).count())
# Q1
# (0, 60]      2
# (60, 100]    4
# Name: Q1, dtype: int64

print()
print('# Dataframe使用')
print(df.groupby(pd.cut(df.Q1,bins=[0,60,100])).count())
#            name  team  Q1  Q2  Q3  Q4
# Q1
# (0, 60]       2     2   2   2   2   2
# (60, 100]     4     4   4   4   4   4

print()
print('# 其它参数示例')
print('# 不显示区间，使用数字作为每个箱子的标签，形式如0，1，2，n等')
print(pd.cut(df.Q1,bins=[0,60,100],labels=False))
# 0    1
# 1    0
# 2    0
# 3    1
# 4    1
# 5    1
# Name: Q1, dtype: int64

print()
print('# 指定标签名')
print(pd.cut(df.Q1,bins=[0,60,100],labels=['不及格','及格',]))
# 0     及格
# 1    不及格
# 2    不及格
# 3     及格
# 4     及格
# 5     及格
# Name: Q1, dtype: category
# Categories (2, object): ['不及格' < '及格']

# print(pd.cut(df.Q1,bins=[0,60,100],labels=['不及格','及格'])) # 结果同上


# print(df.groupby(pd.cut(df.Q1,bins=[0,60,100],labels=['不及格','及格'])).sum())
#                   name  team   Q1   Q2   Q3   Q4
# Q1
# 不及格            ArryAck    CA   93   97   55  141
# 及格   LiverEorgeOahRick  ECDB  347  265  253  328

print()
print('# 包含最低部分')
print(pd.cut(df.Q1,bins=[0,60,100],include_lowest=True))
# 0     (60.0, 100.0]
# 1    (-0.001, 60.0]
# 2    (-0.001, 60.0]
# 3     (60.0, 100.0]
# 4     (60.0, 100.0]
# 5     (60.0, 100.0]
# Name: Q1, dtype: category
# Categories (2, interval[float64, right]): [(-0.001, 60.0] < (60.0, 100.0]]

print()
print('# 是否为右闭区间，下例为[89, 100)')
print(pd.cut(df.Q1,bins=[0,89,100],right=False))
# 0    [89.0, 100.0)
# 1      [0.0, 89.0)
# 2      [0.0, 89.0)
# 3    [89.0, 100.0)
# 4      [0.0, 89.0)
# 5              NaN
# Name: Q1, dtype: category
# Categories (2, interval[int64, left]): [[0, 89) < [89, 100)]


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.5 数据分箱')
print('\t6.5.2 等宽分箱pd.qcut()')
print()
# update20240429
# 小节注释
'''
pd.qcut()可以指定所分区间的数量，Pandas会自动进行分箱：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 按Q1成绩分为两组')
print(pd.qcut(df.Q1,q=2))
# 0     (77.0, 100.0]
# 1    (35.999, 77.0]
# 2    (35.999, 77.0]
# 3     (77.0, 100.0]
# 4    (35.999, 77.0]
# 5     (77.0, 100.0]
# Name: Q1, dtype: category
# Categories (2, interval[float64, right]): [(35.999, 77.0] < (77.0, 100.0]]

print()
print('# 查看分组区间')
print(pd.qcut(df.Q1,q=2).unique())
# [(77.0, 100.0], (35.999, 77.0]]
# Categories (2, interval[float64, right]): [(35.999, 77.0] < (77.0, 100.0]]

print()
print('# 应用到分组中：')
# Series使用
print(df.Q1.groupby(pd.qcut(df.Q1,q=2)).count())
# Q1
# (35.999, 77.0]    3
# (77.0, 100.0]     3
# Name: Q1, dtype: int64

print()
# DataFrame使用
print(df.groupby(pd.qcut(df.Q1,q=2)).max())
#                 name team   Q1  Q2  Q3   Q4
# Q1
# (35.999, 77.0]   Oah    D   65  60  61   86
# (77.0, 100.0]   Rick    E  100  99  97  100

print()
print('# 其它参数如下：')
print(pd.qcut(range(5),4))
# [(-0.001, 1.0], (-0.001, 1.0], (1.0, 2.0], (2.0, 3.0], (3.0, 4.0]]
# Categories (4, interval[float64, right]): [(-0.001, 1.0] < (1.0, 2.0] < (2.0, 3.0] < (3.0, 4.0]]
print(pd.qcut(range(5),4,labels=False))
# [0 0 1 2 3]

print()
print('# 指定标签名')
print(pd.qcut(range(5),3,labels=["good","medium","bad"]))
# ['good', 'good', 'medium', 'bad', 'bad']
# Categories (3, object): ['good' < 'medium' < 'bad']

print()
print('# 返回箱子标签 array([1. , 51.5, 98.]))')
print(pd.qcut(df.Q1,q=2,retbins=True))
# (0     (77.0, 100.0]
# 1    (35.999, 77.0]
# 2    (35.999, 77.0]
# 3     (77.0, 100.0]
# 4    (35.999, 77.0]
# 5     (77.0, 100.0]
# Name: Q1, dtype: category
# Categories (2, interval[float64, right]): [(35.999, 77.0] < (77.0, 100.0]], array([ 36.,  77., 100.])

print()
print('# 分箱位小数位数')
print(pd.qcut(df.Q1,q=2,precision=3))
# 0     (77.0, 100.0]
# 1    (35.999, 77.0]
# 2    (35.999, 77.0]
# 3     (77.0, 100.0]
# 4    (35.999, 77.0]
# 5     (77.0, 100.0]
# Name: Q1, dtype: category
# Categories (2, interval[float64, right]): [(35.999, 77.0] < (77.0, 100.0]]

print()
print('# 排名分3个层次')
print(pd.qcut(df.Q1.rank(method='first'),3))
# 0    (2.667, 4.333]
# 1    (0.999, 2.667]
# 2    (0.999, 2.667]
# 3      (4.333, 6.0]
# 4    (2.667, 4.333]
# 5      (4.333, 6.0]
# Name: Q1, dtype: category
# Categories (3, interval[float64, right]): [(0.999, 2.667] < (2.667, 4.333] < (4.333, 6.0]]

'''
6.5.3 小结
本节介绍的分箱也是一种数据分组方式，经常用在数据建模、机器
学习中，与传统的分组相比，它更适合离散数据。
'''