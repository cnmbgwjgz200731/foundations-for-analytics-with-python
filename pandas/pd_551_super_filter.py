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
print('第5章 pandas高级操作')
print('\t5.5 高级过滤')
print('\t5.5.1 df.where()')
print()
# update20240408
'''
本节介绍几个非常好用的数据过滤输出方法，它们经常用在一些复杂的数据处理过程中。
df.where()和df.mask()通过给定的条件对原数据是否满足条件进行筛选，最终返回与原数据形状相同的数据。
为了方便讲解，我们仅取我们的数据集的数字部分，即只有Q1到Q4列：
'''
# 小节注释
'''
df.where()中可以传入一个布尔表达式、布尔值的Series/DataFrame、序列或者可调用的对象，
然后与原数据做对比，返回一个行索引与列索引与原数据相同的数据，且在满足条件的位置保留原值，
在不满足条件的位置填充NaN。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

# 只保留数字类型列
df1 = df.select_dtypes(include='number')
print(df1)

print()
print('# 数值大于70')
print(df1.where(df1 > 70)) # 不满足条件的填充NaN ,满足条件的 保留原值
#       Q1    Q2    Q3     Q4
# 0   89.0   NaN   NaN    NaN
# 1    NaN   NaN   NaN    NaN
# 2    NaN   NaN   NaN   84.0
# 3   93.0  96.0  71.0   78.0
# 4    NaN   NaN   NaN   86.0
# 5  100.0  99.0  97.0  100.0

print()
print('# 传入一个可调用对象，这里我们用lambda：')
# Q1列大于50
print(df1.where(lambda d:d.Q1 > 50)) # 条件不满足 整行值全部填充为NaN
#       Q1    Q2    Q3     Q4
# 0   89.0  21.0  24.0   64.0
# 1    NaN   NaN   NaN    NaN
# 2   57.0  60.0  18.0   84.0
# 3   93.0  96.0  71.0   78.0
# 4   65.0  49.0  61.0   86.0
# 5  100.0  99.0  97.0  100.0

print()
print('# 条件为一个布尔值的Series：')
# 传入一个布尔值Series，前三个为真
print(df1.Q1.where(pd.Series([True]*3)))
# print(pd.Series([True]*3))
# 0    89.0
# 1    36.0
# 2    57.0
# 3     NaN
# 4     NaN
# 5     NaN
# Name: Q1, dtype: float64

'''上例中不满足条件的都返回为NaN，我们可以指定一个值或者算法来替换NaN：'''

print()
print('# 大于等于60分的显示成绩，小于的显示“不及格”')
print(df1.where(df1>=60,'不及格'))
#     Q1   Q2   Q3   Q4
# 0   89  不及格  不及格   64
# 1  不及格  不及格  不及格  不及格
# 2  不及格   60  不及格   84
# 3   93   96   71   78
# 4   65  不及格   61   86
# 5  100   99   97  100

print()
print('# 给定一个算法，df为偶数时显示原值减去20后的相反数')

# 定义一个数是否为偶数的表达式 c
c = df1%2 == 0
# 传入c, 为偶数时显示原值减去20后的相反数
print(df1.where(~c,-(df1-20)))
#    Q1  Q2  Q3  Q4
# 0  89  21  -4 -44
# 1 -16  37  37  57
# 2  57 -40   2 -64
# 3  93 -76  71 -58
# 4  65  49  61 -66
# 5 -80  99  97 -80

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.5 高级过滤')
print('\t5.5.2 np.where()')
print()
# update20240408
'''
本节介绍几个非常好用的数据过滤输出方法，它们经常用在一些复杂的数据处理过程中。
df.where()和df.mask()通过给定的条件对原数据是否满足条件进行筛选，最终返回与原数据形状相同的数据。
为了方便讲解，我们仅取我们的数据集的数字部分，即只有Q1到Q4列：
'''
# 小节注释
'''
np.where()是NumPy的一个功能，虽然不是Pandas提供的，但可以弥补df.where()的不足，所以有必要一起介绍。
df.where()方法可以将满足条件的值筛选出来，将不满足的值替换为另一个值，但无法对满足条件的值进行替换，
而np.where()就实现了这种功能，达到SQL中if（条件，条件为真的值，条件为假的值）的效果。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

# 只保留数字类型列
df1 = df.select_dtypes(include='number')
print(df1)

print()
print('# 小于60分为不及格')
# np.where()返回的是一个二维array：
print(np.where(df1 >= 60,'合格','不合格'))
# array([0, 0, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5], dtype=int64), array([0, 3, 1, 3, 0, 1, 2, 3, 0, 2, 3, 0, 1, 2, 3], dtype=int64))
# [['合格' '不合格' '不合格' '合格']
#  ['不合格' '不合格' '不合格' '不合格']
#  ['不合格' '合格' '不合格' '合格']
#  ['合格' '合格' '合格' '合格']
#  ['合格' '不合格' '合格' '合格']
#  ['合格' '合格' '合格' '合格']]

# 让df.where()中的条件为假，从而应用np.where()的计算结果
print(df1.where(df1==9999999,np.where(df1>=60,'合格','不合格')))
#     Q1   Q2   Q3   Q4
# 0   合格  不合格  不合格   合格
# 1  不合格  不合格  不合格  不合格
# 2  不合格   合格  不合格   合格
# 3   合格   合格   合格   合格
# 4   合格  不合格   合格   合格
# 5   合格   合格   合格   合格

print()
print('# 包含是、否结果的Series')
'''下例是np.where()对一个Series（d.avg为计算出来的虚拟列）进行判断，返回一个包含是、否结果的Series。'''
print(df1.assign(avg=df1.mean(1))
         .assign(及格=lambda d: np.where(d.avg>=50,'是','否')))
#     Q1  Q2  Q3   Q4    avg 及格
# 0   89  21  24   64  49.50  否
# 1   36  37  37   57  41.75  否
# 2   57  60  18   84  54.75  是
# 3   93  96  71   78  84.50  是
# 4   65  49  61   86  65.25  是
# 5  100  99  97  100  99.00  是

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.5 高级过滤')
print('\t5.5.3 df.mask()')
print()
# update20240409
'''
本节介绍几个非常好用的数据过滤输出方法，它们经常用在一些复杂的数据处理过程中。
df.where()和df.mask()通过给定的条件对原数据是否满足条件进行筛选，最终返回与原数据形状相同的数据。
为了方便讲解，我们仅取我们的数据集的数字部分，即只有Q1到Q4列：
'''
# 小节注释
'''
df.mask()的用法和df.where()基本相同，唯一的区别是df.mask()将满足条件的位置填充为NaN。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

# 只保留数字类型列
df1 = df.select_dtypes(include='number')
print(df1)

print()
print('# 小于60分 保留原值，否则NaN填充')
print(df1.mask(df1 >= 60))
#      Q1    Q2    Q3    Q4
# 0   NaN  21.0  24.0   NaN
# 1  36.0  37.0  37.0  57.0
# 2  57.0   NaN  18.0   NaN
# 3   NaN   NaN   NaN   NaN
# 4   NaN  49.0   NaN   NaN
# 5   NaN   NaN   NaN   NaN

print()
print('# 对满足条件的位置指定填充值')
s = df1.Q1
# print(s)
print(df1.Q1.mask(s>80,'优秀'))
# 0    优秀
# 1    36
# 2    57
# 3    优秀
# 4    65
# 5    优秀
# Name: Q1, dtype: object
print()
print('# 通过数据筛选返回布尔序列')
# df.mask()和df.where()还可以通过数据筛选返回布尔序列：
# 返回布尔序列，符合条件的行值为True
print((df.where((df.team == 'A') & (df.Q1>30)) == df).Q1)
# 0    False
# 1    False
# 2     True
# 3    False
# 4    False
# 5    False
# Name: Q1, dtype: bool

print()
# 返回布尔序列，符合条件的行值为False
print((df.mask((df.team == 'A') & (df.Q1>30)) == df).Q1)
# 0     True
# 1     True
# 2    False
# 3     True
# 4     True
# 5     True
# Name: Q1, dtype: bool