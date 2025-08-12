import numpy as np
import pandas as pd
import warnings

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

# 忽略特定的警告 | 当遇到 openpyxl 中的 UserWarning 类型的警告时，它们将不会被打印出来。
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.4 位置计算')
print('\t4.4.1 位置差值diff')
print()
# update20240228
'''
本节介绍几个经常到用的位置计算操作。diff()和shift()经常用来计算数据的增量变化，rank()用来生成数据的整体排名。

df.diff()可以做位移差操作，经常用来计算一个序列数据中上一个数据和下一个数据之间的差值，如增量研究。
默认被减的数列下移一位，原数据在同位置上对移动后的数据相减，得到一个新的序列，
第一位由于被减数下移，没有数据，所以结果为NaN。可以传入一个数值来规定移动多少位，负数代表移动方向相反。
Series类型如果是非数字，会报错，
DataFrame会对所有数字列移动计算，同时不允许有非数字类型列。

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

s = pd.Series([9, 4, 6, 7, 9])
print(s)
print()
print('# 后一个值与前一个值的差值')
print(s.diff())
print()
# 0    NaN
# 1   -5.0
# 2    2.0
# 3    1.0
# 4    2.0
# dtype: float64


print('# 后方向，移动2位求差值')
print(s.diff(-2))
# 0    3.0
# 1   -3.0
# 2   -3.0
# 3    NaN
# 4    NaN
# dtype: float64

print('# 对于DataFrame，还可以传入axis=1进行左右移动：')
# 只筛选4个季度的5条数据
df1 = df.iloc[:5, 2:6]
print(df1)
print(df1.diff(1, axis=1))
# 上下方向移动 差值
print(df1.diff(1, axis=0))
print(df1.diff(1))  # 效果同上
print(df1.diff())  # 效果同上
print(df1.diff(2))  # 从上向下 间隔2位相减
print(df1.diff(-2))  # 从下向上 间隔2位相减

#    Q1  Q2  Q3  Q4
# 0 NaN -68   3  40
# 1 NaN   1   0  20
# 2 NaN   3 -42  66
# 3 NaN   3 -25   7
# 4 NaN -16  12  25

#      Q1    Q2    Q3    Q4
# 0   NaN   NaN   NaN   NaN
# 1 -53.0  16.0  13.0  -7.0
# 2  21.0  23.0 -19.0  27.0
# 3  36.0  36.0  53.0  -6.0
# 4 -28.0 -47.0 -10.0   8.0

print()
# 计算间隔为2的差分
print(df1.diff(periods=2))  # 效果同 df1.diff(2)
# 输出：
#      Q1    Q2    Q3    Q4
# 0   NaN   NaN   NaN   NaN
# 1   NaN   NaN   NaN   NaN
# 2 -32.0  39.0  -6.0  20.0
# 3  57.0  59.0  34.0  21.0
# 4   8.0 -11.0  43.0   2.0

# 如果dataframe中有非数值类型，运行报错。测试如下
# print(df.diff()) # TypeError: unsupported operand type(s) for -: 'str' and 'str'


print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.4 位置计算')
print('\t4.4.2 位置移动shift')
print()
# update20240229
'''
本节介绍几个经常用到的位置计算操作。diff()和shift()经常用来计算数据的增量变化，rank()用来生成数据的整体排名。

shift()可以对数据进行移位，不做任何计算，也支持上下左右移
动，移动后目标位置的类型无法接收的为NaN。

'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name' ,usecols=[2,3,4,5]
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print('# 整体下移1行，最顶的1行为NaN')
print(df.shift())
#     name  team    Q1    Q2    Q3    Q4
# 0   None  None   NaN   NaN   NaN   NaN
# 1  Liver     E  89.0  21.0  24.0  64.0
# 2   Arry     C  36.0  37.0  37.0  57.0
# 3    Ack     A  57.0  60.0  18.0  84.0
# 4  Eorge     C  93.0  96.0  71.0  78.0
# 5    Oah     D  65.0  49.0  61.0  86.0

print()
print('# 整体下移3行，最顶的3行为NaN')
print(df.shift(3))
#     name  team    Q1    Q2    Q3    Q4
# 0   None  None   NaN   NaN   NaN   NaN
# 1   None  None   NaN   NaN   NaN   NaN
# 2   None  None   NaN   NaN   NaN   NaN
# 3  Liver     E  89.0  21.0  24.0  64.0
# 4   Arry     C  36.0  37.0  37.0  57.0
# 5    Ack     A  57.0  60.0  18.0  84.0

print()
print('# 整体上移一行，最底的一行为NaN')
print(df.Q1.head())
print(df.Q1.head().shift(-1))
# 0    36.0
# 1    57.0
# 2    93.0
# 3    65.0
# 4     NaN
# Name: Q1, dtype: float64

print()
print('# 向右移动1位')
print(df.shift(axis=1))
#    name   team Q1   Q2  Q3  Q4
# 0  None  Liver  E   89  21  24
# 1  None   Arry  C   36  37  37
# 2  None    Ack  A   57  60  18
# 3  None  Eorge  C   93  96  71
# 4  None    Oah  D   65  49  61
# 5  None   Rick  B  100  99  97
print()
print('# 向右移动3位')
print(df.shift(3, axis=1))  # 向右移动3位
#    name  team    Q1     Q2 Q3   Q4
# 0  None  None  None  Liver  E   89
# 1  None  None  None   Arry  C   36
# 2  None  None  None    Ack  A   57
# 3  None  None  None  Eorge  C   93
# 4  None  None  None    Oah  D   65
# 5  None  None  None   Rick  B  100
print()
print('# 向左移动1位')
print(df.shift(-1, axis=1))
#   name  team  Q1  Q2   Q3  Q4
# 0    E    89  21  24   64 NaN
# 1    C    36  37  37   57 NaN
# 2    A    57  60  18   84 NaN
# 3    C    93  96  71   78 NaN
# 4    D    65  49  61   86 NaN
# 5    B   100  99  97  100 NaN
print()
print('# 实现了df.Q1.diff()')
# print(df.Q1,df.Q1.shift())
print(df.Q1 - df.Q1.shift())
# 0     NaN
# 1   -53.0
# 2    21.0
# 3    36.0
# 4   -28.0
# 5    35.0
# Name: Q1, dtype: float64


print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.4 位置计算')
print('\t4.4.3 位置序号rank()')
print()
# update20240229
'''
本节介绍几个经常用到的位置计算操作。diff()和shift()经常用来计算数据的增量变化，rank()用来生成数据的整体排名。

rank()可以生成数据的排序值替换掉原来的数据值，它支持对所有类型数据进行排序，
如英文会按字母顺序。使用rank()的典型例子有学生的成绩表，给出排名：


'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name' ,usecols=[2,3,4,5]
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print('# 排名，将值变了序号')
# 数值从小到大排序 文本按照abc依次排序，首字母都为a，看第二个字母abc排序
print(df.rank())
print(df.head(3).rank())  # 前3行排序
#    name  team   Q1   Q2   Q3   Q4
# 0   4.0   6.0  4.0  1.0  2.0  2.0
# 1   2.0   3.5  1.0  2.0  3.0  1.0
# 2   1.0   1.0  2.0  4.0  1.0  4.0
# 3   3.0   3.5  5.0  5.0  5.0  3.0
# 4   5.0   5.0  3.0  3.0  4.0  5.0
# 5   6.0   2.0  6.0  6.0  6.0  6.0
#    name  team   Q1   Q2   Q3   Q4
# 0   3.0   3.0  3.0  1.0  2.0  2.0
# 1   2.0   2.0  1.0  2.0  3.0  1.0
# 2   1.0   1.0  2.0  3.0  1.0  3.0

print()
print('# 横向排名')
print(df.loc[:, 'Q1':'Q4'].head().rank(axis=1))  # 同一行排名 数值和文本类型不能同一行 否则报错！
#     Q1   Q2   Q3   Q4
# 0  4.0  1.0  2.0  3.0
# 1  1.0  2.5  2.5  4.0
# 2  2.0  3.0  1.0  4.0
# 3  3.0  4.0  1.0  2.0
# 4  3.0  1.0  2.0  4.0

print()
print('# 参数pct=True将序数转换成0~1的数')  # 适用文本类型
print(df.rank(pct=True).head(3).round(2))
print(df.loc[:, 'Q1':'Q4'].head(3).rank(pct=True, axis=1).round(2))  #
#    name  team    Q1    Q2    Q3    Q4
# 0  0.67  1.00  0.67  0.17  0.33  0.33
# 1  0.33  0.58  0.17  0.33  0.50  0.17
# 2  0.17  0.17  0.33  0.67  0.17  0.67
#      Q1    Q2    Q3    Q4
# 0  1.00  0.25  0.50  0.75
# 1  0.25  0.62  0.62  1.00
# 2  0.50  0.75  0.25  1.00

print()
print('--------method参数----------')
# 创建一个包含重复值的Series
s = pd.Series([7, 3.5, 3.5, 1, np.nan])

# 使用不同的method参数进行排名
print()
print('average：序号的平均值，如并列第1名，则按二次元计算（1+2）/2，都显示1.5，下个数据的值为3。')
print(s.rank(method='average'))  # 平均值
# 0    4.0
# 1    2.5
# 2    2.5
# 3    1.0
# 4    NaN
# dtype: float64


print()
print('min：最小的序数，如并列第1名，则都显示1，下个数据为3。')
print(s.rank(method='min'))  # 最小值
# 0    4.0
# 1    2.0
# 2    2.0
# 3    1.0
# 4    NaN
# dtype: float64


print()
print('max：最大的序数，如并列第1名，则都显示1，下个数据为2。')
print(s.rank(method='max'))  # 最大值
# 0    4.0
# 1    3.0
# 2    3.0
# 3    1.0
# 4    NaN
# dtype: float64

print()
print('first：如并列第1名，则按出现顺序分配排名')
print(s.rank(method='first'))  # 索引顺序
# 输出：
# 0    4.0
# 1    2.0
# 2    3.0
# 3    1.0
# 4    NaN
# dtype: float64

print()
print('\\ndense：并列排名相同，但下一个不同值的排名加1')
print(s.rank(method='dense'))  # 紧密排名
# 输出：
# 0    3.0
# 1    2.0
# 2    2.0
# 3    1.0
# 4    NaN
# dtype: float64

# 处理空值的排名
print(s.rank(method='min', na_option='bottom'))  # 空值放在最后
print(s.rank(method='min', na_option='top'))  # 空值放在前面
# out：
# 0    4.0
# 1    2.0
# 2    2.0
# 3    1.0
# 4    5.0
# dtype: float64
# 0    5.0
# 1    3.0
# 2    3.0
# 3    2.0
# 4    1.0
# dtype: float64
