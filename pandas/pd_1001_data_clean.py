import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import glob
import os
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
csv_feb_file = 'E:/bat/input_files/sales_february_2014.csv'
multindex_file = 'E:/bat/input_files/pandas_out_20240509042.xlsx'


path = 'E:/bat/output_files/pandas_read_csv_20240118.csv'
# df1.to_excel('E:/bat/output_files/pandas_out_20240510053.xlsx')

# TODO 忽略特定的警告 | 当遇到 openpyxl 中的 UserWarning 类型的警告时，它们将不会被打印出来。
warnings.filterwarnings('ignore',category=UserWarning,module='openpyxl')

print()
print('------------------------------------------------------------')
print('第10章 Pandas数据清洗')
print('\t10.1 缺失值的认定')
print('\t10.1.1 缺失值类型')
print()
# update20240517
'''
数据清洗是数据分析的一个重要步骤，关系到数据的质量，而数据的质量又关系到数据分析的效果。
数据清洗一般包括缺失值填充、冗余数据删除、数据格式化、异常值处理、逻辑错误数据检测、数据一致性校验、重复值过滤、数据质量评估等。
Pandas提供了一系列操作方法帮助我们轻松完成这些操作。
'''
# 小节注释
'''
一般使用特殊类型NaN代表缺失值，可以用NumPy定义为np.NaN或np.nan。
在Pandas 1.0以后的版本中，实验性地使用标量pd.NA来代表。

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print('# 原始数据')
df = pd.DataFrame({
    'A':['a1','a1','a2','a2'],
    'B':['b1','b2',None,'b2'],
    'C':[1,2,3,4],
    'D':[5,6,None,8],
    'E':[5,None,7,8],
    })

print(df)
#     A     B  C    D    E
# 0  a1    b1  1  5.0  5.0
# 1  a1    b2  2  6.0  NaN
# 2  a2  None  3  NaN  7.0
# 3  a2    b2  4  8.0  8.0

'''以上数据中，2B、2D、1E为缺失值。如果想把正负无穷也当作缺失值，可以通过以下全局配置来设定：'''

# 将无穷值设置为缺失值
# print()
# pd.options.mode.use_inf_as_na = True
# pd.options.mode.use_inf_as_na = True
# print(df)

print()
print('------------------------------------------------------------')
print('\t10.1.2 缺失值判断')

'''
df.isna()及其别名df.isnull()是Pandas中判断缺失值的主要方法。
对整个数据进行缺失值判断，True为缺失：
'''

# 检测缺失值
print(df.isna())
#        A      B      C      D      E
# 0  False  False  False  False  False
# 1  False  False  False  False   True
# 2  False   True  False   True  False
# 3  False  False  False  False  False

print()
print('# 检测指定列的缺失值')
print(df.D.isna())
# 0    False
# 1    False
# 2     True
# 3    False
# Name: D, dtype: bool

print()
print('# 检测非缺失值')
print(df.notna())
#       A      B     C      D      E
# 0  True   True  True   True   True
# 1  True   True  True   True  False
# 2  True  False  True  False   True
# 3  True   True  True   True   True

print()
print('# 检测某列非缺失值')
print(df.D.notna())
# 0     True
# 1     True
# 2    False
# 3     True
# Name: D, dtype: bool

print()
print('------------------------------------------------------------')
print('\t10.1.3 缺失值统计')

'''如果需要统计一个数据中有多少个缺失值，可利用sum计算，计算时将False当作0、将True当作1的特性：'''

print('# 布尔值的求和')
print(pd.Series([True,True,False]).sum()) # 2

# 如果需要计算数据中的缺失值情况，可以使用以下方法：
print()
print('# 每列有多少个缺失值')
print(df.isnull().sum()) # isna() = isnull()
# A    0
# B    1
# C    0
# D    1
# E    1

print()
print('# 每行有多少个缺失值')
print(df.isnull().sum(1))
# 0    0
# 1    1
# 2    2
# 3    0

print()
print('# 总共有多少个缺失值')
print(df.isna().sum().sum()) # 3

print()
print('------------------------------------------------------------')
print('\t10.1.4 缺失值筛选')

print(df)
print()
print(df.isna().any(axis=1))
# 0    False
# 1     True
# 2     True
# 3    False
# dtype: bool

print()
print('# 有缺失值的行')
print(df.loc[df.isna().any(axis=1)]) # 必须加参数axis=1
#     A     B  C    D    E
# 1  a1    b2  2  6.0  NaN
# 2  a2  None  3  NaN  7.0
print('# 有缺失值的列')
print(df.loc[:,df.isna().any()])
#       B    D    E
# 0    b1  5.0  5.0
# 1    b2  6.0  NaN
# 2  None  NaN  7.0
# 3    b2  8.0  8.0

'''如果要查询没有缺失值的行和列，可以对表达式取反'''
print()
print('# 没有缺失值的行')
print(df.loc[~(df.isna().any(axis=1))])
#     A   B  C    D    E
# 0  a1  b1  1  5.0  5.0
# 3  a2  b2  4  8.0  8.0
print('# 没有缺失值的列')
print(df.loc[:,~(df.isna().any())])
#     A  C
# 0  a1  1
# 1  a1  2
# 2  a2  3
# 3  a2  4

print()
print('------------------------------------------------------------')
print('\t10.1.5 NA标量')
'''
Pandas 1.0以后的版本中引入了一个专门表示缺失值的标量pd.NA，
它代表空整数、空布尔、空字符，这个功能目前处于实验阶段。
pd.NA的目标是提供一个“缺失值”指示器，该指示器可以在各种数据类型中一致使用
（而不是np.nan、None或pd.NaT，具体取决于数据类型）。
'''

s = pd.Series([1,2,None,4],dtype='Int64')
print(s)
print(s[2])
# 0       1
# 1       2
# 2    <NA>
# 3       4
# dtype: Int64
# <NA>

print(s[2] is pd.NA) # True

print(pd.isna(pd.NA)) # True

# 以下是pd.NA参与运算的一些逻辑示例：

print()
print('# 运算')
# 加法
print(pd.NA + 1) # <NA>
# 乘法
print('a' * pd.NA) # <NA>
print('a' + pd.NA) # 同上
print(pd.NA ** 0) # 1
print(1 ** pd.NA) # 1
# 其它示例
print(pd.NA == 1) # <NA>
print(pd.NA == pd.NA) # <NA>
print(pd.NA < 2.5) # <NA>

print()
print('------------------------------------------------------------')
print('\t10.1.6 时间数据中的缺失值')
'''
对于时间数据中的缺失值，Pandas提供了一个NaT来表示，并且NaT和NaN是兼容的：
'''
print('# 时间数据中的缺失值')
s = pd.Series([pd.Timestamp('20200101'),None,pd.Timestamp('20200103')])
# pd.Timestamp('20200103')])
print(s)
# 0   2020-01-01
# 1          NaT
# 2   2020-01-03
# dtype: datetime64[ns]

print()
print('------------------------------------------------------------')
print('\t10.1.7 整型数据中的缺失值')

'''由于NaN是浮点型，因此缺少一个整数的列可以转换为整型。'''

# print(df)
print(type(df.at[2,'D']))
# <class 'numpy.float64'>
print(pd.Series([1,2,np.nan,4],dtype=pd.Int64Dtype()))
# 0       1
# 1       2
# 2    <NA>
# 3       4
# dtype: Int64

print()
print('------------------------------------------------------------')
print('\t10.1.8 插入缺失值')
'''如同修改数据一样，我们可以通过以下方式将缺失值插入数据中：'''
print('# 修改为缺失值')
df.loc[0] = None
df.loc[1] = np.nan
df.A = pd.NA
print(df)
#       A     B    C    D    E
# 0  <NA>  None  NaN  NaN  NaN
# 1  <NA>   NaN  NaN  NaN  NaN
# 2  <NA>  None  3.0  NaN  7.0
# 3  <NA>    b2  4.0  8.0  8.0

'''
10.1.9 小结
本节我们介绍了数据中的None、np.nan和pd.NA，它们都是缺失值的类型，对缺失值的识别和判定非常关键。
只有有效识别出数据的缺失部分，我们才能对这些缺失值进行处理。
'''



print()
print('------------------------------------------------------------')
print('第10章 Pandas数据清洗')
print('\t10.2 缺失值的操作')
print('\t10.2.1 缺失值填充')
print()
# update20240517
'''
对于缺失值，我们通常会根据业务需要进行修补，但对于缺失严重的数据，会直接将其删除。
本节将介绍如何对缺失值进行一些常规的操作。
'''
# 小节注释
'''
对于缺失值，我们常用的一个办法是利用一定的算法去填充它。
这样虽然不是特别准确，但对于较大的数据来说，不会对结果产生太大影响。
df.fillna(x)可以将缺失值填充为指定的值：

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print('# 原始数据')
df = pd.DataFrame({
    'A':['a1','a1','a2','a2'],
    'B':['b1','b2',None,'b2'],
    'C':[1,2,3,4],
    'D':[5,6,None,8],
    'E':[5,None,7,8],
    })

print(df)
#     A     B  C    D    E
# 0  a1    b1  1  5.0  5.0
# 1  a1    b2  2  6.0  NaN
# 2  a2  None  3  NaN  7.0
# 3  a2    b2  4  8.0  8.0

'''以上数据中，2B、2D、1E为缺失值。如果想把正负无穷也当作缺失值，可以通过以下全局配置来设定：'''

print('# 将缺失值填充为0')
print(df.fillna(0))
#     A   B  C    D    E
# 0  a1  b1  1  5.0  5.0
# 1  a1  b2  2  6.0  0.0
# 2  a2   0  3  0.0  7.0
# 3  a2  b2  4  8.0  8.0

print('# 常用的方法还有以下几个：')
'''
# 填充为 0
df.fillna(0)
# 填充为指定字符
df.fillna('missing')
df.fillna('暂无')
df.fillna('待补充')
# 指定字段填充
df.one.fillna('暂无')
# 指定字段填充
df.one.fillna(0, inplace=True)
# 只替换第一个
df.fillna(0, limit=1)
# 将不同列的缺失值替换为不同的值
values = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
df.fillna(value=values)

有时候我们不能填入固定值，而要按照一定的方法填充。
df.fillna()提供了一个method参数，可以指定以下几个方法。
pad / ffill：向前填充，使用前一个有效值填充，
df.fillna(method='ffill')可以简写为df.ffill()。
bfill / backfill：向后填充，使用后一个有效值填充，
df.fillna(method='bfill')可以简写为df.bfill()。


'''
print(df.fillna(method='pad'))
#     A   B  C    D    E
# 0  a1  b1  1  5.0  5.0
# 1  a1  b2  2  6.0  5.0
# 2  a2  b2  3  6.0  7.0
# 3  a2  b2  4  8.0  8.0

'''除了取前后值，还可以取经过计算得到的值，比如常用的平均值填充法：'''

print()
print('# 填充列的平均值')

dff = df.fillna(0).drop(columns=['A','B']).mean()
print(dff)
# C    2.50
# D    4.75
# E    5.00
# dtype: float64

print(df.fillna(dff))
#     A     B  C     D    E
# 0  a1    b1  1  5.00  5.0
# 1  a1    b2  2  6.00  5.0
# 2  a2  None  3  4.75  7.0
# 3  a2    b2  4  8.00  8.0
print(df.fillna(dff.mean()))
#     A         B  C         D         E
# 0  a1        b1  1  5.000000  5.000000
# 1  a1        b2  2  6.000000  4.083333
# 2  a2  4.083333  3  4.083333  7.000000
# 3  a2        b2  4  8.000000  8.000000

print()
print('# 对指定列填充平均值')
# df.loc[:,['B','D']] = df.loc[:,['B','D']].fillna(dff.mean())
# print(df)
#     A         B  C         D    E
# 0  a1        b1  1  5.000000  5.0
# 1  a1        b2  2  6.000000  NaN
# 2  a2  4.083333  3  4.083333  7.0
# 3  a2        b2  4  8.000000  8.0

'''缺失值填充的另一个思路是使用替换方法df.replace()：'''
print()
print('# 将指定列的空值替换成指定值')
print(df.replace({'E':{np.nan:100}}))

print()
print('------------------------------------------------------------')
print('\t10.2.2 插值填充')
'''
插值（interpolate）是离散函数拟合的重要方法，利用它可根据函数在有限个点处的取值状况，
估算出函数在其他点处的近似值。
Series和DataFrame对象都有interpolate()方法，默认情况下，该方法在缺失值处执行线性插值。
它利用数学方法来估计缺失点的值，对于较大的数据非常有用。
'''

s = pd.Series([0,1,np.nan,3])

# 插值填充
print(s.interpolate())
# 0    0.0
# 1    1.0
# 2    2.0
# 3    3.0
# dtype: float64

'''其中默认method ='linear'，即使用线性方法，认为数据呈一条直线。method方法指定的是插值的算法。'''

'''
如果你的数据增长速率越来越快，可以选择method='quadratic'二次插值；
如果数据集呈现出累计分布的样子，推荐选择method='pchip'；
如果需要填补默认值，以平滑绘图为目标，推荐选择method='akima'。

这些都需要你的环境中安装了SciPy库。
'''

print()
print(s.interpolate(method='akima'))

print()
print('------------------------------------------------------------')
print('\t10.2.3 缺失值删除')

print(df)
print('# 删除有缺失值的行')
print(df.dropna())
#     A   B  C    D    E
# 0  a1  b1  1  5.0  5.0
# 3  a2  b2  4  8.0  8.0

print('# 删除有缺失值的列')
print(df.dropna(axis=1)) # axis='columns'
#     A  C
# 0  a1  1
# 1  a1  2
# 2  a2  3
# 3  a2  4

'''
# 删除所有有缺失值的行
df.dropna()
# 删除所有有缺失值的列
df.dropna(axis='columns')
df.dropna(axis=1)
# 删除所有值都缺失的行
df.dropna(how='all')
# 删除至少有两个缺失值的行 / 保留非缺失值大于等于2的行
df.dropna(thresh=2)
# 指定判断缺失值的列范围
df.dropna(subset=['name', 'born'])
# 使删除的结果生效
df.dropna(inplace=True)
# 指定列的缺失值删除
df.col.dropna()

'''
print(df.dropna(thresh=4)) # 保留非缺失值大于等于4的行
#     A   B  C    D    E
# 0  a1  b1  1  5.0  5.0
# 1  a1  b2  2  6.0  NaN
# 3  a2  b2  4  8.0  8.0

print()
# print(df)
print(df.dropna(subset=['E'])) # 指定E列有空值行时 才删除该行
#     A     B  C    D    E
# 0  a1    b1  1  5.0  5.0
# 2  a2  None  3  NaN  7.0
# 3  a2    b2  4  8.0  8.0
# print(df.dropna(subset=['D','E']))

print()
print('# 指定列的缺失值删除')
print(df.E.dropna())
# 0    5.0
# 2    7.0
# 3    8.0
# Name: E, dtype: float64


print()
print('------------------------------------------------------------')
print('\t10.2.4 缺失值参与计算')
# 对所有列求和

print(df.drop(columns='B'))
df1 = df.drop(columns='B') # 如果B列存在 运算会报错！
print(df1.sum())
# A    a1a1a2a2
# C          10
# D        19.0
# E        20.0
# dtype: object

'''加法会忽略缺失值，或者将其按0处理，再试试累加：'''
print(df.D.cumsum())
# 0     5.0
# 1    11.0
# 2     NaN
# 3    19.0
# Name: D, dtype: float64

'''cumsum()和cumprod()会忽略NA值，但值会保留在序列中，可以使用skipna=False跳过有缺失值的计算并返回缺失值：'''

print()
# print(df.D.cumprod()) # 累乘
print(df.D.cumsum(skipna=False)) # 累加，跳过空值
# 0     5.0
# 1    11.0
# 2     NaN
# 3     NaN
# Name: D, dtype: float64

print()
print('# 缺失值不计数')
print(df.count())
# A    4
# B    3
# C    4
# D    3
# E    3
# dtype: int64

'''
再看看缺失值在做聚合分组操作时的情况，如果聚合分组的列里有空值，则会自动忽略这些值（当它不存在）：
'''

print()
print(df.groupby('B').sum())
#        A  C     D    E
# B
# b1    a1  1   5.0  5.0
# b2  a1a2  6  14.0  8.0

print('# 聚合计入缺失值')
print(df.groupby('B',dropna=False).sum())
#         A  C     D    E
# B
# b1     a1  1   5.0  5.0
# b2   a1a2  6  14.0  8.0
# NaN    a2  3   0.0  7.0

# df.drop(columns='A',inplace=True)
# print(df)

'''
10.2.5 小结
本节介绍了缺失值的填充方法。
如果数据质量有瑕疵，在不影响分析结果的前提下，可以用固定值填充、插值填充。
对于质量较差的数据可以直接丢弃。
'''

print()
print('------------------------------------------------------------')
print('第10章 Pandas数据清洗')
print('\t10.3 数据替换')
print('\t10.3.1 指定值替换')
print()
# update20240520
'''
Pandas中数据替换的方法包含数值、文本、缺失值等替换，
经常用于数据清洗与整理、枚举转换、数据修正等情形。
Series和DataFrame中的replace()都提供了一种高效而灵活的方法。
'''
# 小节注释
'''
以下是在Series中将0替换为5：

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print('# 原始数据')
# 以下是在Series中将0替换为5：
ser = pd.Series([0., 1., 2., 3., 4.])
print(ser)
print(ser.replace(0, 5))
# 0    5.0
# 1    1.0
# 2    2.0
# 3    3.0
# 4    4.0
# dtype: float64

print()
# 也可以批量替换：
print('# 一一对应进行替换')
# print(ser.replace([0,1,2,3,4],[4,3,2,1,0]))
# # 用字典映射对应替换值
# ser.replace({0: 10, 1: 100})
# # 将a列的0、b列中的5替换为100
# df.replace({'a': 0, 'b': 5}, 100)
# # 指定列里的替换规则
# df.replace({'a': {0: 100, 4: 400}})

print()
print('------------------------------------------------------------')
print('\t10.3.2 使用替换方式')

'''除了给定指定值进行替换，我们还可以指定一些替换的方法：'''
# 将 1，2，3 替换为它们前一个值
print(ser.replace([1,2,3],method='pad')) # ffill是它同义词
# 将 1，2，3 替换为它们后一个值
print(ser.replace([1,2,3],method='bfill'))

'''
如果指定的要替换的值不存在，则不起作用，也不会报错。以上的替换也适用于字符类型数据。
'''

print()
print('------------------------------------------------------------')
print('\t10.3.3 字符替换')

'''
替换方法默认没有开启正则匹配模式，直接按原字符匹配替换，
如果遇到字符规则比较复杂的内容，可使用正则表达式进行匹配：
'''
df = pd.DataFrame({'A': ['bat', 'foo', 'bar', 'baz', 'foobar'],'B': ['bat', 'foo', 'bar', 'xyz', 'foobar']})
print(df)
# 把bat替换为new，不使用正则表达式
print(df.replace(to_replace='bat',value='new'))
#        A      B
# 0     new     new
# 1     foo     foo
# 2     bar     bar
# 3     baz     xyz
# 4  foobar  foobar

print()
print('# 利用正则表达式将ba开头的值替换为new')
# df.replace(to_replace=r'^ba.$', value='new', regex=True)
print(df.replace(to_replace=r'^ba.$',value='new',regex=True))

print('# 如果多列规则不一，可以按以下格式对应传入')
print(df.replace({'A':r'^ba.$'},{'A':'new'},regex=True))

# 测试不同列 不同修改方式，报错！
# print(df.replace({'A':r'^ba.$'},{'A':'new'},{'B':r'^ba.$'},{'B':'wocao'},regex=True))
print()
# 多个规则均替换为同样的值
# df.replace(regex=[r'^ba.$', 'foo'], value='new')
# 多个正则及对应的替换内容
# df.replace(regex={r'^ba.$': 'new', 'foo': 'xyz'})

print()
print('------------------------------------------------------------')
print('\t10.3.4 缺失值替换')

'''替换可以处理缺失值相关的问题，例如我们可以先将无效的值替换为nan，再进行缺失值处理：'''

d = {'a': list(range(4)),
     'b': list('ab..'),
     'c': ['a', 'b', np.nan, 'd']
     }

df = pd.DataFrame(d)
print(df)

print('# 将.替换为NaN')
# print(df.replace('.',np.nan))

print('# 使用正则表达式，将空格等替换为NaN')
print(df.replace(r'\s*\.\s*',np.nan,regex=True)) # 结果同上  将.替换为NaN
#    a    b    c
# 0  0    a    a
# 1  1    b    b
# 2  2  NaN  NaN
# 3  3  NaN    d

print()
# 对应替换，a换b，点换NaN
# print(df)
print(df.replace(['a', '.'], ['b', np.nan]))
#    a    b    c
# 0  0    b    b
# 1  1    b    b
# 2  2  NaN  NaN
# 3  3  NaN    d
print()
print(df.replace([r'\.',r'(a)'],['dot',r'\1stuff'],regex=True)) # 点换dot，a换astuff
#    a       b       c
# 0  0  astuff  astuff
# 1  1       b       b
# 2  2     dot     NaN
# 3  3     dot       d


'''
# b中的点要替换，将b替换为NaN，可以多列
df.replace({'b': '.'}, {'b': np.nan})
# 使用正则表达式
df.replace({'b': r'\s*\.\s*'}, {'b': np.nan}, regex=True)
# b列的b值换为空
df.replace({'b': {'b': r''}}, regex=True)
# b列的点、空格等替换为NaN
df.replace(regex={'b': {r'\s*\.\s*': np.nan}})
# 在b列的点后加ty，即.ty
df.replace({'b': r'\s*(\.)\s*'},{'b': r'\1ty'},regex=True)
# 多个正则规则
df.replace([r'\s*\.\s*', r'a|b'], np.nan, regex=True)
# 用参数名传参
df.replace(regex=[r'\s*\.\s*', r'a|b'], value=np.nan)

'''

print()
print('------------------------------------------------------------')
print('\t10.3.5 数字替换')

'''将相关数字替换为缺失值：'''
df = pd.DataFrame(np.random.randn(10,2))
# 生成一个布尔索引数组，长度与df的行数相同
mask = np.random.rand(df.shape[0]) > 0.5
print(mask)
df.loc[mask] = 1.5
print(df)
# print(df)

print(df.replace(1.5,None))

'''个人感觉没啥意义'''

print()
print('------------------------------------------------------------')
print('\t10.3.5 数据修剪')
'''
对于数据中存在的极端值，过大或者过小，可以使用df.clip(lower,upper)来修剪。
当数据大于upper时使用upper的值，小于lower时用lower的值，这和numpy.clip方法一样。
'''
# 包含极端值的数据
df = pd.DataFrame({'a':[-1,2,5],'b':[6,1,-3]})
# print(df)
print('# 修剪成最大为3，最小为0')
print(df.clip(0,3))
#    a  b
# 0  0  3
# 1  2  1
# 2  3  0

# 按列指定下限和上限阈值进行修剪，如下例中数据按同索引位的c值和c对应值+1进行修剪
print()
c = pd.Series([-1,1,3])
print(df.clip(c,c+1,axis=0))
#    a  b
# 0 -1  0
# 1  2  1
# 2  4  3

'''
10.3.7 小结
替换数据是数据清洗的一项很普遍的操作，同时也是修补数据的一种有效方法。
df.replace()方法功能强大，在本节中，我们了解了它实现定值替换、定列替换、广播替换、运算替换等功能。
'''

print()
print('------------------------------------------------------------')
print('第10章 Pandas数据清洗')
print('\t10.4 重复值及删除数据')
print('\t10.4.1 重复值识别')
print()
# update20240523
'''
数据在收集、处理过程中会产生重复值，包括行和列，既有完全重复，又有部分字段重复。
重复的数据会影响数据的质量，特别是在它们参与统计计算时。
本节介绍Pandas如何识别重复值、删除重复值，以及如何删除指定的数据。
'''
# 小节注释
'''
df.duplicated()是Pandas用来检测重复值的方法，语法为：：
# 检测重复值语法
df.duplicated(subset=None, keep='first')

它可以返回表示重复行的布尔值序列，默认为一行的所有内容，
subset可以指定列。keep参数用来确定要标记的重复值，可选的值有：
first：将除第一次出现的重复值标记为True，默认。
last：将除最后一次出现的重复值标记为True。
False：将所有重复值标记为True。
▶ 以下是一些具体的使用方法举例：
'''
df = pd.DataFrame({'A':['x','x','z'],
                   'B':['x','x','x'],
                   'C':[1,1,2],
                   })

print('# 全行检测，除第一次出现的外，重复的为True')

print(df.duplicated())
# 0    False
# 1     True
# 2    False
# dtype: bool

print('# 除最后一次出现的外，重复的为True')
print(df.duplicated(keep='last'))
# 0     True
# 1    False
# 2    False
# dtype: bool

print('# 所有重复的都为True')
print(df.duplicated(keep=False))
# 0     True
# 1     True
# 2    False
# dtype: bool
print('# 指定列检测')
print(df.duplicated(subset=['B'],keep=False))
# 0    True
# 1    True
# 2    True
# dtype: bool

'''重复值的检测可用于数据的查询和筛选，示例如下：'''
print(df[df.duplicated()])
#    A  B  C
# 1  x  x  1

print()
print('------------------------------------------------------------')
print('\t10.4.2 删除重复值')

'''
删除重复值的语法如下：
     df.drop_duplicates(subset=None,
                         keep='first',
                         inplace=False,
                         ignore_index=False)

参数说明如下。
subset：指定的标签或标签序列，仅删除这些列重复值，默认情况为所有列。
keep：确定要保留的重复值，有以下可选项。
     first：保留第一次出现的重复值，默认。
     last：保留最后一次出现的重复值。
     False：删除所有重复值。
inplace：是否生效。
ignore_index：如果为True，则重新分配自然索引（0，1，…，n–1）。

'''

print(df)
print('# 删除重复行')
print(df.drop_duplicates())
#    A  B  C
# 0  x  x  1
# 2  z  x  2

print('# 删除指定列')
print(df.drop_duplicates(subset=['A'])) # 结果同上
#    A  B  C
# 0  x  x  1
# 2  z  x  2

print('# 保留最后一个')
print(df.drop_duplicates(subset=['A'],keep='last'))
#    A  B  C
# 1  x  x  1
# 2  z  x  2

print('# 删除全部重复值')
print(df.drop_duplicates(subset=['A'],keep=False))
#    A  B  C
# 2  z  x  2

print()
print('------------------------------------------------------------')
print('\t10.4.3 删除数据')
'''
df.drop()通过指定标签名称和相应的轴，或直接给定索引或列名称来删除行或列。
使用多层索引时，可以通过指定级别来删除不同级别上的标签。

# 语法
df.drop(labels=None, axis=0,
          index=None, columns=None,
          level=None, inplace=False,
          errors='raise')

参数说明如下：
     labels：要删除的列或者行，如果要删除多个，传入列表。
     axis：轴的方向，0为行，1为列，默认为0。
     index：指定的一行或多行。
     column：指定的一列或多列。
     level：索引层级，将删除此层级。
     inplace：布尔值，是否生效。
     errors：ignore或者raise，默认为raise，如果为ignore，则容忍错误，仅删除现有标签。
'''

print(df)
print('# 删除指定行')
print(df.drop([0,1]))
#    A  B  C
# 2  z  x  2

print('# 删除指定列')
print(df.drop(['B','C'],axis=1))
print(df.drop(columns=['B','C'])) # 结果同上
#    A
# 0  x
# 1  x
# 2  z

'''
10.4.4 小结
本节介绍了三个重要的数据清洗工具：
df.duplicated()能够识别出重复值，返回一个布尔序列，用于查询和筛选重复值；
df.drop_duplicates()可以直接删除指定的重复数据；
df.drop()能够灵活地按行或列删除指定的数据，可以通过计算得到异常值所在的列和行再执行删除。

'''

print()
print('------------------------------------------------------------')
print('第10章 Pandas数据清洗')
print('\t10.5 NumPy格式转换')
print('\t10.5.1 转换方法')
print()
# update20240523
'''
2.5节介绍过可以将一个NumPy数据转换为DataFrame或者Series数据。
在特征处理和数据建模中，很多库使用的是NumPy中的ndarray数据类型，
Pandas在对数据进行处理后，要将其应用到上述场景，就需要将类型转为NumPy的ndarray。
本节就来介绍一下如何将Pandas的数据类型转换为NumPy的类型。
'''
# 小节注释
'''
Pandas 0.24.0引入了两种从Pandas对象中获取NumPy数组的新方
法。
ds.to_numpy()：可以用在Index、Series和DataFrame对象；
s.array：为PandasArray，用在Index和Series，它封装了numpy.ndarray接口。

有了以上方法，不再推荐使用Pandas的values和as_matrix()。
上述这两个函数旨在提高API的一致性，是Pandas官方未来支持的方向，
values和as_matrix()虽然在近期的版本中不会被弃用，
但可能会在将来的某个版本中被取消，因此官方建议用户尽快迁移到较新的API。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.DataFrame({'A':['x','x','z'],
                   'B':['x','x','x'],
                   'C':[1,1,2],
                   })

print()
print('------------------------------------------------------------')
print('\t10.5.2 DataFrame转为ndarray')
'''df.values和df.to_numpy()返回的是一个array类型：'''

df = pd.read_excel(team_file)
print(df)

print(df.values) # 不推荐
print(df.to_numpy()) # 推荐，结果同上
# [['Liver' 'E' 89 21 24 64]
#  ['Arry' 'C' 36 37 37 57]
#  ['Ack' 'A' 57 60 18 84]
#  ['Eorge' 'C' 93 96 71 78]
#  ['Oah' 'D' 65 49 61 86]
#  ['Rick' 'B' 100 99 97 100]]

print(type(df.to_numpy()))
# <class 'numpy.ndarray'>

print(df.to_numpy().dtype) # object

print('# 转换指定的列')
print(df[['name','Q1']].to_numpy())
# [['Liver' 89]
#  ['Arry' 36]
#  ['Ack' 57]
#  ['Eorge' 93]
#  ['Oah' 65]
#  ['Rick' 100]]

print()
print('------------------------------------------------------------')
print('\t10.5.3 Series转为ndarray')

'''对Series使用s.values和s.to_numpy()返回的是一个array类型：'''
# df.Q1.values # 不推荐
# df.Q1.to_numpy()
print(df.Q1.to_numpy())
# [ 89  36  57  93  65 100]
print(type(df.Q1.to_numpy()))
# <class 'numpy.ndarray'>
print(df.Q1.to_numpy().dtype)
# int64
print(type(df.Q1.to_numpy().dtype))
# <class 'numpy.dtype[int64]'>
print(df.Q1.array)
# Length: 6, dtype: int64
print(type(df.Q1.array))
# <class 'pandas.core.arrays.numpy_.PandasArray'>

print()
print('------------------------------------------------------------')
print('\t10.5.4 df.to_records()')

'''可以使用to_records()方法，但是如果数据类型不是你想要的，则必须对它们进行一些处理。'''

# 转为NumPy record array
print(df.to_records())
# [(0, 'Liver', 'E',  89, 21, 24,  64) (1, 'Arry', 'C',  36, 37, 37,  57)
#  (2, 'Ack', 'A',  57, 60, 18,  84) (3, 'Eorge', 'C',  93, 96, 71,  78)
#  (4, 'Oah', 'D',  65, 49, 61,  86) (5, 'Rick', 'B', 100, 99, 97, 100)]
print(type(df.to_records()))
# <class 'numpy.recarray'>
print('# 转为array')
print(np.array(df.to_records())) # 看起来结果和上述一致，但是数据类型不同
# [(0, 'Liver', 'E',  89, 21, 24,  64) (1, 'Arry', 'C',  36, 37, 37,  57)
#  (2, 'Ack', 'A',  57, 60, 18,  84) (3, 'Eorge', 'C',  93, 96, 71,  78)
#  (4, 'Oah', 'D',  65, 49, 61,  86) (5, 'Rick', 'B', 100, 99, 97, 100)]
print(type(np.array(df.to_records()))) # <class 'numpy.ndarray'>

'''上例中，to_records()将数据转为了NumPy的record array类型，然后再用NumPy的np.array读取一下，转为array类型。'''
print(df.to_records()[0])
print(np.array(df.to_records())[0]) # 结果同上
# (0, 'Liver', 'E', 89, 21, 24, 64)

print()
print('------------------------------------------------------------')
print('\t10.5.5 np.array读取')
'''可以用np.array直接读取DataFrame或者Series数据，最终也会转换为array类型：'''

print(np.array(df)) # Dataframe转
# [['Liver' 'E' 89 21 24 64]
#  ['Arry' 'C' 36 37 37 57]
#  ['Ack' 'A' 57 60 18 84]
#  ['Eorge' 'C' 93 96 71 78]
#  ['Oah' 'D' 65 49 61 86]
#  ['Rick' 'B' 100 99 97 100]]
print(np.array(df.Q1)) # 直接转
# [ 89  36  57  93  65 100]
print(np.array(df.Q1.array)) # PandasArray转  结果同上
print(np.array(df.to_records().view(type=np.matrix))) # 转为矩阵
# [[(0, 'Liver', 'E',  89, 21, 24,  64) (1, 'Arry', 'C',  36, 37, 37,  57)
#   (2, 'Ack', 'A',  57, 60, 18,  84) (3, 'Eorge', 'C',  93, 96, 71,  78)
#   (4, 'Oah', 'D',  65, 49, 61,  86) (5, 'Rick', 'B', 100, 99, 97, 100)]]

'''
10.5.6 小结
本节介绍了如何将Pandas的两大数据类型DataFrame和Series转为NumPy的格式，推荐使用to_numpy()方法。
关于NumPy的更多操作可以访问笔者的NumPy在线教程，地址为https://www.gairuo.com/p/numpytutorial。

10.6 本章小结
数据清洗是我们获取到数据集后要做的第一件事，处理缺失数据和缺失值是数据清洗中最棘手的部分。
只有保证数据的高质量才有可能得出高质量的分析结论，
一些数据建模和机器学习的场景对数据质量有严格的要求，甚至不允许有缺失值。

本章介绍了在Pandas中缺失值的表示方法以及如何找到缺失值，
重复值的筛选方法以及如何对它们进行删除、替换和填充等操作。
完成这些工作，将得到一个高质量的数据集，为下一步数据分析做好准备。
'''