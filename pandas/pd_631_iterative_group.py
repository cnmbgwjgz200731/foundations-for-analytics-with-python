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
print('\t6.3 分组对象的操作')
print('\t6.3.2 迭代分组')
print()
# update20240423
'''
上一节完成了分组对象的创建，分组对象包含数据的分组情况，
接下来就来对分组对象进行操作，获取其相关信息，为最后的数据聚合统计打好基础。
'''
# 小节注释
'''
分组对象的groups方法会生成一个字典（其实是Pandas定义的PrettyDict），
这个字典包含分组的名称和分组的内容索引列表，然后我们可以使用字典的.keys()方法取出分组名称：
▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

# 分组，为了方便案例介绍，删去name列，分组后全为数字
grouped = df.drop('name',axis=1).groupby('team')
# print(grouped.sum())
#        Q1   Q2   Q3   Q4
# team
# A      57   60   18   84
# B     100   99   97  100
# C     129  133  108  135
# D      65   49   61   86
# E      89   21   24   64

print()
print('# 查看分组内容')
print(df.groupby('team').groups)
# {'A': [2], 'B': [5], 'C': [1, 3], 'D': [4], 'E': [0]}  列表值为行索引值

print()
print('# 查看分组名')
print(df.groupby('team').groups.keys())
# dict_keys(['A', 'B', 'C', 'D', 'E'])
# print(df.groupby('team') .groups.items())

print()
print('# 多层索引，可以使用元组（）')
grouped2 = df.groupby(['team',df.name.str[0]])
# print(grouped2.get_group(('B','A'))) # 报错 因为没有这种组合 所以会报错！
print(grouped2.get_group(('B','R')))
#    name team   Q1  Q2  Q3   Q4
# 5  Rick    B  100  99  97  100

print()
print('# 获取分组字典数据')
'''grouped.indices返回一个字典，其键为组名，值为本组索引的array格式，可以实现对单分组数据的选取：'''
print(grouped.indices)
# {'A': array([2], dtype=int64), 'B': array([5], dtype=int64), 'C': array([1, 3], dtype=int64), 'D': array([4], dtype=int64), 'E': array([0], dtype=int64)}
print()
print('# 选择A组')
print(grouped.indices['A'])
# [2]


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.3 分组对象的操作')
print('\t6.3.2 迭代分组')
print()
# update20240423
# 小节注释
'''
我们对分组对象grouped进行迭代，看每个元素是什么数据类型：
▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

# 分组，为了方便案例介绍，删去name列，分组后全为数字
grouped = df.drop('name',axis=1).groupby('team')
# print(grouped.sum())
#        Q1   Q2   Q3   Q4
# team
# A      57   60   18   84
# B     100   99   97  100
# C     129  133  108  135
# D      65   49   61   86
# E      89   21   24   64

print()
print('# 迭代')

for g in grouped:
    print(type(g))

# <class 'tuple'>
# <class 'tuple'>
# <class 'tuple'>
# <class 'tuple'>
# <class 'tuple'>

print('# 迭代元素的数据类型')
for name,group in grouped:
    print(type(name))
    print(type(group))

# <class 'str'>
# <class 'pandas.core.frame.DataFrame'>
# <class 'str'>
# <class 'pandas.core.frame.DataFrame'>
# <class 'str'>
# <class 'pandas.core.frame.DataFrame'>
# <class 'str'>
# <class 'pandas.core.frame.DataFrame'>
# <class 'str'>
# <class 'pandas.core.frame.DataFrame'>

# for name,group in grouped:
#     print(name)
#     print(group)
'''因此我们可以通过以上方式迭代分组对象。'''

for name,group in grouped:
    print(group)

#   team  Q1  Q2  Q3  Q4
# 2    A  57  60  18  84
#   team   Q1  Q2  Q3   Q4
# 5    B  100  99  97  100
#   team  Q1  Q2  Q3  Q4
# 1    C  36  37  37  57
# 3    C  93  96  71  78
#   team  Q1  Q2  Q3  Q4
# 4    D  65  49  61  86
#   team  Q1  Q2  Q3  Q4
# 0    E  89  21  24  64


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.3 分组对象的操作')
print('\t6.3.3 选择列')
print()
# update20240423

# 小节注释
'''
我们对分组对象grouped进行迭代，看每个元素是什么数据类型：
▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

# 分组，为了方便案例介绍，删去name列，分组后全为数字
grouped = df.drop('name',axis=1).groupby('team')
# print(grouped.sum())
#        Q1   Q2   Q3   Q4
# team
# A      57   60   18   84
# B     100   99   97  100
# C     129  133  108  135
# D      65   49   61   86
# E      89   21   24   64

print()
print('# 选择分组后的某一列')
print(grouped.Q1)
# <pandas.core.groupby.generic.SeriesGroupBy object at 0x000001A66458BB50>
print(grouped['Q1']) # 结果同上

print()
print('# 选择多列')
print(grouped[['Q1','Q2']])
# <pandas.core.groupby.generic.DataFrameGroupBy object at 0x0000024301F8B0D0>

print('# 对多列进行聚合运算')
print(grouped[['Q1','Q2']].sum())
#        Q1   Q2
# team
# A      57   60
# B     100   99
# C     129  133
# D      65   49
# E      89   21

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.3 分组对象的操作')
print('\t6.3.4 应用函数apply()')
print()
# update20240423
# 小节注释
'''
分组对象使用apply()调用一个函数，传入的是DataFrame，
返回一个经过函数计算后的DataFrame、Series或标量，然后再把数据组合。
▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 将所有的元素乘以2')
print(df.groupby('team').apply(lambda x:x * 2))
#               name team   Q1   Q2   Q3   Q4
# team
# A    2      AckAck   AA  114  120   36  168
# B    5    RickRick   BB  200  198  194  200
# C    1    ArryArry   CC   72   74   74  114
#      3  EorgeEorge   CC  186  192  142  156
# D    4      OahOah   DD  130   98  122  172
# E    0  LiverLiver   EE  178   42   48  128

# print(df.groupby('team',as_index=False).apply(lambda x:x * 2))
#            name team   Q1   Q2   Q3   Q4
# 0 2      AckAck   AA  114  120   36  168
# 1 5    RickRick   BB  200  198  194  200
# 2 1    ArryArry   CC   72   74   74  114
#   3  EorgeEorge   CC  186  192  142  156
# 3 4      OahOah   DD  130   98  122  172
# 4 0  LiverLiver   EE  178   42   48  128

print()
print('# 按分组将一列输出位列表')
print(df.groupby('team').apply(lambda x: x['name'].to_list()))

# team
# A            [Ack]
# B           [Rick]
# C    [Arry, Eorge]
# D            [Oah]
# E          [Liver]
# dtype: object

print()
print('# 查看某个组')
print(df.groupby('team').apply(lambda x:x['name'].to_list()).C)
# ['Arry', 'Eorge']

# 调用函数，实现每组Q1成绩最高的前三个：
print()
print('# 各组Q1（为参数）成绩最高的前三个')

def first_3(df_,c):
    return df_[c].sort_values(ascending=False).head(1)

# 调用函数
print(df.set_index('name').groupby('team').apply(first_3,'Q1'))
'''
本df数据量少，所以看不出效果。
如果数据足够多 会看到如下：
team  name 
A     Ack       57
      Rick     100
      Eorge     93
B     Rick     100
      Rick     100
      Eorge     93
C     Eorge     93
      Arry      36
      Rick     100
D     Oah       65
      Rick     100
      Eorge     93
E     Liver     89
Name: Q1, dtype: int64
'''

# 通过设置group_keys，可以使分组字段不作为索引
print(df.set_index('name')
        .groupby('team',group_keys=False)
        .apply(first_3,'Q1')
        )

print()
# 传入一个Series，映射系列不同的聚合统计算法：
print(df.groupby('team')
      .apply(lambda x: pd.Series({
        'Q1_sum' : x['Q1'].sum(),
        'Q1_max' : x['Q1'].max(),
        'Q2_mean': x['Q2'].mean(),
        'Q4_prodsum' : (x['Q4'] * x['Q4']).sum()
        }

      )))


#       Q1_sum  Q1_max  Q2_mean  Q4_prodsum
# team
# A       57.0    57.0     60.0      7056.0
# B      100.0   100.0     99.0     10000.0
# C      129.0    93.0     66.5      9333.0
# D       65.0    65.0     49.0      7396.0
# E       89.0    89.0     21.0      4096.0

print()
def f_mi(x):
    d = []
    d.append(x['Q1'].sum())
    d.append(x['Q2'].max())
    d.append(x['Q3'].mean())
    d.append((x['Q4'] * x['Q4']).sum())
    return pd.Series(d,index=[['Q1','Q2','Q3','Q4'],['sum','max','mean','prodsum']])

print(df.groupby('team').apply(f_mi))  # 同比上述代码 数值相同 列标签不同

#          Q1    Q2    Q3       Q4
#         sum   max  mean  prodsum
# team
# A      57.0  60.0  18.0   7056.0
# B     100.0  99.0  97.0  10000.0
# C     129.0  96.0  54.0   9333.0
# D      65.0  49.0  61.0   7396.0
# E      89.0  21.0  24.0   4096.0



print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.3 分组对象的操作')
print('\t6.3.5 管道方法pipe()')
print()
# update20240424
# 小节注释
'''
类似于DataFrame的管道方法，分组对象的管道方法是接收之前的分组对象，
将同组的所有数据应用在方法中，最后返回的是经过函数处理过的返回数据格式。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 每组最大值和最小值之和')
print(df.groupby('team').pipe(lambda x: x.max() + x.min()))
#             name   Q1   Q2   Q3   Q4
# team
# A         AckAck  114  120   36  168
# B       RickRick  200  198  194  200
# C      EorgeArry  129  133  108  135
# D         OahOah  130   98  122  172
# E     LiverLiver  178   42   48  128

print()
print('# 定义了A组和B组平均值的差值')

# 原函数 报错！TypeError: Could not convert ['A'] to numeric
# def mean_diff(x):
#     return x.get_group('A').mean() - x.get_group('B').mean()

# 修改代码
def mean_diff(x):
    a_mean = x.get_group('A').select_dtypes(include=np.number).mean()
    b_mean = x.get_group('B').select_dtypes(include=np.number).mean()
    return a_mean - b_mean

df1 = df.drop(['name'],axis=1)
print(df1)
# 使用函数
print(df1.groupby('team').pipe(mean_diff))
# Q1   -43.0
# Q2   -39.0
# Q3   -79.0
# Q4   -16.0
# dtype: float64


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.3 分组对象的操作')
print('\t6.3.6 转换方法transform()')
print()
# update20240424
# 小节注释
'''
transform()类似于agg()，但与agg()不同的是它返回的是一个与原始数据相同形状的DataFrame，
会将每个数据原来的值一一替换成统计后的值。例如按组计算平均成绩，
那么返回的新DataFrame中每个学生的成绩就是它所在组的平均成绩。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 将所有数据替换成分组中的平均成绩')
print(df.drop(['name'],axis=1).groupby('team').transform(np.mean))
#       Q1    Q2    Q3     Q4
# 0   89.0  21.0  24.0   64.0
# 1   64.5  66.5  54.0   67.5
# 2   57.0  60.0  18.0   84.0
# 3   64.5  66.5  54.0   67.5
# 4   65.0  49.0  61.0   86.0
# 5  100.0  99.0  97.0  100.0

print()
print('# 其它方法')
print(df.groupby('team').transform(max))
#     name   Q1  Q2  Q3   Q4
# 0  Liver   89  21  24   64
# 1  Eorge   93  96  71   78
# 2    Ack   57  60  18   84
# 3  Eorge   93  96  71   78
# 4    Oah   65  49  61   86
# 5   Rick  100  99  97  100

print(df.drop(['name'],axis=1).groupby('team').transform(np.std)) # 标准差
#           Q1       Q2         Q3         Q4
# 0        NaN      NaN        NaN        NaN
# 1  40.305087  41.7193  24.041631  14.849242
# 2        NaN      NaN        NaN        NaN
# 3  40.305087  41.7193  24.041631  14.849242
# 4        NaN      NaN        NaN        NaN
# 5        NaN      NaN        NaN        NaN

print()
print('使用函数，和上一个学生的差值（没有处理姓名列）')
print(df.groupby('team').transform(lambda x: x.shift(-1)))
#     name    Q1    Q2    Q3    Q4
# 0   None   NaN   NaN   NaN   NaN
# 1  Eorge  93.0  96.0  71.0  78.0
# 2   None   NaN   NaN   NaN   NaN
# 3   None   NaN   NaN   NaN   NaN
# 4   None   NaN   NaN   NaN   NaN
# 5   None   NaN   NaN   NaN   NaN

def score(gb):
    return (gb - gb.mean()) / gb.std() * 10

# 调用
grouped = df.drop(['name'],axis=1).groupby('team')
# print(grouped)
print(grouped.transform(score))
#          Q1        Q2        Q3        Q4
# 0       NaN       NaN       NaN       NaN
# 1 -7.071068 -7.071068 -7.071068 -7.071068
# 2       NaN       NaN       NaN       NaN
# 3  7.071068  7.071068  7.071068  7.071068
# 4       NaN       NaN       NaN       NaN
# 5       NaN       NaN       NaN       NaN

print()
# 也可以用它来进行按组筛选：
print('# Q1成绩大于60的组的所有成员')
print(df[df.drop(['name'],axis=1).groupby('team').transform('mean').Q1 > 60])
#     name team   Q1  Q2  Q3   Q4
# 0  Liver    E   89  21  24   64
# 1   Arry    C   36  37  37   57
# 3  Eorge    C   93  96  71   78
# 4    Oah    D   65  49  61   86
# 5   Rick    B  100  99  97  100

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.3 分组对象的操作')
print('\t6.3.7 筛选方法filter()')
print()
# update20240424
'''
上一节完成了分组对象的创建，分组对象包含数据的分组情况，
接下来就来对分组对象进行操作，获取其相关信息，为最后的数据聚合统计打好基础。
'''
# 小节注释
'''
使用filter()对组作为整体进行筛选，如果满足条件，则整个组会被显示。
传入它调用函数中的默认变量为每个分组的DataFrame，
经过计算，最终返回一个布尔值（不是布尔序列），为真的DataFrame全部显示。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 每组4个季度的平均分的平均分为本组的总平均分')
print(df.drop(['name'],axis=1).groupby('team').mean().mean(1))
# team
# A    54.750
# B    99.000
# C    63.125
# D    65.250
# E    49.500
# dtype: float64

print()
print('# 筛选出所在组总平均分大于51的成员')
print(df.drop(['name'],axis=1).groupby('team').filter(lambda x: x.select_dtypes(include='number').mean(1).mean() > 51))
#   team   Q1  Q2  Q3   Q4
# 1    C   36  37  37   57
# 2    A   57  60  18   84
# 3    C   93  96  71   78
# 4    D   65  49  61   86
# 5    B  100  99  97  100

print()
print('# Q1成绩至少有一个大于92的组')
print(df.drop(['name'],axis=1).groupby('team').filter(lambda x: (x['Q1'] > 92).any()))
#   team   Q1  Q2  Q3   Q4
# 1    C   36  37  37   57
# 3    C   93  96  71   78
# 5    B  100  99  97  100
print()
print('# Q1成绩全部大于92的组')
print(df.drop(['name'],axis=1).groupby('team').filter(lambda x: (x['Q1'] > 92).all()))
#   team   Q1  Q2  Q3   Q4
# 5    B  100  99  97  100


print()
# print(df.groupby('team').first(2))
print(df.groupby('team').rank())

'''
# 其它功能！ 

# 所有成员平均成绩大于60的组
df.groupby(['team']).filter(lambda x: (x.mean() >= 60).all())
# Q1所有成员成绩之和超过1060的组
df.groupby('team').filter(lambda g: g.Q1.sum() > 1060)

df.groupby('team').first() # 组内第一个
df.groupby('team').last() # 组内最后一个
df.groupby('team').ngroups # 5（分组数）
df.groupby('team').ngroup() # 分组序号
grouped.backfill()
grouped.bfill()
df.groupby('team').head() # 每组显示前5个
grouped.tail(1) # 每组最后一个
grouped.rank() # 排序值
grouped.fillna(0)
grouped.indices() # 组名:索引序列组成的字典
# 分组中的第几个值

gp.nth(1) # 第一个
gp.nth(-1) # 最后一个
gp.nth([-2, -1])
# 第n个非空项
gp.nth(0, dropna='all')
gp.nth(0, dropna='any')
df.groupby('team').shift(-1) # 组内移动
grouped.tshift(1) # 按时间周期移动
df.groupby('team').any()
df.groupby('team').all()
df.groupby('team').rank() # 在组内的排名
# 仅 SeriesGroupBy 可用
df.groupby("team").Q1.nlargest(2) # 每组最大的两个
df.groupby("team").Q1.nsmallest(2) # 每组最小的两个
df.groupby("team").Q1.nunique() # 每组去重数量
df.groupby("team").Q1.unique() # 每组去重值
df.groupby("team").Q1.value_counts() # 每组去重值及数量
df.groupby("team").Q1.is_monotonic_increasing # 每组值是否单调递增
df.groupby("team").Q1.is_monotonic_decreasing # 每组值是否单调递减
# 仅 DataFrameGroupBy 可用
df.groupby("team").corrwith(df2) # 相关性

'''