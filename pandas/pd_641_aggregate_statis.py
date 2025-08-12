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
print('\t6.4 聚合统计')
print('\t6.4.1 描述统计')
print()
# update20240425
'''
本节主要介绍对分组完的数据的统计工作，这是分组聚合的最后一步。
通过最终数据的输出，可以观察到业务的变化情况，体现数据的价值。

'''
# 小节注释
'''
分组对象如同df.describe()，也支持.describe()，用来对数据的总体进行描述：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 描述统计')
# print(df.groupby('team').Q1.describe())
#       count   mean        std    min     25%    50%     75%    max
# team
# A       1.0   57.0        NaN   57.0   57.00   57.0   57.00   57.0
# B       1.0  100.0        NaN  100.0  100.00  100.0  100.00  100.0
# C       2.0   64.5  40.305087   36.0   50.25   64.5   78.75   93.0
# D       1.0   65.0        NaN   65.0   65.00   65.0   65.00   65.0
# E       1.0   89.0        NaN   89.0   89.00   89.0   89.00   89.0

print(df.groupby('team').describe())
#         Q1                           ...      Q4
#      count   mean        std    min  ...     25%    50%     75%    max
# team                                 ...
# A      1.0   57.0        NaN   57.0  ...   84.00   84.0   84.00   84.0
# B      1.0  100.0        NaN  100.0  ...  100.00  100.0  100.00  100.0
# C      2.0   64.5  40.305087   36.0  ...   62.25   67.5   72.75   78.0
# D      1.0   65.0        NaN   65.0  ...   86.00   86.0   86.00   86.0
# E      1.0   89.0        NaN   89.0  ...   64.00   64.0   64.00   64.0
#
# [5 rows x 32 columns]

print()
print('# 由于列过多，我们进行转置')
print(df.groupby('team').describe().T)
# 由于列过多，我们进行转置
# team         A      B          C     D     E
# Q1 count   1.0    1.0   2.000000   1.0   1.0
#    mean   57.0  100.0  64.500000  65.0  89.0
# ...
# Q4 count   1.0    1.0   2.000000   1.0   1.0
#    mean   84.0  100.0  67.500000  86.0  64.0
#    std     NaN    NaN  14.849242   NaN   NaN
#    min    84.0  100.0  57.000000  86.0  64.0
#    25%    84.0  100.0  62.250000  86.0  64.0
#    50%    84.0  100.0  67.500000  86.0  64.0
#    75%    84.0  100.0  72.750000  86.0  64.0
#    max    84.0  100.0  78.000000  86.0  64.0

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.4 聚合统计')
print('\t6.4.2 统计函数')
print()
# update20240426
# 小节注释
'''
对分组对象直接使用统计函数，对分组内的所有数据进行此计算，最终以DataFrame形式显示数据。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 各组平均数')
grouped = df.drop('name',axis=1).groupby('team')
print(grouped.mean())
#          Q1    Q2    Q3     Q4
# team
# A      57.0  60.0  18.0   84.0
# B     100.0  99.0  97.0  100.0
# C      64.5  66.5  54.0   67.5
# D      65.0  49.0  61.0   86.0
# E      89.0  21.0  24.0   64.0

print()
print('# 其它统计')

print(df.groupby('team').size())
# team
# A    1
# B    1
# C    2
# D    1
# E    1
# dtype: int64

print()
print(df.drop('name',axis=1).groupby('team').prod())
#          Q1    Q2    Q3     Q4
# team
# A      57.0  60.0  18.0   84.0
# B     100.0  99.0  97.0  100.0
# C      64.5  66.5  54.0   67.5
# D      65.0  49.0  61.0   86.0
# E      89.0  21.0  24.0   64.0

# df.groupby('team').describe() # 描述性统计
# df.groupby('team').sum() # 求和
# df.groupby('team').count() # 每组数量，不包括缺失值
# df.groupby('team').max() # 求最大值
# df.groupby('team').min() # 求最小值
# df.groupby('team').size() # 分组数量
# df.groupby('team').mean() # 平均值
# df.groupby('team').median() # 中位数
# df.groupby('team').std() # 标准差
# df.groupby('team').var() # 方差
# grouped.corr() # 相关性系数
# grouped.sem() # 标准误差
# grouped.prod() # 乘积
# grouped.cummax() # 每组的累计最大值
# grouped.cumsum() # 累加
# grouped.mad() # 平均绝对偏差


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.4 聚合统计')
print('\t6.4.3 聚合方法agg()')
print()
# update20240426
# 小节注释
'''
对分组对象直接使用统计函数，对分组内的所有数据进行此计算，最终以DataFrame形式显示数据。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 所有列使用一个计算方法')
grouped = df.drop('name',axis=1).groupby('team')
# print(grouped.mean())
print(df.groupby('team').aggregate(sum))
print(df.groupby('team').agg(sum)) # 结果同上

print(grouped.agg(np.size))
#       Q1  Q2  Q3  Q4
# team
# A      1   1   1   1
# B      1   1   1   1
# C      2   2   2   2
# D      1   1   1   1
# E      1   1   1   1
print(grouped['Q1'].agg(np.mean))
# team
# A     57.0
# B    100.0
# C     64.5
# D     65.0
# E     89.0
# Name: Q1, dtype: float64

'''我们使用它主要是为了实现一个字段使用多种统计方法，不同字段使用不同方法：'''

print()
print('# 每个字段使用多个计算方法')
print(grouped[['Q1','Q3']].agg([np.sum,np.mean,np.std]))
#        Q1                     Q3
#       sum   mean        std  sum  mean        std
# team
# A      57   57.0        NaN   18  18.0        NaN
# B     100  100.0        NaN   97  97.0        NaN
# C     129   64.5  40.305087  108  54.0  24.041631
# D      65   65.0        NaN   61  61.0        NaN
# E      89   89.0        NaN   24  24.0        NaN

print()
# 不同列使用不同计算方法，且一个列用多个计算方法
print(df.groupby('team').agg({'Q1':['min','max'],'Q2':'sum'}))
#        Q1        Q2
#       min  max  sum
# team
# A      57   57   60
# B     100  100   99
# C      36   93  133
# D      65   65   49
# E      89   89   21

# 类似于我们之前学过的增加新列的方法df.assign()，agg()可以指定新列的名字：
print()
print('# 相同列，不同统计函数')
print(df.groupby('team').Q1.agg(Mean='mean',Sum='sum'))
#        Mean  Sum
# team
# A      57.0   57
# B     100.0  100
# C      64.5  129
# D      65.0   65
# E      89.0   89

print()
print('# 不同列 不同统计函数')
print(df.groupby('team').agg(Mean=('Q1','mean'),Sum=('Q2','sum')))
#        Mean  Sum
# team
# A      57.0   60
# B     100.0   99
# C      64.5  133
# D      65.0   49
# E      89.0   21

print()
print(df.groupby('team').agg(
                            Q1_max=pd.NamedAgg(column='Q1',aggfunc='max'),
                            Q2_min=pd.NamedAgg(column='Q2',aggfunc='min')
                        ))
#       Q1_max  Q2_min
# team
# A         57      60
# B        100      99
# C         93      37
# D         65      49
# E         89      21

print()
print('# 如果列名不是有效的Python变量格式，则可以用以下方法：')
print(df.groupby('team').agg(**{
    '1_max':pd.NamedAgg(column='Q1',aggfunc='max')
}))

#       1_max
# team
# A        57
# B       100
# C        93
# D        65
# E        89

print()
print('# 聚合结果使用函数')
# lambda函数，所有方法都可以使用
def max_min(x):
    return x.max() - x.min()

# 定义函数
print(df.groupby('team').Q1.agg(Mean='mean',
                          Sum='sum',
                          Diff=lambda x:x.max() - x.min(),
                          Max_min=max_min)
)
#        Mean  Sum  Diff  Max_min
# team
# A      57.0   57     0        0
# B     100.0  100     0        0
# C      64.5  129    57       57
# D      65.0   65     0        0
# E      89.0   89     0        0


print(df.groupby('team')[['Q1', 'Q2']].agg(
    Mean_Q1=('Q1', 'mean'),
    Sum_Q1=('Q1', 'sum'),
    Diff_Q1=('Q1', lambda x: x.max() - x.min()),
    Max_min_Q1=('Q1', max_min),
    Mean_Q2=('Q2', 'mean'),
    Sum_Q2=('Q2', 'sum'),
    Diff_Q2=('Q2', lambda x: x.max() - x.min()),
    Max_min_Q2=('Q2', max_min)
))


# print(df.groupby('team').agg(Mean='mean',
#                           Sum='sum',
#                           Diff=lambda x:x.max() - x.min(),
#                           Max_min=max_min)
# )

# print(df.groupby('team').agg(max_min))  报错！


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.4 聚合统计')
print('\t6.4.4 时序重采样方法resample()')
print()
# update20240428

# 小节注释
'''
针对时间序列数据，resample()将分组后的时间索引按周期进行聚合统计。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

idx = pd.date_range('1/1/2024',periods=100,freq='T')
df2 = pd.DataFrame(data={'a':[0,1]*50,'b':1},
                   index=idx)
print(df2)
'''
                     a  b
2024-01-01 00:00:00  0  1
2024-01-01 00:01:00  1  1
2024-01-01 00:02:00  0  1
2024-01-01 00:03:00  1  1
2024-01-01 00:04:00  0  1
...                 .. ..
2024-01-01 01:35:00  1  1
2024-01-01 01:36:00  0  1
2024-01-01 01:37:00  1  1
2024-01-01 01:38:00  0  1
2024-01-01 01:39:00  1  1

[100 rows x 2 columns]
'''

print()
print('# 每20分钟聚合一次')
print(df2.groupby('a').resample('20T').sum())
'''
                        a   b
a                            
0 2024-01-01 00:00:00   0  10
  2024-01-01 00:20:00   0  10
  2024-01-01 00:40:00   0  10
  2024-01-01 01:00:00   0  10
  2024-01-01 01:20:00   0  10
1 2024-01-01 00:00:00  10  10
  2024-01-01 00:20:00  10  10
  2024-01-01 00:40:00  10  10
  2024-01-01 01:00:00  10  10
  2024-01-01 01:20:00  10  10
'''

print()
print('# 其他案例')
print('# 三个周期一聚合（一分钟一个周期）')
print(df2.groupby('a').resample('3T').sum()) # [67 rows x 2 columns]

print()
print('# 30秒一分组')
print(df2.groupby('a').resample('30S').sum()) # [394 rows x 2 columns]

print()
print('# 每月')
print(df2.groupby('a').resample('M').sum())
#                a   b
# a
# 0 2024-01-31   0  50
# 1 2024-01-31  50  50

print()
print('# 以右边时间点为标识')
print(df2.groupby('a').resample('3T',closed='right').sum()) # [67 rows x 2 columns]

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.4 聚合统计')
print('\t6.4.5 组内头尾值')
print()
# update20240428

# 小节注释
'''
在一个组内，如果希望取第一个值和最后一个值，可以使用以下方法。
当然，定义第一个和最后一个是你需要事先完成的工作。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print(df.groupby('team').first())
#        name   Q1  Q2  Q3   Q4
# team
# A       Ack   57  60  18   84
# B      Rick  100  99  97  100
# C      Arry   36  37  37   57
# D       Oah   65  49  61   86
# E     Liver   89  21  24   64

print()
print(df.groupby('team').last())
#        name   Q1  Q2  Q3   Q4
# team
# A       Ack   57  60  18   84
# B      Rick  100  99  97  100
# C     Eorge   93  96  71   78
# D       Oah   65  49  61   86
# E     Liver   89  21  24   64


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.4 聚合统计')
print('\t6.4.6 组内分位数')
print()
# update20240428

# 小节注释
'''
在一个组内，如果希望取第一个值和最后一个值，可以使用以下方法。
当然，定义第一个和最后一个是你需要事先完成的工作。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 二分位数，即中位数')
print(df.drop('name',axis=1).groupby('team').median())
#          Q1    Q2    Q3     Q4
# team
# A      57.0  60.0  18.0   84.0
# B     100.0  99.0  97.0  100.0
# C      64.5  66.5  54.0   67.5
# D      65.0  49.0  61.0   86.0
# E      89.0  21.0  24.0   64.0

print(df.drop('name',axis=1).groupby('team').quantile()) # 结果同上
print(df.drop('name',axis=1).groupby('team').quantile(0.5)) # 结果同上

print()
print('# 三分位数 、 四分位数')
print(df.drop('name',axis=1).groupby('team').quantile(0.33)) # 三分位数
#           Q1     Q2     Q3      Q4
# team
# A      57.00  60.00  18.00   84.00
# B     100.00  99.00  97.00  100.00
# C      54.81  56.47  48.22   63.93
# D      65.00  49.00  61.00   86.00
# E      89.00  21.00  24.00   64.00

print(df.drop('name',axis=1).groupby('team').quantile(0.25)) # 四分位数
#           Q1     Q2    Q3      Q4
# team
# A      57.00  60.00  18.0   84.00
# B     100.00  99.00  97.0  100.00
# C      50.25  51.75  45.5   62.25
# D      65.00  49.00  61.0   86.00
# E      89.00  21.00  24.0   64.00

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.4 聚合统计')
print('\t6.4.7 组内差值')
print()
# update20240428
# 小节注释
'''
和DataFrame的diff()一样，分组对象的diff()方法会在组内进行前后数据的差值计算，并以原DataFrame形状返回数据：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('#  grouped为全数字列，计算在组内的前后差值')
grouped = df.drop('name',axis=1).groupby('team')
print(grouped.diff())

#      Q1    Q2    Q3    Q4
# 0   NaN   NaN   NaN   NaN
# 1   NaN   NaN   NaN   NaN
# 2   NaN   NaN   NaN   NaN
# 3  57.0  59.0  34.0  21.0
# 4   NaN   NaN   NaN   NaN
# 5   NaN   NaN   NaN   NaN

'''
6.4.8 小结
本节介绍的功能是将分组的结果最终统计并展示出来。
我们需要掌握常见的数学统计函数，另外也可以使用NumPy的大量统计方法。
特别是要熟练使用agg()方法，它功能强大，显示功能完备，是在我们今后的
数据分析中最后的数据分组聚合工具。
'''


