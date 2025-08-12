import numpy as np
import pandas as pd
import warnings
import jinja2

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
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.3 数据排序')
print('\t5.3.1 索引排序')
print()
# update20240318
'''
df.sort_index()实现按索引排序，默认以从小到大的升序方式排列。如希望按降序排序，传入ascending=False：

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

# TODO 索引降序
print('索引降序')
print(df.sort_index())  # 默认按 行索引 排序
print(df.sort_index(ascending=True))  # 同上
print(df.sort_index(ascending=False))  # 索引降序
#     name team   Q1  Q2  Q3   Q4
# 5   Rick    B  100  99  97  100
# 4    Oah    D   65  49  61   86
# ...

print()
print('在列索引方向上排序')
print(df.sort_index(axis=1, ascending=False))
#   team   name   Q4  Q3  Q2   Q1
# 0    E  Liver   64  24  21   89
# ...

print()
print('更多方法')
# 创建一个无序的 Series 对象
s = pd.Series([3, 1, 4, 1, 5], index=[5, 3, 4, 1, 2])

# 创建一个无序的 DataFrame 对象
df = pd.DataFrame({
    'team': ['B', 'A', 'D', 'C'],
    'score': [85, 90, 78, 88]
}, index=[2, 3, 1, 4])

# print(s)
# print(df)
print()

print('# 对 Series 对象升序排序')
s_sorted = s.sort_index()
print(s_sorted)
print()
# 1    1
# 2    5
# 3    1
# 4    4
# 5    3
# dtype: int64

print('# 对 DataFrame 对象升序排序')
df_sorted = df.sort_index()
print(df_sorted)
print()
#   team  score
# 1    D     78
# 2    B     85
# 3    A     90
# 4    C     88

print('# 只对 DataFrame 的某一列按索引排序')
team_sorted = df.team.sort_index()
print(team_sorted)
print()
# 1    D
# 2    B
# 3    A
# 4    C
# Name: team, dtype: object

print('# 对 Series 对象降序排序')
s_sorted_desc = s.sort_index(ascending=False)
print(s_sorted_desc)
print()
# 5    3
# 4    4
# 3    1
# 2    5
# 1    1
# dtype: int64

print('# 对原 Series 对象就地排序')
s.sort_index(inplace=True)
print(s)
print()
# 1    1
# 2    5
# 3    1
# 4    4
# 5    3
# dtype: int64

print('# 对 Series 对象排序并重置索引')
s_sorted_reset = s.sort_index(ignore_index=True)
print(s_sorted_reset)
print()
# 0    1
# 1    5
# 2    1
# 3    4
# 4    3
# dtype: int64


print("sort_values() 使用 na_position='first' 参数排序并将空值放在前面")

# 创建一个包含空值的 Series 对象
s_with_na = pd.Series([np.nan, 1, 3, np.nan, 2])
print(s_with_na)
# 对 Series 对象排序并将空值放在前面
s_sorted_na_first = s_with_na.sort_values(na_position='first')
print(s_sorted_na_first)
print()
# 0    NaN
# 3    NaN
# 1    1.0
# 4    2.0
# 2    3.0
# dtype: float64
s_sorted_na_first = s_with_na.sort_values()
print(s_sorted_na_first)
print()
# 1    1.0
# 4    2.0
# 2    3.0
# 0    NaN
# 3    NaN
# dtype: float64

print()
print('sort_index() 参数level=0 和 sort_remaining=False')
# 创建一个无序的 Series 对象
s = pd.Series([3, 1, 4, 1, 5], index=[5, 3, 4, 1, 2])

# 创建一个无序的 DataFrame 对象
df = pd.DataFrame({
    'team': ['B', 'A', 'D', 'C'],
    'score': [85, 90, 78, 88]
}, index=[2, 3, 1, 4])

# print(s)
# print(df)
print()

# 创建一个多层索引的 Series 对象
s_multi_index = pd.Series([3, 1, 4, 1, 5],
                          index=pd.MultiIndex.from_tuples([(1, 'c'), (1, 'b'), (1, 'a'), (2, 'a'), (2, 'b')]))
print(s_multi_index)
# 1  c    3
#    b    1
#    a    4
# 2  a    1
#    b    5
# dtype: int64
# 对 Series 对象的第一级索引排序
s_sorted_level = s_multi_index.sort_index(level=0)
print(s_sorted_level)
# 1  a    4
#    b    1
#    c    3
# 2  a    1
#    b    5
# dtype: int64

# 在多层索引中使用 level 和 sort_remaining=False 参数
# 对 Series 对象的第二级索引排序，但不排序其余部分
s_sorted_level_no_remaining = s_multi_index.sort_index(level=1, sort_remaining=False)
print(s_sorted_level_no_remaining)
# 1  a    4
# 2  a    1
# 1  b    1
# 2  b    5
# 1  c    3
# dtype: int64

s_sorted_level_no_remaining = s_multi_index.sort_index(level=1, sort_remaining=True)
print(s_sorted_level_no_remaining)
# 1  a    4
# 2  a    1
# 1  b    1
# 2  b    5
# 1  c    3
# dtype: int64

'''
。sort_index 方法用于根据索引排序，level 参数用于指定多级索引的哪个级别进行排序，
而 sort_remaining 参数用于控制是否对除了指定级别之外的其他级别的索引也进行排序。

当 sort_remaining=True（默认值）时，会在对指定索引级别排序后，对剩余的索引级别按照字典顺序进行排序。
当 sort_remaining=False 时，只会对指定的索引级别进行排序，而不对剩余的索引级别进行排序。
对于 sort_values 方法，它并没有 sort_remaining 参数，它只用于根据数据值进行排序，并可以通过 na_position 参数控制 NaN 值的位置。
'''

print()
print('df.reindex()指定自己定义顺序的索引，实现行和列的顺序重新定义：')
df1 = pd.DataFrame({
    'A': [1, 2, 4],
    'B': [3, 5, 6]
}, index=['a', 'b', 'c'])
print(df1)
#    A  B
# a  1  3
# b  2  5
# c  4  6

print()
print('# 按要求重新指定索引顺序')
print(df1.reindex(['c', 'b', 'a']))
#    A  B
# c  4  6
# b  2  5
# a  1  3

print()
print('# 指定列顺序')
print(df1.reindex(['B', 'A'], axis=1))
#    B  A
# a  3  1
# b  5  2
# c  6  4


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.3 数据排序')
print('\t5.3.2 数值排序')
print()
# update20240318
'''
数据值的排序主要使用sort_values()，数字按大小顺序，字符按字母顺序。Series和DataFrame都支持此方法：

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print(df.Q1.sort_values())  # 单列按照数值 升序排序
# 1     36
# 2     57
# 4     65
# 0     89
# 3     93
# 5    100
# Name: Q1, dtype: int64

print()
print('# DataFrame需要传入一个或多个排序的列名：')
print(df.sort_values('Q4'))
#     name team   Q1  Q2  Q3   Q4
# 1   Arry    C   36  37  37   57
# 0  Liver    E   89  21  24   64
# 3  Eorge    C   93  96  71   78
# 2    Ack    A   57  60  18   84
# 4    Oah    D   65  49  61   86
# 5   Rick    B  100  99  97  100

# TODO 默认排序是升序，但可以指定排序方式，下例先按team升序排列，如遇到相同的team再按name降序排列。
print()
print(df.sort_values(['team', 'name'], ascending=[True, False]))
#     name team   Q1  Q2  Q3   Q4
# 2    Ack    A   57  60  18   84
# 5   Rick    B  100  99  97  100
# 3  Eorge    C   93  96  71   78
# 1   Arry    C   36  37  37   57
# 4    Oah    D   65  49  61   86
# 0  Liver    E   89  21  24   64

print()
print('# 其它常用方法')
# s.sort_values(ascending=False) # 降序
# s.sort_values(inplace=True) # 修改生效
# s.sort_values(na_position='first') # 空值在前
# # df按指定字段排列
# df.sort_values(by=['team'])
# df.sort_values('Q1')
# # 按多个字段，先排team，在同team内再看Q1
# df.sort_values(by=['team', 'Q1'])
# # 全降序
# df.sort_values(by=['team', 'Q1'], ascending=False)
# # 对应指定team升Q1降
# df.sort_values(by=['team', 'Q1'], ascending=[True, False])
# 索引重新0-(n-1)排
print(df.sort_values('team', ignore_index=True))
#     name team   Q1  Q2  Q3   Q4
# 0    Ack    A   57  60  18   84
# 1   Rick    B  100  99  97  100
# 2   Arry    C   36  37  37   57
# 3  Eorge    C   93  96  71   78
# 4    Oah    D   65  49  61   86
# 5  Liver    E   89  21  24   64


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.3 数据排序')
print('\t5.3.3 混合排序')
print()
# update20240320
'''
有时候需要用索引和数据值混合排序。下例中假如name是索引，我们需要先按team排名，再按索引排名：

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

# df.set_index('name',inplace=True) # 设置那么列为索引
df1 = df.set_index('name')
# print(df1)
df1.index.names = ['s_name']  # 给索引起名
print(df1.sort_values(by=['s_name', 'team']))  # 排序 sort_index() 无 by= 参数；sort_values()有 by= 参数
#        team   Q1  Q2  Q3   Q4
# s_name
# Ack       A   57  60  18   84
# Arry      C   36  37  37   57
# Eorge     C   93  96  71   78
# Liver     E   89  21  24   64
# Oah       D   65  49  61   86
# Rick      B  100  99  97  100
'''以下方法也可以实现上述需求，不过要注意顺序：'''

print()
# 设置索引，按team排序，再按索引排序
print(df.set_index('name').sort_values('team').sort_index())  # 结果同上

'''看结果 应该是先按照team排序后 在按照index排序 实际还是以name索引列排序为准'''
# print()
# print(df.set_index('name').sort_values('team'))
# print()
# print(df.set_index('name').sort_index()) # 结果等于print(df.set_index('name').sort_values('team').sort_index())
print()
print('# 按姓名排序后取出排名后的索引列表')
print(df.name.sort_values().index)  # Index([2, 1, 3, 0, 4, 5], dtype='int64')

print()
print('# 将新的索引应用到数据中')
print(df.reindex(df.name.sort_values().index))  # 将新的索引应用到数据中
#     name team   Q1  Q2  Q3   Q4
# 2    Ack    A   57  60  18   84
# 1   Arry    C   36  37  37   57
# 3  Eorge    C   93  96  71   78
# 0  Liver    E   89  21  24   64
# 4    Oah    D   65  49  61   86
# 5   Rick    B  100  99  97  100


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.3 数据排序')
print('\t5.3.4 按值大小排序')
print()
# update20240320
'''
nsmallest()和nlargest()用来实现数字列的排序，并可指定返回的个数：

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 先按Q1最小在前，如果相同，Q2小的在前')
print(df.nsmallest(5, ['Q1', 'Q2']))  # 取前5行数据
#     name team  Q1  Q2  Q3  Q4
# 1   Arry    C  36  37  37  57
# 2    Ack    A  57  60  18  84
# 4    Oah    D  65  49  61  86
# 0  Liver    E  89  21  24  64
# 3  Eorge    C  93  96  71  78

'''仅支持数字类型的排序。下面是几个其他示例：'''

print()
s = pd.Series([9, 4, 5, 5, 2, 3, 0], name='num')
print(s)
print(s.nsmallest(3))  # 最小的3个
print(s.nlargest(3))  # 最大的3个

print()
print('指定列')
print(df.nlargest(3, 'Q1'))  # Q1列最大的前3行数据
print(df.nlargest(5, ['Q1', 'Q2']))  # Q1列最大的前5行数据 如果有相同值，看Q2列数据较大的
print(df.nsmallest(5, ['Q1', 'Q2']))  # Q1列最小的前5行数据 如果有相同值，看Q2列数据较小的

'''
5.3.5 小结
本节介绍了索引的排序、数值的排序以及索引和数值混合的排序方法。
在实际需求中，更加复杂的排序可能需要通过计算增加辅助列来实现。
'''
