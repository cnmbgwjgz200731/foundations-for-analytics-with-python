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
print('\t5.6 数据迭代')
print('\t5.6.1 迭代Series')
print()
# update20240410
'''
数据迭代和数据遍历都是按照某种顺序逐个对数据进行访问和操作，
在Python中大多由for语句来引导。Pandas中的迭代操作可以将数据按行或者按列遍历，
我们可以进行更加细化、个性化的数据处理。
'''
# 小节注释
'''
Series本身是一个可迭代对象，Series df.name.values返回array结构数据可用于迭代，
不过可直接对Series使用for语句来遍历它的值：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print()
print('# 迭代指定的列')

print(df.name)

for i in df['name']:
    print(i)

for i,n,q in zip(df.index,df.name,df.Q1):
    print(i,n,q)
# 0 Liver 89
# 1 Arry 36
# 2 Ack 57
# 3 Eorge 93
# 4 Oah 65
# 5 Rick 100


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.6 数据迭代')
print('\t5.6.2 df.iterrows()')
print()
# update20240410
# 小节注释
'''
df.iterrows()生成一个可迭代对象，将DataFrame行作为（索引，行数据）组成的Series数据对进行迭代。
在for语句中需要两个变量来承接数据：一个为索引变量，即使索引在迭代中不会使用（这种情况可用useless作为变量名）；
另一个为数据变量，读取具体列时，可以使用字典的方法和对象属性的方法。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print()
print('# 迭代，使用name、Q1数据')
for index,row in df.iterrows():
    print(index,row['name'],row.Q1)
# 0 Liver 89
# 1 Arry 36
# 2 Ack 57
# 3 Eorge 93
# 4 Oah 65
# 5 Rick 100
'''df.iterrows()是最常用、最方便的按行迭代方法。'''


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.6 数据迭代')
print('\t5.6.3 df.itertuples()')
print()
# update20240410
# 小节注释
'''
df.itertuples()生成一个namedtuples类型数据，name默认名为Pandas，可以在参数中指定。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print()
print('# 迭代，使用name、Q1数据')
for row in df.itertuples():
    print(row)
# Pandas(Index=0, name='Liver', team='E', Q1=89, Q2=21, Q3=24, Q4=64)
# Pandas(Index=1, name='Arry', team='C', Q1=36, Q2=37, Q3=37, Q4=57)
# Pandas(Index=2, name='Ack', team='A', Q1=57, Q2=60, Q3=18, Q4=84)
# Pandas(Index=3, name='Eorge', team='C', Q1=93, Q2=96, Q3=71, Q4=78)
# Pandas(Index=4, name='Oah', team='D', Q1=65, Q2=49, Q3=61, Q4=86)
# Pandas(Index=5, name='Rick', team='B', Q1=100, Q2=99, Q3=97, Q4=100)

print()
print('# 以下是其它一些使用方法示例：')
# 不包含索引
for row in df.itertuples(index=False):
    print(row)
# Pandas(name='Liver', team='E', Q1=89, Q2=21, Q3=24, Q4=64)
# Pandas(name='Arry', team='C', Q1=36, Q2=37, Q3=37, Q4=57)
# ...

print()
# 自定义name
for row in df.itertuples(index=False,name='Gairuo'):
    print(row)
# Gairuo(name='Liver', team='E', Q1=89, Q2=21, Q3=24, Q4=64)
# Gairuo(name='Arry', team='C', Q1=36, Q2=37, Q3=37, Q4=57)
# ...

print()
# 使用数据
for row in df.itertuples():
    print(row.Index,row.name)
# 0 Liver
# 1 Arry
# 2 Ack
# 3 Eorge
# 4 Oah
# 5 Rick

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.6 数据迭代')
print('\t5.6.4 df.items()')
print()
# update20240410
# 小节注释
'''
df.items()和df.iteritems()功能相同，它迭代时返回一个（列名，本列的Series结构数据），实现对列的迭代：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print()
print('# 迭代，使用name、Q1数据')

for label,ser in df.items():
    print(label)
    print(ser[:3],end='\n\n')

# name
# 0    Liver
# 1     Arry
# 2      Ack
# Name: name, dtype: object
#
# team
# 0    E
# 1    C
# 2    A
# Name: team, dtype: object
# ........


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.6 数据迭代')
print('\t5.6.5 按列迭代')
print()
# update20240410
# 小节注释
'''
除了df.items()，如需要迭代一个DataFrame的列，可以直接对DataFrame迭代，会循环得到列名：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 直接对DataFrame迭代')

for column in df:
    print(column)
# name
# team
# Q1
# Q2
# Q3
# Q4

print()
print('# 再利用df [列名]的方法迭代列：')

for column in df:
    print(df[column])


print()
print('# 可对每个列的内容进行迭代')

for column in df:
    for i in df[column]:
        print(i)
# Liver
# Arry
# Ack
# Eorge
# Oah
# Rick
# E
# C
# ...

print()
print('# 可以迭代指定列')

for i in df.name:
    print(i)

print()
print('# 只迭代想要的列')
l = ['name','Q1']
cols = df.columns.intersection(l)
for col in cols:
    print(col)

# name
# Q1

'''
与df.iterrows()相比，df.itertuples()运行速度会更快一些，推荐在数据量庞大的情况下优先使用。
'''
