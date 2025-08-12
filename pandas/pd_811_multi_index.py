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


path = 'E:/bat/output_files/pandas_read_csv_20240118.csv'

# TODO 忽略特定的警告 | 当遇到 openpyxl 中的 UserWarning 类型的警告时，它们将不会被打印出来。
warnings.filterwarnings('ignore',category=UserWarning,module='openpyxl')

print()
print('------------------------------------------------------------')
print('第8章 Pandas多层索引')
print('\t8.1 概述')
print('\t8.1.1 什么是多层索引')
print()
# update20240509
'''
截至目前，我们处理的数据基本都是一列索引和一行表头，但在实际业务中会存在有多层索引的情况。
多层索引（Hierarchical indexing）又叫多级索引，
它为一些非常复杂的数据分析和操作（特别是处理高维数据）提供了方法。
从本质上讲，它使你可以在Series（一维）和DataFrame（二维）等较低维度的数据结构中存储和处理更高维度的数据。

'''
# 小节注释
'''
本节介绍多层数据的一些基本概念和使用场景，以及如何创建多层索引数据。
理解了多层数据的基本概念和使用场景，我们就能更好地应用它的特性来解决实际数据分析中的问题。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

'''看了图 个人感觉 就是一个 有合并同类项的 表格'''

print()
print('------------------------------------------------------------')
print('第8章 Pandas多层索引')
print('\t8.1 概述')
print('\t8.1.2 通过分组产生多层索引')
print()
# update20240509

# 小节注释
'''
在之前讲过的数据分组案例中，多个分组条件会产生多层索引的情况，如：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print('# 按团队分组，各团队中平均成绩及格的人数')
# 行多层索引
print(df.groupby(['team',df.select_dtypes('number').mean(1) > 60]).count())
#             name  Q1  Q2  Q3  Q4
# team
# A    False     1   1   1   1   1
# B    True      1   1   1   1   1
# C    False     1   1   1   1   1
#      True      1   1   1   1   1
# D    True      1   1   1   1   1
# E    False     1   1   1   1   1

print()
print('# 列 多层索引')
print(df.groupby('team').agg({'Q1':['max','min'],'Q2':['sum','count']}))
#        Q1        Q2
#       max  min  sum count
# team
# A      57   57   60     1
# B     100  100   99     1
# C      93   36  133     2
# D      65   65   49     1
# E      89   89   21     1

print()
print('# 行列多层索引！')
print(df.groupby(['team',df.select_dtypes('number').mean(1) > 60]).agg({'Q1':['max','min'],'Q2':['sum','count']}))
#              Q1       Q2
#             max  min sum count
# team
# A    False   57   57  60     1
# B    True   100  100  99     1
# C    False   36   36  37     1
#      True    93   93  96     1
# D    True    65   65  49     1
# E    False   89   89  21     1

'''
这样就清晰地表达了业务意义。在处理复杂数据时常常会出现多层索引，
相当于我们对Excel同样值的表头进行了合并单元格。
'''

print()
print('------------------------------------------------------------')
print('第8章 Pandas多层索引')
print('\t8.1 概述')
print('\t8.1.3 由序列创建多层索引')
print()
# update20240509

# 小节注释
'''
MultiIndex对象是Pandas标准Index的子类，由它来表示多层索引业务。
可以将MultiIndex视为一个元组对序列，其中每个元组对都是唯一的。可以通过以下方式生成一个索引对象。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print('# 序列数据')
# 定义一个序列
arrays = [[1,1,2,2],['A','A','B','B']]
# 生成多层索引
index = pd.MultiIndex.from_arrays(arrays,names=('class','team'))
print(index)
# MultiIndex([(1, 'A'),
#             (1, 'A'),
#             (2, 'B'),
#             (2, 'B')],
#            names=['class', 'team'])

print()
print('# 指定的索引是多层索引')
print(pd.DataFrame([{'Q1':60,'Q2':70}],index=index))
#             Q1  Q2
# class team
# 1     A     60  70
#       A     60  70
# 2     B     60  70
#       B     60  70

# 个人测试
# print(pd.DataFrame([{'Q1':[50,60],'Q2':[70,80]}],index=index))
# print(pd.DataFrame({'Q1':[30,40,50,60],'Q2':[70,80,90,100]},index=index))

print()
print('------------------------------------------------------------')
print('第8章 Pandas多层索引')
print('\t8.1 概述')
print('\t8.1.4 由元组创建多层索引')
print()
# update20240509

# 小节注释
'''
可以使用pd.MultiIndex.from_tuples()将由元组组成的序列转换为多层索引

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print('# 定义一个两层的序列')
# 定义一个序列
arrays = [[1,1,2,2],['A','B','A','B']]
# 转换为元组
tuples = list(zip(*arrays)) # *意思是解包 zip(*arrays)类似zip(arrays[0],arrays[1])
print(tuples)
# [(1, 'A'), (1, 'B'), (2, 'A'), (2, 'B')]
# 将元组转换为多层索引对象
index = pd.MultiIndex.from_tuples(tuples,names=['class','team'])
#  使用多层索引对象
print(pd.Series(np.random.randn(4),index=index))
# 1      A       0.868282
#        B       0.721167
# 2      A       2.634965
#        B      -1.010468
# dtype: float64

print()
print('------------------------------------------------------------')
print('第8章 Pandas多层索引')
print('\t8.1 概述')
print('\t8.1.5 可迭代对象的笛卡儿积')
print()
# update20240509

# 小节注释
'''
使用上述方法时我们要将所有层的所有值都写出来，
而pd.MultiIndex.from_product()可以做笛卡儿积计算，将所有情况排列组合出来，如：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print('# 生成多层索引对象')
_class = [1,2]
team = ['A','B']
index = pd.MultiIndex.from_product([_class,team],
                                   names=['class','team'])

# Series应用多层索引对象
print(pd.Series(np.random.randn(4),index=index))
# class  team
# 1      A      -0.211555
#        B      -1.193187
# 2      A      -1.333777
#        B       0.729419
# dtype: float64


print()
print('------------------------------------------------------------')
print('第8章 Pandas多层索引')
print('\t8.1 概述')
print('\t8.1.6 将DataFrame转为多层索引对象')
print()
# update20240509

# 小节注释
'''
pd.MultiIndex.from_frame()可以将DataFrame的数据转换为多层索引对象，如：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print('# 生成多层索引对象')

df_i = pd.DataFrame([['1','A'],['1','B'],['2','B'],['2','A']],
                    columns=['class','team'])

print(df_i)
#   class team
# 0     1    A
# 1     1    B
# 2     2    B
# 3     2    A
# 将DataFrame中的数据转换成多层索引对象
index = pd.MultiIndex.from_frame(df_i)
# print(index)
# 应用多层对象
print(pd.Series(np.random.randn(4),index=index))
# class  team
# 1      A      -0.234935
#        B       1.176749
# 2      B       0.132324
#        A      -0.774019
# dtype: float64

'''
8.1.7 小结
多层索引最为常见的业务场景是数据分组聚合，它一般会产生多层索引的数据。
本节介绍了什么是多层索引、多层索引的业务意义，以及如果创建多层索引，如何将多层索引应用到DataFrame和Series上。
'''


print()
print('------------------------------------------------------------')
print('第8章 Pandas多层索引')
print('\t8.2 多层索引操作')
print('\t8.2.1 生成数据')
print()
# update20240509

# 小节注释
'''
索引的常规操作也适用于多层索引，但多层索引还有一些特定的操作需要我们熟练掌握，以便更加灵活地运用它。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print('# 典型的多层索引数据的生成过程')
# 索引
index_arrays = [[1,1,2,2],['男','女','男','女']]
# 列名
columns_arrays = [['2019','2019','2020','2020'],
                  ['上半年','下半年','上半年','下半年']]

# 索引转换为多层
index = pd.MultiIndex.from_arrays(index_arrays,
                                  names=('班级','性别'))

# 列名转换为多层
columns = pd.MultiIndex.from_arrays(columns_arrays,
                                    names=('年份','学期'))
# 应用到Dataframe中
df = pd.DataFrame([(88,99,88,99),(77,88,97,98),
                    (67,89,54,78),(34,67,89,54)],
                   columns=columns,index=index)

# df.to_excel('E:/bat/output_files/pandas_out_20240509043.xlsx') # 默认 index=True || ,index=False 会报错！

print(df)
# 年份    2019     2020
# 学期     上半年 上半年  上半年 上半年
# 班级 性别
# 1  男    88  99   88  99
#    女    77  88   97  98
# 2  男    67  89   54  78
#    女    34  67   89  54

'''在行和列上可以分别定义多层索引。'''

print('\t8.2.2 索引信息')

print()
print(df.index)
'''
MultiIndex([(1, '男'),
            (1, '女'),
            (2, '男'),
            (2, '女')],
           names=['班级', '性别'])
'''

print(df.columns)
'''
MultiIndex([('2019', '上半年'),
            ('2019', '上半年'),
            ('2020', '上半年'),
            ('2020', '上半年')],
           names=['年份', '学期'])
'''

print()
print('# 查看行、列索引的名称')
print(df.index.names)
print(df.columns.names)
# ['班级', '性别']
# ['年份', '学期']


print()
print('\t8.2.3 查看层级')
'''多层索引由于层级较多，在数据分析时需要查看它共有多少个层级。'''
print(df.index.nlevels) # 行层级数
# 2
print(df.index.levels) # 行的层级
# [[1, 2], ['女', '男']]
print(df.columns.nlevels) # 列层级数
# 2
print(df.columns.levels) # 列的层级
# [['2019', '2020'], ['上半年', '下半年']]

print(df[['2019','2020']].index.levels) # 筛选后的层级
# [[1, 2], ['女', '男']]

# print()
# print(df['2019'])


print()
print('\t8.2.4 索引内容')
'''可以取指定层级的索引内容，也可以按索引名取索引内容：'''
print('# 获取索引第2层内容')
print(df.index.get_level_values(1))
# Index(['男', '女', '男', '女'], dtype='object', name='性别')

print('# 获取列索引第1层内容')
print(df.columns.get_level_values(0))
# Index(['2019', '2019', '2020', '2020'], dtype='object', name='年份')

print()
print('#  按索引名称取索引内容')
print(df.index.get_level_values('班级'))
# Index([1, 1, 2, 2], dtype='int64', name='班级')
print(df.columns.get_level_values('年份'))
# Index(['2019', '2019', '2020', '2020'], dtype='object', name='年份')


print()
print('\t8.2.5 排序')
'''多层索引可以根据需要实现较为复杂的排序操作'''
print('# 使用索引名可进行排序，可以指定具体的列')

print(df.sort_values(by=['性别',('2020','下半年')])) # 必须是个元组
# 年份    2019     2020
# 学期     上半年 下半年  上半年 下半年
# 班级 性别
# 2  女    34  67   89  54
# 1  女    77  88   97  98
# 2  男    67  89   54  78
# 1  男    88  99   88  99

# print(df.sort_values(by=[('2','性别'),'下半年']))  # 也报错！
print(df.index.reorder_levels([1,0])) # 等级顺序，互换
# MultiIndex([('男', 1),
#             ('女', 1),
#             ('男', 2),
#             ('女', 2)],
#            names=['性别', '班级'])

# 个人测试 替换原df的行索引 成功
df.index = df.index.reorder_levels([1,0])
print(df)
# 年份    2019     2020
# 学期     上半年 下半年  上半年 下半年
# 性别 班级
# 男  1    88  99   88  99
# 女  1    77  88   97  98
# 男  2    67  89   54  78
# 女  2    34  67   89  54

print()
print(df.index.sortlevel(level=0,ascending=True)) # 按指定级别排序
# (MultiIndex([('女', 1),
#             ('女', 2),
#             ('男', 1),
#             ('男', 2)],
#            names=['性别', '班级']), array([1, 3, 0, 2], dtype=int64))

print(df.index.sortlevel(level=1,ascending=True)) # 个人测试 按照班级顺序排列
# (MultiIndex([('女', 1),
#             ('男', 1),
#             ('女', 2),
#             ('男', 2)],
#            names=['性别', '班级']), array([1, 0, 3, 2], dtype=int64))

print()
print(df.index.reindex(df.index[::-1])) # 更换顺序，或者指定一个顺序
# (MultiIndex([('女', 2),
#             ('男', 2),
#             ('女', 1),
#             ('男', 1)],
#            names=['性别', '班级']), array([3, 2, 1, 0], dtype=int64))

# str_list = ['a','b','c','d']
# print(str_list[::-1])


print()
print('\t8.2.6 其他操作')
'''以下是一些其他操作'''
print(df.index.to_numpy()) # 生成一个笛卡儿积的元组对序列
# [('男', 1) ('女', 1) ('男', 2) ('女', 2)]

print(df.index.remove_unused_levels())  # 返回没有使用的层级 | 搞不懂
# MultiIndex([('男', 1),
#             ('女', 1),
#             ('男', 2),
#             ('女', 2)],
#            names=['性别', '班级'])

print(df.swaplevel(0,1)) # 交换索引
# 年份    2019     2020
# 学期     上半年 下半年  上半年 下半年
# 班级 性别
# 1  男    88  99   88  99
#    女    77  88   97  98
# 2  男    67  89   54  78
#    女    34  67   89  54
# 个人测试
# print(df.columns.swaplevel(0,1)) # 测试能交换 返回列索引

print()
# TODO print(df.to_frame()) # # 转为DataFrame  运行报错

# 个人测试 有效
# 假设df是一个Series
# df = pd.Series([1, 2, 3, 4],name='num')
# print(df)
#
# # 将Series转换为DataFrame
# df_df = df.to_frame()
#
# # 打印转换后的DataFrame
# print(df_df)

'''
在Pandas中，to_frame()方法用于将Series对象转换为DataFrame。
如果尝试在一个已经为DataFrame类型的对象上调用此方法，会引发错误。
解决这个问题的关键是确认您正在操作的对象类型，并根据其类型选择正确的操作。
如果对象是Series，使用to_frame()是合适的；
如果对象是DataFrame，则不需要使用此方法。
'''

print()
print('# 删除指定等级')
print(df.index.droplevel(0))
# Index([1, 1, 2, 2], dtype='int64', name='班级')

# TODO 个人测试 有效
# df.index = df.index.droplevel(0)
# print(df)
# 年份 2019     2020
# 学期  上半年 下半年  上半年 下半年
# 班级
# 1    88  99   88  99
# 1    77  88   97  98
# 2    67  89   54  78
# 2    34  67   89  54

print()
print('# 返回索引的位置')
print(df.index.get_locs(('女',2))) # [3]
print(df.index.get_loc(('女',2))) # 3
# [3]

'''
8.2.7 小结
多层索引的基础操作与普通索引的操作一样。
本节介绍了多层索引的一些特殊操作，如查看索引的信息、索引内容、索引的层级及排序等。
在Pandas中，大多数方法都针对多层索引进行了适配，可传入类似level的参数对指定层级进行操作。
'''

print()
print('------------------------------------------------------------')
print('第8章 Pandas多层索引')
print('\t8.3 数据查询')
print('\t8.3.1 查询行')
print()
# update20240510
'''
截至目前，我们处理的数据基本都是一列索引和一行表头，但在实际业务中会存在有多层索引的情况。
多层索引（Hierarchical indexing）又叫多级索引，
它为一些非常复杂的数据分析和操作（特别是处理高维数据）提供了方法。
从本质上讲，它使你可以在Series（一维）和DataFrame（二维）等较低维度的数据结构中存储和处理更高维度的数据。

'''
# 小节注释
'''
多层索引组成的数据相对复杂，在确定需求后我们要清晰判断是哪个层级下的数据，
并充分运用本节的内容进行各角度的数据筛选。
需要注意的是，如果行或列中有一个是单层索引，那么与之前介绍过的单层索引一样操作。
本节中的行和列全是多层索引。

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print('# 测试读取 含有多重索引的excel') # 测试成功！
df = pd.read_excel(multindex_file,header=[0,1],index_col=[0,1])
print(df.index)
# print(df)

print()
print('# 典型的多层索引数据的生成过程')
# 索引
index_arrays = [[1,1,2,2],['男','女','男','女']]
# 列名
columns_arrays = [['2019','2019','2020','2020'],
                  ['上半年','下半年','上半年','下半年']]

# 索引转换为多层
index = pd.MultiIndex.from_arrays(index_arrays,
                                  names=('班级','性别'))

# 列名转换为多层
columns = pd.MultiIndex.from_arrays(columns_arrays,
                                    names=('年份','学期'))
# 应用到Dataframe中
df = pd.DataFrame([(88,99,88,99),(77,88,97,98),
                    (67,89,54,78),(34,67,89,54)],
                   columns=columns,index=index)

print(df)

print()
print('# 查看一班的数据')
print(df.loc[1])
# 年份 2019     2020
# 学期  上半年 下半年  上半年 下半年
# 性别
# 男    88  99   88  99
# 女    77  88   97  98

print(df.loc[1:2]) #  查询1班和2班的数据

'''如果我们要同时根据一二级索引查询，可以将需要查询的索引条件组成一个元组：'''
print()
print(df.loc[(1,'男')])

# 个人测试 输出转换为dataframe 存入文件不用使用to_frame()  有效果！
# df1 = df.loc[(1,'男')]
# print(df1.to_frame())
# df1.to_excel('E:/bat/output_files/pandas_out_20240510052.xlsx')


print()
print('\t8.3.2 查询列')
'''
查询列时，可以直接用切片选择需要查询的列，使用元组指定相关的层级数据：
'''

print(df['2020']) # 整个一级索引下
# 学期     上半年  下半年
# 班级 性别
# 1  男    88   99
#    女    97   98
# 2  男    54   78
#    女    89   54
print()
print(df[('2020','上半年')]) # # 指定二级索引
print(df['2020']['上半年']) # 结果同上

print()
print('\t8.3.3 行列查询')
'''
行列查询和单层索引一样，指定层内容也用元组表示。slice(None)可以在元组中占位，表示本层所有内容：
'''
print(df.loc[(1,'男'),'2020']) # 只显示2020年1班男生
# 学期
# 上半年    88
# 下半年    99
# Name: (1, 男), dtype: int64

# df1 = df.loc[(1,'男'),'2020']  || 文件不显示年份
# df1.to_excel('E:/bat/output_files/pandas_out_20240510053.xlsx')

print()
print('# 只看下半年')
print(df.loc[:,(slice(None),'下半年')]) #
# 年份    2019 2020
# 学期     下半年  下半年
# 班级 性别
# 1  男    99   99
#    女    88   98
# 2  男    89   78
#    女    67   54

# TODO 个人测试
# df1 = df.loc[:,(slice(None),'下半年')]  # || 文件如输出显示
# df1.to_excel('E:/bat/output_files/pandas_out_20240510053.xlsx')

print()
print('# 只看女生')
print(df.loc[(slice(None),'女'),:])
# 年份    2019     2020
# 学期     上半年 下半年  上半年 下半年
# 班级 性别
# 1  女    77  88   97  98
# 2  女    34  67   89  54

print('# 只看一班')
print(df.loc[1,(slice(None)),:])
# 年份 2019     2020
# 学期  上半年 下半年  上半年 下半年
# 性别
# 男    88  99   88  99
# 女    77  88   97  98

print()
print(df.loc[(1,slice(None)),:])  # 逻辑同上 显示班级
# 年份    2019     2020
# 学期     上半年 下半年  上半年 下半年
# 班级 性别
# 1  男    88  99   88  99
#    女    77  88   97  98

print()
print('# 只看2020年的数据')
print(df.loc[:,('2020',slice(None))])
# 年份    2020
# 学期     上半年 下半年
# 班级 性别
# 1  男    88  99
#    女    97  98
# 2  男    54  78
#    女    89  54

print()
print('\t8.3.4 条件查询')
'''
按照一定的条件查询数据，和单层索引的数据查询一样，不过在选择列上要按多层的规则.
'''

print(df[df[('2020','上半年')]>80])
# 年份    2019     2020
# 学期     上半年 下半年  上半年 下半年
# 班级 性别
# 1  男    88  99   88  99
#    女    77  88   97  98
# 2  女    34  67   89  54
print(df[df.loc[:,('2020','上半年')]>80]) # 结果同上！

print()
print('\t8.3.5 用pd.IndexSlice索引数据')
'''
pd.IndexSlice可以创建一个切片对象，轻松执行复杂的索引切片操作：
'''
# idx = pd.IndexSlice
# idx[0] # 0
# idx[:] # slice(None, None, None)
# idx[0,'x'] # (0, 'x')
# idx[0:3] # slice(0, 3, None)
# idx[0.1:1.5] # slice(0.1, 1.5, None)
# idx[0:5,'x':'y'] # (slice(0, 5, None), slice('x', 'y', None))

# 应用在查询中：
idx = pd.IndexSlice
print(df.loc[idx[:,['男']],:]) # 只显示男生
# 年份    2019     2020
# 学期     上半年 下半年  上半年 下半年
# 班级 性别
# 1  男    88  99   88  99
# 2  男    67  89   54  78
print(df.loc[:,idx[:,['上半年']]]) # 只显示上半年
# 年份    2019 2020
# 学期     上半年  上半年
# 班级 性别
# 1  男    88   88
#    女    77   97
# 2  男    67   54
#    女    34   89

print()
print('\t8.3.6 df.xs()')
''' 使用df.xs()方法采用索引内容作为参数来选择多层索引数据中特定级别的数据：'''

print(df.xs((1,'男'))) # 1班男生
# 年份    学期
# 2019  上半年    88
#       下半年    99
# 2020  上半年    88
#       下半年    99
# Name: (1, 男), dtype: int64

print()
print(df.xs('2020',axis=1)) # 2020年

print()
print(df.xs('男',level=1)) # 所有男生

# print()
# print(df.xs(1)) # 个人测试 1班 运行成功  print(df.xs('男')) 报错！

print()
print('参数drop_level=0')
print(df.xs(1,drop_level=0)) # 在返回的结果中保留层级
# 年份    2019     2020
# 学期     上半年 下半年  上半年 下半年
# 班级 性别
# 1  男    88  99   88  99
#    女    77  88   97  98

print(df.xs(1)) # 无参数 drop_level=false  或 drop_level=0
# 年份 2019     2020
# 学期  上半年 下半年  上半年 下半年
# 性别
# 男    88  99   88  99
# 女    77  88   97  98

'''
xs 只能选择索引的一个层级，如果需要进行更复杂的选择，可能需要使用 loc 或 iloc 等其他方法。

在实际业务中，xs 方法特别适用于需要快速选择特定层级数据的场景。
例如，在金融数据分析中，可能需要选择某一特定日期或某一特定证券的数据，这时 xs 就非常有用。
然而，对于更加复杂的数据选择任务，可能需要使用其他更加灵活的方法。

8.3.7 小结
本节介绍了多层索引的数据查询操作，这些操作让我们可以方便地对于复杂的多层数据按需求进行查询。
和单层索引数据一样，多层索引数据也可以使用切片、loc、iloc等操作，只是需要用元组表达出层级。

'''