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

# TODO 忽略特定的警告 | 当遇到 openpyxl 中的 UserWarning 类型的警告时，它们将不会被打印出来。
warnings.filterwarnings('ignore',category=UserWarning,module='openpyxl')

print()
print('------------------------------------------------------------')
print('第9章 Pandas数据重塑与透视')
print('\t9.1 数据查询')
print('\t9.1.1 整理透视')
print()
# update20240513
'''
数据透视表，顾名思义，是指它有“透视”数据的能力，可以找出大量复杂无关的数据的内在关系，
将数据转化为有意义、有价值的信息，从而看到它所代表的事物的规律和本质。

数据透视是最常用的数据汇总工具，Excel中经常会做数据透视，它可以根据一个或多个指定的维度来聚合数据。
Pandas也提供了数据透视函数来实现这些功能。
'''
# 小节注释
'''
要实现基础的透视操作，可以使用df.pivot()返回按给定的索引、列值重新组织整理后的DataFrame。
df.pivot()有3个参数:

index：作为新DataFrame的索引，取分组去重的值；如果不传入，则取现有索引。
columns：作为新DataFrame的列，取去重的值，当列和索引的组合有多个值时会报错，需要使用pd.pivot_table()进行操作。
values：作为新DataFrame的值，如果指定多个，会形成多层索引；如果不指定，会默认为所有剩余的列。

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print('------------------------------------------------------------')
print('\t9.1.2 整理透视操作')
print()
'''

'''
# 构造数据
df = pd.DataFrame({'A':['a1','a1','a2','a2','a3','a3'],
                   'B':['b1','b2','b3','b1','b2','b3'],
                   'C':['c1','c2','c3','c4','c5','c6'],
                   'D':['d1','d2','d3','d4','d5','d6'],})
# 展示数据
# print(df)
#     A   B   C   D
# 0  a1  b1  c1  d1
# 1  a1  b2  c2  d2
# 2  a2  b3  c3  d3
# 3  a2  b1  c4  d4
# 4  a3  b2  c5  d5
# 5  a3  b3  c6  d6

print('# 透视，指定索引、列、值')
print(df.pivot(index='A',columns='B',values='C'))
# B    b1   b2   b3
# A
# a1   c1   c2  NaN
# a2   c4  NaN   c3
# a3  NaN   c5   c6

'''
上面的代码取A列的去重值作为索引，取B列的去重值作为列，取C列内容作为具体数据值。
两个轴交叉处的取值方法是原表中A与B对应C列的值，如果无值则显示NaN。
'''

print()
print('# 不指定值内容')
print(df.pivot(index='A',columns='B'))
#       C              D
# B    b1   b2   b3   b1   b2   b3
# A
# a1   c1   c2  NaN   d1   d2  NaN
# a2   c4  NaN   c3   d4  NaN   d3
# a3  NaN   c5   c6  NaN   d5   d6

'''其效果和以下代码相同：'''
print(df.pivot(index='A',columns='B',values=['C','D'])) # 结果同上

print()
print('------------------------------------------------------------')
print('\t9.1.3 聚合透视')

'''
df.pivot()只是对原数据的结构、显示形式做了变换，在实现业务中，
往往还需要在数据透视过程中对值进行计算，这时候就要用到pd.pivot_table()。
它可以实现类似Excel那样的高级数据透视功能。

pd.pivot_table()有以下几个关键参数。
    data：要透视的DataFrame对象。
    index：在数据透视表索引上进行分组的列。
    columns：在数据透视表列上进行分组的列。
    values：要聚合的一列或多列。
    aggfunc：用于聚合的函数，默认是平均数mean。
    fill_value：透视会以空值填充值。
    margins：是否增加汇总行列。
'''

print()
print('------------------------------------------------------------')
print('\t9.1.4 聚合透视操作')

df = pd.DataFrame({
                    'A':['a1', 'a1', 'a1', 'a2', 'a2', 'a2'],
                    'B':['b2', 'b2', 'b1', 'b1', 'b1', 'b1'],
                    'C':['c1', 'c1', 'c2', 'c2', 'c1', 'c1'],
                    'D':[1, 2, 3, 4, 5, 6],
                    })

print(df)
#     A   B   C  D
# 0  a1  b2  c1  1
# 1  a1  b2  c1  2
# 2  a1  b1  c2  3
# 3  a2  b1  c2  4
# 4  a2  b1  c1  5
# 5  a2  b1  c1  6

# 报错！ValueError: Index contains duplicate entries, cannot reshape
# print(df.pivot(index='A',columns='B',values='D'))

'''
问题解释
例如，在原始数据中，当 A=a1 和 B=b2 时，D 有两个值（1 和 2），这导致了重复，因为 pivot 期望每个 A-B 组合只对应一个 D 值。
'''

print()
print('# pd.pivot_table()透视')
print(pd.pivot_table(df,index='A',columns='B',values='D'))
# B    b1   b2
# A
# a1  3.0  1.5
# a2  5.0  NaN

'''需要将这些重复数据按一定的算法计算出来，pd.pivot_table()默认的算法是取平均值。'''

# 验证数据
# 筛选a2和b1的数据
print(df.loc[(df.A == 'a2') & (df.B == 'b1')])
# 对D求平均值
print(df.loc[(df.A == 'a2') & (df.B=='b1')].D.mean())
# 5.0

print()
print('------------------------------------------------------------')
print('\t9.1.5 聚合透视高级操作')
print('# 高级聚合')
print(pd.pivot_table(df,index=['A','B'], # 指定多个索引
                     columns='C', # 指定列
                     values='D', # 指定数据值
                     fill_value=0, # 将聚合为空的值填充为0
                     aggfunc=np.sum, # 指定聚合方法为求和 默认aggfunc='mean'  aggfunc='sum'/np.sum 结果相同
                     margins=True # 增加行列汇总
                     ))

# C       c1  c2  All
# A   B
# a1  b1   0   3    3
#     b2   3   0    3
# a2  b1  11   4   15
# All     14   7   21

print()
print('# 使用多个聚合计算')
print(pd.pivot_table(df,index=['A','B'],
                     columns=['C'],
                     values='D',
                     aggfunc=[np.mean,np.sum]
                     ))

#       mean        sum
# C       c1   c2    c1   c2
# A  B
# a1 b1  NaN  3.0   NaN  3.0
#    b2  1.5  NaN   3.0  NaN
# a2 b1  5.5  4.0  11.0  4.0

print()
print('# 为各列分别指定计算方法')
df = pd.DataFrame({
                'A':['a1', 'a1', 'a1', 'a2', 'a2', 'a2'],
                'B':['b2', 'b2', 'b1', 'b1', 'b1', 'b1'],
                'C':['c1', 'c1', 'c2', 'c2', 'c1', 'c1'],
                'D':[1, 2, 3, 4, 5, 6],
                'E':[9, 8, 7, 6, 5, 4],
                })

print(pd.pivot_table(df,
                     index=['A','B'],
                     columns=['C'],
                     aggfunc={'D':np.mean,'E':np.sum},
                     # fill_value=0
                     ))

#          D          E
# C       c1   c2    c1   c2
# A  B
# a1 b1  NaN  3.0   NaN  7.0
#    b2  1.5  NaN  17.0  NaN
# a2 b1  5.5  4.0   9.0  6.0

'''最终形成的数据，D列取平均，E列取和。'''

'''
9.1.6 小结
本节介绍了Pandas如何进行透视操作。df.pivot()是对数据进行整理，变换显示方式，
而pd.pivot_table()会在整理的基础上对重复的数据进行相应的计算。
Pandas透视表功能和Excel类似，但Pandas对数据的聚合方法更加灵活，
能够实现更加复杂的需求，特别是在处理庞大的数据时更能展现威力。
'''

print()
print('------------------------------------------------------------')
print('第9章 Pandas数据重塑与透视')
print('\t9.2 数据堆叠')
print('\t9.2.1 理解堆叠')
print()
# update20240514
'''
数据的堆叠也是数据处理的一种常见方法。在多层索引的数据中，通常为了方便查看、对比，会将所有数据呈现在一列中；
相反，对于层级过多的数据，我们可以实施解堆操作，让它呈现多列的状态。
'''
# 小节注释
'''
数据堆叠可以简单理解成将多列数据转为一列数据（见图9-3）。
如果原始数据有多个数据列，堆叠（stack）的过程表示将这些数据列的所有数据表全部旋转到行上；
类似地，解堆（unstack）的过程表示将在行上的索引旋转到列上。
解堆是堆叠的相反操作。


堆叠和解堆的本质如下。
堆叠：“透视”某个级别的（可能是多层的）列标签，返回带有索引的DataFrame，
      该索引带有一个新的行标签，这个新标签在原有索引的最右边。

解堆：将（可能是多层的）行索引的某个级别“透视”到列轴，从
     而生成具有新的最里面的列标签级别的重构的DataFrame。
     堆叠过程将数据集的列转行，解堆过程为行转列。
     上例中，原始数据集索引有两层，堆叠过程就是将最左的列转到最内层的行上，
     解堆是将最内层的行转移到最内层的列索引中。

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()


print()
print('------------------------------------------------------------')
print('\t9.2.2 堆叠操作df.stack()')

df = pd.DataFrame({
    'A':['a1','a1','a2','a2'],
    'B':['b1','b2','b1','b2'],
    'C':[1,2,3,4],
    'D':[5,6,7,8],
    'E':[5,6,7,8],
})

# 设置多层索引
df.set_index(['A','B'],inplace=True)
print(df)
#        C  D  E
# A  B
# a1 b1  1  5  5
#    b2  2  6  6
# a2 b1  3  7  7
#    b2  4  8  8

print()
print('# 进行堆叠')
print(df.stack())
# A   B
# a1  b1  C    1
#         D    5
#         E    5
#     b2  C    2
#         D    6
#         E    6
# a2  b1  C    3
#         D    7
#         E    7
#     b2  C    4
#         D    8
#         E    8
# dtype: int64

# df.stack().to_excel('E:/bat/output_files/pandas_out_20240514023.xlsx')

# 查看类型
# print(type(df.stack())) # <class 'pandas.core.series.Series'>
'''我们看到生成了一个Series，所有的列都透视在了多层索引的新增列中。'''

print()
print('------------------------------------------------------------')
print('\t9.2.3 解堆操作df.unstack()')
print('# 将原来的数据堆叠并赋值给s')

s = df.stack()
# print(s)
print('# 操作解堆')
print(s.unstack())
#        C  D  E
# A  B
# a1 b1  1  5  5
#    b2  2  6  6
# a2 b1  3  7  7
#    b2  4  8  8

'''解堆后生成的是一个DataFrame。'''

'''
9.2.4 小结
数据的堆叠和解堆分别用来解决数据的展开和收缩问题。
堆叠让数据变成一维数据，可以让我们从不同维度来观察和使用数据。
解堆和堆叠互为相反操作。
'''

print()
print('------------------------------------------------------------')
print('第9章 Pandas数据重塑与透视')
print('\t9.3 交叉表')
print('\t9.3.1 基本语法')
print()
# update20240514
'''
交叉表（cross tabulation）是一个很有用的分析工具，是用于统计分组频率的特殊透视表。
简单来说，交叉表就是将两列或多列中不重复的元素组成一个新的DataFrame，新数据的行和列交叉部分的值为其组合在原数据中的数量。
'''
# 小节注释
'''
交叉表的基本语法如下：
    # 基本语法
    pd.crosstab(index, columns, values=None, rownames=None,
                colnames=None, aggfunc=None, margins=False,
                margins_name: str = 'All', dropna: bool = True,
                normalize=False)

参数说明如下。
    index：传入列，如df['A']，作为新数据的索引。
    columns：传入列，作为新数据的列，新数据的列为此列的去重值。
    values：可选，传入列，根据此列的数值进行计算，计算方法取aggfunc参数指定的方法，此时aggfunc为必传。
    aggfunc：函数，values列计算使用的计算方法。
    rownames：新数据和行名，一个序列，默认值为None，必须与传递的行数、组数匹配。
    colnames：新数据和列名，一个序列，默认值为None；如果传递，则必须与传递的列数、组数匹配。
    margins：布尔值，默认值为False，添加行/列边距（小计）。
    normalize：布尔值，{'all'，'index'，'columns'}或{0,1}，默认值为False。通过将所有值除以值的总和进行归一化。

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()


print()
print('------------------------------------------------------------')
print('\t9.3.2 生成交叉表')

# 原数据
df = pd.DataFrame({
                    'A':['a1', 'a1', 'a2', 'a2', 'a1'],
                    'B':['b2', 'b1', 'b2', 'b2', 'b1'],
                    'C':[1, 2, 3, 4, 5],
                    })

# 生成交叉表
print(pd.crosstab(df['A'],df['B']))
'''
B   b1  b2
A         
a1   2   1
a2   0   2
'''

'''
在上例中，A、B两列进行交叉，A有不重复的值a1和a2，B有不重
复的值b1和b2。交叉后组成了新的数据，行和列的索引分别是a1、a2和
b1、b2，它们交叉位上对应的值为此组合的数量，如a1、b1组合有两
个，所以它们的交叉位上的值为2；没有a2、b1组合，所以对应的值为
0。
'''

print()
print('# 对分类数据做交叉')
one = pd.Categorical(['a','b'],categories=['a','b','c'])
two = pd.Categorical(['d','e'],categories=['d','e','f'])
print(pd.crosstab(one,two))
# col_0  d  e
# row_0
# a      1  0
# b      0  1

print()
# print(one)
# print(two)
# ['a', 'b']
# Categories (3, object): ['a', 'b', 'c']
# ['d', 'e']
# Categories (3, object): ['d', 'e', 'f']


print()
print('------------------------------------------------------------')
print('\t9.3.3 归一化')

'''
normalize参数可以帮助我们实现数据归一化，算法为对应值除以所
有值的总和，让数据处于0～1的范围，方便我们观察此位置上的数据在
全体中的地位。
'''

print(pd.crosstab(df['A'],df['B']))
# B   b1  b2
# A
# a1   2   1
# a2   0   2

print('#  交叉表，归一化')
print(pd.crosstab(df['A'],df['B'],normalize=True))
# B    b1   b2
# A
# a1  0.4  0.2
# a2  0.0  0.4

print('交叉表，按列归一化')
print(pd.crosstab(df['A'],df['B'],normalize='columns'))
# B    b1        b2
# A
# a1  1.0  0.333333
# a2  0.0  0.666667
print('交叉表，按行归一化')
print(pd.crosstab(df['A'],df['B'],normalize='index'))
# B         b1        b2
# A
# a1  0.666667  0.333333
# a2  0.000000  1.000000

print()
print('------------------------------------------------------------')
print('\t9.3.4 指定聚合方法')

'''
用aggfunc指定聚合方法对values指定的列进行计算：
'''

print('# 交叉表，按C列的和进行求和聚合')
print(pd.crosstab(df['A'],df['B'],values=df['C'],aggfunc=np.sum))
# B    b1   b2
# A
# a1  7.0  1.0
# a2  NaN  7.0

print()
print('------------------------------------------------------------')
print('\t9.3.5 汇总')

'''
margins=True可以增加行和列的汇总，按照行列方向对数据求和，
类似margins_name='total'可以定义这个汇总行和列的名称：
'''

print('# 交叉表，增加汇总')
print(pd.crosstab(df['A'],df['B'],
                  values=df['C'],
                  aggfunc=np.sum,
                  margins=True,
                  margins_name='total', # 定义汇总行列名称
                  ))
# B       b1   b2  total
# A
# a1     7.0  1.0      8
# a2     NaN  7.0      7
# total  7.0  8.0     15

'''
9.3.6 小结
交叉表将原始数据中的两个列铺开，形成这两列所有不重复值的交叉位，
在交叉位上填充这个值在原始数据中的组合数。
交叉表可以帮助我们了解标签数据的构成情况。
'''


print()
print('------------------------------------------------------------')
print('第9章 Pandas数据重塑与透视')
print('\t9.4 数据转置df.T')
print('\t9.4.1 理解转置')
print()
# update20240515
'''
数据的转置是指将数据的行与列进行互换，它会使数据的形状和逻辑发生变化。
Pandas提供了非常便捷的df.T操作来进行转置。
'''
# 小节注释
'''
在数据处理分析过程中，为了充分利用行列的关系表达，
我们需要将原数据的行与列进行互换。转置其实是将数据沿着左上与右下形成的对角线进行翻转，

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()


print()
print('------------------------------------------------------------')
print('\t9.4.2 转置操作')

'''df.T属性是df.transpose()方法的简写形式，我们只要记住.T就可以快速进行转置操作。'''
# 原数据
df = pd.DataFrame({
                    'A':['a1', 'a2', 'a3', 'a4', 'a5'],
                    'B':['b1', 'b2', 'b3', 'b4', 'b5'],
                    'C':[1, 2, 3, 4, 5],
                    })

# 转置
print(df.T)
#     0   1   2   3   4
# A  a1  a2  a3  a4  a5
# B  b1  b2  b3  b4  b5
# C   1   2   3   4   5

'''
我们观察到，a1和5这两个值始终在左上和右下，
以它们连成的对角线作为折线对数据进行了翻转，让列成为了行，行成为了列。
Series也支持转置，不过它返回的是它自己，没有变化。
'''

print()
print('------------------------------------------------------------')
print('\t9.4.3 类型变化')
'''一般情况下数据转置后，列的数据类型会发生变化：'''
print('# 原始数据的数据类型')
print(df.dtypes)
# A    object
# B    object
# C     int64
# dtype: object

print('# 转置后的数据类型')
print(df.T.dtypes)
# 0    object
# 1    object
# 2    object
# 3    object
# 4    object
# dtype: object

'''这是因为，数据结构带来的巨大变化会让数据类型重新得到确定。'''

print()
print('------------------------------------------------------------')
print('\t9.4.4 轴交换df.swapaxes()')

'''
Pandas提供了df.swapaxes(axis1, axis2, copy=True)方法，
该方法用来进行轴（行列）交换，如果是进行行列交换，它就相当于df.T。
'''
print(df)
print(df.swapaxes('index','columns')) # 行列交换，相当于df.T
print(df.swapaxes('columns','index')) # 同上

print('copy=True')
print(df.swapaxes('index','columns',copy=True)) # 结果同上
#     0   1   2   3   4
# A  a1  a2  a3  a4  a5
# B  b1  b2  b3  b4  b5
# C   1   2   3   4   5

print('# 无变化')
print(df.swapaxes('index','index'))
#     A   B  C
# 0  a1  b1  1
# 1  a2  b2  2
# 2  a3  b3  3
# 3  a4  b4  4
# 4  a5  b5  5

'''
9.4.5 小结
转置操作让我们能够以另一个角度看数据。随着行列的交换，
数据的意义也会发生变化，让数据的内涵表达增添了一种可能性。
'''

print()
print('------------------------------------------------------------')
print('第9章 Pandas数据重塑与透视')
print('\t9.5 数据融合')
print('\t9.5.1 基本语法')
print()
# update20240515
'''
数据融合df.melt()是df.pivot()的逆操作函数，简单来说，
它是将指定的列铺开，放到行上名为variable（可指定）、值为value（可指定）列。
'''
# 小节注释
'''
具体的语法结构如下：
    pd.melt(frame: pandas.core.frame.DataFrame,
            id_vars=None, value_vars=None,
            var_name='variable', value_name='value',
            col_level=None)

参数说明如下:
    id_vars：tuple、list或ndarray（可选），用作标识变量的列。
    value_vars：tuple、list或ndarray（可选），要取消透视的列。如果未指定，则使用未设置为id_vars的所有列。
    var_name：scalar，用于“变量”列的名称。如果为None，则使用frame.columns.name或“variable”。
    value_name：scalar，默认为“value”，用于“value”列的名称。
    col_level：int或str（可选），如果列是多层索引，则使用此级别来融合。
▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print()
print('------------------------------------------------------------')
print('\t9.5.2 融合操作')

# 原数据
df = pd.DataFrame({
                    'A':['a1', 'a2', 'a3', 'a4', 'a5'],
                    'B':['b1', 'b2', 'b3', 'b4', 'b5'],
                    'C':[1, 2, 3, 4, 5],
                    })

# 数据融合
print(pd.melt(df))
   # variable value
# 0         A    a1
# 1         A    a2
# ...
# 13        C     4
# 14        C     5
'''以上操作将A、B、C三列的标识和值展开，一列为标签，默认列名variable，另一列为值，默认列名为value。'''

print()
print('------------------------------------------------------------')
print('\t9.5.3 标识和值')
'''
数据融合时，可以指定标识，下例指定A、B两列作为融合操作后的标识，保持不变，对其余列（C列）展开数据：
'''

print('# 数据融合，指定标识')
print(pd.melt(df,id_vars=['A','B']))
#     A   B variable  value
# 0  a1  b1        C      1
# 1  a2  b2        C      2
# 2  a3  b3        C      3
# 3  a4  b4        C      4
# 4  a5  b5        C      5

print('# 数据融合，指定值列')
print(pd.melt(df,value_vars=['B','C']))
'''
  variable value
0        B    b1
1        B    b2
2        B    b3
3        B    b4
4        B    b5
5        C     1
6        C     2
7        C     3
8        C     4
9        C     5
'''

# print()
# print(pd.melt(df,id_vars=['C','B']))

'''如果想保留标识，可以使用id_vars同时指定。'''

print()
print('------------------------------------------------------------')
print('\t9.5.4 指定名称')

'''标识的名称和值的名称默认分别是variable和value，我们可以指定它们的名称：'''

print('# 指定标识和值列的名称')
print(pd.melt(df,id_vars=['A'],value_vars=['B'],
              var_name='B_lable',value_name='B_value'
              ))
'''
    A B_lable B_value
0  a1       B      b1
1  a2       B      b2
2  a3       B      b3
3  a4       B      b4
4  a5       B      b5
'''

'''
9.5.5 小结
数据融合让数据无限度地铺开，它是透视的反向操作。
另外，对于原始数据是多层索引的，可以使用col_level=0参数指定融合时的层级。

'''


print()
print('------------------------------------------------------------')
print('第9章 Pandas数据重塑与透视')
print('\t9.6 虚拟变量')
print('\t9.6.1 基本语法')
print()
# update20240515
'''
虚拟变量（Dummy Variable）又称虚设变量、名义变量或哑变量，
是一个用来反映质的属性的人工变量，是量化了的自变量，通常取值为0或1，常被用于one-hot特征提取。
'''
# 小节注释
'''
简单来说，生成虚拟变量的方法pd.get_dummies()是将一列或多列的去重值作为新表的列，
每列的值由0和1组成：如果原来位置的值与列名相同，则在新表中该位置的值为1，否则为0。
这样就形成了一个由0和1组成的特征矩阵。


# 基本语法
pd.get_dummies(data, 
                prefix=None,
                prefix_sep='_', 
                dummy_na=False,
                columns=None, 
                sparse=False,
                drop_first=False, 
                dtype=None)

参数如下：
data：被操作的数据，DataFrame或者Series。
prefix：新列的前缀。
prefix_sep：新列前缀的连接符。

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print()
print('------------------------------------------------------------')
print('\t9.6.2 生成虚拟变量')

# 原数据
df = pd.DataFrame({
                    'a':list('adcb'),
                    'b':list('fehg'),
                    'a1':range(4),
                    'b1':range(4,8),
                    })

print(df)
# print(df.dtypes)
#    a  b  a1  b1
# 0  a  f   0   4
# 1  d  e   1   5
# 2  c  h   2   6
# 3  b  g   3   7

print('# 生成虚拟变量')
print(pd.get_dummies(df.a))
#        a      b      c      d
# 0   True  False  False  False
# 1  False  False  False   True
# 2  False  False   True  False
# 3  False   True  False  False

'''
分析一下执行结果，只关注df.a列：a列共有a、b、c、d四个值，
故新数据有此四列；
    索引和列名交叉的位置如果为1，说明此索引位上的值为列名，为0则表示不为列名。
    例如，在索引2上，1在c列，对比原数据发现df.loc[2, ‘a’]为c。
'''

print()
print('------------------------------------------------------------')
print('\t9.6.3 列前缀')

'''
有的原数据的部分列（如下例中的a1列）值为纯数字，为了方便使用，
需要给生成虚拟变量的列名增加一个前缀，用prefix来定义：
'''

print(pd.get_dummies(df.a1,prefix='a1'))
#     a1_0   a1_1   a1_2   a1_3
# 0   True  False  False  False
# 1  False   True  False  False
# 2  False  False   True  False
# 3  False  False  False   True

print()
print('------------------------------------------------------------')
print('\t9.6.4 从DataFrame生成')

'''可以直接对DataFrame生成虚拟变量，会将所有非数字列生成虚拟变量（数字列保持不变）：'''

print(pd.get_dummies(df))
#    a1  b1    a_a    a_b    a_c    a_d    b_e    b_f    b_g    b_h
# 0   0   4   True  False  False  False  False   True  False  False
# 1   1   5  False  False  False   True   True  False  False  False
# 2   2   6  False  False   True  False  False  False  False   True
# 3   3   7  False   True  False  False  False  False   True  False

print('# 只指定一列：')
# 只生成b列的虚拟变量
print(pd.get_dummies(df,columns=['b']))
#    a  a1  b1    b_e    b_f    b_g    b_h
# 0  a   0   4  False   True  False  False
# 1  d   1   5   True  False  False  False
# 2  c   2   6  False  False  False   True
# 3  b   3   7  False  False   True  False


print()
print('个人测试')
print(pd.get_dummies(df))
print(pd.get_dummies(df,drop_first=True)) # 删除非数字列的第一个
print(pd.get_dummies(df,drop_first=True,dtype=float)) # 值的显示格式  默认是布尔值
#    a1  b1  a_b  a_c  a_d  b_f  b_g  b_h
# 0   0   4  0.0  0.0  0.0  1.0  0.0  0.0
# 1   1   5  0.0  0.0  1.0  0.0  0.0  0.0
# 2   2   6  0.0  1.0  0.0  0.0  0.0  1.0
# 3   3   7  1.0  0.0  0.0  0.0  1.0  0.0

'''
9.6.5 小结
虚拟变量生成操作将数据进行变形，形成一个索引与值（变形后成为列）的二维矩阵，
在对应交叉位上用1表示有此值，0表示无此值。虚拟变量经常用于与特征提取相关的机器学习场景。
'''


print()
print('------------------------------------------------------------')
print('第9章 Pandas数据重塑与透视')
print('\t9.7 因子化')
print('\t9.7.1 基本方法')
print()
# update20240515
'''
因子化是指将一个存在大量重复值的一维数据解析成枚举值的过程，这样可以方便我们进行分辨。
factorize既可以用作顶层函数pd.factorize()，也可以用作Series.factorize()和Index.factorize()方法。
'''
# 小节注释
'''
对数据因子化后返回两个值，一个是因子化后的编码列表，另一个是原数据的去重值列表：

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print('# 数据')
data = ['b','b','a','c','b']

print('# 因子化')
codes,uniques = pd.factorize(data)

# 编码
print(codes) # [0 0 1 2 0]
# 去重值
print(uniques) # ['b' 'a' 'c']

'''
上例中，我们将列表data进行了因子化，返回一个由两个元素组成的元组，我们用codes和uniques分别来承接这个元组的元素。
codes：数字编码表，将第一个元素编为0，其后依次用1、2等表示，但遇到相同元素使用相同编码。
       这个编码表的长度与原始数据相同，用编码对原始数据进行一一映射。
uniques：去重值，就是因子。

以上数据都是可迭代的array类型，需要注意的是codes和uniques是我们定义的变量名，
强烈推荐大家这么命名，这样代码会变得容易理解。

对Series操作后唯一值将生成一个index对象：
'''
print('示例2')
cat = pd.Series(['a','a','c'])
codes,uniques = pd.factorize(cat)
print(codes) # [0 0 1]
print(uniques) # Index(['a', 'c'], dtype='object')

print()
print('------------------------------------------------------------')
print('\t9.7.2 排序')

'''
使用sort=True参数后将对唯一值进行排序，编码列表将继续与原值保持对应关系，但从值的大小上将体现出顺序。
'''
codes,uniques = pd.factorize(['b','b','a','c','b',],sort=True)
print(codes)
print(uniques)
# [1 1 0 2 1]
# ['a' 'b' 'c']

# 非排序
codes,uniques = pd.factorize(['b','b','a','c','b',])
# [0 0 1 2 0]
# ['b' 'a' 'c']

print()
print('------------------------------------------------------------')
print('\t9.7.3 缺失值')
'''缺失值不会出现在唯一值列表中'''

codes,uniques = pd.factorize(['b',None,'a','c','b',])
print(codes)
print(uniques)
# [ 0 -1  1  2  0]
# ['b' 'a' 'c']

print()
print('------------------------------------------------------------')
print('\t9.7.4 枚举类型')
'''Pandas的枚举类型数据（Categorical）也可以使用此方法：'''

cat = pd.Categorical(['a','a','c',],categories=['a','b','c'])
codes,uniques = pd.factorize(cat)
# print(cat)
print(codes)
print(uniques)
# [0 0 1]
# ['a', 'c']
# Categories (3, object): ['a', 'b', 'c']

'''
9.7.5 小结
简单来说，因子化方法pd.factorize()做了两件事：
    一是对数据进行数字编码，二是对数据进行去重。
    在大序列数据中，因子化能帮助我们抽取数据特征，将数据变成类别数据再进行分析。
'''

print()
print('------------------------------------------------------------')
print('第9章 Pandas数据重塑与透视')
print('\t9.8 爆炸列表')
print('\t9.8.1 基本功能')
print()
# update20240515
'''
爆炸这个词非常形象，在这里是指将类似列表的每个元素转换为一行，
索引值是相同的。这在我们处理一些需要展示的数据时非常有用。
'''
# 小节注释
'''
下面的两行数据中有类似列表的值，我们将它们炸开后，它们排好了队，但是依然使用原来的索引：

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print('# 原始数据')
s = pd.Series([[1,2,3],'foo',[],[3,4]])
print(s)
# 0    [1, 2, 3]
# 1          foo
# 2           []
# 3       [3, 4]
# dtype: object

print()
print('# 爆炸列')
print(s.explode())
# 0      1
# 0      2
# 0      3
# 1    foo
# 2    NaN
# 3      3
# 3      4
# dtype: object

'''
注意观察会发现，每行列表中的元素都独自占用了一行，
而索引保持不变，空列表值变成了NaN，非列表的元素没有变化。
这就是爆炸的作用，像一颗炸弹一样，将列表打散。
'''
print()
print('------------------------------------------------------------')
print('\t9.8.2 DataFrame的爆炸')
'''DataFrame可以对指定列实施爆炸'''

df = pd.DataFrame({'A':[[1,2,3],'foo',[],[3,4]],'B':range(4)})
print(df)
#            A  B
# 0  [1, 2, 3]  0
# 1        foo  1
# 2         []  2
# 3     [3, 4]  3

print('# 爆炸指定列')
print(df.explode('A'))
#      A  B
# 0    1  0
# 0    2  0
# 0    3  0
# 1  foo  1
# 2  NaN  2
# 3    3  3
# 3    4  3

'''在DataFrame中爆炸指定列后，其他列的值会保持不变。'''

print()
print('------------------------------------------------------------')
print('\t9.8.3 非列表格式')

'''对于不是列表但具有列表特质的数据，我们也可以在处理之后让其完成爆炸，如下面的数据：'''

# 原数据
df = pd.DataFrame([{'var1':'a,b,c','var2':1},
                   {'var1':'d,e,f','var2':2},
                   ])

print(df)
#     var1  var2
# 0  a,b,c     1
# 1  d,e,f     2

'''var1列的数据虽然是按逗号隔开的，但它不是列表，这时候我们先将其处理成列表，再实施爆炸：'''

print()
# print(df.explode('var1')) # 成功运行，但结果同上 无爆炸

print('# 使用指定同名列的方式对列进行修改')
df = df.assign(var1=df.var1.str.split(','))
print(df)
#         var1  var2
# 0  [a, b, c]     1
# 1  [d, e, f]     2
print(df.explode('var1'))
#   var1  var2
# 0    a     1
# 0    b     1
# 0    c     1
# 1    d     2
# 1    e     2
# 1    f     2

'''
9.8.4 小结
在实际的数据处理和办公环境中，经常会遇到用指定符号隔开的数据，我们分析之前要先进行数据的二维化。
Pandas提供的s.explode()列表爆炸功能可以方便地实施这一操作，我们需要灵活掌握。

9.9 本章小结
本章介绍了Pandas提供的数据变换操作，通过这些数据变换，可以观察数据的另一面，
探究数据反映出的深层次的业务意义。

另外一些操作（如虚拟变量、因子化等）可帮助我们提取出数据的特征，
为下一步数据建模、数据分析、机器学习等操作打下良好基础。
'''