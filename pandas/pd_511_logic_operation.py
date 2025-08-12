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

# 忽略特定的警告 | 当遇到 openpyxl 中的 UserWarning 类型的警告时，它们将不会被打印出来。
warnings.filterwarnings('ignore',category=UserWarning,module='openpyxl')


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.1 复杂查询')
print('\t5.1.1 逻辑运算')
print()
# update20240307
'''
在数据分析和数据建模的过程中需要对数据进行清洗和整理等工作，有时需要对数据增删字段。
本章将介绍Pandas对数据的复杂查询、数据类型转换、数据排序、数据的修改、数据迭代以及函数的使用。


第4章介绍了.loc[]等几个简单的数据筛选操作，但实际业务需求往往需要按照一定的条件甚至复杂的组合条件来查询数据。
本节将介绍如何发挥Pandas数据筛选的无限可能，随心所欲地取用数据。

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print('# dataframe单列逻辑运算') # 输出一个布尔值组成的series
print(df.Q1 > 36)
# 0     True
# 1    False
# 2     True
# 3     True
# 4     True
# 5     True
# Name: Q1, dtype: bool
print()
print('# 索引==1') # 输出一个array类型数组
print(df.index == 1)
# [False  True False False False False]
print(type(df.index == 1)) # <class 'numpy.ndarray'>

print()
print('')
print(df.head(2).loc[:,'Q1':'Q4'] > 60) # 只取数字部分，否则会因字符无大于运算而报错
#       Q1     Q2     Q3     Q4
# 0   True  False  False   True
# 1  False  False  False  False

print('# ~ & | 运算符号')
print(df.loc[(df.Q1 < 60) & (df['team'] == 'C')]) # Q1成绩小于60分，并且是C组成员
#    name team  Q1  Q2  Q3  Q4
# 1  Arry    C  36  37  37  57
print(df.loc[~(df.Q1 < 60) & (df['team'] == 'C')]) # Q1成绩不小于60分，并且是C组成员
#     name team  Q1  Q2  Q3  Q4
# 3  Eorge    C  93  96  71  78

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.1 复杂查询')
print('\t5.1.2 逻辑筛选数据')
print()
# update20240307
'''
切片（[]）、.loc[]和.iloc[]均支持上文所介绍的逻辑表达式。
通过逻辑表达式进行复杂条件的数据筛选时需要注意，表达式输出的结果必须是一个布尔序列或者符合其格式要求的数据形式。
例如，df.iloc[1+1]和df.iloc[lambda df: len(df)-1]计算出一个数值，符合索引的格式，
df.iloc[df.index==8]返回的是一个布尔序列，df.iloc[df.index]返回的是一个索引，它们都是有效的表达式。

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()


print('# 是切片（[]）的一些逻辑筛选的示例') # 输出一个布尔值组成的series

print()
print('# Q1等于36')
print(df[df['Q1'] == 36]) # Q1等于36
#    name team  Q1  Q2  Q3  Q4
# 1  Arry    C  36  37  37  57
print()
print('# Q1不等于36')
print(df[~(df['Q1'] == 36)]) # Q1不等于36
#     name team   Q1  Q2  Q3   Q4
# 0  Liver    E   89  21  24   64
# 2    Ack    A   57  60  18   84
# 3  Eorge    C   93  96  71   78
# 4    Oah    D   65  49  61   86
# 5   Rick    B  100  99  97  100
print()
print(df[df.name == 'Rick']) # 姓名为rick
print(df[df.Q1 > df.Q2])

print()
print('# 以下是.loc[]和.lic[]的一些示例：')
print(df.loc[df['Q1'] > 90,'Q1']) # Q1大于90，仅显示Q1列
print(df.loc[df['Q1'] > 90,'Q1':]) # Q1大于90，显示Q1及其后面所有的列
print(df.loc[(df.Q1 > 80) & (df.Q2 < 30)]) # and关系
print(df.loc[(df.Q1 > 90) | (df.Q2 < 30)]) # or关系
print(df.loc[df.Q1 == 100]) # 等于100
print(df.loc[df['Q1'] == 100]) # 等于100
'''
需要注意的是在进行或（|）、与（&）、非（~）运算时，各个独立逻辑表达式需要用括号括起来。
'''


print()
print('# any() 和 all() 函数')
print('# Q1、Q2成绩全为超过80分的')
print(df.loc[:,['Q1','Q2']])
print((df.loc[:,['Q1','Q2']] > 80))
# print(df[(df.loc[:,['Q1','Q2']] > 80)])
print((df.loc[:,['Q1','Q2']] > 80).all(axis=1))
print((df.loc[:,['Q1','Q2']] > 80).all(axis=0))


print()
# print('# Q1、Q2成绩全为超过80分的')
print(df[(df.loc[:,['Q1','Q2']] > 80).all(axis=1)])
#     name team   Q1  Q2  Q3   Q4
# 3  Eorge    C   93  96  71   78
# 5   Rick    B  100  99  97  100
print()
print('# Q1、Q2成绩至少有一个超过80分的')
print(df[(df.loc[:,['Q1','Q2']] > 80).any(axis=1)]) #any(1) 会报错！
#     name team   Q1  Q2  Q3   Q4
# 0  Liver    E   89  21  24   64
# 3  Eorge    C   93  96  71   78
# 5   Rick    B  100  99  97  100
print()
# print(df[(df.loc[:,['Q1','Q2']] > 80).all(0)])
# print(df[(df.loc[:,['Q1','Q2']] > 80).all(axis=0)])
# print(df[(df.loc[:,['Q1','Q2']] > 80).all(axis=0),['Q1','Q2']])


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.1 复杂查询')
print('\t5.1.3 函数筛选')
print()
# update20240307
'''
可以在表达式处使用lambda函数，默认变量是其操作的对象。如果操作的对象是一个DataFrame，那么变量就是这个DataFrame；
如果是一个Series，那么就是这个Series。可以看以下例子，s就是指df.Q1这个Series：

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print()
print('# 查询最大索引的值')
print(df.Q1[lambda s: max(s.index)]) # 值为100
print()
print('# 计算最大值')
print(max(df.Q1.index)) # 最大索引值 为 5

print()
# print(df[5]) # 报错！
print(df.Q1[5]) # 100
print(df.Q1[df.index==5])
# 5    100
# Name: Q1, dtype: int64
print(df.index==5) # [False False False False False  True]

print()
print('# 下面是一些lambda示例：')
print(df[lambda df: df['Q1'] == 100]) # Q1为100的
#    name team   Q1  Q2  Q3   Q4
# 5  Rick    B  100  99  97  100
print(df[df['Q1'] == 100]) # 结果同上

print()
print(df.loc[lambda df: df.Q1 == 100,'Q1':'Q2']) # Q1为100的,进显示'Q1','Q2' 列
#     Q1  Q2
# 5  100  99

print()
print('# 由真假值组成的序列')
print(df.loc[:,lambda df: df.columns.str.len()==4]) # 由真假值组成的序列
#     name team
# 0  Liver    E
# 1   Arry    C
# ....
print()
print(df.loc[:,lambda df: [i for i in df.columns if 'Q' in i]])
#     Q1  Q2  Q3   Q4
# 0   89  21  24   64
# 1   36  37  37   57
# ....
print()
print(df.iloc[:3,lambda df: df.columns.str.len() == 2])
#    Q1  Q2  Q3  Q4
# 0  89  21  24  64
# ...



print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.1 复杂查询')
print('\t5.1.4 比较函数')
print()
# update20240314
'''
Pandas提供了一些比较函数，使我们可以将逻辑表达式替换为函数形式。

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print()
print('# 以下相当于 df[df.Q1 == 60]')
print(df[df.Q1.eq(65)])
#   name team  Q1  Q2  Q3  Q4
# 4  Oah    D  65  49  61  86

print()
print('# 其它比较函数')
print(df[df.Q1.ne(60)]) # 不等于60  !=
print(df[df.Q1.le(57)]) # 小于等于57 <=
print(df[df.Q1.lt(57)]) # 小于57 <
print(df[df.Q1.ge(93)]) # 大于等于93 >=
print(df[df.Q1.gt(93)]) # 大于93 >
print(df.loc[df.Q1.gt(80) & df.Q2.lt(90)]) # Q1大于80 且 Q2小于90
print(df.loc[(df.Q1.gt(80)) & (df.Q2.lt(90))]) # 同上

print()
'''
这些函数可以传入一个定值、数列、布尔序列、Series或DataFrame，来与原数据比较。
另外还有一个. isin()函数，用于判断数据是否包含指定内容。可以传入一个列表，原数据只需要满足其中一个存在即可；
也可以传入一个字典，键为列名，值为需要匹配的值，以实现按列个性化匹配存在值。
'''
print('# eq() 字符串测试')
print(df[df.name.eq('Rick')]) # 字符串也行
#    name team   Q1  Q2  Q3   Q4
# 5  Rick    B  100  99  97  100

print()
print('# isin')
print(df[df.team.isin(['A','B'])]) # 包含A、B两组的
#    name team   Q1  Q2  Q3   Q4
# 2   Ack    A   57  60  18   84
# 5  Rick    B  100  99  97  100
print(df[df.isin({'team':['C','D'],'Q1':[36,96]})]) # 复杂查询 其他值为NaN
#   name team    Q1  Q2  Q3  Q4
# 0  NaN  NaN   NaN NaN NaN NaN
# 1  NaN    C  36.0 NaN NaN NaN
# 2  NaN  NaN   NaN NaN NaN NaN
# 3  NaN    C   NaN NaN NaN NaN
# 4  NaN    D   NaN NaN NaN NaN
# 5  NaN  NaN   NaN NaN NaN NaN

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.1 复杂查询')
print('\t5.1.5 查询df.query()')
print()
# update20240314
'''
df.query(expr)使用布尔表达式查询DataFrame的列，表达式是一个字符串，类似于SQL中的where从句，不过它相当灵活。

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print()
print('# df.query()')
print(df.query('Q1>Q2>90')) # 直接写类型SQL where语句
#    name team   Q1  Q2  Q3   Q4
# 5  Rick    B  100  99  97  100
print(df.query('Q1+Q2 > 180'))
#     name team   Q1  Q2  Q3   Q4
# 3  Eorge    C   93  96  71   78
# 5   Rick    B  100  99  97  100
print()
print(df.query('Q1==Q4'))
#    name team   Q1  Q2  Q3   Q4
# 5  Rick    B  100  99  97  100
print(df.query('Q1<50 & Q2>30 and Q3>20'))
print(df.query('(Q1<50) & (Q2>30) and (Q3>20)')) # 同上
#    name team  Q1  Q2  Q3  Q4
# # 1  Arry    C  36  37  37  57
print(df.query('Q1>Q2>Q3>Q4'))
print(df.query('team != "C"'))
print(df.query('team not in ("E","A","B")'))

print()
print('# 对于名称中带有空格的列，可以使用反引号引起来')
# 创建一个示例 DataFrame
df2 = pd.DataFrame({
    'team name': ['Team A', 'Team B', 'Team C'],
    'score': [85, 90, 95]
})

# 使用 query 方法查询 "team name" 为 "Team A" 的行
result = df2.query("`team name` == 'Team A'")
print(result)
#   team name  score
# 0    Team A     85

'''
请注意，在 query() 方法中，字符串值需要用单引号或双引号括起来，以区分字符串和列名。
同时，反引号用于包围含有空格或特殊字符的列名。如果列名不包含这些字符，就不需要使用反引号。
'''

print()
print('支持使用@符引入变量')
a = df.Q1.mean()
print(a)
print(df.query('Q1 > @a+4')) # 支持传入变量，如大于平均分4分的行
# print(df.query('Q1 > a + 10'))  # 直接引用a 会报错！name 'a' is not defined
print(df.query('Q1 > `Q2` + @a'))
print(df.query('Q1 > Q2 + @a')) # 同上！

print()
print('df.eval()与df.query()类似，也可以用于表达式筛选：')
# df.eval()用法与df.query类似
print(df[df.eval('Q1 > 90 > Q3 > 10')])
#     name team  Q1  Q2  Q3  Q4
# 3  Eorge    C  93  96  71  78
print(df[df.eval('Q1 < Q2 + @a')])
#     name team   Q1  Q2  Q3   Q4
# 0  Liver    E   89  21  24   64
# ....


print(df.query('Q1 > 90').loc[:,'Q1':'Q4']) # 筛选行列



print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.1 复杂查询')
print('\t5.1.6 查询df.filter()')
print()
# update20240314
'''
df.filter()可以对行名和列名进行筛选，支持模糊匹配、正则表达式。

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print()
print('# df.filter(items)')
print(df.filter(items=['Q1','Q2'])) # 选择2列
#     Q1  Q2
# 0   89  21
# ...

print()
print('# df.filter(regex)') # 正则参数
print(df.filter(regex='Q',axis=1)) # 列名包含Q的列
#     Q1  Q2  Q3   Q4
# 0   89  21  24   64
# ...
print(df.filter(regex='e$',axis=1)) # 以e结尾的列
#     name
# 0  Liver
# ...
print()
print(df.filter(regex='1$',axis=0)) # 正则，索引名以1结尾

print()
print('# 模糊匹配 like')
print(df.filter(like='2',axis=0)) # 索引中有2的
#   name team  Q1  Q2  Q3  Q4
# 2  Ack    A  57  60  18  84
# ...
# 索引中以2开头的，列名有Q的
print(df.filter(regex='^2',axis=0).filter(like='Q', axis=1))
#    Q1  Q2  Q3  Q4
# 2  57  60  18  84


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.1 复杂查询')
print('\t5.1.7 按数据类型查询')
print()
# update20240314
'''
Pandas提供了一个按列数据类型筛选的功能df.select_dtypes(include=None, exclude=None),
它可以指定包含和不包含的数据类型，如果只有一个类型，传入字符；如果有多个类型，传入列表。

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print()
print('# df.select_dtypes(include=None, exclude=None)')
print(df.select_dtypes(include='bool'))
# Empty DataFrame
# Columns: []
# Index: [0, 1, 2, 3, 4, 5]
'''如果没有满足条件的数据，会返回一个仅有索引的DataFrame。'''

print()
print('include')
print(df.select_dtypes(include=['int64'])) # 只选择int64型数据
print(df.select_dtypes(include=['int'])) # 同上
print(df.select_dtypes(include=['number'])) # 同上
# print(df.dtypes) # object int64
#     Q1  Q2  Q3   Q4
# 0   89  21  24   64
# ...
print()
print('exclude')
print(df.select_dtypes(exclude=['number'])) # 排除数字类型数据
#     name team
# 0  Liver    E
# ...

