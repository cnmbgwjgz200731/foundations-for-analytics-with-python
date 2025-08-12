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
print('\t6.1 概述')
print('\t6.1.2 groupby语法')
print()
# update20240418
'''

'''
# 小节注释
'''

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# groupby 语法')


# def groupby(self,
#             by: Any = None,
#             axis: Literal["index", "columns", "rows"] | int = 0,
#             level: Hashable | Sequence[Hashable] | None = None,
#             as_index: bool = True,
#             sort: bool = True,
#             group_keys: bool = True,
#             observed: bool = False,
#             dropna: bool = True) -> DataFrameGroupBy
'''
by：代表分组的依据和方法。如果by是一个函数，则会在数据的索引的每个值去调用它，从而产生值，按这些值进行分组。
    如果传递dict或Series，则将使用dict或Series的值来确定组；
    如果传递ndarray，则按原样使用这些值来确定组。传入字典，键为原索引名，值为分组名。
axis：沿行（0）或列（1）进行拆分。也可传入index或columns，默认是0。
level：如果轴是多层索引（MultiIndex），则按一个或多个特定的层级进行拆分，支持数字、层名及序列。
as_index：数据分组聚合输出，默认返回带有组标签的对象作为索引，传False则不会。
sort：是否对分组进行排序。默认会排序，传False会让数据分组中第一个出现的值在前，同时会提高分组性能。
group_keys：调用函数时，将组键添加到索引中进行识别。
observed：仅当分组是分类数据时才适用。如果为True，仅显示分类分组数据的显示值；如果为False，显示分类分组数据的所有值。
dropna：如果为True，并且组键包含NA值，则NA值及行/列将被删除；如果为False，则NA值也将被视为组中的键。

以上大多参数对于Series也是适用的，如果对DataFrame进行分组会返回DataFrame-GroupBy对象，对Series分组会返回SeriesGroupBy对象。
'''

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.1 概述')
print('\t6.1.3 dataframe应用分组')
print()
# update20240418
'''

'''
# 小节注释
'''

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 按team分组对应列并相加')

print(df.groupby('team').sum())
#            name   Q1   Q2   Q3   Q4
# team
# A           Ack   57   60   18   84
# B          Rick  100   99   97  100
# C     ArryEorge  129  133  108  135
# D           Oah   65   49   61   86
# E         Liver   89   21   24   64

print()
print('# 对不同列使用不同的计算方法')
print(df.groupby('team').agg({'Q1':'sum',
                              'Q2':'count',  # 必须加引号 否则报错
                              'Q3':'mean',   # 必须加引号 否则报错
                              'Q4':'max'
                                }))
#        Q1  Q2    Q3   Q4
# team
# A      57   1  18.0   84
# B     100   1  97.0  100
# C     129   2  54.0   78
# D      65   1  61.0   86
# E      89   1  24.0   64

print()
print('# 对同一列使用不同的计算方法')
print(df.groupby('team').agg({'Q1':[sum,'std','max'], #
                              'Q2':'count',
                              'Q3':'mean',
                              'Q4':'max'}))
#        Q1                    Q2    Q3   Q4
#       sum        std  max count  mean  max
# team
# A      57        NaN   57     1  18.0   84
# B     100        NaN  100     1  97.0  100
# C     129  40.305087   93     2  54.0   78
# D      65        NaN   65     1  61.0   86
# E      89        NaN   89     1  24.0   64

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.1 概述')
print('\t6.1.4 series应用分组')
print()
# update20240418
'''

'''
# 小节注释
'''

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 对Series df.Q1按team分组，求和')
# 对Series也可以使用分组聚合，但相对来说场景比较少。
print(df.Q1.groupby(df.team).sum()) #
# team
# A     57
# B    100
# C    129
# D     65
# E     89
# Name: Q1, dtype: int64

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.2 分组')
print('\t6.2.2 按标签分组')
print()
# update20240419
'''
df.groupby()方法可以先按指定字段对DataFrame进行分组，生成一个分组器对象，
然后把这个对象的各个字段按一定的聚合方法输出。
本节将针对分组对象介绍什么是分组对象，分组对象的创建可以使用哪些方法。
'''
# 小节注释
'''
最简单的分组方法是指定DataFrame中的一列，按这列的去重数据分组。
也可以指定多列，按这几列的排列组合去重进行分组。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

# print(df.drop(columns=['name']))

# grouped = df.groupby('col') # 单列
# grouped = df.groupby('col', axis='columns') # 按行
# grouped = df.groupby(['col1', 'col2']) # 多列

grouped = df.groupby('team')
print(grouped)
# <pandas.core.groupby.generic.DataFrameGroupBy object at 0x000001BE3686DF40>

# 可以使用get_group()查看分组对象单个分组的内容：
print(grouped.get_group('D'))
#   name team  Q1  Q2  Q3  Q4
# 4  Oah    D  65  49  61  86

print()
print('# 多列分组')
grouped2 = df.groupby(['name','team'])
# print(grouped2)
# <pandas.core.groupby.generic.DataFrameGroupBy object at 0x0000020EDD0EC2B0>

# ValueError: must supply a tuple to get_group with multiple grouping keys
# print(grouped2.get_group(['Arry','C'])) # 运行失败

print(grouped2.get_group(('Arry','C'))) # 必须使用元组才运行成功！
#    name team  Q1  Q2  Q3  Q4
# 1  Arry    C  36  37  37  57

print()
print('# 按行分组')
grouped3 = df.groupby('team',axis='columns')
# print(grouped3)
# <pandas.core.groupby.generic.DataFrameGroupBy object at 0x0000020EC4155640>

# print(grouped3.get_group('D')) # 运行报错，KeyError: 'D'

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.2 分组')
print('\t6.2.3 表达式')
print()
# update20240419
'''
df.groupby()方法可以先按指定字段对DataFrame进行分组，生成一个分组器对象，
然后把这个对象的各个字段按一定的聚合方法输出。
本节将针对分组对象介绍什么是分组对象，分组对象的创建可以使用哪些方法。
'''
# 小节注释
'''
通过行和列的表达式，生成一个布尔数据的序列，从而将数据分为True和False两组。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 索引值是否为偶数，分成两组')
print(df.groupby(lambda x:x%2==0).sum())
#                 name team   Q1   Q2   Q3   Q4
# False  ArryEorgeRick  CCB  229  232  205  235
# True     LiverAckOah  EAD  211  130  103  234
print(df.groupby(df.index%2==0).sum()) # 结果同上

print()
print('# 以下为按索引值是否大于等于4为标准分为True和False两组：')
print(df.groupby(lambda x:x >=4).sum())
#                     name  team   Q1   Q2   Q3   Q4
# False  LiverArryAckEorge  ECAC  275  214  150  283
# True             OahRick    DB  165  148  158  186
print(df.groupby(df.index >=4).sum()) # 结果同上

print()
print('# 列名包含Q的分成一组')
print(df.groupby(lambda x:'Q' in x,axis=1).sum())
#     False  True
# 0  LiverE    198
# 1   ArryC    167
# 2    AckA    219
# 3  EorgeC    338
# 4    OahD    261
# 5   RickB    396

print()
print('# 按姓名首字母分组')
print(df.groupby(df.name.str[0]).sum())
#          name team   Q1  Q2  Q3   Q4
# name
# A     ArryAck   CA   93  97  55  141
# E       Eorge    C   93  96  71   78
# L       Liver    E   89  21  24   64
# O         Oah    D   65  49  61   86
# R        Rick    B  100  99  97  100

print()
print('# 按A及B、其他团队分组')
print(df.groupby(df.team.isin(['A','B'])).sum())
#                     name  team   Q1   Q2   Q3   Q4
# team
# False  LiverArryEorgeOah  ECCD  283  203  193  285
# True             AckRick    AB  157  159  115  184

print()
print('# 按姓名第一个字母和第二个字母分组')
# df.groupby([df.name.str[0], df.name.str[1]])
print(df.groupby([df.name.str[0],df.name.str[1]]).sum())
#             name team   Q1  Q2  Q3   Q4
# name name
# A    c       Ack    A   57  60  18   84
#      r      Arry    C   36  37  37   57
# E    o     Eorge    C   93  96  71   78
# L    i     Liver    E   89  21  24   64
# O    a       Oah    D   65  49  61   86
# R    i      Rick    B  100  99  97  100

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.2 分组')
print('\t6.2.4 函数分组')
print()
# update20240419
# 小节注释
'''
通过行和列的表达式，生成一个布尔数据的序列，从而将数据分为True和False两组。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 按姓名首字母为元音、辅音分组')
# by参数可以调用一个函数来通过计算返回一个分组依据。

def get_letter_type(letter):
    if letter[0].lower() in 'aeiou':
        return '元音'
    else:
        return '辅音'

# 使用函数
print(df.set_index('name').groupby(get_letter_type).sum())
#       team   Q1   Q2   Q3   Q4
# name
# 元音    CACD  251  242  187  305
# 辅音      EB  189  120  121  164


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.2 分组')
print('\t6.2.5 多种方法混合')
print()
# update20240422

# 小节注释
'''
由于分组可以按多个依据，在同一次分组中可以混合使用不同的分组方法。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 按team、姓名首字母是否为元音分组')
# by参数可以调用一个函数来通过计算返回一个分组依据。

def get_letter_type(letter):
    if letter[0].lower() in 'aeiou':
        return '元音'
    else:
        return '辅音'

# 使用函数
# print(df.set_index('name').groupby(get_letter_type).sum())
#       team   Q1   Q2   Q3   Q4
# name
# 元音    CACD  251  242  187  305
# 辅音      EB  189  120  121  164

print(df.groupby(['team',df.name.apply(get_letter_type)]).sum())
#                 name   Q1   Q2   Q3   Q4
# team name
# A    元音          Ack   57   60   18   84
# B    辅音         Rick  100   99   97  100
# C    元音    ArryEorge  129  133  108  135
# D    元音          Oah   65   49   61   86
# E    辅音        Liver   89   21   24   64


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.2 分组')
print('\t6.2.6 用pipe调用分组方法')
print()
# update20240422

# 小节注释
'''
我们之前了解过df.pipe()管道方法，可以调用一个函数对DataFrame进行处理，
我们发现Pandas的groupby是一个函数——pd.DataFrame.groupby：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 使用pipe调用分组函数')
print(df.pipe(pd.DataFrame.groupby,'team').sum())
#            name   Q1   Q2   Q3   Q4
# team
# A           Ack   57   60   18   84
# B          Rick  100   99   97  100
# C     ArryEorge  129  133  108  135
# D           Oah   65   49   61   86
# E         Liver   89   21   24   64

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.2 分组')
print('\t6.2.7 分组器Grouper')
print()
# update20240422

# 小节注释
'''
Pandas提供了一个分组器pd.Grouper()，它也能帮助我们完成数据分组的工作。
有了分组器，我们可以复用分组工作。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 分组器语法')
# 分组器语法
# pandas.Grouper(key=None, level=None, freq=None, axis=0, sort=False)
# print(df.groupby(pd.Grouper('team')))

print(df.groupby(pd.Grouper('team')).sum())
#            name   Q1   Q2   Q3   Q4
# team
# E         Liver   89   21   24   64
# C     ArryEorge  129  133  108  135
# A           Ack   57   60   18   84
# D           Oah   65   49   61   86
# B          Rick  100   99   97  100


# df.groupby('team')
# df.groupby(pd.Grouper('team')).sum()
# # 如果是时间，可以60秒一分组
# df.groupby(Grouper(key='date', freq='60s'))
# # 轴方向
# df.groupby(Grouper(level='date', freq='60s', axis=1))
# # 按索引
# df.groupby(pd.Grouper(level=1)).sum()
# # 多列
# df.groupby([pd.Grouper(freq='1M', key='Date'), 'Buyer']).sum()
# df.groupby([pd.Grouper('dt', freq='D'),
# pd.Grouper('other_column')
# ])
# # 按轴层级
# df.groupby([pd.Grouper(level='second'), 'A']).sum()
# df.groupby([pd.Grouper(level=1), 'A']).sum()
# # 按时间周期分组
# df['column_name'] = pd.to_datetime(df['column_name'])
# df.groupby(pd.Grouper(key='column_name', freq="M")).mean()
# # 10年一个周期
# df.groupby(pd.cut(df.date,
# pd.date_range('1970', '2020', freq='10YS'),
# right=False)
# ).mean()


data = {'date':pd.date_range(start='2024-04-22 00:00:00',periods=10,freq='15s'),
        'value':np.random.randint(0,100,size=10)
        }

df1 = pd.DataFrame(data)

print(df1)
#                  date  value
# 0 2024-04-22 00:00:00     47
# 1 2024-04-22 00:00:15     79
# 2 2024-04-22 00:00:30     74
# 3 2024-04-22 00:00:45     85
# 4 2024-04-22 00:01:00     36
# 5 2024-04-22 00:01:15     47
# 6 2024-04-22 00:01:30     35
# 7 2024-04-22 00:01:45     13
# 8 2024-04-22 00:02:00     48
# 9 2024-04-22 00:02:15     39

print()
print('# 使用 Grouper 进行分组')
grouped = df1.groupby(pd.Grouper(key='date',freq='60s'))
print(grouped)

# 计算每组的平均值
result = grouped['value'].mean()
print(result)
# date
# 2024-04-22 00:00:00    71.25
# 2024-04-22 00:01:00    32.75
# 2024-04-22 00:02:00    43.50
# Freq: 60S, Name: value, dtype: float64

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.2 分组')
print('\t6.2.8 索引')
print()
# update20240422

# 小节注释
'''
groupby操作后分组字段会成为索引，如果不想让它成为索引，可以使用as_index=False进行设置：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 取消分组列为索引！')
print(df.groupby('team',as_index=False).sum())
#   team       name   Q1   Q2   Q3   Q4
# 0    A        Ack   57   60   18   84
# 1    B       Rick  100   99   97  100
# 2    C  ArryEorge  129  133  108  135
# 3    D        Oah   65   49   61   86
# 4    E      Liver   89   21   24   64


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.2 分组')
print('\t6.2.9 排序')
print()
# update20240422

# 小节注释
'''
groupby操作后分组字段会成为索引，数据会对索引进行排序，如果不想排序，可以使用sort=False进行设置。
不排序的情况下会按索引出现的顺序排列：
▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 不对索引排序')
print(df.groupby('team',sort=False,as_index=False).sum())
#   team       name   Q1   Q2   Q3   Q4
# 0    E      Liver   89   21   24   64
# 1    C  ArryEorge  129  133  108  135
# 2    A        Ack   57   60   18   84
# 3    D        Oah   65   49   61   86
# 4    B       Rick  100   99   97  100