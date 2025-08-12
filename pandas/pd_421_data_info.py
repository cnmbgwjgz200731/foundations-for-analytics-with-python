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

path = 'E:/bat/output_files/pandas_read_csv_20240118.csv'

# 忽略特定的警告 | 当遇到 openpyxl 中的 UserWarning 类型的警告时，它们将不会被打印出来。
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.2 数据的信息')
print('\t4.2.1 查看样本')
print()
# update20240226
'''
加载完的数据可能由于量太大，我们需要查看部分样本数据，
Pandas提供了三个常用的样式查看方法。
df.head()：前部数据，默认5条，可指定条数。
df.tail()：尾部数据，默认5条，可指定条数。
df.sample()：一条随机数据，可指定条数。
以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file, index_col='name')  # Q1 name
print(df)
print()

print('df.head() 读取头部数据，默认前5行')
print(df.head())  # 默认读取前5行
print()
print(df.head(3))  # 读取前3行
print()

print('df.tail() 读取尾部数据，默认后5行')
print(df.tail())  # 默认读取后5行
print()
print(df.tail(2))  # 默认读取后2行
print()

print('df.sample() 随机读取数据，默认1行')
print(df.sample())  # 默认随机读取1行
print()
print(df.sample(2))  # 默认随机读取2行
print()

print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.2 数据的信息')
print('\t4.2.2 数据形状')
print()
# update20240226
'''
执行df.shape会返回一个元组，
该元组的第一个元素代表行数，第二个元素代表列数，
这就是这个数据的基本形状，也是数据的大小。

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file)  # Q1 name
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print('df.shape 读取数据形状 n行n列')
print(df.shape)  # out: (6, 6)
print()

s = pd.Series([1, 2, 3], name='row')
print(s)
print(s.shape)  # out:(3,)

print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.2 数据的信息')
print('\t4.2.3 基础信息')
print()
# update20240226
'''
执行df.info() 会显示所有数据的类型、索引情况、行列数、各字段数据类型、内存占用等。Series不支持。

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file, index_col='name')  # Q1 name
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print('df.info 读取数据基础信息 xlsx文件')
print(df.info())  # out: (6, 6)
print()

s = pd.Series([1, 2, 3], name='row')
print(s)
print(s.shape)  # out:(3,)
print()
print(s.info())
print()

print('# 读取csv文件 基础信息')
df = pd.read_csv(m_file, delimiter=';')  # Q1 name
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
print()

# print('df.info csv 基础信息')
print(df.info())  # out: (6, 6)
print()

print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.2 数据的信息')
print('\t4.2.4 数据类型')
print()
# update20240227
'''
df.dtypes会返回每个字段的数据类型及DataFrame整体的类型。

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print('df.dtypes 读取数据类型 xlsx文件')
print(df.dtypes)  # out: (6, 6)
print()

s = pd.Series([1, 2, 3], name='row')
print(s.dtypes)
print()

print('# 读取csv文件 数据类型')
df = pd.read_csv(m_file, delimiter=';')  # Q1 name
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
print()

# print('df.info csv 基础信息')
print(df.dtypes)  # out: (6, 6)
print()

print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.2 数据的信息')
print('\t4.2.5 行列索引内容')
print()
# update20240227
'''
df.axes会返回一个列内容和行内容组成的列表[列索引, 行索引]。

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print('df.axes xlsx文件')
print(df.axes)  # out: (6, 6)
print()

print('# 读取series ')
s = pd.Series([1, 2, 3], name='row')
print(s.axes)
print()

print('# 读取csv文件 ')
df = pd.read_csv(m_file, delimiter=';')  # Q1 name
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
print()

# print('df.info csv 基础信息')
print(df.axes)  # out: (6, 6)
print()

'''
df.axes xlsx文件
[Index(['Liver', 'Arry', 'Ack', 'Eorge', 'Oah', 'Rick'], dtype='object', name='name'), Index(['team', 'Q1', 'Q2', 'Q3', 'Q4'], dtype='object')]

# 读取数据时 无参数 index='name'
[RangeIndex(start=0, stop=6, step=1), Index(['name', 'team', 'Q1', 'Q2', 'Q3', 'Q4'], dtype='object')]

# 读取series
[RangeIndex(start=0, stop=3, step=1)]

# 读取csv文件 

[RangeIndex(start=0, stop=1599, step=1), Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality'],
      dtype='object')]
'''

print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.2 数据的信息')
print('\t4.2.6 其它信息')
print()
# update20240227
'''
其它信息

本节数据信息的操作让我们对数据有了一个全面的认识，这对数据的下一步分析至关重要，
加载完数据后，推荐先进行以下操作，以便及早找到数据的质量问题。

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print('读取 xlsx文件')
print(df.index)  # 行索引
print(df.columns)  # 列索引 列名
print(df.values)  # 所有值的列表矩阵 （除了列名以外的所有值）
print(df.ndim)  # 维度数 out：2
print(df.size)  # 行*列的总数 out:36
# 是否为空，注意，有空值不认为是空
print(df.empty)  # out:False
# Series的索引，DataFrame的列名
print(df.keys())  # out: Index(['name', 'team', 'Q1', 'Q2', 'Q3', 'Q4'], dtype='object')
print()
print('df.name')
print(df.name)  # 这个是特例 因为数据有列名为name 实际执行调用了该列数据 dataframe无此参数 会报错！
print()

print('# 读取series ')
s = pd.Series([1, 2, 3], name='row')
print(s.index)
# print(s.columns) # out:报错！
print(s.values)  # 所有值的列表矩阵 （除了列名以外的所有值）
print(s.ndim)  # 维度数 out：1
print(s.size)  # 行*列的总数 out:3
print(s.empty)  # out:False
# Series的索引，DataFrame的列名
print(s.keys())  # out: RangeIndex(start=0, stop=3, step=1)
print()

print('series特有的参数')
print(s.name)  # out:row || None（无名称时显示）
print(s.array)
'''
out:
<PandasArray>
[1, 2, 3]
Length: 3, dtype: int64
'''
print(s.dtype)  # out: int64
print(s.dtypes)  # out:int64 # 这个不是特有，和dtype作为对比
print(s.hasnans)  # 检查 Series 是否包含 NaN | out:False/True

print()

print('# 读取csv文件 ')
df = pd.read_csv(m_file, delimiter=';')  # Q1 name
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
print()

# print('df.info csv 基础信息')
print(df.index)  # out:
print(df.columns)
print(df.values)  # 所有值的列表矩阵 （除了列名以外的所有值）
print(df.ndim)  # 维度数 out：2
print(df.size)  # 行*列的总数 out:19188
print(df.empty)  # out:False
# Series的索引，DataFrame的列名
print(df.keys())  # out:
print()

'''
读取 xlsx文件
RangeIndex(start=0, stop=6, step=1)
Index(['name', 'team', 'Q1', 'Q2', 'Q3', 'Q4'], dtype='object')

# 读取series 
RangeIndex(start=0, stop=3, step=1)

# 读取csv文件 

RangeIndex(start=0, stop=1599, step=1)
Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality'],
      dtype='object')


进程已结束,退出代码0
'''
