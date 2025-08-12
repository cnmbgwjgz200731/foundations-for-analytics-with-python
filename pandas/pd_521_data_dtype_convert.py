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
warnings.filterwarnings('ignore',category=UserWarning,module='openpyxl')


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.2 数据类型转换')
print('\t5.2.1 推断类型')
print()
# update20240315
'''
在开始数据分析前，我们需要为数据分配好合适的类型，这样才能够高效地处理数据。
不同的数据类型适用于不同的处理方法。之前的章节中介绍过，加载数据时可以指定数据各列的类型：

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print(df.dtypes)
print()

# TODO 加载数据时指定数据类型
print()
print('对指定字段分别指定 数据类型')
df1 = pd.read_excel(team_file,dtype={'name':'string','Q1':'string'})
# print(df1)
print(df1.dtypes)
# name    string[python]
# team            object
# Q1      string[python]
# Q2               int64
# Q3               int64
# Q4               int64
# dtype: object
print()
print('对所有字段指定统一类型')
df2 = pd.read_excel(team_file,dtype='string')
# print(df2)
print(df2.dtypes)
# name    string[python]
# team    string[python]
# Q1      string[python]
# Q2      string[python]
# Q3      string[python]
# Q4      string[python]
# dtype: object

print()
'''
Pandas可以用以下方法智能地推断各列的数据类型，会返回一个按推断修改后的DataFrame。
如果需要使用这些类型的数据，可以赋值替换。
'''
print('自动转换合适的数据类型')
print(df.infer_objects()) # 推断后的DataFrame
print(df.infer_objects().dtypes)
# name    object
# team    object
# Q1       int64
# Q2       int64
# Q3       int64
# Q4       int64
# dtype: object

print()
print('# 推荐这个新方法，它支持string类型')
print(df.convert_dtypes()) # 推断后的DataFrame
print(df.convert_dtypes().dtypes)
# name    string[python]
# team    string[python]
# Q1               Int64
# Q2               Int64
# Q3               Int64
# Q4               Int64
# dtype: object


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.2 数据类型转换')
print('\t5.2.2 指定类型')
print()
# update20240315
'''
pd.to_XXX系统方法可以将数据安全转换，errors参数可以实现无法转换则转换为兜底类型：

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 按大体类型推定')
m = ['1',2,3]
s = pd.to_numeric(m) # 转成数字
print(s) # [1 2 3]
print(s.dtype)
print(m)
# print(m.dtype) # AttributeError: 'list' object has no attribute 'dtype'
# TODO 转成时间 m变量会报错
print(pd.to_datetime(s))
# DatetimeIndex(['1970-01-01 00:00:00.000000001',
#                '1970-01-01 00:00:00.000000002',
#                '1970-01-01 00:00:00.000000003'],
#               dtype='datetime64[ns]', freq=None)
# TODO 转成时间差
print(pd.to_timedelta(s))
# TimedeltaIndex(['0 days 00:00:00.000000001', '0 days 00:00:00.000000002',
#                 '0 days 00:00:00.000000003'],
#                dtype='timedelta64[ns]', freq=None)
# TODO 错误处理
print(pd.to_datetime(m,errors='coerce')) # 错误处理
# DatetimeIndex(['NaT', '1970-01-01 00:00:00.000000002',
#                '1970-01-01 00:00:00.000000003'],
#               dtype='datetime64[ns]', freq=None)

print("errors='ignore'")
print(pd.to_numeric(m,errors='ignore'))
# print(pd.to_numeric(m,errors='ignore').dtype)
# [1 2 3]
# int64

print()
# print(pd.to_numeric(m,errors='coerce').fillna(0)) # 兜底填充
# AttributeError: 'numpy.ndarray' object has no attribute 'fillna'

# 错误处理，如果解析失败则使用 NaN，并使用 fillna() 填充 NaN 为 0
# 需要先将 numpy.ndarray 转换为 pandas.Series 对象
print(pd.Series(pd.to_numeric(m,errors='coerce')).fillna(0)) # 运行正常
'''
pd.to_numeric(m, errors='coerce').fillna(0) 的错误信息： 
当使用 pd.to_numeric() 并设置 errors='coerce' 时，如果有无法转换为数字的元素，这些元素会被设置为 NaN。
然后，由于 pd.to_numeric() 返回的是 numpy.ndarray，而 numpy.ndarray 没有 fillna() 方法，因此会引发 AttributeError
'''

print()
print('# 组合成日期')
# 创建一个示例 DataFrame
df3 = pd.DataFrame({
    'year': [2020, 2021, 2022],
    'month': [1, 2, 3],
    'day': [15, 16, 17]
})

# 将年、月、日三列组合成日期
df3['date'] = pd.to_datetime(df3[['year', 'month', 'day']])

# 打印结果
print(df3)
#    year  month  day       date
# 0  2020      1   15 2020-01-15
# 1  2021      2   16 2021-02-16
# 2  2022      3   17 2022-03-17

# TODO --------------------------------------------------------
'''
转换为数字类型时，默认返回的dtype是float64还是int64取决于提供的数据。使用downcast参数获得向下转换后的其他类型。
'''

# 最低期望
print()
print('# # 最低期望')
print(pd.to_numeric(m,downcast='integer')) # 至少为有符号int数据类型
print(pd.to_numeric(m,downcast='signed')) # 同上
# [1 2 3]
print(pd.to_numeric(m,downcast='unsigned')) # 至少为无符号int数据类型 [1 2 3] ||它适用于所有数据都是非负的情况。
print(pd.to_numeric(m,downcast='float')) # 至少为float浮点类型 [1. 2. 3.]

'''
downcast 参数告诉 pd.to_numeric() 尝试将数据转换为更小的类型以节省内存。
在实际应用中，通过 downcast 参数，你可以优化数据存储，并减少内存占用，特别是在处理大型数据集时。
需要注意的是，downcast 操作可能会导致数据溢出，如果转换后的数据类型无法容纳原始数据的值，
你可能会得到不正确的结果。因此，在使用 downcast 参数时，应确保数据的范围与目标数据类型兼容。

'''
print()
print('# 可以应用在函数中：')
df4 = df.select_dtypes(include='number')
# print(df4)
print(df4.apply(pd.to_numeric).dtypes)


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.2 数据类型转换')
print('\t5.2.3 类型转换astype()')
print()
# update20240315
'''
astype()是最常见也是最通用的数据类型转换方法，一般我们使用astype()操作数据转换就可以了。

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print(df.dtypes)
print()

print(df.Q1.astype('int32').dtypes) # int32
print(df.astype({'Q1':'int32','Q2':'int16'}).dtypes)
# name    object
# team    object
# Q1       int32
# Q2       int16
# Q3       int64
# Q4       int64
# dtype: object

print()
print(df.index.dtype) # int64  # 索引类型转换
print(df.index.astype('int32'))  # 所有数据转换为int32 | Index([0, 1, 2, 3, 4, 5], dtype='int32')
print(df.astype({'Q1':'int16'}).dtypes) # 指定字段转指定类型

s= pd.Series([1.2,3.4,2.56,2.456,4,-0.12]) # {'amount':[1.2,3.4,2.56,2.456,4,-0.12]}
print(s)
# 0    1.200
# 1    3.400
# 2    2.560
# 3    2.456
# 4    4.000
# 5   -0.120
# dtype: float64
print(s.astype('int64'))
# 0    1
# 1    3
# 2    2
# 3    2
# 4    4
# 5    0
# dtype: int64
print()
print(s.astype('float32',copy=False)) # 不与原数据关联
'''
使用copy=False参数指示pandas在转换类型时不创建数据的副本。
如果转换不需要复制任何数据（例如，如果原始数据已经是目标类型），
则直接在原对象上进行更改，否则，pandas仍会创建一个新的对象。
'''
print()
print('# astype(uint8)')
# print(s.astype(np.uint8)) # ValueError: Cannot losslessly cast from float64 to uint8
# np.uint8 是无符号8位整数类型，它的取值范围是 0 到 255。

# 创建一个包含整数的 pandas Series 对象
s1 = pd.Series([10, 200, 300, 256, 500])

# 将 Series 对象的数据类型转换为 np.uint8
s_uint8 = s1.astype(np.uint8)
print(s_uint8)
# 0     10
# 1    200
# 2     44  # 注意：300 转换为 uint8 后溢出，变成了 44
# 3      0  # 注意：256 转换为 uint8 后溢出，变成了 0
# 4    244  # 注意：500 转换为 uint8 后溢出，变成了 244
# dtype: uint8

print()
print(df['name'].astype('string'))
print(df['Q4'].astype('float'))
# df['Q4'] = df['Q4'].astype('float')
# print(df['Q4'])
print(s.astype('datetime64[ns]'))
# 0   1970-01-01 00:00:00.000000001
# 1   1970-01-01 00:00:00.000000003
# 2   1970-01-01 00:00:00.000000002
# 3   1970-01-01 00:00:00.000000002
# 4   1970-01-01 00:00:00.000000004
# 5   1970-01-01 00:00:00.000000000
# dtype: datetime64[ns]

print()
print('# astype(bool)')
# 创建一个DataFrame
data = pd.DataFrame({
    'status': [1, 0, 'True', 'False', 'yes', 'no', '', None]
})

# 将'status'列转换为布尔类型
data['status'] = data['status'].astype('bool')

# 打印DataFrame
print(data)

'''
在实际应用中，你可能需要将数据转换为布尔类型来表示开关状态、是否条件等。
使用astype('bool')是执行此类转换的快速方法。
然而，在转换之前，应确保你的数据在逻辑上适合表示为布尔值，
并且注意到在pandas中，除了0、False、空字符串和None之外，几乎所有其他值都会被转换为True，包括非空字符串。

在进行转换时，如果数据列包含NaN或None等缺失值，它们也会被转换为False。
如果这不是你预期的行为，你可能需要在转换之前处理这些缺失值。
'''

# TODO ---------------------------------------
print()
print('# 将"23.45%"这样的文本转为浮点数')
s = pd.Series(['23.45%',89.23])
da = pd.DataFrame({'num':['23.45%']})
print(s)
print(da.dtypes)
# data.rate.apply(lambda x: x.replace('%', '')).astype('float')/100
print(da.num.apply(lambda x: x.replace('%', '')))
print(da.num.apply(lambda x: x.replace('%', '')).astype('float'))
print(da.num.apply(lambda x: x.replace('%', '')).astype('float')/100)
# 0    23.45
# Name: num, dtype: object
# 0    23.45
# Name: num, dtype: float64
# 0    0.2345
# Name: num, dtype: float64


db = pd.DataFrame({'num':['23.45%',89.23]})
# AttributeError: 'float' object has no attribute 'replace'
db = db.astype('string') # 如果不转换文本类型 下一步会报错！
print(db.num.apply(lambda x: x.replace('%', '')))
# 0    23.45
# 1    89.23
# Name: num, dtype: object

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.2 数据类型转换')
print('\t5.2.4 转为时间类型')
print()
# update20240315
'''
我们通常使用pd.to_datetime()和s.astype('datetime64[ns]')来做时间类型转换，第14章会专门介绍这两个函数。

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print(df.dtypes)
print()

t = pd.Series(['20200801', '20200802'])
print(t)
# 0    20200801
# 1    20200802
# dtype: object
print(pd.to_datetime(t))
# 0   2020-08-01
# 1   2020-08-02
# dtype: datetime64[ns]
print(t.astype('datetime64[ns]')) # 同上
