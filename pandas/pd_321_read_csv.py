import numpy as np
import pandas as pd

from io import StringIO
from io import BytesIO

print()
print('3.2 读取CSV')
print('\t3.2.2 数据内容')
print()
# update20240111

# 可以传数据字符串，即csv中的数据字符以字符串形式直接传入：
# from io import StringIO
data = ('col1,col2,col3\n'
        'a,b,1\n'
        'a,b,2\n'
        'c,d,3')
df = pd.read_csv(StringIO(data))
print(df)
print(df.dtypes)
'''
out:
  col1 col2  col3
0    a    b     1
1    a    b     2
2    c    d     3
col1    object
col2    object
col3     int64
dtype: object
'''
# 指定字段数据类型
df1 = pd.read_csv(StringIO(data), dtype=object)
print(df1)
print(df1.dtypes)
'''
out:
  col1 col2 col3
0    a    b    1
1    a    b    2
2    c    d    3
col1    object
col2    object
col3    object
dtype: object
'''

# 也可以传入字节数据：
# from io import BytesIO
print()
print('也可以传入字节数据：')
print()

data = (b'word,length\n'
        b'Tr\xc3\xa4umen,7\n'
        b'Gr\xc3\xbc\xc3\x9fe,5')

df = pd.read_csv(BytesIO(data))
print(df)
'''
out:
      word  length
0  Träumen       7
1    Grüße       5
'''

print()
print('\t3.2.3 分隔符')
print()

# 数据分隔符默认是逗号，可以指定为其它符号
df = pd.read_csv('E:/bat/input_files/sales_january_2014.csv')  # 不指定分隔符，默认逗号为分隔符
print(df)
df = pd.read_csv('E:/bat/input_files/sales_january_2014.csv', sep='\t')  # 制表符分隔tab
print(df)
df = pd.read_table('E:/bat/input_files/sales_january_2014.csv')  # read_table 默认是制表符分隔tab 效果同上！
print(df)
# df = pd.read_csv(data,sep='\t')
print(df)
'''
pd.read_csv还提供了一个参数名为delimiter的定界符，
这是一个备选分隔符，是sep的别名，效果和sep一样。
如果指定该参数，则sep参数失效。
'''

print()
print('------------------------------------------------------------')
print('\t3.2.4 表头')
print()

# header参数支持整数型和由整数型组成的列表，指定第几行是表头，
# 默认会自动推断把第一行作为表头。

input_file = 'E:/bat/input_files/sales_january_2014.csv'
lx_test = 'E:/bat/input_files/lxtest_sales_january_2014.csv'

df = pd.read_csv(input_file, header=0)  # 第一行为表头
print(df)
print()

df = pd.read_csv(input_file, header=None)  # 没有表头，列索引依次为阿拉伯数字
print(df)
print()

df = pd.read_csv(input_file, header=[0, 1, 3])  # 多层索引MultiIndex
print(df)
print()

df = pd.read_csv(lx_test, header=[0], skip_blank_lines=True)  # skip_blank_lines=True
print(df)
print()

'''
如果skip_blank_lines=True，header参数将忽略空行和注释行，
因此header=0表示第一行数据而非文件的第一行。
'''

print()
print('------------------------------------------------------------')
print('\t3.2.5 列名')
print()

'''
names用来指定列的名称，它是一个类似列表的序列，与数据一一对应。
如果文件不包含列名，那么应该设置header=None，
列名列表中不允许有重复值。
'''
df1 = pd.read_csv(input_file, names=['a', 'b', 'c', 'd', 'e'])  # 指定列名列表 | 如果指定列名，原来的列名默认为第一行数据
print(df1)
print()

df2 = pd.read_csv(input_file, names=['a', 'b', 'c', 'd', 'e'], header=None)
print(df2)
print()

print()
print('------------------------------------------------------------')
print('\t3.2.6 索引')
print()

'''
index_col用来指定索引列，可以是行索引的列编号或者列名，
如果给定一个序列，则有多个行索引。
Pandas不会自动将第一列作为索引，不指定时会自动使用以0开始的自然索引。
'''

# 支持int、str、int序列、str序列、False，默认为None
print(pd.read_csv(input_file))

print(pd.read_csv(input_file, index_col=False))  # 列索引默认为阿拉伯数字 | 效果同上

print(pd.read_csv(input_file, index_col=0))  # 第1列为索引

print(pd.read_csv(input_file, index_col='Customer Name'))  # 指定列名 该列为索引

print(pd.read_csv(input_file, index_col=['Customer ID', 'Invoice Number']))  # 多个索引 指定列名

print(pd.read_csv(input_file, index_col=[2, 3]))  # 多个索引 指定默认阿拉伯数字的列索引

print()
print('------------------------------------------------------------')
print('\t3.2.7 使用部分列')
print()

'''
如果只使用数据的部分列，可以用usecols来指定，
这样可以加快加载速度并降低内存消耗。
'''
df = pd.read_csv(input_file, usecols=[0, 4, 3])  # 按索引只读取指定列，与顺序无关
print(df)
print(df.columns)

print('# 查询列名长度 指定变量 使用切片')
len_th = len(df.columns)
print(len_th)

df = pd.read_csv(input_file, usecols=range(len_th))
print(df)
print(df.columns)

df.columns = [heading.lower() for heading in df.columns.str.replace(' ', '_')]
print(df.columns)
print(df)

print()
print('# 按列名，列名必须存在')
# print(df[['sale_amount','purchase_date']]) # 按列名，列名必须存在
print(pd.read_csv(input_file, usecols=['Sale Amount', 'Purchase Date']))

print()
print('# 指定列顺序，其实是df的筛选功能')
print(pd.read_csv(input_file, usecols=['Sale Amount', 'Purchase Date'])[['Purchase Date', 'Sale Amount']])

df = pd.read_csv(input_file)
df.columns = [heading.lower() for heading in df.columns.str.replace(' ', '_')]
print(df.columns)

print()
print('# 以下用callable方式可以巧妙指定顺序，in后面的是我们要的顺序')
data = ('col1,col2,col3\n'
        'a,b,1\n'
        'a,b,2\n'
        'c,d,3')

df = pd.read_csv(StringIO(data))
print(df)

df = pd.read_csv(StringIO(data), usecols=lambda x: x.upper() in ['COL3', 'COL2', 'COL1'])
print(df)
'''
out:
col1 col2  col3
0    a    b     1
1    a    b     2
2    c    d     3
  col1 col2  col3
0    a    b     1
1    a    b     2
2    c    d     3

# 顺序没变啊？
'''

input_file = 'E:/bat/input_files/sales_january_2014.csv'
lx_test = 'E:/bat/input_files/lxtest_sales_january_2014.csv'
m_file = 'E:/bat/input_files/winequality-red.csv'
big_file = 'E:/bat/input_files/dq_split_file.csv'

print()
print('------------------------------------------------------------')
print('\t3.2.8 返回序列')
print()

'''
将squeeze设置为True，如果文件只包含一列，则返回一个Series，
如果有多列，则还是返回DataFrame。
'''

# 布尔值，默认为False
# 下例只取一列，会返回一个Series
# df = pd.read_csv(input_file,usecols=[0],squeeze=True)
# print(type(df))
# 有两列则还是df
# df = pd.read_csv(input_file,usecols=[0,2],squeeze=True)
# print(type(df))
# help(pd.read_csv)

# 替代方案代码：
df = pd.read_csv(input_file, usecols=[0])
if df.shape[1] == 1:
    # 如果 DataFrame 只有一列，则将其转换为 Series
    df = df.iloc[:, 0]

print(type(df))

'''
提示已移除该关键字-_-
安装1.5.3版本后
out:
<class 'pandas.core.series.Series'>
<class 'pandas.core.frame.DataFrame'>
替代方案运行out:
<class 'pandas.core.series.Series'>
'''
print()
print('------------------------------------------------------------')
print('\t3.2.9 表头前缀')
print()
'''
如果原始数据没有列名，可以指定一个前缀加序数的名称，
如n0、n1，通过prefix参数指定前缀。
'''

input_file = 'E:/bat/input_files/supplier_data_no_header_row.csv'

df = pd.read_csv(input_file, header=None)
print(df)

print()
# df = pd.read_csv(input_file,prefix='c_',header=None)
# print(df)
'''
正常输出后 添加警告：
out:
FutureWarning: 
The prefix argument has been deprecated and will be removed in a future version. 
Use a list comprehension on the column names in the future.
'''

print()
print('未来版本中prefix参数已弃用，使用列表推导式可实现相同效果')
df = pd.read_csv(input_file, header=None)
df.columns = ['c_' + str(i) for i in range(len(df.columns))]
print(df)

# LHGGR9845R8017882

print()
print('------------------------------------------------------------')
print('\t3.2.10 处理重复列名')
print()

'''
如果该参数为True，当列名有重复时，解析列名将变为X, X.1, …,
X.N，而不是X, …, X。如果该参数为False，那么当列名中有重复时，
前列将会被后列覆盖。
'''

# 布尔值，默认为True
data = 'a,b,a\n0,1,2\n3,4,5'
df = pd.read_csv(StringIO(data))
# df = pd.read_csv(StringIO(data),mangle_dupe_clos=False) # 参数无效
print(df)
# 表头为a b a.1
# False会报ValueError错误
'''
out:
   a  b  a.1
0  0  1    2
1  3  4    5

不用mangle_dupe_clos参数，运行后第二个列名a自动成为a.1
'''

print()
print('------------------------------------------------------------')
print('\t3.2.11 数据类型')
print()

print('# 查看数据类型')
df = pd.read_csv(StringIO(data))
print(df.dtypes)
print()
'''
out:
a      int64
b      int64
a.1    int64
dtype: object
'''

print('# 所有数据列均为此数据类型')
df = pd.read_csv(StringIO(data), dtype=float)  # 所有数据列均为此数据类型
print(df.dtypes)
print()
'''
out:
a      float64
b      float64
a.1    float64
dtype: object
'''

print('# 指定字段的数据类型')
data = 'a,b,c\n0,1,2\n3,4,5'
df = pd.read_csv(StringIO(data), dtype={'b': float, 'a': str})  # 指定字段的数据类型
print(df.dtypes)
print()
'''
out:
a     object
b    float64
c      int64
dtype: object
'''

print('# 依次指定 ')
data = 'a,b,c,d\n0,1,2,4\n3,4,5,6'
df = pd.read_csv(StringIO(data))
print(df.dtypes)

# df = pd.read_csv(StringIO(data),dtype=[str,str,float,float])
# print(df.dtypes)
'''
# df = pd.read_csv(StringIO(data),dtype=[str,str,float,float])
# print(df.dtypes)
运行后 没有显示...不懂！
'''

print()
print('------------------------------------------------------------')
print('\t3.2.12 引擎')
print()

'''
使用的分析引擎可以选择C或Python。C语言的速度最快，
Python语言的功能最为完善，一般情况下，不需要另行指定。
'''

# 格式为engine=None，其中可选值有{'c', 'python'}
# df = pd.read_csv(big_file,encoding='gbk',engine='c')
# print(df.head())
# print(df.groupby('user_id').count())

'''
out:
测试两种引擎  engine='c' 的读取速度比 engine='python' 更快！
'''

print()
print('------------------------------------------------------------')
print('\t3.2.13 列数据处理')
print()

'''
使用converters参数对列的数据进行转换，
参数中指定列名与针对此列的处理函数，
最终以字典的形式传入，字典的键可以是列名或者列的序号。
'''

# 字典格式，默认为none
data = 'x,y\na,1\nb,2'


def foo(p):
    return p + 's'


df = pd.read_csv(StringIO(data))
print(df)

print()
print('# x应用函数，y使用lambda')
df = pd.read_csv(StringIO(data), converters={'x': foo, 'y': lambda x: x * 3})  # 使用列名
print(df)
'''
out:
    x    y
0  as  111
1  bs  222
'''

print()
print('# 使用列索引')
df = pd.read_csv(StringIO(data), converters={0: foo, 1: lambda x: int(x) * 3})  # 使用列索引
print(df)
'''
out:
    x  y
0  as  3
1  bs  6
'''

print()
print('------------------------------------------------------------')
print('\t3.2.14 真假值转换')
print()

'''
使用true_values和false_values将指定的文本内容转换为True或False，可以用列表指定多个值。
'''
# 列表，默认为None
data = 'a,b,c\n1,Yes,2\n3,No,4'
df = pd.read_csv(StringIO(data))
print(df)
df = pd.read_csv(StringIO(data), true_values=['Yes'], false_values=['No'])
print(df)
'''
out:
   a    b  c
0  1  Yes  2
1  3   No  4

   a      b  c
0  1   True  2
1  3  False  4
'''

print()
print('------------------------------------------------------------')
print('\t3.2.15 跳过指定行')
print()

'''
如下跳过需要忽略的行数（从文件开始处算起）或需要忽略的行号列表（从0开始）：
'''
df = pd.read_csv(input_file)
print(df)
print()

print('# 跳过前2行 原第3行默认为列名')
df = pd.read_csv(input_file, skiprows=2)
print(df)
print()

print('# 跳过前2行 原第3行默认为列名')
df = pd.read_csv(input_file, skiprows=range(2))  # 效果同上
print(df)
print()

print('# 跳过指定行')
df = pd.read_csv(input_file, skiprows=[2, 4])
print(df)
print()

print('# 跳过指定行-数组')
df = pd.read_csv(input_file, skiprows=np.array([2, 5, 11]))
print(df)
print()

print('# 隔行跳过')
df = pd.read_csv(input_file, skiprows=lambda x: x % 2 != 0)
print(df)
print()

print('# 尾部跳过')
# 尾部跳过，从文件尾部开始忽略，C引擎不支持。
df = pd.read_csv(input_file, skipfooter=1)  # 最后1行不加载
print(df)
print()

print('# 跳过空行')
df = pd.read_csv(lx_test, skip_blank_lines=True)  # 跳过空行
print(df)
df = pd.read_csv(lx_test, skip_blank_lines=False)  # 不跳过空行
print(df)
print()
'''
skip_blank_lines指定是否跳过空行，如果为True，则跳过空行，否则数据记为NaN。
如果skip_blank_lines=True，header参数将忽略空行和注释行，因此
header=0表示第一行数据而非文件的第一行。
'''

print()
print('------------------------------------------------------------')
print('\t3.2.16 读取指定行')
print()

'''
nrows参数用于指定需要读取的行数，从文件第一行算起，
经常用于较大的数据，先取部分进行代码编写。
'''
# int类型，默认为None
df = pd.read_csv(big_file, nrows=10, encoding='gbk')  # 添加参数encoding='gbk' 不然报错！
print(df)

print()
print('------------------------------------------------------------')
print('\t3.2.17 空值替换')
print()
'''
update20240116 
na_values参数的值是一组用于替换NA/NaN的值。
如果传参，需要指定特定列的空值。以下值默认会被认定为空值：

使用na_values时需要关注下面keep_default_na的配合使用和影响：
'''

data = 'a,b,c\n0,1,2\n3,4,5'
df = pd.read_csv(StringIO(data))
print(df)
# 可传入标量、字符串、类似列表序列和字典，默认为None
print('# 5和5.0会被认为是NaN')
df = pd.read_csv(StringIO(data), na_values=[5])
print(df)
print()
'''
out:
   a  b  c
0  0  1  2
1  3  4  5
   a  b    c
0  0  1  2.0
1  3  4  NaN
'''

print('# ?会被认为是NaN')
data = 'a,b,c\n0,1,2\n3,4,?'
df = pd.read_csv(StringIO(data))
print(df)
df = pd.read_csv(StringIO(data), na_values='?')
print(df)
print()
'''
out:
   a  b  c
0  0  1  2
1  3  4  ?
   a  b    c
0  0  1  2.0
1  3  4  NaN
'''

df = pd.read_csv(lx_test)
df.columns = [column.lower() for column in df.columns.str.replace(' ', '_')]
# print(df)
print(df)  # 默认跳过空行
print()

df = pd.read_csv(lx_test, skip_blank_lines=False)
print(df)
print()

df = pd.read_csv(lx_test, skip_blank_lines=True)
print(df)
print()

print('# 空值为NaN')
df = pd.read_csv(lx_test, keep_default_na=False, na_values=[""])
print(df)
print()
# keep_default_na=True
df = pd.read_csv(lx_test)  # 无论有没有参数keep_default_na=False/True,na_values=[""] | 输出都是跳过空行
print(df)
print()

print('# 字符NA和字符ds会被认为是NaN')
df = pd.read_csv(lx_test, keep_default_na=True, na_values=["NA", "ds"])
print(df)
print()

print('# Nope会被认为是NaN')
df = pd.read_csv(StringIO(data), na_values=["Nope"])
print(df)
print()

print('# Nope会被认为是NaN')
data = 'a,b,c\n0,1,2\n3,4,Nope'
df = pd.read_csv(StringIO(data))
print(df)
df = pd.read_csv(StringIO(data), na_values=["Nope"])
print(df)
print()
'''
out:
   a  b     c
0  0  1     2
1  3  4  Nope
   a  b    c
0  0  1  2.0
1  3  4  NaN
'''

print("# a、b、c均被认为是NaN，等于na_values=['a','b','c']")
data = 'a,b,c\n0,1,2\na,b,c'
df = pd.read_csv(StringIO(data))
print(df)
df = pd.read_csv(StringIO(data), na_values='abc')  # 实际输出 无效
print(df)
df = pd.read_csv(StringIO(data), na_values=['a', 'b', 'c'])  # 有效
print(df)
print()
'''
out:
  a  b  c
0  0  1  2
1  a  b  c
   a  b  c
0  0  1  2
1  a  b  c
     a    b    c
0  0.0  1.0  2.0
1  NaN  NaN  NaN

'''
print("# # 指定列的指定值会被认为是NaN")
data = 'a,b,c,1\n0,1,2,3\n1,2,3,5\n2,5,3,4'
df = pd.read_csv(StringIO(data))
print(df)
print(pd.read_csv(StringIO(data), na_values={'c': 3, 1: [2, 5]}))  # c列数值为3 和 第1列数值为2或5 的指定值为NaN
'''
out:
   a  b  c  1
0  0  1  2  3
1  1  2  3  5
2  2  5  3  4
   a    b    c  1
0  0  1.0  2.0  3
1  1  NaN  NaN  5
2  2  NaN  NaN  4
'''

print()
print('------------------------------------------------------------')
print('\t3.2.18 保留默认空值')
print()
'''
分析数据时是否包含默认的NaN值，是否自动识别。如果指定na_values参数，
并且keep_default_na=False，那么默认的NaN将被覆盖，否则添加。

keep_default_na	   na_values	逻辑
TRUE	           指定	        na_values的配置附加处理
TRUE	           未指定	    自动识别
FALSE	           指定	        使用na_values的配置
FALSE	           未指定	    不做处理

说明：
如果na_filter为False（默认为True），那么keep_default_na和na_values参数均无效。
'''

# 布尔型，默认为True
print('# 不自动识别空值')
# pd.read_csv(data, keep_default_na=False)
data = 'a,b,c\n0,1,2\n3,4,'
df = pd.read_csv(StringIO(data))
print(df)
print()
df = pd.read_csv(StringIO(data), keep_default_na=False)
print(df)
print()
df = pd.read_csv(StringIO(data), keep_default_na=True)
print(df)
print()
'''
out:
# 不自动识别空值
   a  b    c
0  0  1  2.0
1  3  4  NaN

   a  b  c
0  0  1  2
1  3  4   

   a  b    c
0  0  1  2.0
1  3  4  NaN 
'''

'''
na_filter为是否检查丢失值（空字符串或空值）。对于大文件来
说，数据集中没有空值，设定na_filter=False可以提升读取速度。
'''

print('# 布尔型，默认为True')
print(pd.read_csv(StringIO(data), na_filter=False))  # 不检查
print(pd.read_csv(big_file, encoding='gbk', na_filter=False))  # 检查有无参数na_filter=False 读取速度|| 好像没啥差别！

print()
print('------------------------------------------------------------')
print('\t3.2.19 日期时间解析')
print()
'''
update20240117

data = 'a,b,c,1\n0,1,2,3\n1,2,3,5\n2,5,3,4'
'''

data = '''a,b,c,date\n
0,1,2,01Jan2020\n
1,2,3,02Feb2020\n
2,5,3,03Mar2020'''

df = pd.read_csv(StringIO(data))
print(df)
print(df.dtypes)
print()
# df = pd.read_csv(StringIO(data),parse_dates=['date'],date_parser=date_parser) # 运行成功同时FutureWarning 修改如下
df = pd.read_csv(StringIO(data), parse_dates=['date'], dtype={'date': 'object'})
# df = pd.read_csv(StringIO(data),dtype={'date':'object'})
print(df)
print(df.dtypes)
df['date'] = pd.to_datetime(df['date'], format='%d%b%Y')  # 使用 pd.to_datetime 转换日期列
print(df)
print(df.dtypes)
print()
# parse_dates参数用于对时间日期进行解析。
# 布尔型、整型组成的列表、列表组成的列表或者字典，默认为False
print('# 自动解析日期时间格式')
df = pd.read_csv(StringIO(data), parse_dates=True)
print(df)
print()
print('# 指定日期时间字段进行解析')
df = pd.read_csv(StringIO(data), parse_dates=['date'])
print(df)
print()
print('# 将第1、3列合并解析成名为“datetime”的时间类型列')
df = pd.read_csv(StringIO(data), parse_dates={'datetime': [1, 3]})
print(df)
print()
# df = pd.read_csv(StringIO(data),parse_dates={'tie':[1,4]})
# print(df)
# print()

# 使用
print('# 使用input_file文件测试')
df = pd.read_csv(input_file)
# df.columns = [column.lower() for column in df.columns.str.replace(' ','_')]
print(df)
print(df.dtypes)
print()

# date_parser = lambda x: pd.to_datetime(x,format='%m/%d/%Y')
df = pd.read_csv(input_file, parse_dates=['Purchase Date'])  # ,dtype={'Purchase Date':'object'}
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], format='%m/%d/%Y', errors='coerce')  # ,errors='coerce'消除警告
df.columns = [column.lower() for column in df.columns.str.replace(' ', '_')]
print(df)
print(df.dtypes)

df = pd.read_csv(StringIO(data))
print(df)
print(df.dtypes)
print()
# df = pd.read_csv(StringIO(data),parse_dates=['date'],date_parser=date_parser) # 运行成功同时FutureWarning 修改如下
df = pd.read_csv(StringIO(data), parse_dates=['date'], dtype={'date': 'object'})
# df = pd.read_csv(StringIO(data),dtype={'date':'object'})
print(df)
print(df.dtypes)
df['date'] = pd.to_datetime(df['date'], format='%d%b%Y')  # 使用 pd.to_datetime 转换日期列
print(df)
print(df.dtypes)
print()

print('# 将第1、3列合并解析成名为“datetime”的时间类型列')
df = pd.read_csv(StringIO(data), parse_dates={'datetime': [1, 3]})
print(df)
print()

'''
如果用上文中的parse_dates参数将多列合并并解析成一个时间列，
设置keep_date_col的值为True时，会保留这些原有的时间组成列；如果
设置为False，则不保留这些列。

# 布尔型，默认为False
'''

print('# 会保留这些原有的时间组成列')  # keep_date_col=True
df = pd.read_csv(StringIO(data), parse_dates=[[1, 2], [1, 3]], keep_date_col=True)
print(df)
print()

df = pd.read_csv(StringIO(data), parse_dates={'datetime': [1, 3]}, keep_date_col=True)
print(df)
print()

print('# 不保留这些原有的时间组成列')  # keep_date_col=False
df = pd.read_csv(StringIO(data), parse_dates=[[1, 2], [1, 3]], keep_date_col=False)
print(df)
print()

# 如果infer_datetime_format被设定为True并且parse_dates可用，那么Pandas将尝试转换为日期类型。
# 布尔型，默认为False
# infer_datetime_format 参数将移除 默认
# df = pd.read_csv(StringIO(data),parse_dates=True,infer_datetime_format=True) # 输出 日期无变动
# print(df)

'''
对于DD/MM格式的日期类型，如日期2020-01-06，
如果dayfirst=True，则会转换成2020-06-01。
'''
print('# dayfirst参数')
data = '''a,b,c,date\n
0,1,2,01/03/2020\n
1,2,3,02/07/2020\n
2,5,3,03/02/2020'''

df = pd.read_csv(StringIO(data))
print(df)
print()

df = pd.read_csv(StringIO(data), parse_dates=[3])
print(df)
print()

# dayfirst 布尔型，默认为False
df = pd.read_csv(StringIO(data), dayfirst=True, parse_dates=[3])
print(df)
print()

'''
out:
   a  b  c        date
0  0  1  2  01/03/2020
1  1  2  3  02/07/2020
2  2  5  3  03/02/2020

   a  b  c       date
0  0  1  2 2020-01-03
1  1  2  3 2020-02-07
2  2  5  3 2020-03-02

   a  b  c       date
0  0  1  2 2020-03-01
1  1  2  3 2020-07-02
2  2  5  3 2020-02-03
'''

'''
cache_dates如果为True，则使用唯一的转换日期缓存来应用datetime
转换。解析重复的日期字符串，尤其是带有时区偏移的日期字符串时，
可能会大大提高速度。
'''
print('# cache_dates参数 布尔型，默认为False')
df = pd.read_csv(StringIO(data), parse_dates=[3], cache_dates=True)
print(df)
print()

print()
print('------------------------------------------------------------')
print('\t3.2.20 文件处理')
print()
'''
update20240118

data = 'a,b,c,1\n0,1,2,3\n1,2,3,5\n2,5,3,4'

以下是一些对读取文件对象的处理方法。iterator参数如果设置为True，
则返回一个TextFileReader对象，并可以对它进行迭代，以便逐块处理文件。
'''

# data = 'a,b,c,date\n0,1,2,01Jan2020\n1,2,3,02Feb2020\n2,5,3,03Mar2020'
data = '''a,b,c,date
0,1,2,01Jan2020
1,2,3,02Feb2020
2,5,2,03Mar2020
3,6,1,10Mar2020
4,7,0,11Mar2020
5,8,9,12Mar2020
6,9,8,13Mar2020
7,0,6,02Mar2020
'''

# 布尔型，默认为False
# df = pd.read_csv(StringIO(data),iterator=True)
# print(df)

print('示例 1: 分块读取 CSV 数据')
# 创建1个迭代器对象
df = pd.read_csv(StringIO(data), parse_dates=[3], iterator=True)
print(df)
print()

# 使用 get_chunk 方法分块读取数据
chunk_size = 2  # 每次读取的行数

while True:
    try:
        chunk = df.get_chunk(chunk_size)
        print(chunk)
    except StopIteration:
        print("Reached end of file")
        break

'''
out:
   a  b  c       date
0  0  1  2 2020-01-01
1  1  2  3 2020-02-02
   a  b  c       date
2  2  5  2 2020-03-03
3  3  6  1 2020-03-10
   a  b  c       date
4  4  7  0 2020-03-11
5  5  8  9 2020-03-12
   a  b  c       date
6  6  9  8 2020-03-13
7  7  0  6 2020-03-02
Reached end of file
'''

print()
print('使用 for 循环分块迭代 CSV 数据')
print()

df = pd.read_csv(StringIO(data), chunksize=3)  # 指定每个块的大小

for chunk in df:
    print(chunk)
    # 在这里可以对每个块进行处理

'''
out:
   a  b  c       date
0  0  1  2  01Jan2020
1  1  2  3  02Feb2020
2  2  5  2  03Mar2020
   a  b  c       date
3  3  6  1  10Mar2020
4  4  7  0  11Mar2020
5  5  8  9  12Mar2020
   a  b  c       date
6  6  9  8  13Mar2020
7  7  0  6  02Mar2020
'''

print()
print('------------------------------------------------------------')
print('\t3.2.20 文件处理')
print()
'''
update20240118

data = 'a,b,c,1\n0,1,2,3\n1,2,3,5\n2,5,3,4'

以下是一些对读取文件对象的处理方法。iterator参数如果设置为True，
则返回一个TextFileReader对象，并可以对它进行迭代，以便逐块处理文件。
'''

# data = 'a,b,c,date\n0,1,2,01Jan2020\n1,2,3,02Feb2020\n2,5,3,03Mar2020'
data = '''a,b,c,date
0,1,2,01Jan2020
1,2,3,02Feb2020
2,5,2,03Mar2020
3,6,1,10Mar2020
4,7,0,11Mar2020
5,8,9,12Mar2020
6,9,8,13Mar2020
7,0,6,02Mar2020
'''

print()
print('# 分块处理大文件')
print()
# df = pd.read_csv(StringIO(data),chunksize=4)
# 分块处理大文件
path = 'E:/bat/output_files/pandas_read_csv_20240118.csv'

df_iterator = pd.read_csv(StringIO(data), chunksize=2, parse_dates=[3])


def process_dataframe(df):
    pass
    return df


for index, df_tmp in enumerate(df_iterator):
    # print(index,process_dataframe(df_tmp))
    df_processed = process_dataframe(df_tmp)  # 这个函数是不是没啥用？
    if index == 0:
        df_processed.to_csv(path, mode='w', header=True, index=False)
    else:
        df_processed.to_csv(path, mode='a', header=False, index=False)

print()
print('# tar.gz文件超大文件分块处理')
print()
'''
compression（压缩格式）用于对磁盘数据进行即时解压缩。如果为“infer”,
且filepath_or_buffer是以.gz、.bz2、.zip或.xz结尾的字符串，
则使用gzip、bz2、zip或xz，否则不进行解压缩。
如果使用zip，则ZIP文件必须仅包含一个要读取的数据文件。设置为None将不进行解压缩。

# 可选值有'infer'、'gzip'、'bz2'、'zip'、'xz'和None，默认为'infer'
'''
in_file = 'E:/bat/input_files/jiangshuang/mchnt.tar.gz'
in_file_thr = 'E:/bat/input_files/jiangshuang/48200000_20231225_091446_jiangs_imchnt'

chunk_size = 5000  # 你可以根据你的内存大小调整这个值
# encoding='gbk'报错 || encoding='utf-8'也报错

df = pd.read_csv(in_file, compression='gzip', encoding='ISO-8859-1',
                 chunksize=chunk_size, usecols=[0], skiprows=1, names=['id'],
                 header=None, low_memory=False,
                 skip_blank_lines=True, engine='c')
# 运行成功  耗时大概2min 有800万+
for index, chunk in enumerate(df):
    #     # 处理每个 chunk 的代码
    #     print(chunk.head())
    #     print(chunk.dtypes)
    #     print(chunk.columns)
    #     print(chunk.shape)
    #     # print(chunk)
    #     # chunk = chunk.dtypes
    #     # chunk = chunk.head()
    if index == 0:
        chunk.to_csv('E:/bat/output_files/pandas_out_20240118041.csv', mode='w', header=True, index=False)
    else:
        chunk.to_csv('E:/bat/output_files/pandas_out_20240118041.csv', mode='a', header=False, index=False)
#     # chunk.to_csv('E:/bat/output_files/pandas_out_20231226023.csv')
#     break  # 如果你只想看第一个 chunk 的话，可以使用 break 语句

print()
print('------------------------------------------------------------')
print('\t3.2.21 符号')
print()
'''
update20240118

data = 'a,b,c,1\n0,1,2,3\n1,2,3,5\n2,5,3,4'

以下是对文件中的一些数据符号进行的特殊识别处理。如下设置千分位分隔符thousands：
'''

data = """a,b,c,date
0,'1,234','2,467,997',01Jan2020
'14,879','21','346,574,948',02Feb2020
2,'465','5,798,334',03Mar2020
"""

print()
print('# 千分位分隔符thousands：')
print()

# 逗号分隔
df = pd.read_csv(StringIO(data), thousands=',')
print(df)
print()

data1 = """a,b,c
        0,'1.234','2.467.997'
        1,'4.879','21'
        2,'465','5.798.334'
        """

# 英文句号分隔
df = pd.read_csv(StringIO(data1), thousands='.')
print(df)
print()

# 小数点decimal，识别为小数点的字符。
# 字符串，默认为'.'
pd.read_csv(data, decimal=",")

# 行结束符lineterminator，将文件分成几行的字符，仅对C解析器有效。
# 长度为1的字符串，默认为None
data = 'a,b,c~1,2,3~4,5,6'
df = pd.read_csv(StringIO(data), lineterminator='~')
