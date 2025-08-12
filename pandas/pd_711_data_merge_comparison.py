import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
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
print('第7章 Pandas数据合并与对比')
print('\t7.1 数据追加df.append')
print('\t7.1.1 基本语法')
print()
# update20240430
'''
在实际的应用中，数据可能分散在不同的文件、不同的数据库表中，也可能有不同的存储形式，
为了方便分析，需要将它们拼合在一起。一般有这样几种情况：
一是两份数据的列名完全相同，把其中一份数据追加到另一份的后面；
二是两份数据的列名有些不同，把这些列组合在一起形成多列；
三是以上两种情况混合。同时，在合并过程中还需要做些计算。

Pandas提供的各种功能能够轻而易举地完成这些工作。

'''
# 小节注释
'''
有这样一种场景，我们从数据库或后台系统的页面中导出数据，
由于单次操作数据量太大，会相当耗时，也容易超时失败，这时可以分多次导出，然后再进行合并。
df.append()可以将其他DataFrame附加到调用方的末尾，并返回一个新对象。

它是最简单、最常用的数据合并方式。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

# 语法结构
df._append(self, other, ignore_index=False,verify_integrity=False, sort=False)

'''
其中的各个参数说明如下。
other：调用方要追加的其他DataFrame或者类似序列内容。
       可以放入一个由DataFrame组成的列表，将所有DataFrame追加起来。
ignore_index：如果为True，则重新进行自然索引。
verify_integrity：如果为True，则遇到重复索引内容时报错。
sort：进行排序。
'''


print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.1 数据追加df.append')
print('\t7.1.2 相同结构')
print()
# update20240430
# 小节注释
'''
如果数据的字段相同，直接使用第一个DataFrame的append()方法，传入第二个DataFrame。
如果需要追加多个DataFrame，可以将它们组成一个列表再传入。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 代码示例')
df1 = pd.DataFrame({'x':[1,2],'y':[3,4]})
print(df1)

df2 = pd.DataFrame({'x':[5,6],'y':[7,8]})

# 追加合并
print(df1._append(df2))
#    x  y
# 0  1  3
# 1  2  4
# 0  5  7
# 1  6  8
print(df1._append(df2,ignore_index=True,verify_integrity=True,sort=True))
#    x  y
# 0  1  3
# 1  2  4
# 2  5  7
# 3  6  8

print()
print('# 追加多个数据')
print(df1._append([df2,df2,df2]))
#    x  y
# 0  1  3
# 1  2  4
# 0  5  7
# 1  6  8
# 0  5  7
# 1  6  8
# 0  5  7
# 1  6  8

print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.1 数据追加df.append')
print('\t7.1.3 不同结构')
print()
# update20240430
# 小节注释
'''
对于不同结构的追加，一方有而另一方没有的列会增加，没有内容的位置会用NaN填充。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 代码示例')
df1 = pd.DataFrame({'x':[1,2],'y':[3,4]})
print(df1)

df2 = pd.DataFrame({'x':[5,6],'y':[7,8]})

df3 = pd.DataFrame({'y':[5,6],'z':[7,8]})

# 追加合并
print(df1._append(df3))
#      x  y    z
# 0  1.0  3  NaN
# 1  2.0  4  NaN
# 0  NaN  5  7.0
# 1  NaN  6  8.0
print(df1._append(df3,ignore_index=True))
#      x  y    z
# 0  1.0  3  NaN
# 1  2.0  4  NaN
# 2  NaN  5  7.0
# 3  NaN  6  8.0

print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.1 数据追加df.append')
print('\t7.1.4 忽略索引')
print()
# update20240430

# 小节注释
'''
追加操作索引默认为原数据的，不会改变，如果需要忽略，可以传入ignore_index=True：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 忽略索引')
df1 = pd.DataFrame({'x':[1,2],'y':[3,4]})
print(df1)

df2 = pd.DataFrame({'x':[5,6],'y':[7,8]})

df3 = pd.DataFrame({'y':[5,6],'z':[7,8]})

# 追加合并
print(df1._append(df3))
#      x  y    z
# 0  1.0  3  NaN
# 1  2.0  4  NaN
# 0  NaN  5  7.0
# 1  NaN  6  8.0
print(df1._append(df3,ignore_index=True))
#      x  y    z
# 0  1.0  3  NaN
# 1  2.0  4  NaN
# 2  NaN  5  7.0
# 3  NaN  6  8.0

'''或者，可以根据自己的需要重新设置索引。对索引的操作前面介绍过。'''
print()
print('# 修改索引')
df4 = df1._append(df3,ignore_index=True)
print(df4.set_axis(['a','b','c','d'],axis='index'))


print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.1 数据追加df.append')
print('\t7.1.5 重复内容')
print()
# update20240430
# 小节注释
'''
重复内容默认是可以追加的，如果传入verify_integrity=True参数和值，则会检测追加内容是否重复，如有重复会报错。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 验证内容是否重复（包括索引值）')
df1 = pd.DataFrame({'x':[1,2],'y':[3,4]})
# print(df1)
df2 = pd.DataFrame({'x':[5,6],'y':[7,8]})

df3 = pd.DataFrame({'y':[5,6],'z':[7,8]})

# 追加合并
print(df1._append(df2,ignore_index=False,verify_integrity=True)) # 报错！
print(df1._append(df2,ignore_index=True,verify_integrity=True)) # 忽略索引，则运行正常


df4 = pd.DataFrame({'y':[4,5],'z':[7,8]},index=['a','b']) # 行索引值不同，数值相同，运行正常。
print(df1._append(df4,verify_integrity=True))
#      x  y    z
# 0  1.0  3  NaN
# 1  2.0  4  NaN
# a  NaN  4  7.0
# b  NaN  5  8.0

print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.1 数据追加df.append')
print('\t7.1.6 追加序列')
print()
# update20240430
# 小节注释
'''
append()除了追加DataFrame外，还可以追加一个Series，经常用于数据添加更新场景。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 追加序列_添加series')
print(df.tail(3))
#     name team   Q1  Q2  Q3   Q4
# 3  Eorge    C   93  96  71   78
# 4    Oah    D   65  49  61   86
# 5   Rick    B  100  99  97  100

# 定义新同学的信息

lily = pd.Series(['lily','C',55,56,57,58],
                 index=['name','team','Q1','Q2','Q3','Q4'])

# print(lily)
df = df._append(lily,ignore_index=True)
print(df.tail(3))
#    name team   Q1  Q2  Q3   Q4
# 4   Oah    D   65  49  61   86
# 5  Rick    B  100  99  97  100
# 6  lily    C   55  56  57   58


print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.1 数据追加df.append')
print('\t7.1.7 追加字典')
print()
# update20240430
# 小节注释
'''
append()还可以追加字典。我们可以将上面的学生信息定义为一个字典，然后进行追加：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 追加序列_添加字典')
print(df.tail(3))
#     name team   Q1  Q2  Q3   Q4
# 3  Eorge    C   93  96  71   78
# 4    Oah    D   65  49  61   86
# 5   Rick    B  100  99  97  100

# 定义新同学的信息

# lily = pd.Series(['lily','C',55,56,57,58],
#                  index=['name','team','Q1','Q2','Q3','Q4'])

lily = {'name':'lily','team':'C','Q1':55,'Q2':56,'Q3':57,'Q4':58}

# print(lily)
print()
df = df._append(lily,ignore_index=True)
print(df.tail(3))
#    name team   Q1  Q2  Q3   Q4
# 4   Oah    D   65  49  61   86
# 5  Rick    B  100  99  97  100
# 6  lily    C   55  56  57   58

'''以上操作更加直观简洁，推荐在需要增加单条数据的时候使用。'''

'''
7.1.8 小结
df.append()方法可以轻松实现数据的追加和拼接。如果列名相同，会追加一行到数据后面；
如果列名不同，会将新的列添加到原数据。
数据的追加是Pandas数据合并操作中最基础、最简单的功能，需要熟练掌握。

'''


print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.2 数据连接pd.concat')
print('\t7.2.1 基本语法')
print()
# update20240430
'''
Pandas数据可以实现纵向和横向连接，将数据连接后会形成一个新对象——Series或DataFrame。
连接是最常用的多个数据合并操作。
pd.concat()是专门用于数据连接合并的函数，它可以沿着行或者列进行操作，
同时可以指定非合并轴的合并方式（如合集、交集等）。

'''
# 小节注释
'''
以下为pd.concat()的基本语法：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

# 语法
pd.concat(objs, axis=0, join='outer',ignore_index=False, keys=None,
          levels=None, names=None, sort=False,verify_integrity=False, copy=True)

'''
其中主要的参数如下。
objs：需要连接的数据，可以是多个DataFrame或者Series。它是必传参数。
axis：连接轴的方法，默认值是0，即按列连接，追加在行后面。值为1时追加到列后面。
join：合并方式，其他轴上的数据是按交集（inner）还是并集（outer）进行合并。
ignore_index：是否保留原来的索引。
keys：连接关系，使用传递的键作为最外层级别来构造层次结构索引，就是给每个表指定一个一级索引。
names：索引的名称，包括多层索引。
verify_integrity：是否检测内容重复。参数为True时，如果合并的数据与原数据包含索引相同的行，则会报错。
copy：如果为False，则不要深拷贝。
pd.concat()会返回一个合并后的DataFrame。


默认情况下，copy=True，这意味着pd.concat()操作会创建原始数据的副本，然后在这些副本上执行连接操作。
如果设置为copy=False，则尽可能不创建数据副本，这可以减少内存消耗和提高性能，但这可能会导致原始数据被修改。
'''

print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.2 数据连接pd.concat')
print('\t7.2.2 简单连接')
print()
# update20240430
# 小节注释
'''
pd.concat()的基本操作可以实现前文讲的df.append()功能。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()


df1 = pd.DataFrame({'x':[1,2],'y':[3,4]})
# print(df1)
df2 = pd.DataFrame({'x':[5,6],'y':[7,8]})

df3 = pd.DataFrame({'y':[5,6],'z':[7,8]})

print(pd.concat([df1,df2]))
#    x  y
# 0  1  3
# 1  2  4
# 0  5  7
# 1  6  8

print()
print(df1._append(df2)) # 结果同上
'''操作中ignore_index和sort参数的作用一样。'''


print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.2 数据连接pd.concat')
print('\t7.2.3 按列连接')
print()
# update20240506

# 小节注释
'''
如果要将多个DataFrame按列拼接在一起，可以传入axis=1参数，
这会将不同的数据追加到列的后面，索引无法对应的位置上将值填充为NaN

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()


df1 = pd.DataFrame({'x':[1,2],'y':[3,4]})

df2 = pd.DataFrame({'x':[5,6,0],'y':[7,8,0]})

print(pd.concat([df1,df2],axis=1))
#      x    y  x  y
# 0  1.0  3.0  5  7
# 1  2.0  4.0  6  8
# 2  NaN  NaN  0  0
'''上例中，df2的行数比df1多一行，合并后df1的部分为NaN。'''

print()
# df2 = pd.DataFrame({'x':[5,6,0],'z':[7,8,0]})
# print(pd.concat([df1,df2],ignore_index=True,sort=True)) # 测试z列 结果如下
#    x    y    z
# 0  1  3.0  NaN
# 1  2  4.0  NaN
# 2  5  NaN  7.0
# 3  6  NaN  8.0
# 4  0  NaN  0.0

print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.2 数据连接pd.concat')
print('\t7.2.4 合并交集')
print()
# update20240506

# 小节注释
'''
以上连接操作会得到两个表内容的并集（默认是join ='outer'），那如果我们需要交集呢？

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()


df1 = pd.DataFrame({'x':[1,2],'y':[3,4]})

df2 = pd.DataFrame({'x':[5,6,0],'y':[7,8,0]})

print(pd.concat([df1,df2],axis=1,join='inner'))
#    x  y  x  y
# 0  1  3  5  7
# 1  2  4  6  8

'''
传入join='inner'取得两个DataFrame的共有部分，去除了df1没有的第三行内容。

另外，我们可以通过reindex()方法实现以上取交集功能：
'''

print()
print('# 两种方法')
print(pd.concat([df1,df2],axis=1).reindex(df1.index)) # 结果同上
print(pd.concat([df1,df2.reindex(df1.index)],axis=1)) # 结果同上
# print(df2.reindex(df1.index))

print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.2 数据连接pd.concat')
print('\t7.2.5 与序列合并')
print()
# update20240506
'''
Pandas数据可以实现纵向和横向连接，将数据连接后会形成一个新对象——Series或DataFrame。
连接是最常用的多个数据合并操作。
pd.concat()是专门用于数据连接合并的函数，它可以沿着行或者列进行操作，
同时可以指定非合并轴的合并方式（如合集、交集等）。

'''
# 小节注释
'''
如同df.append()一样，DataFrame也可以用以下方法与Series合并：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()


df1 = pd.DataFrame({'x':[1,2],'y':[3,4]})

df2 = pd.DataFrame({'x':[5,6,0],'y':[7,8,0]})

z = pd.Series([9,9],name='z')


print(pd.concat([df1,z],axis=1))
#    x  y  z
# 0  1  3  9
# 1  2  4  9

'''
但是，还是建议使用df.assign()来定义一个新列，逻辑会更加简单：
'''
print()
print('# 增加新列')
print(df1.assign(z=z)) # 结果同上
#    x  y  z
# 0  1  3  9
# 1  2  4  9

print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.2 数据连接pd.concat')
print('\t7.2.6 指定索引')
print()
# update20240506
# 小节注释
'''
我们可以再给每个表一个一级索引，形成多层索引，这样可以清晰地看到合成后的数据分别来自哪个DataFrame。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()
print('# 指定索引名')

df1 = pd.DataFrame({'x':[1,2],'y':[3,4]})

df2 = pd.DataFrame({'x':[5,6,0],'y':[7,8,0]})

print(pd.concat([df1,df2],keys=['a','b'])) # ,ignore_index=True 如果添加该参数 则没有多层索引效果
#      x  y
# a 0  1  3
#   1  2  4
# b 0  5  7
#   1  6  8
#   2  0  0

print()
print('# 以字典形式传入')
pieces = {'a':df1,'b':df2}
print(pd.concat(pieces)) # 结果同上

print()
print('# 横向合并，指定索引')
print(pd.concat([df1,df2],axis=1,keys=['a','b']))
#      a       b
#      x    y  x  y
# 0  1.0  3.0  5  7
# 1  2.0  4.0  6  8
# 2  NaN  NaN  0  0

print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.2 数据连接pd.concat')
print('\t7.2.7 多文件合并')
print()
# update20240506

# 小节注释
'''
多文件合并在实际工作中比较常见，汇总表往往费时费力，如果用代码来实现就会省力很多，而且还可以复用，
从而节省大量时间。最简单的方法是先把数据一个个地取出来，然后进行合并.

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()
print('# 通过各种方式读取数据')

# df1 = pd.DataFrame({'x':[1,2],'y':[3,4]})

df2 = pd.read_csv(csv_file) # pd.DataFrame({'x':[5,6,0],'y':[7,8,0]})
df2 = df2.assign(type='df2') # 新增列 区分文件来源

df3 = pd.read_excel(input_file)
df3 = df3.assign(type='df3') # 新增列 区分文件来源

# 合并数据
merged_df = pd.concat([df3,df2],keys=['a','b'])
print(merged_df)
#      Customer ID       Customer Name  ...        Purchase Date type
# a 0         1234          John Smith  ...  2013-01-01 00:00:00  df3
#   1         2345       Mary Harrison  ...  2013-01-06 00:00:00  df3
# ...
#   5         6789  Samantha Donaldson  ...  2013-01-31 00:00:00  df3
# b 0         1234          John Smith  ...               1/1/14  df2
#   1         2345       Mary Harrison  ...               1/6/14  df2
# ...
#   5         6789  Samantha Donaldson  ...              1/31/14  df2
#
# [12 rows x 6 columns]

'''
注意，不要一个表格用一次concat，这样性能会很差，可以先把所有表格加到列表里，然后一次性合并：

# process_your_file(f)方法将文件读取为DataFrame
frames = [process_your_file(f) for f in files]
# 合并
result = pd.concat(frames)
'''

print()
print('# 读取同类型文件')

for file in glob.glob('E:/bat/input_files/*.xlsx'):
    print(file)
'''    
E:/bat/input_files\dq_split_file.xlsx
E:/bat/input_files\import_data_source.xlsx
E:/bat/input_files\sales_2013.xlsx
E:/bat/input_files\sales_2014.xlsx
E:/bat/input_files\sales_2015.xlsx
E:/bat/input_files\split_mpos.xlsx
E:/bat/input_files\split_mpos_less.xlsx
E:/bat/input_files\team.xlsx
'''


# print(glob.glob('E:/bat/input_files/*.csv')) # 输出列表

print()
print('# 自测 读取指定文件夹 指定文件名结尾（sale_*） ，然后合并文件 ')
files = [pd.read_excel(file) for file in glob.glob('E:/bat/input_files/sales_*.xlsx')]

# print(files) # 输出列表
print()
result = pd.concat(files)
print(result) # 运行成功
'''
files = [pd.read_excel(file) for file in glob.glob('E:/bat/input_files/*.xlsx')]
            日期       代理商户号          代理商名称  ...    Q2    Q3     Q4
0   2023-05-06  31018821.0  江西瑞康达电子科技有限公司  ...   NaN   NaN    NaN
....
5          NaN         NaN            NaN  ...  99.0  97.0  100.0

[427377 rows x 61 columns]
'''

print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.2 数据连接pd.concat')
print('\t7.2.8 目录文件合并')
print()
# update20240506

# 小节注释
'''
有时会将体量比较大的数据分片放到同一个硬盘目录下，在使用时进行合并。
可以使用Python的官方库glob来识别目录文件：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()
print('# 通过各种方式读取数据')
# 取出目录下所有XLSX格式的文件
files = glob.glob('E:/bat/input_files/sales_*.xlsx')
cols = ['Customer ID','Customer Name'] # 只取这2列
# 列表推导出对象
dflist = [pd.read_excel(i,usecols=cols) for i in files]
df1 = pd.concat(dflist)
print(df1)

'''    
   Customer ID       Customer Name
0         1234          John Smith
1         2345       Mary Harrison
........
3         4567        Rupert Jones
4         5678       Jenny Walters
5         6789  Samantha Donaldson
'''

print()
print('# 使用Python内置map函数进行操作：')
# 使用pd.read_csv逐一读取文件，然后合并
pd.concat(map(pd.read_csv, ['data/d1.csv',
                            'data/d2.csv',
                            'data/d3.csv']))
# 使用pd.read_excel逐一读取文件，然后合并
pd.concat(map(pd.read_excel, ['data/d1.xlsx',
                              'data/d2.xlsx',
                              'data/d3.xlsx']))


# 目录下的所有文件
from os import listdir
filepaths = [f for f in listdir("./data") if f.endswith('.csv')]
df = pd.concat(map(pd.read_csv, filepaths))

# 其他方法
import glob
df = pd.concat(map(pd.read_csv, glob.glob('data/*.csv')))
df = pd.concat(map(pd.read_excel, glob.glob('data/*.xlsx')))

'''在实际使用中，熟练掌握其中一个方法即可。

7.2.9 小结
相比pd.append()，pd.concat()的功能更为丰富，它是Pandas的一个通用方法，
可以灵活地合并DataFrame的各种序列数据，从而方便地取交集和并集数据。
这个方法在多文件数据合并方面也能让人得心应手。
'''


print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.3 数据合并pd.merge')
print('\t7.3.1 基本语法')
print()
# update20240506
'''
Pandas提供了一个pd.merge()方法，可以实现类似SQL的join操作，功能更全、性能更优。
通过pd.merge()方法可以自由灵活地操作各种逻辑的数据连接、合并等操作。

'''
# 小节注释
'''
基本语法

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()
# 基本语法
pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True,
         suffixes=('_x', '_y'), copy=True, indicator=False,
         validate=None)

# TODO 可以将两个DataFrame或Series合并，最终返回一个合并后的DataFrame。其中的主要参数如下。
'''
left、right：需要连接的两个DataFrame或Series，一左一右。
how：两个数据连接方式，默认为inner，可设置为inner、outer、left或right。
on：作为连接键的字段，左右数据中都必须存在，否则需要用left_on和right_on来指定。
left_on：左表的连接键字段。
right_on：右表的连接键字段。
left_index：为True时将左表的索引作为连接键，默认为False。
right_index：为True时将右表的索引作为连接键，默认为False。
suffixes：如果左右数据出现重复列，新数据表头会用此后缀进行区分，默认为_x和_y。
'''

print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.3 数据合并pd.merge')
print('\t7.3.2 连接键')
print()
# update20240506

# 小节注释
'''
在数据连接时，如果没有指定根据哪一列（连接键）进行连接，
Pandas会自动找到相同列名的列进行连接，并按左边数据的顺序取交集数据。
为了代码的可阅读性和严谨性，推荐通过on参数指定连接键。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()
# 基本语法

# df1 = pd.read_csv(csv_file)
# df2 = pd.read_csv(csv_feb_file)

df1 = pd.DataFrame({'a':[1,2],'x':[5,6]})
print(df1)

df2 = pd.DataFrame({'a':[2,1,0],'y':[6,7,8]})
print(df2)

print(pd.merge(df1,df2,on='a'))
#    a  x  y
# 0  1  5  7
# 1  2  6  6
print(pd.merge(df1,df2,left_on='a',right_on='a')) # 结果同上
#    a  x  y
# 0  1  5  7
# 1  2  6  6

'''以上按a列进行连接，数据顺序取了df1的顺序。'''


print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.3 数据合并pd.merge')
print('\t7.3.3 索引连接')
print()
# update20240506

# 小节注释
'''
可以直接按索引进行连接，将left_index和right_index设置为True，
会以两个表的索引作为连接键，示例代码如下。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()
# 基本语法

# df1 = pd.read_csv(csv_file)
# df2 = pd.read_csv(csv_feb_file)

df1 = pd.DataFrame({'a':[1,2],'x':[5,6]})
print(df1)

df2 = pd.DataFrame({'a':[2,1,0],'y':[6,7,8]})
print(df2)

print(pd.merge(df1,df2,left_index=True,right_index=True,suffixes=('_1','_2'))) # ,suffixes=('_1','_2')
#    a_1  x  a_2  y
# 0    1  5    2  6
# 1    2  6    1  7
# -- 无参数,suffixes=('_1','_2')
#    a_x  x  a_y  y
# 0    1  5    2  6
# 1    2  6    1  7
'''本例中，两个表都有同名的a列，用suffixes参数设置了后缀来区分。'''

print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.3 数据合并pd.merge')
print('\t7.3.4 多连接键')
print()
# update20240506

# 小节注释
'''
如果在合并数据时需要用多个连接键，可以以列表的形式将这些连接键传入on中。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()
# 基本语法

# df1 = pd.read_csv(csv_file)
# df2 = pd.read_csv(csv_feb_file)

df3 = pd.DataFrame({'a':[1,2],'b':[3,4],'x':[5,6]})
df4 = pd.DataFrame({'a':[1,2,3],'b':[3,4,5],'y':[6,7,8]})

print(pd.merge(df3,df4,on=['a','b']))
#    a  b  x  y
# 0  1  3  5  6
# 1  2  4  6  7

'''本例中，a和b列中的（1，3）和（2，4）作为连接键将两个数据进行了连接。'''

print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.3 数据合并pd.merge')
print('\t7.3.5 连接方法')
print()
# update20240507

# 小节注释
'''
how参数可以指定数据用哪种方法进行合并，可以设置为inner、outer、left或right，实现类似SQL的join操作。
默认的方式是inner join，取交集，也就是保留左右表的共同内容；
如果是left join，左边表中所有的内容都会保留；
如果是right join，右表全部保留；
如果是outer join，则左右表全部保留。关联不上的内容为NaN

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()
# 基本语法

df3 = pd.DataFrame({'a':[1,2],'b':[3,4],'x':[5,6]})
df4 = pd.DataFrame({'a':[1,2,3],'b':[3,4,5],'y':[6,7,8]})

print('# 以左表为基表')
print(pd.merge(df3,df4,how='left',on=['a','b']))
#    a  b  x  y
# 0  1  3  5  6
# 1  2  4  6  7

print()
print('# 以右表为基表')
print(pd.merge(df3,df4,how='right',on=['a','b']))
#    a  b    x  y
# 0  1  3  5.0  6
# 1  2  4  6.0  7
# 2  3  5  NaN  8
# print(pd.merge(df4,df3,how='left',on=['a','b']))

# 以下是一些其他的案例。
print()
print('# 取两个表的并集')
print(pd.merge(df3,df4,how='outer',on=['a','b']))
#    a  b    x  y
# 0  1  3  5.0  6
# 1  2  4  6.0  7
# 2  3  5  NaN  8

print()
print('# 取两个表的交集')
print(pd.merge(df3,df4,how='inner',on=['a','b']))
#    a  b  x  y
# 0  1  3  5  6
# 1  2  4  6  7


print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.3 数据合并pd.merge')
print('\t7.3.6 连接指示')
print()
# update20240507

# 小节注释
'''
如果想知道数据连接后是左表内容还是右表内容，可以使用indicator参数显示连接方式。
如果将indicator设置为True，则会增加名为_merge的列，显示这列是从何而来。
_merge列有以下三个取值。
left_only：只在左表中。
right_only：只在右表中。
both：两个表中都有。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()
# 基本语法

df3 = pd.DataFrame({'a':[1,2],'b':[3,4],'x':[5,6]})
df4 = pd.DataFrame({'a':[1,2,3],'b':[3,4,5],'y':[6,7,8]})

print('# 显示连接指示列')
print(pd.merge(df3,df4,how='outer',on=['a','b'],indicator=True))
#    a  b    x  y      _merge
# 0  1  3  5.0  6        both
# 1  2  4  6.0  7        both
# 2  3  5  NaN  8  right_only

'''
7.3.7 小结
pd. merge()方法非常强大，可以实现SQL的join操作。
可以用它来进行复杂的数据合并和连接操作，
但不建议用它来进行简单的追加、拼接操作，因为理解起来有一定的难度。
'''


print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.4 按元素合并')
print('\t7.4.1 df.combine_first()')
print()
# update20240507
'''
在数据合并过程中需要对对应位置的数值进行计算，比如相加、平均、对空值补齐等，
Pandas提供了df.combine_first()和df.combine()等方法进行这些操作。

'''
# 小节注释
'''
使用相同位置的值更新空元素，只有在df1有空元素时才能替换值。
如果数据结构不一致，所得DataFrame的行索引和列索引将是两者的并集。示例代码如下。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()
# 基本语法

df1 = pd.DataFrame({'A':[None,1],'B':[None,2]})
df2 = pd.DataFrame({'A':[3,3],'B':[4,4]})

print('# 数据结构一致_合并')
print(df1.combine_first(df2))
#      A    B
# 0  3.0  4.0
# 1  1.0  2.0

'''在上例中，df1中的A和B的空值被df2中相同位置的值替换。下面是另一个示例。'''

print()
print('# 数据结构不一致_合并')

df1 = pd.DataFrame({'A':[None,1],'B':[2,None]})
df2 = pd.DataFrame({'A':[3,3],'C':[4,4]},index=[1,2])
print(df1.combine_first(df2))
#      A    B    C
# 0  NaN  2.0  NaN
# 1  1.0  NaN  4.0
# 2  3.0  NaN  4.0
'''在上例中，df1中的A中的空值由于没有B中相同位置的值来替换，仍然为空。'''


print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.4 按元素合并')
print('\t7.4.2 df.combine()')
print()
# update20240507

# 小节注释
'''
可以与另一个DataFrame进行按列组合。
使用函数通过计算将一个DataFrame与其他DataFrame合并，以逐元素方式合并列。
所得DataFrame的行索引和列索引将是两者的并集。
这个函数中有两个参数，分别是两个df中对应的Series，计算后返回一个Series或者标量。
下例中，合并时取对应位置大的值作为合并结果。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()
# 基本语法

df1 = pd.DataFrame({'A':[1,2],'B':[3,4]})
df2 = pd.DataFrame({'A':[0,3],'B':[2,1]})

print('# 合并，方法为：s1和s2对应位置上哪个值大就返回哪个值')
print(df1.combine(df2,lambda s1,s2: np.where(s1>s2,s1,s2)))
#    A  B
# 0  1  3
# 1  3  4

print()
print('# 取最大值，即上例的实现')
# 也可以直接使用NumPy的函数：
print(df1.combine(df2,np.maximum)) # 结果同上！
#    A  B
# 0  1  3
# 1  3  4
print('# 取对应最小值')
print(df1.combine(df2,np.minimum))
#    A  B
# 0  0  2
# 1  2  1

print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.4 按元素合并')
print('\t7.4.3 df.update()')
print()
# update20240507

# 小节注释
'''
利用update()方法，可以使用来自另一个DataFrame的非NaN值来修改DataFrame，
而原DataFrame被更新，示例代码如下。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()
# 基本语法

df1 = pd.DataFrame({'a':[None,2],'B':[5,6]})
df2 = pd.DataFrame({'a':[0,2],'B':[None,7]})

print('# 合并，所有空值被对应位置的值替换')
df1.update(df2) # 默认overwrite = True  有空值会覆盖 非空值也覆盖！
print(df1)
#      a  B
# 0  0.0  5
# 1  2.0  7

# df1.update(df2,overwrite=False)
#      a  B
# 0  0.0  5
# 1  2.0  6

'''上例中，如果不想让df1被更新，可以传入参数overwrite=True。'''

'''
7.4.4 小结
之前我们了解的都是对数据整体性的合并，而本节我们介绍的几个方法是真正的元素级的合并，
它们可以按照复杂的规则对两个数据进行合并.
'''


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
print('第7章 Pandas数据合并与对比')
print('\t7.5 数据对比df.compare')
print('\t7.5.1 简单对比')
print()
# update20240508
'''
df.compare()和s.compare()方法分别用于比较两个DataFrame和Series，并总结它们之间的差异。
Pandas在V1.1.0版本中添加了此功能，
如果想使用它，需要先将Pandas升级到此版本及以上。

'''
# 小节注释
'''
在DataFrame上使用compare()传入对比的DataFrame可进行数据对比，如：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

# TODO 基本语法

# def compare(self,
#             other: DataFrame,
#             align_axis: Literal["index", "columns", "rows"] | int = 1,
#             keep_shape: bool = False,
#             keep_equal: bool = False,
#             result_names: Tuple[str | None, str | None] = ("self", "other"))


df1 = pd.DataFrame({'a':[1,2],'B':[5,6]})
df2 = pd.DataFrame({'a':[0,2],'B':[5,7]})
# 对比数据
print(df1.compare(df2))
#      a          B
#   self other self other
# 0  1.0   0.0  NaN   NaN
# 1  NaN   NaN  6.0   7.0

'''
上例中，由于两个表的a列第一行和b列第二行数值有差异，故在各列二级索引中用self和other分别显示数值用于对比；
对于相同的部分，由于不用关心，用NaN表示。
'''

print()
print('# 另一个例子')
df1 = pd.DataFrame({'a': [1, 2], 'b': [5, 6]})
df2 = pd.DataFrame({'a': [1, 2], 'b': [5, 7]})
# 对比数据
print(df1.compare(df2))
#      b
#   self other
# 1  6.0   7.0

'''上例中，a列数据相同，不显示，仅显示不同的b列第二行。另外请注意，只能对比形状相同的两个数据。'''

# 不同参数效果
# print(df1.compare(df2,keep_shape=True,align_axis="columns",result_names=("df1","df2")))


print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.5 数据对比df.compare')
print('\t7.5.2 对齐方式')
print()
# update20240508

# 小节注释
'''
默认情况下，将不同的数据显示在列方向上，我们还可以传入参数
align_axis=0将不同的数据显示在行方向上：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

df1 = pd.DataFrame({'a':[1,2],'B':[5,6]})
df2 = pd.DataFrame({'a':[0,2],'B':[5,7]})
# 对比数据
print(df1.compare(df2,align_axis=0))
#            a    B
# 0 self   1.0  NaN
#   other  0.0  NaN
# 1 self   NaN  6.0
#   other  NaN  7.0

print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.5 数据对比df.compare')
print('\t7.5.3 显示相同值')
print()
# update20240508

# 小节注释
'''
在对比时也可以将相同的值显示出来，方法是传入参数keep_equal=True，示例代码如下：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

df1 = pd.DataFrame({'a':[1,2],'B':[5,6]})
df2 = pd.DataFrame({'a':[0,2],'B':[5,7]})
# 对比数据
print(df1.compare(df2,keep_equal=True))
#      a          B
#   self other self other
# 0    1     0    5     5
# 1    2     2    6     7

'''这样，对比结果数据即使相同，也会显示出来。'''

print()
print('------------------------------------------------------------')
print('第7章 Pandas数据合并与对比')
print('\t7.5 数据对比df.compare')
print('\t7.5.4 保持形状')
print()
# update20240508
'''
df.compare()和s.compare()方法分别用于比较两个DataFrame和Series，并总结它们之间的差异。
Pandas在V1.1.0版本中添加了此功能，
如果想使用它，需要先将Pandas升级到此版本及以上。

'''
# 小节注释
'''
对比时，为了方便知道不同的数据在什么位置，
可以用keep_shape=True来显示原来数据的形态，
不过相同的数据会被替换为NaN进行占位：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

df1 = pd.DataFrame({'a':[1,2],'B':[5,6]})
df2 = pd.DataFrame({'a':[0,2],'B':[5,7]})
# 对比数据
print(df1.compare(df2,keep_shape=True))
#      a          B
#   self other self other
# 0  1.0   0.0  NaN   NaN
# 1  NaN   NaN  6.0   7.0

'''如果想看到原始值，可同时传入keep_equal=True：'''

print(df1.compare(df2,keep_shape=True,keep_equal=True))
#      a          B
#   self other self other
# 0    1     0    5     5
# 1    2     2    6     7

'''
7.5.5 小结
对比数据非常有用，我们在日常办公、数据分析时需要核对数据，
可以使用它来帮助我们自动处理，特别是在数据量比较大的场景，这样能够大大节省人力资源。


7.6 本章小结
本章介绍了数据的合并和对比操作。
对比相对简单，用df.compare()操作进行对比后，能清晰地看到两个数据之间的差异。
合并有df.append()、pd.concat()和pd.merge()三个方法：
df.append()适合在原数据上做简单的追加，一般用于数据内容的追加；
pd.concat()既可以合并多个数据，也可以合并多个数据文件；
pd.merge()可以做类似SQL语句中的join操作，功能最为丰富。

以上几个方法可以帮助我们对多个数据进行整理并合并成一个完整的DataFrame，以便于我们对数据进行整体分析。
'''
