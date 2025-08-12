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
# df1.to_excel('E:/bat/output_files/pandas_out_20240510053.xlsx')

# TODO 忽略特定的警告 | 当遇到 openpyxl 中的 UserWarning 类型的警告时，它们将不会被打印出来。
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

print()
print('------------------------------------------------------------')
print('第11章 Pandas文本处理')
print('\t11.1 数据类型')
print('\t11.1.1 文本数据类型')
print()
# update20240524
'''
我们知道Pandas能够非常好地处理数值信息，对文本信息也有良好的处理能力。
我们日常接触到的大量信息为文本信息，可以在文本中解析出数据信息，然后再进行计算分析。

文本信息是我们在日常办公中遇到的主要数据类型，在做业务表格时也会有大量的文本信息，
对这些文本的加工处理是一件令人头疼的事。
本章，我们就来一起看看Pandas是如何解决这些文本处理的问题的。
'''
# 小节注释
'''
object和StringDtype是Pandas的两个文本类型。在1.0版本之前，
object是唯一文本类型，Pandas会将混杂各种类型的一列数据归为object，
不过在1.0版本之后，使用官方推荐新的数据类型StringDtype，
这样会使代码更加清晰，处理更加高效。本节，我们就来认识一下文本的数据类型。

默认情况下，文本数据会被推断为object类型：

▶ 以下是一些具体的使用方法举例：
'''
# 原数据
df = pd.DataFrame({
    'A': ['a1', 'a1', 'a2', 'a2'],
    'B': ['b1', 'b2', None, 'b2'],
    'C': [1, 2, 3, 4],
    'D': [5, 6, None, 8],
    'E': [5, None, 7, 8]
})

print(df)
print(df.dtypes)
# A     object
# B     object
# C      int64
# D    float64
# E    float64
# dtype: object

'''如果想使用string类型，需要专门指定：'''

# 指定数据类型
print(pd.Series(['a', 'b', 'c'], dtype='string'))
print(pd.Series(['a', 'b', 'c'], dtype=pd.StringDtype()))  # 结果同上
# 0    a
# 1    b
# 2    c
# dtype: string

print()
print('------------------------------------------------------------')
print('\t11.1.2 类型转换')
'''关于将类型转换为string类型，推荐使用以下转换方法，它能智能地将数据类型转换为最新支持的合适类型：'''
# 类型转换，支持string类型
print(df.convert_dtypes().dtypes)
# A    string[python]
# B    string[python]
# C             Int64
# D             Int64
# E             Int64
# dtype: object

'''当然也可以使用之前介绍过的astype()进行转换：'''
print()
s = pd.Series(['a', 'b', 'c'])
print(s.dtype)
print(s.astype('object').dtype)
print(s.astype("string").dtype)
print(s.dtype)
# object
# object
# string
# object

print()
print('------------------------------------------------------------')
print('\t11.1.3 类型异同')

print('# 数值为Int64')
print(pd.Series(["a", None, "b"]).str.count("a"))  # dtype: float64
# 0    1.0
# 1    NaN
# 2    0.0
# dtype: float64
print(pd.Series(["a", None, "b"], dtype="string").str.count("a"))  # dtype: Int64
# 0       1
# 1    <NA>
# 2       0
# dtype: Int64

print('# 逻辑判断为boolean')
print(pd.Series(["a", None, "b"]).str.isdigit())  # dtype: object
# 0    False
# 1     None
# 2    False
# dtype: object
print(pd.Series(["a", None, "b"], dtype="string").str.isdigit())
# 0    False
# 1     <NA>
# 2    False
# dtype: boolean

'''推荐使用StringDtype。

11.1.4 小结
string和object都是Pandas的字符文本数据类型，在往后的版本中，
Pandas将逐渐提升string类型的重要性，可能将它作为各个场景下的默认字符数据类型。
由于string类型性能更好，功能更丰富，所以推荐大家尽量使用string类型。
'''

print()
print('------------------------------------------------------------')
print('第11章 Pandas文本处理')
print('\t11.2 字符的操作')
print('\t11.2.1 .str访问器')
print()
# update20240524
'''
Series和Index都有一些字符串处理方法，可以方便地进行操作，
这些方法会自动排除缺失值和NA值。可以通过str属性访问它的方法，进行操作。
'''
# 小节注释
'''
可以使用.str.<method>访问器（Accessor）来对内容进行字符操作：

▶ 以下是一些具体的使用方法举例：
'''
# 原数据
s = pd.Series(['A', 'Boy', 'C', np.nan], dtype="string")
# print(s)
# 转为小写
print(s.str.lower())
# 0       a
# 1     boy
# 2       c
# 3    <NA>
# dtype: string

print()
'''对于非字符类型，可以先转换再使用：'''
df = pd.read_excel(team_file, index_col=False)
print(df)
# 转为object
print('# 转为object')
print(df.Q1.astype(str).str)
# <pandas.core.strings.accessor.StringMethods object at 0x00000145A3756CD0>
print('# 转为StringDtype')
print(df.Q1.astype("string").str)
print(df.Q1.astype(str).astype("string").str)
# <pandas.core.strings.accessor.StringMethods object at 0x00000145A3720130>
# <pandas.core.strings.accessor.StringMethods object at 0x0000017513704EB0>
print(df.team.astype("string").str)
# <pandas.core.strings.accessor.StringMethods object at 0x0000017511594CD0>

print()
'''大多数操作也适用于df.index、df.columns索引类型：'''
print('# 对索引进行操作')
# df.set_index('name',inplace=True)
# print(df.index.str.lower()) # 必须设置一个文本列索引 否则运行失败
# Index(['liver', 'arry', 'ack', 'eorge', 'oah', 'rick'], dtype='object', name='name')

# print(df.index.astype(str).lower()) # 默认索引 报错！ AttributeError: 'Index' object has no attribute 'lower'

print(df.columns.str.lower())
# Index(['team', 'q1', 'q2', 'q3', 'q4'], dtype='object')

'''通过.str这个桥梁能让数据获得非常多的字符操作能力。'''

print()
print('------------------------------------------------------------')
print('\t11.2.2 文本格式')
'''以下是一些对文本的格式操作'''

s = pd.Series(['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe'])
df_s = pd.DataFrame(s, columns=['a'])
print(s)
print(df_s)
print()
print(s.str.lower())  # 转为小写
# print(df_s.str.lower()) # 报错！
print(s.str.upper())  # 转为大写
print(s.str.title())  # 标题格式，每个单词首字母大写
print(s.str.swapcase())  # 大小写互换
print(s.str.casefold())  # 转为小写，支持其他语言（如德语）

'''支持大多数Python对字符串的操作。'''
print()
print(df_s)
print(df_s.applymap(lambda x: x.lower()))  # 正常 应用于每个df的元素！
# print(df_s.apply(lambda x: x.lower())) # 报错！

print()
print('------------------------------------------------------------')
print('\t11.2.3 文本对齐')

print("""居中对齐，宽度为10，用'-'填充''""")

print(s.str.center(10, fillchar='-'))
# 0            --lower---
# 1            -CAPITALS-
# 2    this is a sentence
# 3            -SwApCaSe-
# dtype: object
print('# 左对齐')
print(s.str.ljust(10, fillchar='-'))
# 0            lower-----
# 1            CAPITALS--
# 2    this is a sentence
# 3            SwApCaSe--
# dtype: object
print('# 右对齐')
print(s.str.rjust(10, fillchar='-'))  # 结果同默认
# 0            -----lower
# 1            --CAPITALS
# 2    this is a sentence
# 3            --SwApCaSe
# dtype: object
# print(s.str.rjust(10))

print()
print('指定宽度，填充内容对齐方式，填充内容')
# 参数side可取值为left、right或both}, 默认值为left
print(s.str.pad(width=10, side='left', fillchar='-'))
# print(s.str.pad(width=10,side='right',fillchar='-'))

print('# 填充对齐')
# 生成字符，不足30位的在前面加0
print(s.str.zfill(30))
# 0    0000000000000000000000000lower
# 1    0000000000000000000000CAPITALS
# 2    000000000000this is a sentence
# 3    0000000000000000000000SwApCaSe
# dtype: object

print()
print('------------------------------------------------------------')
print('\t11.2.4 计数和编码')

'''以下是文本的计数和内容编码方法：'''
print('# 字符串中指定字母的数量')
print(s.str.count('a'))
# 0    0
# 1    0
# 2    1
# 3    1
# dtype: int64
print('# 字符串长度')
print(s.str.len())
# 0     5
# 1     8
# 2    18
# 3     8
# dtype: int64
print('# 编码')
print(s.str.encode('utf-8'))
# 0                 b'lower'
# 1              b'CAPITALS'
# 2    b'this is a sentence'
# 3              b'SwApCaSe'
# dtype: object
print('# 解码')
print(s.str.decode('utf-8'))
# 0   NaN
# 1   NaN
# 2   NaN
# 3   NaN
# dtype: float64
print('# 字符串的Unicode普通格式')
# form{'NFC', 'NFKC', 'NFD', 'NFKD'}
print(s.str.normalize('NFC'))
# 0                 lower
# 1              CAPITALS
# 2    this is a sentence
# 3              SwApCaSe
# dtype: object
print()
print('------------------------------------------------------------')
print('\t11.2.5 格式判定')

print('以下是与文本格式相关的判断：')
print(s.str.isalpha())
# 0     True
# 1     True
# 2    False
# 3     True
# dtype: bool

# s.str.isalpha # 是否为字母
# s.str.isnumeric # 是否为数字0～9
# s.str.isalnum # 是否由字母或数字组成
# s.str.isdigit # 是否为数字
# s.str.isdecimal # 是否为小数
# s.str.isspace # 是否为空格
# s.str.islower # 是否小写
# s.str.isupper # 是否大写
# s.str.istitle # 是否标题格式
'''
11.2.6 小结
文本类型的数据都支持.str.<method>访问器，访问器给文本带来了大量的实用功能，
.str.<method>访问器几乎支持Python所有对字符的操作。
'''

print()
print('------------------------------------------------------------')
print('第11章 Pandas文本处理')
print('\t11.3 文本高级处理')
print('\t11.3.1 文本分隔')
print()
# update20240527
'''
本节介绍Pandas的.str访问器处理文本的一些高级方法，结合使用这些方法，可以完成复杂的文本信息处理和分析。
'''
# 小节注释
'''
对文本的分隔和替换是最常用的文本处理方式。
对文本分隔后会生成一个列表，我们对列表进行切片操作，可以找到我们想要的内容。
分隔后还可以将分隔内容展开，形成单独的行。

下例以下划线对内容进行了分隔，分隔后每个内容都成为一个列表。分隔对空值不起作用。

▶ 以下是一些具体的使用方法举例：
'''
# 原数据
s = pd.Series(['天_地_人', '你_我_他', np.nan, '风_水_火'], dtype="string")
print(s)

print('# 用下划线分隔')
print(s.str.split('_'))
# 0    [天, 地, 人]
# 1    [你, 我, 他]
# 2         <NA>
# 3    [风, 水, 火]
# dtype: object

'''
分隔后可以使用get或者[]来取出相应内容，不过[]是Python列表切片操作，
更加灵活，不仅可以取出单个内容，也可以取出由多个内容组成的片段。
'''
print('# 取出每行第二个')
print(s.str.split('_').str[1])
# 0       地
# 1       我
# 2    <NA>
# 3       水
# dtype: object
print('# get只能传一个值')
print(s.str.split('_').str.get(1))
print('# []可以使用切片操作')
print(s.str.split('_').str[1:3])
print(s.str.split('_').str[:-2])
print('# 如果不指定分隔符，会按空格进行分隔')
print(s.str.split())
print('# 限制分隔的次数，从左开始，剩余的不分隔')
print(s.str.split('_', n=1))
# 0    [天, 地_人]
# 1    [你, 我_他]
# 2        <NA>
# 3    [风, 水_火]

print()
print('------------------------------------------------------------')
print('\t11.3.2 字符分隔展开')

'''
在用.str.split()将数据分隔为列表后，如果想让列表共同索引位上的值在同一列，
形成一个DataFrame，可以传入expand=True，还可以通过n参数指定分隔索引位来控制形成几列，见下例：
'''

print('# 分隔后展开为DataFrame')
print(s.str.split('_', expand=True))
#       0     1     2
# 0     天     地     人
# 1     你     我     他
# 2  <NA>  <NA>  <NA>
# 3     风     水     火

print('# 指定展开列数，n为切片右值')
print(s.str.split('_', expand=True, n=1))
#       0     1
# 0     天   地_人
# 1     你   我_他
# 2  <NA>  <NA>
# 3     风   水_火

'''rsplit和split一样，只不过它是从右边开始分隔。如果没有n参数，rsplit和split的输出是相同的。'''
print('# 从右分隔为两部分后展开为DataFrame')
print(s.str.rsplit('_', expand=True, n=1))
#       0     1
# 0   天_地     人
# 1   你_我     他
# 2  <NA>  <NA>
# 3   风_水     火

'''对于比较复杂的规则，分隔符处可以传入正则表达式：'''

s = pd.Series(["你和我及他"])
# 用正则表达式代表分隔位
print(s.str.split(r"\和|及", expand=True))
#    0  1  2
# 0  你  我  他

print()
print('------------------------------------------------------------')
print('\t11.3.3 文本切片选择')

'''使用.str.slice()将指定的内容切除掉，不过还是推荐使用s.str[]来实现，这样我们只学一套内容就可以了：'''
s = pd.Series(['sun', 'moon', 'star'])

print('# 以下切掉第一个字符')
print(s.str.slice(1))
print(s.str.split())  # 啥都没做
split_s = s.str.split('', n=2)
print(split_s.str[2])  # 结果同上
# 0     un
# 1    oon
# 2    tar
# dtype: object

print(s.str.slice(start=1))  # 结果同上
'''以下是一些其他用法的示例：'''

print(s.str.split())  # 不做任何事
print(s.str.slice(start=-1))  # 切除最后一个以前的，留下最后一个
print(s.str[-1])  # 结果同上
print(s.str.slice(stop=2))
# 切除步长为2的内容
print(s.str.slice(step=2))  # s.str[::2]
# 切除从头开始，第4位以后并且步长为3的内容
# 同s.str[0:5:3]

s = pd.Series(['sun', 'moon', 'star', 'dhfjdjhae'])
print(s)
print(s.str.slice(start=0, stop=5, step=3))
# 0     s
# 1    mn
# 2    sr
# 3    dj
# dtype: object

print()
print('------------------------------------------------------------')
print('\t11.3.4 文本划分')

'''.str.partition可以将文本按分隔符号划分为三个部分，形成一个新的DataFrame或者相关数据类型。'''
# 构造数据
s = pd.Series(['How are you', 'What are you doing'])
# print(s)
print('# 划分为三列DataFrame')
print(s.str.partition())
#       0  1              2
# 0   How           are you
# 1  What     are you doing
# 其他的操作方法如下：
print('# 从右开始划分')
print(s.str.rpartition())
#               0  1      2
# 0       How are       you
# 1  What are you     doing
print('# 指定字符')
print(s.str.partition("are"))
#        0    1           2
# 0   How   are         you
# 1  What   are   you doing
print()
# print(type(s.str.partition('are')))
print('# 划分为一个元组列')
print(s.str.partition("you", expand=False))
# 0           (How are , you, )
# 1    (What are , you,  doing)
# dtype: object
print('# 对索引进行划分')
idx = pd.Index(['A 123', 'B 345'])
print(idx.str.partition())
# MultiIndex([('A', ' ', '123'),
#             ('B', ' ', '345')],
#            )

print()
print('------------------------------------------------------------')
print('\t11.3.5 文本替换')
'''
在进行数据处理时我们可以使用替换功能剔除我们不想要的内容，换成想要的内容。
这在数据处理中经常使用，因为经过人工整理的数据往往不理想，需要进行替换操作。
我们使用.str.replace()方法来完成这一操作。

例如，对于以下一些金额数据，我们想去除货币符号，为后续转换为数字类型做准备，
因为非数字元素的字符无法转换为数字类型：
'''

print('# 带有货币符的数据')
s = pd.Series(['10', '-¥20', '¥3,000'], dtype="string")
# print(s)
print('# 将人民币符号替换为空')
print(s.str.replace("¥", ''))
print('# 如果需要数字类型，还需要将逗号剔除')
print(s.str.replace("¥", '').str.replace(',', '').str.replace('-', ''))
print(s.str.replace(r'¥|,', ''))  # 没有变化
# 0        10
# 1      -¥20
# 2    ¥3,000
# dtype: string
print(s.str.replace(r'¥|,', '', regex=True))  # 明确指定regex=True，表示你要使用正则表达式进行替换。执行成功！
# 0      10
# 1     -20
# 2    3000
# dtype: string

'''
注意，.str.replace()方法的两个基本参数中，
第一个是旧内容（希望被替换的已有内容），第二个是新内容（替换成的新内容）。

替换字符默认是支持正则表达式的，如果被替换内容是一个正则表达式，
可以使用regex=False关闭对正则表达式的支持。
在被替换字符位还可以传入一个定义好的函数或者直接使用lambda。
另外，替换工作也可以使用df.replace()和s.replace()完成。
'''

print()
print('------------------------------------------------------------')
print('\t11.3.6 指定替换')
'''str.slice_replace()可实现保留选定内容，替换剩余内容：'''
# 构造数据
s = pd.Series(['ax', 'bxy', 'cxyz'])
# 保留第一个字符，其他的替换或者追加T
print(s.str.slice_replace(1, repl='T'))
# 0    aT
# 1    bT
# 2    cT
# dtype: object

print('# 指定位置前删除并用T替换')
print(s.str.slice_replace(stop=2, repl='T'))
# 0      T
# 1     Ty
# 2    Tyz
# dtype: object

print('# 指定区间的内容被替换')
print(s.str.slice_replace(start=1, stop=3, repl='T'))
# 0     aT
# 1     bT
# 2    cTz
# dtype: object
print()
print('------------------------------------------------------------')
print('\t11.3.7 重复替换')
'''可以使用.str.repeat()方法让原有文本内容重复：'''
print('# 将整体重复两次')
print(pd.Series(['a', 'b', 'c']).repeat(repeats=2))
# 0    a
# 0    a
# 1    b
# 1    b
# 2    c
# 2    c
# dtype: object
print('# 将每一行的内容重复两次')
print(pd.Series(['a', 'b', 'c']).str.repeat(repeats=2))
# 0    aa
# 1    bb
# 2    cc
# dtype: object
print('# 指定每行重复几次')
print(pd.Series(['a', 'b', 'c']).str.repeat(repeats=[1, 2, 3]))
# 0      a
# 1     bb
# 2    ccc
# dtype: object
print()
print('------------------------------------------------------------')
print('\t11.3.8 文本连接')
'''方法s.str.cat()具有文本连接的功能，可以将序列连接成一个文本或者将两个文本序列连接在一起。'''
# 文本序列
s = pd.Series(['x', 'y', 'z'], dtype='string')
# 默认无符号连接
print(s.str.cat())
# xyz
print('# 用逗号连接')
print(s.str.cat(sep=','))
# x,y,z

'''如果序列中有空值，会默认忽略空值，也可以指定空值的占位符号：'''
print('# 包含空值的文本序列')
t = pd.Series(['h', 'i', np.nan, 'k'], dtype='string')
# 用逗号连接
print(t.str.cat(sep=','))
# h,i,k
# 用连字符
print(t.str.cat(sep=',', na_rep='-'))
# h,i,-,k
print(t.str.cat(sep=',', na_rep='j'))
# h,i,j,k
'''当然也可以使用pd.concat()来连接两个序列：'''
print()
print(s)
print('# 连接')
print(pd.concat([s, t], axis=1))
#       0     1
# 0     x     h
# 1     y     i
# 2     z  <NA>
# 3  <NA>     k
# print(type(pd.concat([s,t],axis=1))) # <class 'pandas.core.frame.DataFrame'>
# print(type(pd.concat([s,t]))) # <class 'pandas.core.series.Series'>
print('# 两次连接')
print(s.str.cat(pd.concat([s, t], axis=1), na_rep='-'))
# 0    xxh
# 1    yyi
# 2    zz-
# dtype: string
'''如果连接的两个对象长度不同，超出较短对象索引范围的部分不会被连接和输出。'''
print(s.str.cat(pd.concat([t, s], axis=1), na_rep='-'))
# 0    xhx
# 1    yiy
# 2    z-z
# dtype: string
'''连接的对齐方式：'''
h = pd.Series(['b', 'd', 'a'],
              index=[1, 0, 2],
              dtype='string')
# print(h)
print('# 以左边的索引为准')
print(s.str.cat(h))
# 0    xd
# 1    yb
# 2    za
# dtype: string
print(s.str.cat(t, join='left'))
# 0      xh
# 1      yi
# 2    <NA>
# dtype: string
print('# 以右边的索引为准')
print(s.str.cat(h, join='right'))
# 1    yb
# 0    xd
# 2    za
# dtype: string
print('# 其他')
print(s.str.cat(h, join='outer', na_rep='-'))
# 0    xd
# 1    yb
# 2    za
# dtype: string
print(s.str.cat(h, join='inner', na_rep='-'))  # 结果同上
# print(s)
# print(t)

print(s.str.cat(t, join='outer', na_rep='-'))
# 0    xh
# 1    yi
# 2    z-
# 3    -k
# dtype: string
print(s.str.cat(t, join='inner', na_rep='-'))
# 0    xh
# 1    yi
# 2    z-
# dtype: string

print()
print('------------------------------------------------------------')
print('\t11.3.9 文本查询')

'''
Pandas在文本的查询匹配方面也很强大，可以使用正则表达式来进
行复杂的查询匹配，可以根据需要指定获得匹配后返回的数据。

.str.findall()可以查询文本中包括的内容：
'''
# 字符序列
s = pd.Series(['One', 'Two', 'Three'])
# 查询字符
print(s.str.findall('T'))
# 0     []
# 1    [T]
# 2    [T]
# dtype: object

'''以下是一些操作示例：'''
print('# 区分大小写，不会查出内容')
print(s.str.findall('ONE'))
# 0    []
# 1    []
# 2    []
# dtype: object

print('# 忽略大小写')
import re

print(s.str.findall('ONE', flags=re.IGNORECASE))
# 0    [One]
# 1       []
# 2       []
# dtype: object
print('# 包含o')
print(s.str.findall('o'))
# 0     []
# 1    [o]
# 2     []
# dtype: object
print('# 以o结尾')
print(s.str.findall('o$'))
# 0     []
# 1    [o]
# 2     []
# dtype: object
print('# 包含多个的会形成一个列表')
print(s.str.findall('e'))
# 0       [e]
# 1        []
# 2    [e, e]
# dtype: object

'''使用.str.find()返回匹配结果的位置（从0开始），–1为不匹配：'''
print(s.str.find('One'))
# 0    0
# 1   -1
# 2   -1
# dtype: int64
print(s.str.find('e'))
# 0    2
# 1   -1
# 2    3
# dtype: int64
# 此外，还有.str.rfind()，它是从右开始匹配。

print()
print('------------------------------------------------------------')
print('\t11.3.10 文本包含')

'''
.str.contains()会判断字符是否有包含关系，返回布尔序列，经常用在数据筛选中。
它默认是支持正则表达式的，如果不需要，可以关掉。
na参数可以指定空值的处理方式。
'''
# 原数据
s = pd.Series(['One', 'Two', 'Three', np.NaN])
print('# 是否包含检测')
print(s.str.contains('o', regex=False))  # 大小写敏感
# 0    False
# 1     True
# 2    False
# 3      NaN
# dtype: object
# 大小写不敏感！
print(s.str.contains('o', flags=re.IGNORECASE))  # 识别大写 必须打开正则表达式参数 或默认 不加参数,regex=False
# 0     True
# 1     True
# 2    False
# 3      NaN
# dtype: object

'''用在数据查询中：'''
print()
print('# 名字包含A字母')
df = pd.read_excel(team_file)
# print(df)
print(df.loc[df.name.str.contains('A')])
#    name team  Q1  Q2  Q3  Q4
# 1  Arry    C  36  37  37  57
# 2   Ack    A  57  60  18  84
print('# 包含字母A或者R')
print(df.loc[df.name.str.contains('A|R')])
#    name team   Q1  Q2  Q3   Q4
# 1  Arry    C   36  37  37   57
# 2   Ack    A   57  60  18   84
# 5  Rick    B  100  99  97  100
print('# 忽略大小写')
print(df.loc[df.name.str.contains('A|e', flags=re.IGNORECASE)])
#     name team  Q1  Q2  Q3  Q4
# 0  Liver    E  89  21  24  64
# 1   Arry    C  36  37  37  57
# 2    Ack    A  57  60  18  84
# 3  Eorge    C  93  96  71  78
# 4    Oah    D  65  49  61  86
print(df.loc[df.name.str.contains('A|e')])  # 不忽略大小写
#     name team  Q1  Q2  Q3  Q4
# 0  Liver    E  89  21  24  64
# 1   Arry    C  36  37  37  57
# 2    Ack    A  57  60  18  84
# 3  Eorge    C  93  96  71  78
print('# 包含数字')
print(df.loc[df.name.str.contains('\d')])
# Empty DataFrame
# Columns: [name, team, Q1, Q2, Q3, Q4]
# Index: []

print()
'''此外.str.startswith()和.str.endswith()还可以指定是开头还是结尾包含：'''
# 原数据
s = pd.Series(['One', 'Two', 'Three', np.NaN])
print(s.str.startswith('O'))
# 0     True
# 1    False
# 2    False
# 3      NaN
# dtype: object
print('# 对空值的处理')
print(s.str.startswith('O', na=False))
# 0     True
# 1    False
# 2    False
# 3    False
# dtype: bool
print(s.str.endswith('e'))
# 0     True
# 1    False
# 2     True
# 3      NaN
# dtype: object
print(s.str.endswith('e', na=False))
# 0     True
# 1    False
# 2     True
# 3    False
# dtype: bool

print()
print('.str.match()')
'''用.str.match()确定每个字符串是否与正则表达式匹配：'''
print(pd.Series(['1', '2', '3a', '3b', '03c'],
                dtype='string').str.match(r'[0-9][a-z]'))
# 0    False
# 1    False
# 2     True
# 3     True
# 4    False
# dtype: boolean

print()
print('------------------------------------------------------------')
print('\t11.3.11 文本提取')

'''
.str.extract()可以利用正则表达式将文本中的数据提取出来，形成单独的列。
下列代码中正则表达式将文本分为两部分，第一部分匹配a、b两个字母，
第二部分匹配数字，最终得到这两列。c3由于无法匹配，最终得到两列空值。
'''

print(pd.Series(['a1', 'b2', 'c3'], dtype='string')
      .str
      .extract(r'([ab])(\d)', expand=True))
#       0     1
# 0     a     1
# 1     b     2
# 2  <NA>  <NA>

print(pd.Series(['a1', 'b2', 'c3'], dtype='string')
      .str
      .extract(r'([a-z])(\d)', expand=True))
#    0  1
# 0  a  1
# 1  b  2
# 2  c  3

'''
expand参数如果为真，则返回一个DataFrame，不管是一列还是多列；
如果为假，则仅当只有一列时才会返回一个Series/Index。
'''
print()
s = pd.Series(['a1', 'b2', 'c3'], dtype='string')
print(s.str.extract(r'([ab])?(\d)'))
#       0  1
# 0     a  1
# 1     b  2
# 2  <NA>  3
print(s.str.extract(r'([ab])?([0-9])', expand=True))  # 结果同上！

print('# 取正则组的命名为列名')
# ?P<name>：定义了捕获组的名称。其中P表示命名捕获组，name是捕获组的名称。
print(s.str.extract(r'(?P<letter>[ab])(?P<digit>\d)'))
#   letter digit
# 0      a     1
# 1      b     2
# 2   <NA>  <NA>
print(s.str.extract(r'(?P<letter>[ab])?(?P<digit>\d)'))  # 参数 ? 匹配0次或1次
#   letter digit
# 0      a     1
# 1      b     2
# 2   <NA>     3
print('# 测试数字和字母乱序 拆分')
# 测试失败 搞了半天openai
# s = pd.Series(['a1','bn3','c34534','23feg45'],dtype='string')
# print(s.str.extract(r'(?P<letters>[a-zA-Z]+)?(?P<digits>\d+)'))
# 0       a      1
# 1      bn      3
# 2       c  34534
# 3    <NA>     23

print()
'''匹配全部，会将一个文本中所有符合规则的内容匹配出来，最终形成一个多层索引数据：'''
s = pd.Series(["a1a2", "b1b7", "c1"],
              index=["A", "B", "C"],
              dtype="string")

two_groups = '(?P<letter>[a-z])(?P<digit>[0-9])'
print(s.str.extract(two_groups, expand=True))  # 单次匹配
#   letter digit
# A      a     1
# B      b     1
# C      c     1
print(s.str.extractall(two_groups))
#         letter digit
#   match
# A 0          a     1
#   1          a     2
# B 0          b     1
#   1          b     7
# C 0          c     1
print(type(s.str.extractall(two_groups)))
# <class 'pandas.core.frame.DataFrame'>
# df = s.str.extractall(two_groups)
# df.reset_index(inplace=True)
# print(df.unstack())


print()
print('------------------------------------------------------------')
print('\t11.3.12 提取虚拟变量')

'''可以从字符串列中提取虚拟变量，例如用“/”分隔：'''
s = pd.Series(['a/b', 'b/c', np.nan, 'c'], dtype="string")

print('# 提取虚拟')
print(s.str.get_dummies(sep='/'))
#    a  b  c
# 0  1  1  0
# 1  0  1  1
# 2  0  0  0
# 3  0  0  1

'''也可以对索引进行这种操作：'''
idx = pd.Index(['a/b', 'b/c', np.nan, 'c'])
print(idx.str.get_dummies(sep='/'))
# MultiIndex([(1, 1, 0),
#             (0, 1, 1),
#             (0, 0, 0),
#             (0, 0, 1)],
#            names=['a', 'b', 'c'])

'''
11.3.13 小结
先将数据转换为字符类型，然后就可以随心所欲地使用str访问器了。
这些文本高级功能可以帮助我们完成对于复杂文本的处理，同时完成数据的分析。

11.4 本章小结
文本数据虽然不能参与算术运算，但文本数据具有数据维度高、数据量大且语义复杂等特点，在数据分析中需要得到重视。
本章介绍的Pandas操作的文本数据类型的方法及str访问器，大大提高了文本数据的处理效率。
'''
