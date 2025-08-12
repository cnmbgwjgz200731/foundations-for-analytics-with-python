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
warnings.filterwarnings('ignore',category=UserWarning,module='openpyxl')


print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.1 索引操作')
print('\t4.1.2 建立索引')
print()

'''
行索引是数据的索引，列索引指向的是一个Series；
DataFrame的索引也是系列形成的Series的索引；
建立索引让数据更加直观明确，每行数据是针对哪个主体的；
建立索引方便数据处理；
索引允许重复，但业务上一般不会让它重复。

建立索引可以在数据读取加载中指定索引：
'''
print('# 无索引')
df = pd.read_excel(team_file)
print(df)
print()

print('# 读取时 设置单索引 指定字段名称')
df = pd.read_excel(team_file,index_col='name')
print(df)
print()
print(df[df.index == 'Rick'])
print()

print('# 读取后 设置单索引 指定字段名称')
# 运行后 显示没有设置索引 擦
df = pd.read_excel(team_file)
df.set_index(['name'])

print(df)
print()
print(df[df.index == 'Rick']) # 单索引 取值
print()

print('# 读取时 设置多个索引 指定阿拉伯数字')
df = pd.read_excel(team_file,index_col=[0,1])
print(df)
print()
print(df.index)
print(df[df.index == ('Rick','B')]) # 多索引 取值
print()



'''
df.set_index(['name']) 这行代码本身没有问题，它是用来将 DataFrame 的索引设置为 'name' 列的内容。
然而，set_index 函数默认情况下不会修改原始的 DataFrame，而是返回一个新的 DataFrame。
因此，如果你没有将返回的新 DataFrame 赋值给一个变量，原始的 DataFrame 是不会改变的。

要解决这个问题，你可以将返回的 DataFrame 赋值给原来的变量 df 或者一个新变量，
或者使用 inplace=True 参数来直接在原始 DataFrame 上进行修改。

'''
print('# 使用赋值法')
df = pd.read_excel(team_file)
df = df.set_index(['name']) # 返回新的 DataFrame 并赋值给 df
print(df)
print()

print('# 使用inplace参数')
df = pd.read_excel(team_file)
df.set_index(['name','team'],inplace=True) # 直接在原始 DataFrame 上设置索引
print(df)
print()

# df.set_index(['name', 'team']) # 设置两层索引
# df.set_index([df.name.str[0],'name']) # 将姓名的第一个字母和姓名设置为索引

print('# 将姓名的第一个字母和姓名设置为索引')

df = pd.read_excel(team_file)
df = df.set_index([df.name.str[0],'name']) # 将姓名的第一个字母和姓名设置为索引
print(df)
print(df.index)
print()
'''
out:
           team   Q1  Q2  Q3   Q4
name name                        
L    Liver    E   89  21  24   64
A    Arry     C   36  37  37   57
     Ack      A   57  60  18   84
E    Eorge    C   93  96  71   78
O    Oah      D   65  49  61   86
R    Rick     B  100  99  97  100
MultiIndex([('L', 'Liver'),
            ('A',  'Arry'),
            ('A',   'Ack'),
            ('E', 'Eorge'),
            ('O',   'Oah'),
            ('R',  'Rick')],
           names=['name', 'name'])
'''

df = pd.read_excel(team_file)
df = df.set_index([df.name.str[0]]) # 将姓名的第一个字母设置为索引
print(df)
print()

print(df.index) # Index(['L', 'A', 'A', 'E', 'O', 'R'], dtype='object', name='name')
print(df[df.index == 'A'])
print()
'''
out:
      name team  Q1  Q2  Q3  Q4
name                           
A     Arry    C  36  37  37  57
A      Ack    A  57  60  18  84
'''

print('# 可以将一个Series指定为索引：')

df = pd.read_excel(team_file)
print(df)
print()

s = pd.Series([i for i in range(2,8,1)])
df.set_index(s,inplace=True)
print(s)
print(df)
print()
'''
# 可以将一个Series指定为索引：
0    2
1    3
2    4
3    5
4    6
5    7
dtype: int64
    name team   Q1  Q2  Q3   Q4
2  Liver    E   89  21  24   64
3   Arry    C   36  37  37   57
4    Ack    A   57  60  18   84
5  Eorge    C   93  96  71   78
6    Oah    D   65  49  61   86
7   Rick    B  100  99  97  100
'''

print('# 同时指定Series索引和现有字段')
s = pd.Series([i for i in range(2,8,1)],name='row_no') # Series命名！
df.set_index([s,'name'],inplace=True)
print(s)
print(df)
print()
'''
out:
             team   Q1  Q2  Q3   Q4
row_no name                        
2      Liver    E   89  21  24   64
3      Arry     C   36  37  37   57
4      Ack      A   57  60  18   84
5      Eorge    C   93  96  71   78
6      Oah      D   65  49  61   86
7      Rick     B  100  99  97  100
'''

print('# 计算索引')
s = pd.Series([i for i in range(2,8,1)],name='row_no') # Series命名！
df.set_index([s,s**2],inplace=True)
print(s)
print(df)
print()
'''
out:
                name team   Q1  Q2  Q3   Q4
row_no row_no                              
2      4       Liver    E   89  21  24   64
3      9        Arry    C   36  37  37   57
4      16        Ack    A   57  60  18   84
5      25      Eorge    C   93  96  71   78
6      36        Oah    D   65  49  61   86
7      49       Rick    B  100  99  97  100
'''

print('# drop参数,默认不保留')

df = pd.read_excel(team_file)
print()
df = df.set_index('name',drop=False) # 保留原列
print(df)
print()

df = pd.read_excel(team_file)
df = df.set_index('name',drop=True) # 不保留原列
print(df)
print()
'''
out:
# drop参数

        name team   Q1  Q2  Q3   Q4
name                               
Liver  Liver    E   89  21  24   64
Arry    Arry    C   36  37  37   57
Ack      Ack    A   57  60  18   84
Eorge  Eorge    C   93  96  71   78
Oah      Oah    D   65  49  61   86
Rick    Rick    B  100  99  97  100

      team   Q1  Q2  Q3   Q4
name                        
Liver    E   89  21  24   64
Arry     C   36  37  37   57
Ack      A   57  60  18   84
Eorge    C   93  96  71   78
Oah      D   65  49  61   86
Rick     B  100  99  97  100
'''

print('# append参数,默认不保留')

df = pd.read_excel(team_file)
print()
df = df.set_index('name',append=True) # 保留原来的索引
print(df)
print()

df = pd.read_excel(team_file)
print()
df = df.set_index('name',append=False) # 不保留原来的索引
print(df)
print()
'''
out:
        team   Q1  Q2  Q3   Q4
  name                        
0 Liver    E   89  21  24   64
1 Arry     C   36  37  37   57
2 Ack      A   57  60  18   84
3 Eorge    C   93  96  71   78
4 Oah      D   65  49  61   86
5 Rick     B  100  99  97  100


      team   Q1  Q2  Q3   Q4
name                        
Liver    E   89  21  24   64
Arry     C   36  37  37   57
Ack      A   57  60  18   84
Eorge    C   93  96  71   78
Oah      D   65  49  61   86
Rick     B  100  99  97  100
'''


print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.1 索引操作')
print('\t4.1.3 重置索引')
print()

'''
有时我们想取消已有的索引，可以使用df.reset_index()，
它的操作与set_index相反。以下是一些常用的操作：
'''
df = pd.read_excel(team_file,index_col='name')
print(df)
print()

print('# 清除索引：df.reset_index()')
df = df.reset_index()
# df.reset_index(inplace=True) # 测试效果同上！
print(df)
print()

print('# 清除索引：相当于什么也没做')
df = pd.read_excel(team_file)
df = df.set_index('name').reset_index() # 相当于什么也没做
# df.set_index('name').reset_index(inplace=True) # 效果同上
print(df)
print()

print('# 删除原索引：name列没了')
df = pd.read_excel(team_file)
# df.set_index('name').reset_index(drop=True,inplace=True) # name列还在
df = df.set_index('name').reset_index(drop=True) # name列没了
# df = df.set_index('name',inplace=True).reset_index(drop=True) # 报错！
print(df)
print()

print('# name 一级索引取消')
df = pd.read_excel(team_file)
df = df.set_index(['name','team']).reset_index(level=0) # name 索引取消
# df = df.set_index(['name','team']).reset_index(level=1) # team索引取消
print(df)
print()

print('# 使用 层级索引名')
df = pd.read_excel(team_file,index_col='name')
df = df.reset_index(level='name') # name索引取消
# df.reset_index(level='name',inplace=True) # 效果同上
print(df)
print()

print('# 列索引')

# 创建一个具有多重列索引的 DataFrame
df = pd.DataFrame({
    ('A', 'team'): ['Team1', 'Team2'],
    ('A', 'points'): [5, 10],
    ('B', 'assists'): [7, 8]
}).set_index(('A', 'team'))

print("原始 DataFrame：")
print(df)
print()

# 使用 col_level=1 重置索引
# df_reset_col_level_1 = df.reset_index(level=0, col_level=1)
# print("使用 col_level=1 重置索引后的 DataFrame：")
# print(df_reset_col_level_1)
# print()
# 该段代码运行后报错！

# 使用 col_level=0 重置索引
df_reset_col_level_0 = df.reset_index(level=0, col_level=0)
print("使用 col_level=0 重置索引后的 DataFrame：")
print(df_reset_col_level_0)
'''
out:
原始 DataFrame：
               A       B
          points assists
(A, team)               
Team1          5       7
Team2         10       8

使用 col_level=0 重置索引后的 DataFrame：
       A              B
    team points assists
0  Team1      5       7
1  Team2     10       8

'''


# 创建一个具有多级列索引的DataFrame
df = pd.DataFrame({
    ('A', 'a'): {('X', 'x'): 1, ('Y', 'y'): 2},
    ('B', 'b'): {('X', 'x'): 3, ('Y', 'y'): 4},
    ('C', 'c'): {('X', 'x'): 5, ('Y', 'y'): 6},
    ('C', 'name'): {('X', 'x'): 'one', ('Y', 'y'): 'two'}
})

print(df)
print()

# 设置多级行索引为列C中的'name'数据
df = df.set_index(('C', 'name'))
print(df)
print()
# 重置索引，并将原索引所在的列放置在多级列的第一级别（col_level=0）
df_reset = df.reset_index(col_level=0)

print(df_reset)
'''
out:
     A  B  C     
     a  b  c name
X x  1  3  5  one
Y y  2  4  6  two

           A  B  C
           a  b  c
(C, name)         
one        1  3  5
two        2  4  6

     C  A  B  C
  name  a  b  c
0  one  1  3  5
1  two  2  4  6

'''



print('# 不存在层级名称的填入指定名称')
# update20240221
'''不太明白使用场景'''
import pandas as pd
from io import StringIO
# 创建一个具有多重列索引的 DataFrame
data = StringIO("""
team,points,assists
Team1,5,7
Team2,10,8
""")
df = pd.read_csv(data, header=0)
df = df.set_index('team')
print(df)
print()
df.columns = pd.MultiIndex.from_tuples([('A', 'points'), ('B', 'assists')])

print("原始 DataFrame：")
print(df)
print()

# 重置索引，将 'team' 从行索引转为列索引，并设置 col_level=1 和 col_fill='species'
df_reset = df.reset_index(col_level=1, col_fill='species')

print("使用 col_fill='species' 重置索引后的 DataFrame：")
print(df_reset)
print()

print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.1 索引操作')
print('\t4.1.4 索引类型')
print()
# update20240221
'''
为了适应各种业务数据的处理，索引又针对各种类型数据定义了不同的索引类型。

1、数字索引（NumericIndex）
    RangeIndex：单调整数范围的不可变索引。
    Int64Index：64位整型索引。
    UInt64Index：无符号整数索引。
    Float64Index：64位浮点型索引。

2、类别索引（CategoricalIndex）：
    类别只能包含有限数量的（通常是固定的）可能值（类别）。
    可以理解成枚举，比如性别只有男女，但在数据中每行都有，如果按文本处理会效率不高。
    类别的底层是pandas.Categorical。类别在第12章会专门讲解，
    只有在体量非常大的数据面前才能显示其优势。

3、间隔索引（IntervalIndex）：
    代表每个数据的数值或者时间区间，一般应用于分箱数据。

4、多层索引（MultiIndex）：
    多个层次且有归属关系的索引。

5、时间索引（DatetimeIndex）：
    时序数据的时间。

6、时间差索引（TimedeltaIndex）：
    代表时间长度的数据。

7、周期索引（PeriodIndex）：
    一定频度的时间。
'''
df = pd.read_excel(team_file, index_col='name')
print(df)
print()

print('数字索引')
'''
1、数字索引（NumericIndex）共有以下几种。
    RangeIndex：单调整数范围的不可变索引。
    Int64Index：64位整型索引。
    UInt64Index：无符号整数索引。
    Float64Index：64位浮点型索引。
'''

pd.RangeIndex(1, 100, 2)
# RangeIndex(start=1, stop=100, step=2)
pd.Int64Index([1, 2, 3, -4], name='num')
# Int64Index([1, 2, 3, -4], dtype='int64', name='num')
pd.UInt64Index([1, 2, 3, 4])
# UInt64Index([1, 2, 3, 4], dtype='uint64')
pd.Float64Index([1.2, 2.3, 3, 4])
# Float64Index([1.2, 2.3, 3.0, 4.0], dtype='float64')

print()
print('类别索引')
'''
2、类别索引（CategoricalIndex）：
    类别只能包含有限数量的（通常是固定的）可能值（类别）。
    可以理解成枚举，比如性别只有男女，但在数据中每行都有，如果按文本处理会效率不高。
    类别的底层是pandas.Categorical。类别在第12章会专门讲解，
    只有在体量非常大的数据面前才能显示其优势。
'''
pd.CategoricalIndex(['a', 'b', 'a', 'b'])
# CategoricalIndex(['a', 'b', 'a', 'b'], categories=['a', 'b'], ordered=False, dtype='category')

print()
print('间隔索引')
'''
3、间隔索引（IntervalIndex）：
    代表每个数据的数值或者时间区间，一般应用于分箱数据。
'''

pd.interval_range(start=0, end=5)
'''
IntervalIndex([(0, 1], (1, 2], (2, 3], (3, 4], (4, 5]], dtype='interval[int64, right]')
'''

print()
print('多层索引')
'''
4、多层索引（IntervalIndex）：
    多个层次且有归属关系的索引。
'''
arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))
'''
MultiIndex([(1,  'red'),
            (1, 'blue'),
            (2,  'red'),
            (2, 'blue')],
           names=['number', 'color'])
'''

print()
print('时间索引')
'''
5、时间索引（DatetimeIndex）：
    时序数据的时间。
'''

print()
print('# 从一个日期连续到另一个日期')
pd.date_range(start='1/1/2018', end='1/08/2018')
'''
out:
DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
               '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'],
              dtype='datetime64[ns]', freq='D')
'''

print()
print('# 指定开始时间和周期')
pd.date_range(start='1/1/2018', periods=8)
'''
out:
DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
               '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'],
              dtype='datetime64[ns]', freq='D')
'''

print()
print('# 以月为周期')
print(pd.period_range(start='2017-01-01', end='2018-01-01', freq='M'))
'''
out:
PeriodIndex(['2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06',
             '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12',
             '2018-01'],
            dtype='period[M]')
'''

print()
print('# 周期嵌套')
print(pd.period_range(start=pd.Period('2017Q1', freq='Q'), end=pd.Period('2017Q2', freq='Q'), freq='M'))
'''
out:
PeriodIndex(['2017-03', '2017-04', '2017-05', '2017-06'], dtype='period[M]')
'''

print()
print('时间差索引')
'''
6、时间差索引（TimedeltaIndex）：
    代表时间长度的数据。
'''

print(pd.TimedeltaIndex(data=['06:05:01.000030', '+23:59:59.999999',
                              '22 day 2 min 30s 10ns', '+23:29:59.999999', '+12:19:59.999999']))
'''
out:
TimedeltaIndex([    '0 days 06:05:01.000030',     '0 days 23:59:59.999999',
                '22 days 00:02:30.000000010',     '0 days 23:29:59.999999',
                    '0 days 12:19:59.999999'],
               dtype='timedelta64[ns]', freq=None)
'''

print()
print('# 使用datetime')
print(pd.TimedeltaIndex(['1 days', '1 days, 00:00:05', np.timedelta64(2, 'D'), datetime.timedelta(days=2, seconds=2)]))
'''
out:
TimedeltaIndex(['1 days 00:00:00', '1 days 00:00:05', '2 days 00:00:00',
                '2 days 00:00:02'],
               dtype='timedelta64[ns]', freq=None)
'''

print()
print('周期索引')
'''
7、周期索引（PeriodIndex）：
    一定频度的时间。
'''

t = pd.period_range('2020-5-1 10:00:05', periods=8, freq='S')
pd.PeriodIndex(t, freq='S')
'''
out:
PeriodIndex(['2020-05-01 10:00:05', '2020-05-01 10:00:06',
             '2020-05-01 10:00:07', '2020-05-01 10:00:08',
             '2020-05-01 10:00:09', '2020-05-01 10:00:10',
             '2020-05-01 10:00:11', '2020-05-01 10:00:12'],
            dtype='period[S]')
'''


print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.1 索引操作')
print('\t4.1.5 索引对象')
print()
# update20240221
'''
行和列的索引在Pandas里其实是一个Index对象，以下是创建一个Index对象的方法：
'''
df = pd.read_excel(team_file, index_col='name')
print(df)
print()

pd.Index([1, 2, 3])
# Int64Index([1, 2, 3], dtype='int64')
pd.Index(list('abc'))
# Index(['a', 'b', 'c'], dtype='object')
# 可以用name指定一个索引名称
pd.Index(['e', 'd', 'a', 'b'], name='something')
# Index(['e', 'd', 'a', 'b'], dtype='object', name='something')


'''
索引对象可以传入构建数据和读取数据的操作中。
可以查看索引对象，列和行方向的索引对象如下：

'''
df.index
# RangeIndex(start=0, stop=4, step=1)
df.columns
# Index(['month', 'year', 'sale'], dtype='object')
'''
out:
Index(['Liver', 'Arry', 'Ack', 'Eorge', 'Oah', 'Rick'], dtype='object', name='name')
Index(['team', 'Q1', 'Q2', 'Q3', 'Q4'], dtype='object')
'''




print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.1 索引操作')
print('\t4.1.6 索引的属性')
print()
# update20240221
'''
可以通过以下一系列操作查询索引的相关属性，以下方法也适用于
df.columns，因为它们都是index对象。
'''
df = pd.read_excel(team_file, index_col='name')
print(df)
print()


print(df.index) # out:Index(['Liver', 'Arry', 'Ack', 'Eorge', 'Oah', 'Rick'], dtype='object', name='name')
print(df.columns) # out: Index(['team', 'Q1', 'Q2', 'Q3', 'Q4'], dtype='object')

print()
print('1、df.index.name')
print(df.index.name) # 名称 out:name
print(df.columns.name) # out:None

print()
print('2、df.index.array')
print(df.index.array)  # array数组
'''
out:
<PandasArray>
['Liver', 'Arry', 'Ack', 'Eorge', 'Oah', 'Rick']
Length: 6, dtype: object
'''

print(df.columns.array) # array数组
'''
out:
<PandasArray>
['team', 'Q1', 'Q2', 'Q3', 'Q4']
Length: 5, dtype: object
'''

print()
print('3、df.index.dtype')
print(df.index.dtype) # 数据类型  out：object
print(df.columns.dtype) # 数据类型  out：object

print()
print('4、df.index.shape') # 形状
print(df.index.shape) # out：(6,)
print(df.columns.shape) # out：(5,)

print()
print('5、df.index.size') # 元素数量
print(df.index.size) # out：6
print(df.columns.size) # out：5

print()
print('6、df.index.values') # array数组
print(df.index.values) # out：['Liver' 'Arry' 'Ack' 'Eorge' 'Oah' 'Rick']
print(df.columns.values) # out：['team' 'Q1' 'Q2' 'Q3' 'Q4']

# 其他，不常用

print()
print('7、df.index.empty') # 是否为空
print(df.index.empty) # out：False
print(df.columns.empty) # out：False

print()
print('8、df.index.is_unique') # 是否不重复
print(df.index.is_unique) # out：True
print(df.columns.is_unique) # out：True

print()
print('9、df.index.names') # 名称列表
print(df.index.names) # out：['name']
print(df.columns.names) # out：[None]

print()
print('10、df.index._is_all_dates') # 是否全是日期时间
print(df.index._is_all_dates) # out：False
print(df.columns._is_all_dates) # out：False

print()
print('11、df.index.has_duplicates') # 是否有重复值
print(df.index.has_duplicates) # out：False
print(df.columns.has_duplicates) # out：False



print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.1 索引操作')
print('\t4.1.7 索引的操作')
print()
# update20240222
'''
以下是索引的常用操作，这些操作会在我们今后处理数据中发挥作用。
以下方法也适用于df.columns，因为都是index对象。
'''
df = pd.read_excel(team_file, index_col='Q1') # Q1 name
print(df)
print()

# 常用方法
print()
print('# 转换类型')
print(df.index)
print(df.index.astype('int')) # 转换类型
'''
out:
Index([89, 36, 57, 93, 65, 100], dtype='int64', name='Q1')
Index([89, 36, 57, 93, 65, 100], dtype='int32', name='Q1')

如果是文本类型转换为数值 会报错！
'''

print()
print('# 是否存在')
print(df.index.isin(['36']))
print(df.index.isin([36]))
'''
out:
[False False False False False False]
[False  True False False False False]
'''

print()
print('# 修改索引名称')
print(df.index.rename('number')) # 修改索引名称
print(df.index)
'''
举例示意：
out:

# 创建一个简单的 DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# 打印原始 DataFrame 的索引名称（默认为 None）
print("原始 DataFrame 的索引名称：", df.index.name)

# 使用 index.rename 方法给索引设置新的名称 'number'
df_renamed_index = df.index.rename('number')

# 打印更改名称后的索引
print("更改名称后的索引：", df_renamed_index)

# 为了反映更改，需要将重命名后的索引赋值回 DataFrame
df.index = df_renamed_index

# 打印更新后的 DataFrame
print("更新后的 DataFrame：")
print(df)
'''

#
# df.index.nunique() # 不重复值的数量
# df.index.sort_values(ascending=False,) # 排序，倒序
# df.index.map(lambda x:x+'_') # map函数处理
# df.index.str.replace('_', '') # str替换
# df.index.str.split('_') # 分隔
# df.index.to_list() # 转为列表
# df.index.to_frame(index=False, name='a') # 转成DataFrame
# df.index.to_series() # 转为series
# df.index.to_numpy() # 转为numpy
# df.index.unique() # 去重
# df.index.value_counts() # 去重及计数
# df.index.where(df.index=='a') # 筛选
# df.index.rename('grade', inplace=False) # 重命名索引
# df.index.rename(['species', 'year']) # 多层，重命名索引
# df.index.max() # 最大值
# df.index.argmax() # 最大索引值
# df.index.any()
# df.index.all()
# df.index.T # 转置，在多层索引里很有用

# 以下是一些不常用但很重要的操作：
# df.index.append(pd.Index([4,5])) # 追加
# df.index.repeat(2) # 重复几次
# df.index.inferred_type # 推测数据类型
# df.index.hasnans # 有没有空值
# df.index.is_monotonic_decreasing # 是否单调递减
# df.index.is_monotonic # 是否有单调性
# df.index.is_monotonic_increasing # 是否单调递增
# df.index.nbytes # 基础数据中的字节数
# df.index.ndim # 维度数，维数
# df.index.nlevels # 索引层级数，通常为1
# df.index.min() # 最小值
# df.index.argmin() # 最小索引值
# df.index.argsort() # 顺序值组成的数组
# df.index.asof(2) # 返回最近的索引
# # 索引类型转换
# df.index.astype('int64', copy=True) # 深拷贝
# # 拷贝
# df.index.copy(name='new', deep=True, dtype='int64')
# df.index.delete(1) # 删除指定位置
# # 对比不同
# df.index.difference(pd.Index([1,2,4]), sort=False)
# df.index.drop('a', errors='ignore') # 删除
# df.index.drop_duplicates(keep='first') # 去重值
# df.index.droplevel(0) # 删除层级
# df.index.dropna(how='all') # 删除空值
# df.index.duplicated(keep='first') # 重复值在结果数组中为True
# df.index.equals(df.index) # 与另一个索引对象是否相同
# df.index.factorize() # 分解成（array:0-n, Index）
# df.index.fillna(0, {0:'nan'}) # 填充空值
# # 字符列表，把name值加在第一位，每个值加10
# df.index.format(name=True, formatter=lambda x:x+10)
# # 返回一个array，指定值的索引位数组，不在的为-1
# df.index.get_indexer([2,9])
# # 获取指定层级Index对象
# df.index.get_level_values(0)
# # 指定索引的位置，见示例
# df.index.get_loc('b')
# df.index.insert(2, 'f') # 在索引位2插入f
# df.index.intersection(df.index) # 交集
# df.index.is_(df.index) # 类似is检查
# df.index.is_categorical() # 是否分类数据
# df.index.is_type_compatible(df.index) # 类型是否兼容
# df.index.is_type_compatible(1) # 类型是否兼容
# df.index.isna() # array是否为空
# df.index.isnull() # array是否缺失值
# df.index.join(df.index, how='left') # 连接
# df.index.notna() # 是否不存在的值
# df.index.notnull() # 是否不存在的值
# df.index.ravel() # 展平值的ndarray
# df.index.reindex(['a','b']) # 新索引 (Index,array:0-n)
# df.index.searchsorted('f') # 如果插入这个值，排序后在哪个索引位
# df.index.searchsorted([0, 4]) # array([0, 3]) 多个
# df.index.set_names('quarter') # 设置索引名称
# df.index.set_names('species', level=0)
# df.index.set_names(['kind', 'year'], inplace=True)
# df.index.shift(10, freq='D') # 日期索引向前移动10天
# idx1.symmetric_difference(idx2) # 两个索引不同的内容
# idx1.union(idx2) # 拼接
# df.add_prefix('t_') # 表头加前缀
# df.add_suffix('_d') # 表头加后缀
# df.first_valid_index() # 第一个有值的索引
# df.last_valid_index() # 最后一个有值的索引


print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.1 索引操作')
print('\t4.1.8 索引重命名')
print()
# update20240223
'''
将一个数据列置为索引后，就不能再像修改列名那样修改索引的名称了，
需要使用df.rename_axis方法。它不仅可以修改索引名，还可以修改列名。
需要注意的是，这里修改的是索引名称，不是索引或者列名本身。

'''
df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print('修改前：')
print(df.index)
print(df.index.name)

print('修改后：')
df.rename_axis('student_name',inplace=True)
print(df.index)
print(df)

'''
out:
修改前：
Index(['Liver', 'Arry', 'Ack', 'Eorge', 'Oah', 'Rick'], dtype='object', name='name')
name
修改后：
Index(['Liver', 'Arry', 'Ack', 'Eorge', 'Oah', 'Rick'], dtype='object', name='student_name')
             team   Q1  Q2  Q3   Q4
student_name                       
Liver           E   89  21  24   64
Arry            C   36  37  37   57
'''

# df.rename_axis(["dow", "hr"]) # 多层索引修改索引名
# df.rename_axis('info', axis="columns") # 修改行索引名

# 修改多层列索引名
# df.rename_axis(index={'a': 'A', 'b': 'B'})
'''
example:

import pandas as pd

# 创建一个具有多层行索引的 DataFrame
df = pd.DataFrame({
    'value': [1, 2, 3, 4]
})

print(df)
print()

df.index = pd.MultiIndex.from_product([['a', 'b'], [1, 2]], names=['outer', 'inner'])

print(df)
print()
# 使用 rename_axis 修改多层行索引的名称
df = df.rename_axis(index={'outer': 'A', 'inner': 'B'})

print("修改多层行索引名称后的 DataFrame：")
print(df)

------------------------------
out:
   value
0      1
1      2
2      3
3      4

             value
outer inner       
a     1          1
      2          2
b     1          3
      2          4

修改多层行索引名称后的 DataFrame：
     value
A B       
a 1      1
  2      2
b 1      3
  2      4

进程已结束,退出代码0
'''

# 修改多层列索引名
# df.rename_axis(columns={'name': 's_name', 'b': 'B'} #  测试好像没效果
# df.rename_axis(columns=str.upper) # 行索引名变大写 #  测试好像没效果 替换为df.rename 有效
# 下文为示例

'''
df.rename_axis(columns=str.upper)


example:
import pandas as pd

# 创建一个简单的 DataFrame
df = pd.DataFrame({
    'temperature': [22, 23],
    'humidity': [45, 50]
})

print("原始 DataFrame：")
print(df)
print()

# 使用 rename 方法将所有列标签变为大写
df = df.rename(columns=str.upper)

print("列标签变为大写后的 DataFrame：")
print(df)

-----------------------------------
out:
原始 DataFrame：
   temperature  humidity
0           22        45
1           23        50

列标签变为大写后的 DataFrame：
   TEMPERATURE  HUMIDITY
0           22        45
1           23        50

进程已结束,退出代码0
'''


print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.1 索引操作')
print('\t4.1.9 修改索引内容')
print()
# update20240223
'''
用来修改行和列的索引名的主要函数是df.rename和df.set_axis。
df.rename可以给定一个字典，键是原名称，值是想要修改的名称，
还可以传入一个与原索引等长度序列进行覆盖修改，用一个函数处理原索引名。
以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

# 一一对应修改列索引
df.rename(columns={"A": "a", "B": "c"})
df.rename(str.lower, axis='columns')  # 如果我们想要将所有列名转换为小写：
'''
example:
import pandas as pd

# 假设我们有以下 DataFrame：
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
print(df)
print()
# 我们想要将列名 "A" 改为 "a"，"B" 改为 "b"：
df = df.rename(columns={"A": "a", "B": "b"})
print(df)
print()

df = df.rename(str.lower, axis='columns')
print(df)
print()
-------------------------
out:
   A  B
0  1  3
1  2  4

   a  b
0  1  3
1  2  4


'''

# 修改行索引
df.rename(index={0: "x", 1: "y", 2: "z"})
df.rename({1: 2, 2: 4}, axis='index')
""" 
import pandas as pd

# 假设我们有以下 DataFrame：
df = pd.DataFrame({'A': [1, 2, 3], 'B': [3, 4, 5]})
# df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
print(df)
print()


print('# 修改行索引')
df = df.rename(index={0: "x", 1: "y", 2: "z"})
print(df)
'''
out:
   A  B
x  1  3
y  2  4
z  3  5
'''

df = df.rename({1: 2, 2: 4}, axis='index')
print(df)
'''
out:
   A  B
0  1  3
2  2  4
'''
"""

# 修改数据类型
df.rename(index=str)
"""
example:

import pandas as pd

# 假设我们有以下 DataFrame：
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['0', '1'])
print(df.index)
print()
# 我们想要将字符串类型的行索引转换为整数类型：
df = df.rename(index=int)
print(df.index)
print()
'''
out:
Index(['0', '1'], dtype='object')
Index([0, 1], dtype='int64')
'''

"""

# 重新修改索引
replacements = {l1:l2 for l1, l2 in zip(list1, list2)}
df.rename(replacements)
"""
example:
# 假设我们有以下 DataFrame：
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['a', 'b'])

print(df)
print()

list1 = ['a', 'b']
list2 = ['A', 'B']
# 我们想要根据两个列表的对应关系来批量修改索引：
replacements = {l1:l2 for l1, l2 in zip(list1, list2)}
df = df.rename(replacements)
print(df)
print()

--------------------------------------
out:
   A  B
a  1  3
b  2  4

   A  B
A  1  3
B  2  4
"""

# 列名加前缀
df.rename(lambda x:'t_' + x, axis=1)
"""

example:
# 假设我们有以下 DataFrame：
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
print(df)
print()
# 我们想要给所有列名添加前缀 "t_"：
df = df.rename(lambda x:'t_' + x, axis=1)
print(df)
print()

'''
out:
   A  B
0  1  3
1  2  4

   t_A  t_B
0    1    3
1    2    4
'''

"""
# 利用iter()函数的next特性修改
df.rename(lambda x, y=iter('abcdef'): next(y), axis=1)
# 修改列名，用解包形式生成新旧字段字典
df.rename(columns=dict(zip(df, list('abcd'))))

'''
df.set_axis可以将所需的索引分配给给定的轴，
通过分配类似列表或索引的方式来更改列标签或行标签的索引。
'''

# 修改索引
df.set_axis(['a', 'b', 'c'], axis='index')
"""
example:
# 假设我们有以下 DataFrame：
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
print(df)
print()


# 假设我们有以下 DataFrame：
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
# 我们想要将行索引修改为 'a', 'b', 'c'：
# df.set_axis(['a', 'b', 'c'], axis='index', inplace=True) # TypeError: set_axis() got an unexpected keyword argument 'inplace'
df = df.set_axis(['a', 'b', 'c'], axis='index')
print(df)
print()
'''
out:
   A  B
0  1  3
1  2  4

   A  B
a  1  4
b  2  5
c  3  6
'''

"""



# 修改列名
df.set_axis(list('cd'), axis=1)
"""
example:
# 假设我们有以下 DataFrame：
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df = df.set_axis(list('cd'), axis=1) # axis=1
print(df)
print()
'''
out:
   A  B
0  1  3
1  2  4

   c  d
0  1  4
1  2  5
2  3  6
'''
"""
# 使修改生效
df = df.set_axis(['a', 'b'], axis='columns')
"""
example:
# 假设我们有以下 DataFrame：
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df)
print()
# 如果我们想要将列名修改为 'a', 'b' 并且使修改生效：
df = df.set_axis(['a', 'b'], axis='columns')
print(df)
print()
'''
-------------------------------
out:
   A  B
0  1  4
1  2  5
2  3  6

   a  b
0  1  4
1  2  5
2  3  6
'''
"""
# 传入索引内容
df.set_axis(pd.Index(list('abcde')), axis=0)
"""
example:
# 如果我们想要将行索引修改为 'c', 'd', 'e'：
df = df.set_axis(pd.Index(list('cde')), axis=0) # list('cde')提供的元素不能超限 否则报错
print(df)
print()
'''
out:
   A  B
0  1  4
1  2  5
2  3  6

   A  B
c  1  4
d  2  5
e  3  6
'''
"""
