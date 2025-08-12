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
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.4 添加修改')
print('\t5.4.1 修改数值')
print()
# update20240320
'''
对数据的修改、增加和删除在数据整理过程中时常发生。修改的情况一般是修改错误，
还有一种情况是格式转换，如把中文数字修改为阿拉伯数字。修改也会涉及数据的类型修改。

删除一般会通过筛选的方式，筛选完成后将最终的结果重新赋值给变量，达到删除的目的。
增加行和列是最为常见的操作，数据分析过程中会计算出新的指标以新列展示。
'''
# 小节注释
'''
在Pandas中修改数值非常简单，先筛选出需要修改的数值范围，再为这个范围重新赋值。

以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 示例1:修改一个具体的值')
print(df.iloc[0, 0])  # 查询值
# out:Liver
df.iloc[0, 0] = 'Lily'  # 修改值
print(df.iloc[0, 0])
# out:Lily

print()
print('# 修改大范围值--定值')
# 将小于60分的成绩修改为32.5
df.loc[df['Q1'] < 60, 'Q1'] = 32.5  # 修改指定Q1列 | 如果不限制列 符合条件行的所有列数据都会变更！
print(df)
#     name team     Q1  Q2  Q3   Q4
# 0   Lily    E   89.0  21  24   64
# 1   Arry    C   32.5  37  37   57
# 2    Ack    A   32.5  60  18   84
# 3  Eorge    C   93.0  96  71   78
# 4    Oah    D   65.0  49  61   86
# 5   Rick    B  100.0  99  97  100

'''
以上操作df变量的内容被修改，这里指定的是一个定值，所有满足条件的数据均被修改为这个定值。
还可以传一个同样形状的数据来修改值：
'''
print()
print('# 生成1个长度为6的列表')
ls = [4, 5] * 3  # out:[4, 5, 4, 5, 4, 5]
print(ls)
# 修改
df.Q1 = ls
print(df)
#     name team  Q1  Q2  Q3   Q4
# 0   Lily    E   4  21  24   64
# 1   Arry    C   5  37  37   57
# 2    Ack    A   4  60  18   84
# 3  Eorge    C   5  96  71   78
# 4    Oah    D   4  49  61   86
# 5   Rick    B   5  99  97  100
'''对于修改DataFrame，会按对应的索引位进行修改：'''
print()
print('# 修改指定范围--字典')
print(df.loc[1:3, 'Q1':'Q2'])

print()
df1 = pd.DataFrame({'Q1': [1, 2, 3], 'Q2': [4, 5, 6]})
print(df1)
#    Q1  Q2
# 0   1   4
# 1   2   5
# 2   3   6

# 执行修改
df.loc[1:3, 'Q1':'Q2'] = df1  # 对应索引修改

print(df)  # 结果显示 对应的索引修改
#     name team   Q1    Q2  Q3   Q4
# 0   Lily    E  4.0  21.0  24   64
# 1   Arry    C  2.0   5.0  37   57
# 2    Ack    A  3.0   6.0  18   84
# 3  Eorge    C  NaN   NaN  71   78
# 4    Oah    D  4.0  49.0  61   86
# 5   Rick    B  5.0  99.0  97  100


print()
print('# 选择索引和修改索引一致')
df2 = pd.DataFrame({'Q1': [1, 2, 3], 'Q2': [4, 5, 6]}, index=[3, 4, 5])
print(df2)
df.loc[3:5, 'Q1':'Q2'] = df2
print(df)
#     name team   Q1    Q2  Q3   Q4
# 0   Lily    E  4.0  21.0  24   64
# 1   Arry    C  2.0   5.0  37   57
# 2    Ack    A  3.0   6.0  18   84
# 3  Eorge    C  1.0   4.0  71   78
# 4    Oah    D  2.0   5.0  61   86
# 5   Rick    B  3.0   6.0  97  100


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.4 添加修改')
print('\t5.4.2 替换数据')
print()
# update20240320
'''
对数据的修改、增加和删除在数据整理过程中时常发生。修改的情况一般是修改错误，
还有一种情况是格式转换，如把中文数字修改为阿拉伯数字。修改也会涉及数据的类型修改。

删除一般会通过筛选的方式，筛选完成后将最终的结果重新赋值给变量，达到删除的目的。
增加行和列是最为常见的操作，数据分析过程中会计算出新的指标以新列展示。
'''
# 小节注释
'''
replace方法可以对数据进行批量替换：

以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

# s.replace(0, 5) # 将列数据中的0换为5
# df.replace(0, 5) # 将数据中的所有0换为5

# df.replace([0, 1, 2, 3], 4) # 将0～3全换成4
# df.replace([0, 1, 2, 3], [4, 3, 2, 1]) # 对应修改

# # {'pad', 'ffill', 'bfill', None} 试试
# s.replace([1, 2], method='bfill') # 向下填充

# df.replace({0: 10, 1: 100}) # 字典对应修改
# df.replace({'Q1': 0, 'Q2': 5}, 100) # 将指定字段的指定值修改为100
# df.replace({'Q1': {0: 100, 4: 400}}) # 将指定列里的指定值替换为另一个指定的值

# # 使用正则表达式
# df.replace(to_replace=r'^ba.$', value='new', regex=True)
# df.replace({'A': r'^ba.$'}, {'A': 'new'}, regex=True)
# df.replace(regex={r'^ba.$': 'new', 'foo': 'xyz'})
# df.replace(regex=[r'^ba.$', 'foo'], value='new')

print()
print('# 将Series中的所有0替换为5')
s = pd.Series([0, 1, 2, 0, 3])
print(s.replace(0, 5))  # 将Series中的所有0替换为5
# 0    5
# 1    1
# 2    2
# 3    5
# 4    3
# dtype: int64

print()
print('# 将DataFrame中的所有0换为5：')
df = pd.DataFrame({'A': [0, 1, 2], 'B': [5, 0, 3]})
print(df.replace(0, 5))  # 将DataFrame中的所有0替换为5
#    A  B
# 0  5  5
# 1  1  5
# 2  2  3

print()
print('# 将0～3全换成4：')
df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [4, 3, 2, 1]})
print(df.replace([0, 1, 2, 3], 4))  # 将0～3全换成4
#    A  B
# 0  4  4
# 1  4  4
# 2  4  4
# 3  4  4

print()
print('# 对应修改，0换成4，1换成3，以此类推：')
df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [4, 3, 2, 1]})
print(df.replace([0, 1, 2, 3], [4, 3, 2, 1]))  # 对应修改，0换成4，1换成3，以此类推
# print(df.replace([0,1,2,3],[9,8,7,6]))
#    A  B
# 0  4  4
# 1  3  1
# 2  2  2
# 3  1  3

print()
print("# 5、向下填充：{'pad', 'ffill', 'bfill', None} 试试")  # replace() openai提示一般不使用这种填充方法！
s = pd.Series([1, 2, 3, 4, 1, 2])
s1 = s.replace([1, 2], method='bfill')  # 使用后面的值填充1和2
print(s1)
# 0    3
# 1    3
# 2    3
# 3    4
# 4    2
# 5    2
s2 = s.replace([1, 2], method='ffill')
print(s2)
# 0    1
# 1    1
# 2    3
# 3    4
# 4    4
# 5    4
s3 = s.replace([1, 2], method='pad')
print(s3)
# 0    1
# 1    1
# 2    3
# 3    4
# 4    4
# 5    4

print()
print('# 6、字典对应修改：')
df = pd.DataFrame({'A': [0, 1, 2], 'B': [1, 0, 3]})
print(df.replace({0: 10, 1: 100}))  # 使用字典进行替换，0换成10，1换成100
#      A    B
# 0   10  100
# 1  100   10
# 2    2    3

print()
print('# 7、将指定字段的指定值修改为100:')
df = pd.DataFrame({'Q1': [0, 1, 2], 'Q2': [5, 0, 3]})
print(df.replace({'Q1': 0, 'Q2': 5}, 100))  # 在列Q1中将0替换为100，在列Q2中将5替换为100
#     Q1   Q2
# 0  100  100
# 1    1    0
# 2    2    3

print()
print('# 8、将指定列里的指定值替换为另一个指定的值：')
df = pd.DataFrame({'Q1': [0, 4, 2], 'Q2': [1, 4, 3]})
print(df.replace({'Q1': {0: 100, 4: 400}}))  # 只在列Q1中进行替换，将0替换为100，将4替换为400
#     Q1  Q2
# 0  100   1
# 1  400   4
# 2    2   3

print('# 9、使用正则表达式进行替换')
# 假设我们有以下DataFrame
df = pd.DataFrame({'Q3': ['bat', 'foo', 'bar', 'baz', 'foobar'], 'Q4': ['bat', 'foo', 'bar', 'xyz', 'foobar']})
print(df)

print()
print("# 使用正则表达式替换")
# 将 df 中全部以'ba'开头的字符串替换为'new'
df_replaced_regex = df.replace(to_replace=r'^ba.', value='new', regex=True)
print(df_replaced_regex)
#        Q3      Q4
# 0     new     new
# 1     foo     foo
# 2     new     new
# 3     new     xyz
# 4  foobar  foobar

print()
print('# 只在列 Q3 中应用正则表达式替换')
df_replaced_col_regex = df.replace({'Q3': r'^ba.'}, {'Q3': 'new'}, regex=True)
print(df_replaced_col_regex)
#        Q3      Q4
# 0     new     bat
# 1     foo     foo
# 2     new     bar
# 3     new     xyz
# 4  foobar  foobar

print()
print('# 使用正则表达式字典进行替换:')
# 将 Q3 Q4 中以'ba'开头后跟任意字符的字符串替换为'new'，将'foo'替换为'xyz'
df_replaced_regex_dict = df.replace(regex={r'^ba.': 'new', 'foo': 'xyz'})
print(df_replaced_regex_dict)
#        Q3      Q4
# 0     new     new
# 1     xyz     xyz
# 2     new     new
# 3     new     xyz
# 4  xyzbar  xyzbar

print()
print('# 使用正则表达式列表进行替换:')
# 将 Q3 Q4中以'ba'开头后跟任意字符的字符串和'foo'都替换为'new'
df_replaced_regex_list = df.replace(regex=[r'^ba.', 'foo'], value='new')
print(df_replaced_regex_list)
#        Q3      Q4
# 0     new     new
# 1     new     new
# 2     new     new
# 3     new     xyz
# 4  newbar  newbar

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.4 添加修改')
print('\t5.4.3 填充空值')
print()
# update20240321
'''
对数据的修改、增加和删除在数据整理过程中时常发生。修改的情况一般是修改错误，
还有一种情况是格式转换，如把中文数字修改为阿拉伯数字。修改也会涉及数据的类型修改。

删除一般会通过筛选的方式，筛选完成后将最终的结果重新赋值给变量，达到删除的目的。
增加行和列是最为常见的操作，数据分析过程中会计算出新的指标以新列展示。
'''
# 小节注释
'''
fillna对空值填入指定数据，通常应用于数据清洗。还有一种做法是删除有空值的数据，后文会介绍。

以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

df = pd.DataFrame([[np.nan, 2, np.nan, 0],
                   [3, 4, np.nan, 1],
                   [np.nan, np.nan, np.nan, np.nan],
                   [np.nan, 3, np.nan, 4]],
                  columns=list("ABCD"))

print(df)
#      A    B   C    D
# 0  NaN  2.0 NaN  0.0
# 1  3.0  4.0 NaN  1.0
# 2  NaN  NaN NaN  NaN
# 3  NaN  3.0 NaN  4.0
print()

print('1、# 将空值全修改为0：')
# 使用fillna(0)可以将DataFrame中所有的空值（NaN）替换为0。
# df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
df_filled = df.fillna(0)
print(df_filled)
#      A    B    C    D
# 0  0.0  2.0  0.0  0.0
# 1  3.0  4.0  0.0  1.0
# 2  0.0  0.0  0.0  0.0
# 3  0.0  3.0  0.0  4.0

print()
print('2、将空值都修改为其前一个值：')
# 使用fillna(method='ffill')可以将空值替换为它前面的非空值。
df_filled_ffill = df.fillna(method='ffill')
print(df_filled_ffill)
#      A    B   C    D
# 0  NaN  2.0 NaN  0.0
# 1  3.0  4.0 NaN  1.0
# 2  3.0  4.0 NaN  1.0
# 3  3.0  3.0 NaN  4.0

print()
print('3、为各列填充不同的值：')
# 通过传递一个字典给value参数，可以为不同的列指定不同的填充值。
values = {'A': 0, 'B': 1, 'C': 2, 'D': 3}  # {{'A':0,'B':1,'C':2,'D':3}}
df_filled_values = df.fillna(value=values)
print(df_filled_values)
#      A    B    C    D
# 0  0.0  2.0  2.0  0.0
# 1  3.0  4.0  2.0  1.0
# 2  0.0  1.0  2.0  3.0
# 3  0.0  3.0  2.0  4.0

print()
print('4、只替换第一个空值：')
# limit参数可以指定每列替换的最大空值数量
df_filled_limit = df.fillna(value=values, limit=1)
print(df_filled_limit)
#      A    B    C    D
# 0  0.0  2.0  2.0  0.0
# 1  3.0  4.0  NaN  1.0
# 2  NaN  1.0  NaN  3.0
# 3  NaN  3.0  NaN  4.0
# df_filled_limit = df.fillna(value=values,limit=2)
# 替换每列前2个空值
#      A    B    C    D
# 0  0.0  2.0  2.0  0.0
# 1  3.0  4.0  2.0  1.0
# 2  0.0  1.0  NaN  3.0
# 3  NaN  3.0  NaN  4.0

print()
print('# 注意df2没有D列')
# When filling using a DataFrame, replacement happens along the same column names and same indices
df2 = pd.DataFrame(np.zeros((4, 4)), columns=list("ABCE"))
print(df2)
#      A    B    C    E
# 0  0.0  0.0  0.0  0.0
# 1  0.0  0.0  0.0  0.0
# 2  0.0  0.0  0.0  0.0
# 3  0.0  0.0  0.0  0.0
# df.fillna(df2)
df_filled_df2 = df.fillna(df2)
print(df_filled_df2)
#      A    B    C    D
# 0  0.0  2.0  0.0  0.0
# 1  3.0  4.0  0.0  1.0
# 2  0.0  0.0  0.0  NaN
# 3  0.0  3.0  0.0  4.0


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.4 添加修改')
print('\t5.4.4 修改索引名')
print()
# update20240321
'''
对数据的修改、增加和删除在数据整理过程中时常发生。修改的情况一般是修改错误，
还有一种情况是格式转换，如把中文数字修改为阿拉伯数字。修改也会涉及数据的类型修改。

删除一般会通过筛选的方式，筛选完成后将最终的结果重新赋值给变量，达到删除的目的。
增加行和列是最为常见的操作，数据分析过程中会计算出新的指标以新列展示。
'''
# 小节注释
'''
修改索引名最简单也最常用的办法就是将df.index和df.columns重新赋值为一个类似于列表的序列值，这会将其覆盖为指定序列中的名称。
使用df.rename和df.rename_axis对轴名称进行修改。以下案例将列名team修改为class：

以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 将列名team修改为class：')
print(df.rename(columns={'team': 'class'}))
#     name class   Q1  Q2  Q3   Q4
# 0  Liver     E   89  21  24   64
# ...
# 5   Rick     B  100  99  97  100

print()
print('# 对行、列索引进行修改')
print(df.rename(columns={'Q1': 'a', 'Q2': 'b'}))  # 对表头进行修改
#     name team    a   b  Q3   Q4
# 0  Liver    E   89  21  24   64
# ...
print(df.rename(index={0: "x", 1: "y", 2: "z"}))  # 对索引进行修改
#     name team   Q1  Q2  Q3   Q4
# x  Liver    E   89  21  24   64
# y   Arry    C   36  37  37   57
# z    Ack    A   57  60  18   84
# 3  Eorge    C   93  96  71   78
# 4    Oah    D   65  49  61   86
# 5   Rick    B  100  99  97  100

print()
print('')
print(df.index.dtype)  # int64
print(df.columns.dtype)  # object
df1 = df.rename(index=str)  # 修改类型索引
print(df1.index.dtype)  # object

print()
print(df.rename(str.lower, axis='columns'))  # 传索引类型 | 表头修改为小写
#     name team   q1  q2  q3   q4
# 0  Liver    E   89  21  24   64
# ...

print(df.rename({1: 2, 2: 4}, axis='index'))
#     name team   Q1  Q2  Q3   Q4
# 0  Liver    E   89  21  24   64
# 2   Arry    C   36  37  37   57
# 4    Ack    A   57  60  18   84
# 3  Eorge    C   93  96  71   78
# 4    Oah    D   65  49  61   86
# 5   Rick    B  100  99  97  100

print()
print('# 对索引名进行修改')
# 使用rename_axis可以修改轴标签的名称。
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
print(s)
print(s.name)  # None
s_rename = s.rename_axis('animal')
print(s_rename)
# animal
# a    1
# b    2
# c    3
# dtype: int64
print()
print(s.index.name)  # None
# print(s.animal) 报错！

print()
print('# 修改DataFrame的列索引名：')
# 默认情况下，rename_axis会修改列索引的名称。
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
print(df1)
#    A  B
# 0  1  3
# 1  2  4
df1_renamed = df1.rename_axis('animal')
print(df1_renamed)
#         A  B
# animal
# 0       1  3
# 1       2  4
print(df1_renamed.columns)  # Index(['A', 'B'], dtype='object')
print(df1_renamed.index)  # RangeIndex(start=0, stop=2, step=1, name='animal')

print()
print('# 修改DataFrame的行索引名：')
# 使用axis参数指定轴，可以修改行索引的名称。
df1_renamed_axis = df1.rename_axis("limbs", axis="columns")
print(df1_renamed_axis)
# limbs  A  B
# 0      1  3
# 1      2  4

print()
print(df1_renamed_axis.columns)  # Index(['A', 'B'], dtype='object', name='limbs')
print(df1_renamed_axis.index)  # RangeIndex(start=0, stop=2, step=1)

print()
print('# 修改多层索引名：')
# 如果DataFrame有多层索引，可以使用字典形式的参数来修改特定层级的索引名。
# 不懂 以后再说
df_multi = pd.DataFrame({'A': [1, 2], 'B': [3, 4]},
                        index=pd.MultiIndex.from_arrays([[0, 1], ['dog', 'cat']], names=['type', 'name']))
df_multi_renamed = df_multi.rename_axis(index={'type': 'class'})
print(df_multi_renamed)
#             A  B
# class name
# 0     dog   1  3
# 1     cat   2  4

print()
print('# 修改多层索引名：')
s_set_axis = s.set_axis(['x', 'y', 'z'], axis=0)
print(s_set_axis)
# x    1
# y    2
# z    3
# dtype: int64

# 测试报错！
# s_set_axis_1 = s.set_axis(['x', 'y', 'z'],axis=1)
# print(s_set_axis_1)

print()
print('# 使用set_axis修改DataFrame的列名：')
df2 = df1.set_axis(['I', 'II'], axis='columns')  # 没有参数 inplace=True
print(df2)
#    I  II
# 0  1   3
# 1  2   4


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.4 添加修改')
print('\t5.4.5 增加列')
print()
# update20240322
'''
对数据的修改、增加和删除在数据整理过程中时常发生。修改的情况一般是修改错误，
还有一种情况是格式转换，如把中文数字修改为阿拉伯数字。修改也会涉及数据的类型修改。

删除一般会通过筛选的方式，筛选完成后将最终的结果重新赋值给变量，达到删除的目的。
增加行和列是最为常见的操作，数据分析过程中会计算出新的指标以新列展示。
'''
# 小节注释
'''
增加列是数据处理中最常见的操作，Pandas可以像定义一个变量一样定义DataFrame中新的列，
新定义的列是实时生效的。与数据修改的逻辑一样，新列可以是一个定值，所有行都为此值，
也可以是一个同等长度的序列数据，各行有不同的值。接下来我们增加总成绩total列：

以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print()
print('# 四个季度的成绩相加为总成绩')
df['total'] = df.Q1 + df.Q2 + df.Q3 + df.Q4
print(df)

# 方法2
df['total2'] = df.loc[:, 'Q1':'Q4'].sum(1)
print(df)  #
#     name team   Q1  Q2  Q3   Q4  total  total2
# 0  Liver    E   89  21  24   64    198     198
# 1   Arry    C   36  37  37   57    167     167
# 2    Ack    A   57  60  18   84    219     219
# 3  Eorge    C   93  96  71   78    338     338
# 4    Oah    D   65  49  61   86    261     261
# 5   Rick    B  100  99  97  100    396     396

print()
df['foo'] = 100  # 增加一列foo，所有值都是100
print(df)
#     name team   Q1  Q2  Q3   Q4  total  total2  foo
# 0  Liver    E   89  21  24   64    198     198  100
# ...
df['foo'] = df.Q1 + df.Q2  # # 新列为两列相加
print(df)

df['foo'] = df['Q1'] + df['Q2']  # 同上
print(df)
#     name team   Q1  Q2  Q3   Q4  total  total2  foo
# 0  Liver    E   89  21  24   64    198     198  110
# ...

print()
print('# 数值列计算')
# 把所有为数字的值加起来
print(df.select_dtypes(include=[int]))
df['total'] = df.select_dtypes(include=[int]).sum(1)
print(df)
#     name team   Q1  Q2  Q3   Q4  total  total2  foo
# 0  Liver    E   89  21  24   64    704     198  110

print()
df['total3'] = df.loc[:, 'Q1':'Q4'].apply(lambda x: sum(x), axis='columns')
print(df)
#     name team   Q1  Q2  Q3   Q4  total  total2  foo  total3
# 0  Liver    E   89  21  24   64    704     198  110     198
# ...

print()
# 新增列
df.loc[:, 'Q10'] = '我是新来的'  # 也可以
print(df)
#     name team   Q1  Q2  Q3   Q4  total  total2  foo  total3    Q10
# 0  Liver    E   89  21  24   64    704     198  110     198  我是新来的
# ...

print()
print('# 增加一列并赋值，不满足条件的为NaN')
df.loc[df.total2 >= 240, '成绩'] = '合格'
print(df)
#     name team   Q1  Q2  Q3   Q4  total  total2  foo  total3    Q10   成绩
# 0  Liver    E   89  21  24   64    704     198  110     198  我是新来的  NaN
# 1   Arry    C   36  37  37   57    574     167   73     167  我是新来的  NaN
# 2    Ack    A   57  60  18   84    774     219  117     219  我是新来的  NaN
# 3  Eorge    C   93  96  71   78   1203     338  189     338  我是新来的   合格
# 4    Oah    D   65  49  61   86    897     261  114     261  我是新来的   合格
# 5   Rick    B  100  99  97  100   1387     396  199     396  我是新来的   合格
df.loc[df.total2 < 240, '成绩'] = '不合格'
print(df)
#     name team   Q1  Q2  Q3   Q4  total  total2  foo  total3    Q10   成绩
# 0  Liver    E   89  21  24   64    704     198  110     198  我是新来的  不合格
# 1   Arry    C   36  37  37   57    574     167   73     167  我是新来的  不合格
# 2    Ack    A   57  60  18   84    774     219  117     219  我是新来的  不合格
# 3  Eorge    C   93  96  71   78   1203     338  189     338  我是新来的   合格
# 4    Oah    D   65  49  61   86    897     261  114     261  我是新来的   合格
# 5   Rick    B  100  99  97  100   1387     396  199     396  我是新来的   合格


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.4 添加修改')
print('\t5.4.6 插入列 df.insert()')
print()
# update20240326
'''
对数据的修改、增加和删除在数据整理过程中时常发生。修改的情况一般是修改错误，
还有一种情况是格式转换，如把中文数字修改为阿拉伯数字。修改也会涉及数据的类型修改。

删除一般会通过筛选的方式，筛选完成后将最终的结果重新赋值给变量，达到删除的目的。
增加行和列是最为常见的操作，数据分析过程中会计算出新的指标以新列展示。
'''
# 小节注释
'''
Pandas提供了insert()方法来为DataFrame插入一个新列。
insert()方法可以传入三个主要参数：
loc是一个数字，代表新列所在的位置，使用列的数字索引，如0为第一列；
第二个参数column为新的列名；
最后一个参数value为列的值，一般是一个Series。

以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()
# out:
#     name team   Q1  Q2  Q3   Q4
# 0  Liver    E   89  21  24   64
# 1   Arry    C   36  37  37   57
# 2    Ack    A   57  60  18   84
# 3  Eorge    C   93  96  71   78
# 4    Oah    D   65  49  61   86
# 5   Rick    B  100  99  97  100

print('# 在第三列的位置上插入新列total列，值为每行的总成绩')
# df1 = df.insert(2,'total',df.select_dtypes(include=[int]).sum(axis=1)) # 不需要赋值 否则 out : None
df.insert(2, 'total', df.select_dtypes(include=[int]).sum(axis=1))  # insert() 直接修改本身
print(df)
#     name team  total   Q1  Q2  Q3   Q4
# 0  Liver    E    198   89  21  24   64
# ...

'''
在 pandas 中，DataFrame.insert() 方法是用来在 DataFrame 中的指定位置插入一列的，
但是这个方法没有返回值，它直接修改了原始的 DataFrame 对象。
因此，不应该将 df.insert() 的结果赋值给 df1，因为 insert() 方法本身就是就地操作，没有返回值（即返回 None）。
'''

'''-----------------------------------'''

'''
如果已经存在相同的数据列，会报错，可传入allow_duplicates=True插入一个同名的列。
如果希望新列位于最后，可以在第一个参数位loc传入len(df.columns)。

示例如下：
'''

print()
print('# 插入重复列名')
s = pd.Series([6, 5, 4, 3, 2, 1], name='Q4')
print(s)
# df.insert(len(df.columns),'Q4',s) # ValueError: cannot insert Q4, already exists
df.insert(len(df.columns), 'Q4', s, allow_duplicates=True)  # 列名重复
print(df)
#     name team  total   Q1  Q2  Q3   Q4  Q4
# 0  Liver    E    198   89  21  24   64   6
# 1   Arry    C    167   36  37  37   57   5
# ...

'''Notice that pandas uses index alignment in case of value from type Series:'''
# 索引一致
print()
print('# 插入索引一一对应')
df2 = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
print(df2)
#    col1  col2
# 0     1     3
# 1     2     4
s1 = pd.Series([9, 8], index=[1, 2])
print(s1)
# 1    9
# 2    8
df2.insert(2, 'new_col', s1)
print(df2)
#    col1  col2  new_col
# 0     1     3      NaN
# 1     2     4      9.0


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.4 添加修改')
print('\t5.4.7 指定列 df.assign()')
print()
# update20240326
'''
对数据的修改、增加和删除在数据整理过程中时常发生。修改的情况一般是修改错误，
还有一种情况是格式转换，如把中文数字修改为阿拉伯数字。修改也会涉及数据的类型修改。

删除一般会通过筛选的方式，筛选完成后将最终的结果重新赋值给变量，达到删除的目的。
增加行和列是最为常见的操作，数据分析过程中会计算出新的指标以新列展示。
'''
# 小节注释
'''
df.assign(k=v)为指定一个新列的操作，k为新列的列名，v为此列的值，v必须是一个与原数据同索引的Series。
今后我们会频繁用到它，它在链式编程技术中相当重要，因此这里专门介绍一下。
我们平时在做数据探索分析时会增加一些临时列，如果新列全部使用赋值的方式生成，则会造成原数据混乱，
因此就需要一个方法来让我们不用赋值也可以创建一个临时的列。
这种思路适用于所有对原数据的操作，建议在未最终确定数据处理方案时，除了必要的数据整理工作，均使用链式方法，
我们在学习完所有的常用功能后会专门介绍这个技术。

以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 计算total列')
df1 = df.assign(total=df.select_dtypes(include=[int]).sum(1))
print(df1)
#     name team   Q1  Q2  Q3   Q4  total
# 0  Liver    E   89  21  24   64    198
# ...

print(df.assign(total=df.select_dtypes(include=[int]).sum(1)))  # 同上
# print(df)

print()
print('# 增加2列')
print(df.assign(total=df.select_dtypes(include=int).sum(1), Q=100))
#     name team   Q1  Q2  Q3   Q4  total    Q
# 0  Liver    E   89  21  24   64    198  100
# 1   Arry    C   36  37  37   57    167  100
# ...
print(df.assign(total=df.select_dtypes(include=int).sum(1)).assign(Q=100))  # 结果同上

print()
print('# 使用了链式方法')
print(
    df.assign(total=df.select_dtypes(include=[int]).sum(1))  # 总成绩
    .assign(Q=100)  # 目标满分值
    .assign(name_len=df.name.str.len())  # 姓名长度
    .assign(avg=df.select_dtypes(include=[int]).mean(1))  # 平均值
    .assign(avg2=lambda d: d.total / 4)  # 平均值2  TODO 注意lambda 用法 调用total列

)
#     name team   Q1  Q2  Q3   Q4  total    Q  name_len    avg   avg2
# 0  Liver    E   89  21  24   64    198  100         5  49.50  49.50
# 1   Arry    C   36  37  37   57    167  100         4  41.75  41.75
# ...

'''
以上是使用了链式方法的典型代码形式，后期会以这种风格进行代码编写。
特别要说明的是avg2列的计算过程，
因为df实际是没有total这一列的，如果我们需要使用total列，就需要用lambda来调用。
lambda中第一个变量d是代码执行到本行前的DataFrame内容，
可以认为是一个虚拟的DataFrame实体，然后用变量d使用这个DataFrame的数据。
作为变量名，d可以替换为其他任意合法的名称，但为了代码可读性，建议使用d，代表它是一个DataFrame。
如果是Series，建议使用s。
'''

# TODO
# 假设 df 是一个已经存在的 DataFrame
df = pd.DataFrame({
    'Q1': [80, 90, 100],
    'Q2': [70, 85, 95]
})

print('# 新增加一列 Q5--列表式')
df_new = df.assign(Q5=[100, 100, 100])
print(df_new)

list1 = [23.5] * 3
print(df.assign(Q11=list1))
# print(df.assign(Q12=[23.15] * 2)) # ValueError: Length of values (2) does not match length of index (3)
print(df.assign(Q12=[23.15] * 3))  # 效果同上！
#     Q1  Q2    Q12
# 0   80  70  23.15
# 1   90  85  23.15
# 2  100  95  23.15

print()
print('# 计算并增加Q6列')
df_new = df.assign(Q6=df.Q2 / df.Q1)
print(df_new)
#     Q1  Q2        Q6
# 0   80  70  0.875000
# 1   90  85  0.944444
# 2  100  95  0.950000

# TODO 使用lambda函数
df_new = df.assign(Q7=lambda d: d.Q1 * 9 / 5 + 32)
print(df_new)
#     Q1  Q2     Q7
# 0   80  70  176.0
# 1   90  85  194.0
# 2  100  95  212.0

# df_new = df.assign(Q7=df.Q1*9/5+32) # 结果同上

print()
print('# 增加布尔类型列')
df_new = df.assign(tag=df.Q1 > df.Q2)
print(df_new)
#     Q1  Q2   tag
# 0   80  70  True
# 1   90  85  True
# 2  100  95  True

# 比较计算，True 为 1，False 为 0
# 将布尔类型转换为整数
print()
df_new = df.assign(tag=df.Q1 > 90).astype(int)
print(df_new)
#     Q1  Q2  tag
# 0   80  70    0
# 1   90  85    0
# 2  100  95    1
# df_new = df.assign(tag=(df.Q1 > df.Q2).astype(int)) # 效果同上
# print(df_new)

print()
print('# map()映射')
# 映射文案，根据 Q1 是否大于 80 来标记 '及格' 或 '不及格'
df_new = df.assign(tag=(df.Q1 > 80).map({True: '及格', False: '不及格'}))
print(df_new)
#     Q1  Q2 tag
# 0   80  70  不及格
# 1   90  85  及格
# 2  100  95  及格

print()
print('# 一次增加多个列')
df_new = df.assign(Q8=lambda d: d.Q1 * 5,
                   Q9=lambda d: d.Q1 + 1)

print(df_new)
#     Q1  Q2   Q8   Q9
# 0   80  70  400   81
# 1   90  85  450   91
# 2  100  95  500  101

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.4 添加修改')
print('\t5.4.8 执行表达式 df.eval()')
print()
# update20240327
'''
对数据的修改、增加和删除在数据整理过程中时常发生。修改的情况一般是修改错误，
还有一种情况是格式转换，如把中文数字修改为阿拉伯数字。修改也会涉及数据的类型修改。

删除一般会通过筛选的方式，筛选完成后将最终的结果重新赋值给变量，达到删除的目的。
增加行和列是最为常见的操作，数据分析过程中会计算出新的指标以新列展示。
'''
# 小节注释
'''
df.eval()与之前介绍过的df.query()一样，可以以字符的形式传入表达式，增加列数据。下面以增加总分为例：

以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 计算total列')
print(df.eval('total = Q1+Q2+Q3+Q4'))
#     name team   Q1  Q2  Q3   Q4  total
# 0  Liver    E   89  21  24   64    198
# ...

print()
print('# 其它常用方法')
df['C1'] = df.eval('Q2+Q3')
print(df)
#     name team   Q1  Q2  Q3   Q4   C1
# 0  Liver    E   89  21  24   64   45
# ...

a = df.Q1.mean()  # 73.33333333333333
# print(a)
print(df.eval("C3 = `Q3` + @a"))  # 使用变量
# print(df.eval("C3 = Q3 + @a")) # 效果同上
#     name team   Q1  Q2  Q3   Q4   C1          C3
# 0  Liver    E   89  21  24   64   45   97.333333
# ...

# TODO 加1个布尔值
print(df.eval('C3 = Q2 > (Q3 + @a)'))
#     name team   Q1  Q2  Q3   Q4   C1     C3
# 0  Liver    E   89  21  24   64   45  False
# ...

# TODO 增加多个列
print(df.eval('''
             c1 = Q1+Q2
             c2 = Q2+Q3
             '''))
#     name team   Q1  Q2  Q3   Q4   C1   c1   c2
# 0  Liver    E   89  21  24   64   45  110   45
# ...


# b= df.select_dtypes(include=int).mean(axis=1) # 按行平均
# print(b)
# 0     48.6
# 1     48.2
# 2     59.4
# 3    101.0
# 4     74.2
# 5    118.4

# b= df.select_dtypes(include=int).mean(axis=0) # 按列平均
# print(b)
# Q1     73.333333
# Q2     60.333333
# Q3     51.333333
# Q4     78.166667
# C1    111.666667
# dtype: float64

df.eval('total = Q3 + Q4', inplace=True)  # 立即生效
print(df)
#     name team   Q1  Q2  Q3   Q4   C1  total
# 0  Liver    E   89  21  24   64   45     88
# ...


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.4 添加修改')
print('\t5.4.9 增加行')
print()
# update20240327
'''
对数据的修改、增加和删除在数据整理过程中时常发生。修改的情况一般是修改错误，
还有一种情况是格式转换，如把中文数字修改为阿拉伯数字。修改也会涉及数据的类型修改。

删除一般会通过筛选的方式，筛选完成后将最终的结果重新赋值给变量，达到删除的目的。
增加行和列是最为常见的操作，数据分析过程中会计算出新的指标以新列展示。
'''
# 小节注释
'''
可以使用loc[]指定索引给出所有列的值来增加一行数据。目前我们的df最大索引是5，增加一条索引为6的数据：

以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 新增索引为6的数据')
df.loc[6] = ['abc', 'z', 88, 88, 88, 88]
print(df)
#     name team   Q1  Q2  Q3   Q4
# 0  Liver    E   89  21  24   64
# ...
# 6    abc    z   88  88  88   88
print()
print('# 其它常用方法')
# TODO 指定列，无数据列值为NaN
df.loc[7] = {'Q1': 89, 'Q2': 90}
print(df)
#     name team   Q1  Q2    Q3     Q4
# 0  Liver    E   89  21  24.0   64.0
# ...
# 6    abc    z   88  88  88.0   88.0
# 7    NaN  NaN   89  90   NaN    NaN

print('# 自动增加索引')
# df.loc[df.shape[0]+1] = {'Q1':89,'Q2':90} # 逻辑相似
df.loc[df.shape[0]] = {'Q1': 89, 'Q2': 90}
print(df)
# ...
# 6    abc    z   88  88  88.0   88.0
# 7    NaN  NaN   89  90   NaN    NaN
# 8    NaN  NaN   89  90   NaN    NaN

# print(len(df)) # out:9
print()
df.loc[len(df)] = {'Q1': 89.65, 'Q2': 90.01}  # 逻辑同上
print(df)
# ...
# 8    NaN  NaN   89.00  90.00   NaN    NaN
# 9    NaN  NaN   89.65  90.01   NaN    NaN

print()
print('# 批量操作，可以使用迭代')  # 限制匹配列数量 否则报错
rows = [[1, 2], [3, 4], [5, 6]]
df1 = df.loc[:, 'Q3':'Q4']
for row in rows:
    # df.loc[:,'Q3':'Q4'].loc[len(df)] = row
    df1.loc[len(df1)] = row

print(df1)
# ...
# 7    NaN    NaN
# 8    NaN    NaN
# 9    NaN    NaN
# 10   1.0    2.0
# 11   3.0    4.0
# 12   5.0    6.0
'''
在实际业务中，当使用循环批量添加新行时，确保新行的结构与 DataFrame 的结构一致是非常重要的。
如果新数据的列数比 DataFrame 多或者少，你需要相应地调整数据结构，或者在 DataFrame 中新增或忽略某些列。
'''

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.4 添加修改')
print('\t5.4.10 追加合并')
print()
# update20240327
'''
对数据的修改、增加和删除在数据整理过程中时常发生。修改的情况一般是修改错误，
还有一种情况是格式转换，如把中文数字修改为阿拉伯数字。修改也会涉及数据的类型修改。

删除一般会通过筛选的方式，筛选完成后将最终的结果重新赋值给变量，达到删除的目的。
增加行和列是最为常见的操作，数据分析过程中会计算出新的指标以新列展示。
'''
# 小节注释
'''
增加行数据的使用场景相对较少，一般是采用数据追加的模式。数据追加会在后续章节中介绍。

以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# # df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# # print(df.dtypes)
# print()

print('# 追加合并')
df1 = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
# print(df1)
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))  # CD

df3 = pd.DataFrame([[11, 12], [13, 14]], columns=list('CD'))
# print(df2)
# df1._append(df2)
print(df1._append(df2))  #
#    A  B
# 0  1  2
# 1  3  4
# 0  5  6
# 1  7  8
print(df1._append(df2, ignore_index=True))  # 重新编制索引
#    A  B
# 0  1  2
# 1  3  4
# 2  5  6
# 3  7  8

print()
print('# 不同列名 合并数据')
print(df1._append(df3))
#      A    B     C     D
# 0  1.0  2.0   NaN   NaN
# 1  3.0  4.0   NaN   NaN
# 0  NaN  NaN  11.0  12.0
# 1  NaN  NaN  13.0  14.0

print()
print('# pd.concat([s1,s2])可以将2个df或s连接起来')
s1 = pd.Series(['a', 'b'])
s2 = pd.Series(['c', 'd'])
print(pd.concat([s1, s2]))
# 0    a
# 1    b
# 0    c
# 1    d
print(pd.concat([s1, s2], ignore_index=True))
# 0    a
# 1    b
# 2    c
# 3    d

# print(pd.concat([df1,df3])) 测试可以合并 和 _append方法类似


print()
print('# 使用 pd.concat() 合并两个 Series，并增加一个多层索引')
s1 = pd.Series([1, 2])
s2 = pd.Series([3, 4])

result = pd.concat([s1, s2], keys=['s1', 's2'])
print(result)
# s1  0    1
#     1    2
# s2  0    3
#     1    4

# 使用 pd.concat() 合并两个 Series，并为多层索引指定名称
print()
print('# 使用 pd.concat() 合并两个 Series，并为多层索引指定名称')
result_named = pd.concat([s1, s2], keys=['s1', 's2'], names=['Series name', 'Row ID'])
print(result_named)
# Series name  Row ID
# s1           0         1
#              1         2
# s2           0         3
#              1         4

df1 = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))

# print(df1)
# print(df2)
# print(pd.concat([df1,df2]))
# 假设 df3 有不同的列顺序或额外的列
df3 = pd.DataFrame({'B': [9, 10], 'C': [11, 12]})

# 使用 pd.concat() 合并 df1 和 df3，不对列进行排序
result_df_nosort = pd.concat([df1, df3], sort=False)  # 默认就是sort=False 不排序
print(result_df_nosort)
#      A   B     C
# 0  1.0   2   NaN
# 1  3.0   4   NaN
# 0  NaN   9  11.0
# 1  NaN  10  12.0

print()
print('# 只合并 DataFrame 中的相同列')
# 使用 pd.concat() 合并 df1 和 df3，仅保留相同的列
result_df_inner = pd.concat([df1, df3], join='inner')  # 默认 join: str = "outer",
print(result_df_inner)
#     B
# 0   2
# 1   4
# 0   9
# 1  10

print()
print('# 按列连接 DataFrame')

df4 = pd.DataFrame({'C': [13, 14, 15], 'D': [16, 17, 18]})

result_df_axis = pd.concat([df1, df4], axis=1)  # ,axis=1 指示索引要 一一对应 多出的索引单独一行
print(result_df_axis)
#      A    B   C   D
# 0  1.0  2.0  13  16
# 1  3.0  4.0  14  17
# 2  NaN  NaN  15  18

# 无参数 axis = 1
#      A    B     C     D
# 0  1.0  2.0   NaN   NaN
# 1  3.0  4.0   NaN   NaN
# 0  NaN  NaN  13.0  16.0
# 1  NaN  NaN  14.0  17.0
# 2  NaN  NaN  15.0  18.0

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.4 添加修改')
print('\t5.4.11 删除')
print()
# update20240328
'''
对数据的修改、增加和删除在数据整理过程中时常发生。修改的情况一般是修改错误，
还有一种情况是格式转换，如把中文数字修改为阿拉伯数字。修改也会涉及数据的类型修改。

删除一般会通过筛选的方式，筛选完成后将最终的结果重新赋值给变量，达到删除的目的。
增加行和列是最为常见的操作，数据分析过程中会计算出新的指标以新列展示。
'''
# 小节注释
'''
删除有两种方法，一种是使用pop()函数。使用pop()，Series会删除
指定索引的数据同时返回这个被删除的值，DataFrame会删除指定列并
返回这个被删除的列。以上操作都是实时生效的。

以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

s = pd.Series([1, 2, 3, 4, 5])
print(s)
s1 = s.pop(3)
print(s1)  # 4
print(s)

# 删除Q1列
df.pop('Q1')
print(df)
#     name team  Q2  Q3   Q4
# 0  Liver    E  21  24   64
# ...

'''还有一种方法是使用反选法，将需要的数据筛选出来赋值给原变量，最终实现删除。'''
# 反选法
print(df.loc[:3, ['name', 'team']])
df1 = df.loc[:3, ['name', 'team']]
print(df1)
#     name team
# 0  Liver    E
# 1   Arry    C
# 2    Ack    A
# 3  Eorge    C

'''
▶ 删除操作是永久性的，一旦执行，原始数据就会丢失。如果后续操作需要这些数据，你需要先将其保存或复制。
▶ pop()只能一次删除一个元素。如果需要删除多个元素或进行更复杂的删除操作，可能需要使用其他方法，比如drop()。
'''

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.4 添加修改')
print('\t5.4.12 删除空值')
print()
# update20240328
'''
对数据的修改、增加和删除在数据整理过程中时常发生。修改的情况一般是修改错误，
还有一种情况是格式转换，如把中文数字修改为阿拉伯数字。修改也会涉及数据的类型修改。

删除一般会通过筛选的方式，筛选完成后将最终的结果重新赋值给变量，达到删除的目的。
增加行和列是最为常见的操作，数据分析过程中会计算出新的指标以新列展示。
'''
# 小节注释
'''
在一些情况下会删除有空值、缺失不全的数据，df.dropna可以执行这种操作：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

df = pd.DataFrame({'A': [1, np.nan, 3],
                   'B': [4, 5, np.nan],
                   'C': [7, np.nan, np.nan],
                   'D': [11, 12, 13]
                   })

print(df)
#      A    B    C   D
# 0  1.0  4.0  7.0  11
# 1  NaN  5.0  NaN  12
# 2  3.0  NaN  NaN  13

print()
print('# 一行中有一个缺失值就删除')
df_cleaned = df.dropna()
print(df_cleaned)
#      A    B    C   D
# 0  1.0  4.0  7.0  11

print()
print('# 只保留全有值的列')
df_cleaned_columns = df.dropna(axis='columns')
print(df_cleaned_columns)
#     D
# 0  11
# 1  12
# 2  13

print()
print('# 行或列全是空值 才删除')  # 行值全为空值 删除；  列值全为空值 不删除
df1 = pd.DataFrame({'A': [np.nan, np.nan, 3],
                    'B': [np.nan, 5, np.nan],
                    'C': [np.nan, np.nan, np.nan]
                    })
print(df1)
df_cleaned_all = df1.dropna(how='all')  # how='any' # 行值有任意一个空值 就删除
print(df_cleaned_all)
#      A    B   C
# 0  NaN  NaN NaN
# 1  NaN  5.0 NaN
# 2  3.0  NaN NaN

# df1.dropna(how='all')
#      A    B   C
# 1  NaN  5.0 NaN
# 2  3.0  NaN NaN

print()
print('# 同一行至少有2个非空值 保留 否则删除')
df2 = pd.DataFrame({'A': [np.nan, np.nan, 3],
                    'B': [np.nan, 5, np.nan],
                    'C': [np.nan, np.nan, 6]
                    })

print(df2)
df_cleaned_thresh = df2.dropna(thresh=2)
print(df_cleaned_thresh)
#      A    B    C
# 0  NaN  NaN  NaN
# 1  NaN  5.0  NaN
# 2  3.0  NaN  6.0

#      A   B    C
# 2  3.0 NaN  6.0

# print(df)
# df_cleaned_thresh_1 = df.dropna(thresh=2)
# print(df_cleaned_thresh_1)

print()
print('# 删除并且是替换生效')
df.dropna(inplace=True)
print(df)
#      A    B    C   D
# 0  1.0  4.0  7.0  11


# 创建一个含有缺失值的 DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', np.nan],
    'age': [24, 27, 22, 31],
    'toy': [np.nan, 'Batmobile', 'Bullwhip', 'Barbie']
})

print(df)
#       name  age        toy
# 0    Alice   24        NaN
# 1      Bob   27  Batmobile
# 2  Charlie   22   Bullwhip
# 3      NaN   31     Barbie
# 仅在 'name' 和 'toy' 列中检查缺失值，如果存在缺失值则删除对应的行
df_cleaned_subset = df.dropna(subset=['name', 'toy'])
print(df_cleaned_subset)
#       name  age        toy
# 1      Bob   27  Batmobile
# 2  Charlie   22   Bullwhip
