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
csv_file = 'E:/bat/input_files/sales_january_2014.csv'

path = 'E:/bat/output_files/pandas_read_csv_20240118.csv'

# 忽略特定的警告 | 当遇到 openpyxl 中的 UserWarning 类型的警告时，它们将不会被打印出来。
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.5 数据选择')
print('\t4.5.1 选择列')
print()
# update20240301
'''
除了上文介绍的查看DataFrame样本数据外，还需要按照一定的条件对数据进行筛选。
通过Pandas提供的方法可以模拟Excel对数据的筛选操作，也可以实现远比Excel复杂的查询操作。
本节将介绍如何选择一列、选择一行、按组合条件筛选数据等操作，
让你对数据的操作得心应手，灵活地应对各种数据查询需求。

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print()
print('# 选取Dataframe中的1列')
print(df['name'])  # 方法1
print(df.name)  # 方法2
print(df.Q1)
print(type(df.Q1))
print(type(df))
'''
注意：
这两种操作方法效果是一样的，切片（[]）操作比较通用，
当列名为一个合法的Python变量时，可以直接使用点操作（.name）为属性去使用。
如列名为1Q、my name等，则无法使用点操作，因为变量不允许以数字开头或存在空格，
如果想使用可以将列名处理，如将空格替换为下划线、增加字母开头前缀，如s_1Q、my_name。
'''

print()
print('----------------------------------------------------')
print('\t4.5 数据选择')
print('\t4.5.2 切片[]')
print('----------------------------------------------------')
print()

'''
我们可以像列表那样利用切片功能选择部分行的数据，但是不支持仅索引一条数据：
'''

print(df[:2])  # 前2行数据
#     name team  Q1  Q2  Q3  Q4
# 0  Liver    E  89  21  24  64
# 1   Arry    C  36  37  37  57
print(df[4:10])  # 索引号为4~10的数据 如果没有索引10 默认从第4到尾部所有数据
print(df[:])  # 所有数据 一般不这么用
print(df[0:5:2])  # 索引内 按照步长取值
#     name team  Q1  Q2  Q3  Q4
# 0  Liver    E  89  21  24  64
# 2    Ack    A  57  60  18  84
# 4    Oah    D  65  49  61  86
print(df[::-1])  # 反转顺序
print(df[::-2])  # 反转顺序后 间隔步长2取值
#     name team   Q1  Q2  Q3   Q4
# 5   Rick    B  100  99  97  100
# 3  Eorge    C   93  96  71   78
# 1   Arry    C   36  37  37   57
# print(df[2]) # 报错！

'''
需要注意的是，切片的逻辑和Python列表的逻辑一样，不包括右边的索引值。
如果切片里是一个列名组成的列表，则可筛选出这些列：
'''

print()
# print(df[['name','Q4']])
print(df[['name', 'Q4']].head(2))
#     name  Q4
# 0  Liver  64
# 1   Arry  57

'''需要区别的是，如果只有一列，则会是一个DataFrame：'''
print()
print('# 需要区别的是，如果只有一列，则会是一个DataFrame：')
print(df[['name']])  # 选择一列，返回DataFrame，注意与下例进行区分
print(type(df[['name']]))
# <class 'pandas.core.frame.DataFrame'>
print()
print(df['name'])
print(type(df['name']))
# <class 'pandas.core.series.Series'>

'''切片中支持条件表达式，可以按条件查询数据，5.1节会详细介绍。'''

print()
print('----------------------------------------------------')
print('\t4.5 数据选择')
print('\t4.5.3 按轴标签.loc')
print('----------------------------------------------------')
print()

'''
df.loc的格式是df.loc[<行表达式>, <列表达式>]，如列表达式部分不传，将返回所有列，
Series仅支持行表达式进行索引的部分。loc操作通过索引和列的条件筛选出数据。
如果仅返回一条数据，则类型为Series
'''
print('# 代表索引，如果是字符，需要加引号')
print(df.loc[0])  # 选择索引为0的行
# name    Liver
# team        E
# Q1         89
# Q2         21
# Q3         24
# Q4         64
# Name: 0, dtype: object
print(type(df.loc[0]))  # <class 'pandas.core.series.Series'>

# print(df.loc[5]) # 选择索引为5的行 || df.loc[8] 超出索引会报错！
print()
print('# 索引为name')
print(df.set_index('name').loc['Rick'])  # 单列
# team      B
# Q1      100
# Q2       99
# Q3       97
# Q4      100
# Name: Rick, dtype: object
print()
print(df.set_index('name').loc[['Rick', 'Ack']])  # 多列
#      team   Q1  Q2  Q3   Q4
# name
# Rick    B  100  99  97  100
# Ack     A   57  60  18   84
print()
print('# 指定索引为0，3，5的行')
print(df.loc[[0, 3, 5]])  # KeyError: '[10] not in index'

print()
print('# 为真的列显示，隔一个显示一个')
# 创建一个布尔型数组，长度与df的行数相同
# 这里我们使用列表推导式生成一个隔行为True的布尔型列表
bool_index = [True if i % 2 == 0 else False for i in range(len(df))]
print(bool_index)
print([False, True] * 3)  # 效果同上
print(df.loc[bool_index])
#     name team  Q1  Q2  Q3  Q4
# 0  Liver    E  89  21  24  64
# 2    Ack    A  57  60  18  84
# 4    Oah    D  65  49  61  86
print(df[[False, True] * 3])  # 效果同上
#     name team   Q1  Q2  Q3   Q4
# 1   Arry    C   36  37  37   57
# 3  Eorge    C   93  96  71   78
# 5   Rick    B  100  99  97  100

print()
print('# 附带列筛选，必须有行筛选')
'''
附带列筛选，必须有行筛选。
列部分的表达式可以是一个由希望筛选的表名组成的列表，
也可以是一个用冒号隔开的切片形式，
来表示从左到右全部包含，左侧和右侧可以分别省略，表示本侧所有列。
'''
print(df.loc[0:5, ['name', 'team']])  # 可以是一个由希望筛选的表名组成的列表

print()
print('# 前3行，Q1和Q2两列')
print(df.loc[0:2, ['Q1', 'Q2']])  # 前3行，Q1和Q2两列
#    Q1  Q2
# 0  89  21
# 1  36  37
# 2  57  60
print(df.loc[:, ['Q1', 'Q2']])  # 所有行，Q1和Q2两列
print(df.loc[:, 'Q1':])  # 所有行，Q1及其后面的所有列 || 列不用中括号[]
print(df.loc[:, :])  # 所有内容
'''以上方法可以混用在行和列表达式，.loc中的表达式部分支持条件表达式，可以按条件查询数据，后续章节会详细介绍。'''

print()
print('----------------------------------------------------')
print('\t4.5 数据选择')
print('\t4.5.4 按数字索引.iloc')
print('----------------------------------------------------')
print()

'''
与loc[]可以使用索引和列的名称不同，
利用df.iloc[<行表达式>, <列表达式>]格式可以使用数字索引（行和列的0～n索引）进行数据筛选，
意味着iloc[]的两个表达式只支持数字切片形式，其他方面是相同的。
'''
print('# 代表索引')
print(df.loc[0])  # 选择索引为0的行
print(df.iloc[0])  # 选择索引为0的行 | 结果同上

print()
print(df.iloc[0:3])  # 前3行
print(df.iloc[:3])  # 前3行 结果同上
print(df.iloc[:])  # 所有数据
print(df.loc[:])  # 同上

print()
print('----------')
print(df.iloc[2:5:2])  # 步长为2
#   name team  Q1  Q2  Q3  Q4
# 2  Ack    A  57  60  18  84
# 4  Oah    D  65  49  61  86
print()
print('----------')
print(df.iloc[:3, [0, 1]])  # 前3行 & 前2列
print(df.iloc[:3, :])  # 前3行 & 所有列
print(df.iloc[:3, :-2])  # 前3行 &  从右向左起 第3列到最左列
#     name team  Q1  Q2
# 0  Liver    E  89  21
# 1   Arry    C  36  37
# 2    Ack    A  57  60
'''以上方法可以混用在行和列表达式，.iloc中的表达式部分支持条件表达式，可以按条件查询数据，后续章节会详细介绍。'''

print()
print('----------------------------------------------------')
print('\t4.5 数据选择')
print('\t4.5.5 取具体值.at/.iat')
print('----------------------------------------------------')
print()

'''
如果需要取数据中一个具体的值，就像取平面直角坐标系中的一个点一样，可以使用.at[]来实现。
.at类似于loc，仅取一个具体的值，结构为df.at[<索引>,<列名>]。
如果是一个Series，可以直接值入索引取到该索引的值。

'''
print('# 注：索引是字符，需要加引号')
print(df.at[4, 'Q1'])  # 65
print(type(df.at[4, 'Q1']))  # <class 'numpy.int64'>

print()
print(df.set_index('name').at['Rick', 'Q1'])  # 100 索引是name

print(df.at[0, 'name'])  # Liver
print(df.loc[0].at['name'])  # Liver | 同上

print()
print('# 指定列的值对应其他列的值')
print(df.set_index('name').at['Liver', 'team'])  # E
print(df.set_index('name').team.at['Rick'])  # B

print()
print('# 指定列的对应索引的值')
print(df.team.at[2])  # A

print()
print('# iat和iloc一样，仅支持数字索引：')
print(df.iat[4, 2])  # 65
print(df.loc[0].iat[1])  # E

print()
print('----------------------------------------------------')
print('\t4.5 数据选择')
print('\t4.5.6 获取数据.get()')
print('----------------------------------------------------')
print()
# update20240304
'''
.get可以做类似字典的操作，如果无值，则返回默认值（下例中是0）。
格式为df.get(key, default=None)，如果是DataFrame，key需要传入列名，返回的是此列的Series；
如果是Series，需要传入索引，返回的是一个定值：

'''
print()
print(df.get('name', 0))  # 返回name列
print(df.get('nameXXX', 0))  # 0, 返回默认值

print()
print('Series传索引返回具体值')
s = pd.Series([9, 4, 6, 7, 9])
print(s.get(3, 0))  # out:7
print()
s1 = pd.Series([9, 4, 6])
print(s1.get(3, 0))  # out:0

print()
print(df.name.get(5, 0))  # out:Rick

print()
print('----------------------------------------------------')
print('\t4.5 数据选择')
print('\t4.5.7 数据截取.truncate()')
print('----------------------------------------------------')
print()
# update20240304
'''
df.truncate()可以对DataFrame和Series进行截取，可以将索引传入before和after参数，将这个区间以外的数据剔除。

'''
print()
print(df.truncate(before=2, after=4))
#     name team  Q1  Q2  Q3  Q4
# 2    Ack    A  57  60  18  84
# 3  Eorge    C  93  96  71  78
# 4    Oah    D  65  49  61  86

print()
s = pd.Series([9, 4, 6, 7, 9])
print(s)
print(s.truncate(before=2))  # 包含索引2
print(s.truncate(after=3))  # 包含索引3

print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.5 数据选择')
print('\t4.5.8 索引选择器')
print()
# update20240304
'''
pd.IndexSlice是一个专门的索引选择器，它的使用方法类似df.loc[]切片中的方法，
常用在多层索引中，以及需要指定应用范围（subset参数）的函数中，特别是在链式方法中。

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print(df.head(3).loc[pd.IndexSlice[:, ['Q1', 'Q2']]])
#    Q1  Q2
# 0  89  21
# 1  36  37
# 2  57  60

print()
print('变量化使用')
s = pd.Series([9, 4, 6, 7, 9])
idx = pd.IndexSlice
print(df.loc[idx[:, ['Q1', 'Q2']]])
#     Q1  Q2
# 0   89  21
# 1   36  37
# ....
print(df.loc[idx[:, 'Q1':'Q4']])
print()

print('# 创建一个多层索引的DataFrame')
# 没看懂
df1 = pd.DataFrame({
    ('Q1', 'A'): [1, 2, 3, 4, 5],
    ('Q2', 'B'): [6, 7, 8, 9, 10],
    ('Q3', 'A'): [11, 12, 13, 14, 15],
    ('Q4', 'B'): [16, 17, 18, 19, 20]
}).T

print(df1)
# 创建IndexSlice对象
idx = pd.IndexSlice

print()
# 使用IndexSlice来选择数据
print(df1.loc[idx[:, 'Q1':'Q4']])
print(df1.loc[idx[:, 'Q1':'Q4'], :])
# Empty DataFrame
# Columns: []
# Index: [(Q1, A), (Q2, B), (Q3, A), (Q4, B)]
# Empty DataFrame
# Columns: [0, 1, 2, 3, 4]
# Index: []


# 还可以按条件查询创建复杂的选择器，以下是几个案例：
print()
print('# 创建复杂条件选择器')
selected = df.loc[(df.team == 'A') & (df.Q1 > 50)]
print(selected)

print()
print(selected.index)
idxs = pd.IndexSlice[selected.index, 'name']
print()
print(idxs)
print('# 应用选择器')
print(df.loc[idxs])
# selected = df.loc[(df.team=='A') & (df.Q1 > 90)]
# Empty DataFrame
# Columns: [name, team, Q1, Q2, Q3, Q4]
# Index: []
#
# Index([], dtype='int64')
#
# (Index([], dtype='int64'), 'name')
# 应用选择器
# Series([], Name: name, dtype: object)
# -----------------------------------------------
# selected = df.loc[(df.team=='A') & (df.Q1 > 50)]
#   name team  Q1  Q2  Q3  Q4
# 2  Ack    A  57  60  18  84
#
# Index([2], dtype='int64')
#
# (Index([2], dtype='int64'), 'name')
# # 应用选择器
# 2    Ack
# Name: name, dtype: object

print()
print('# 选择这部分区域加样式')
# 原书本例子搞不懂 这是ChatGPT提供的示例： 可以运行正常

# 创建一个简单的DataFrame
df = pd.DataFrame({
    'team': ['A', 'A', 'B', 'B'],
    'Q1': [88, 92, 95, 85],
    'name': ['John', 'Alice', 'Bob', 'Eve']
})


# 定义样式函数
def style_fun(val):
    color = 'yellow' if val > 90 else None
    return f'background-color: {color}'


# 应用样式函数到DataFrame
styled_df = df.style.applymap(style_fun, subset=['Q1'])

# 将样式化的DataFrame保存为HTML文件
html = styled_df.to_html()
with open('styled_dataframe.html', 'w') as f:
    f.write(html)

# 现在你可以打开 'styled_dataframe.html' 文件以查看样式化的DataFrame
# 文件路径： C:/Users/liuxin05/PycharmProjects/pythonProject/venv/foundations_for_analytics/first_direc/styled_dataframe.html
