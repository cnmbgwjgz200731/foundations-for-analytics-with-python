import pandas as pd
import sys
import time

print()
print('1.3 Pandas快速入门')
print('\t1.3.3 读取数据')
print()

# 以下两种效果一样，如果是网址，它会自动将数据下载到内存
# df = pd.read_excel('https://www.gairuo.com/file/data/dataset/team.xlsx')
# df = pd.read_excel('team.xlsx') # 文件在notebook文件同一目录下
# 如果是CSV，使用pd.read_csv()，还支持很多类型的数据读取

print()
print('1.3 Pandas快速入门')
print('\t1.3.4 查看数据')
print()

# input_file = sys.argv[1]
# output_file = sys.argv[2]

input_file = 'E:/bat/input_files/supplier_data.csv'
# output_file = 'E:\桌面08\远程\\11pandas_out_20230808027.csv'

df = pd.read_csv(input_file, index_col=False)

# 查看数据
print(df)  # 查看全部数据 数据量大时，中间数据会自动隐藏 显示头尾部分行数据
print(df.head())  # 查看前5条，括号里可以写明你想看的条数
print(df.tail())  # 查看尾部5条
print(df.sample(5))  # 随机查看5条

print()
print('1.3 Pandas快速入门')
print('\t1.3.5 验证数据')
print()
# 验证数据
print(df.shape)  # (12, 5) 查看行数和列数
print(df.info())  # 查看索引、数据类型和内存信息
print(df.describe())  # 查看数值型列的汇总统计
print(df.dtypes)  # 查看各字段类型
print(df.axes)  # 显示数据行和列名
print(df.columns)  # 列名

# 建立索引
print()
print('1.3 Pandas快速入门')
print('\t1.3.6 建立索引')
print()
df.set_index('Supplier Name', inplace=True)
print(df)

# update20231214 个人测试
# 使用pandas的columns属性获取所有的列名，然后对每个列名进行处理：修改列名为小写，并去除空格
# 也可以去除列名中的单引号和问号等 本行代码没有展示
print(df.columns)
df.columns = [heading.lower() for heading in df.columns.str.replace(' ', '_')]
print()
# print(df.columns)
# print(df.sample(5))

print()
print('1.3 Pandas快速入门')
print('\t1.3.7 数据选取')
print()

# 选取列
# 列名称取列
# df.set_index('supplier_name',inplace=True)
# print(df.sample(6))
print(df.loc[:, ['supplier_name']])  # 取单列
print(df.supplier_name)  # 取单列 效果同上
print(df['supplier_name'])  # 取单列 效果同上
# print(df['supplier_name','cost']) # 取多列 运行报错！
print(df.loc[:, ['supplier_name', 'cost']])  # 取多列
print(df[['supplier_name', 'cost']])  # 取多列 效果同上
# print(df.supplier_name)

# 指定索引值取列 | 个人测试 运行成功！
print(df.iloc[:, 1])  # 取单列 | 返回不含列名称
print(df.iloc[:, :3])  # 取多列 | 返回前3列 不含索引值为3的列值
print(df.iloc[:, 0:3])  # 取多列 效果同上
print(df.iloc[:, 0:6:2])  # 取多列 且间隔一列取列值 | 6大于实际列数 也可以跑全部列值

# update20231218
# 选取行
# 用指定索引选取
df.set_index('supplier_name', inplace=True)
print(df[df.index == 'Supplier Z'])  # 指定姓名 || 必须要先设置索引

# 用自然索引选择，类似列表的切片
print(df[0:3])  # 取前三行
print(df[0:10:2])  # 在前10个中每2个取1个
print(df.iloc[:10, :])  # 取前10行

# 指定行和列
# 同时给定行和列的显示范围：
df.set_index('supplier_name', inplace=True)
print(df.loc['Supplier Z', 'part_number':'purchase_date'])  # 只看Supplier Z 的某几列数据 || 必须设置索引列
print(df.loc['Supplier X':'Supplier Z', 'invoice_number':'purchase_date'])  # 指定行区间

print(df.loc['50-9505':'920-4806', 'invoice_number':'purchase_date'])  # 不设置索引 指定行区间 运行成功

# 条件选择
# 按一定的条件显示数据：
# 单一条件
print(df[df.invoice_number == '001-1001'])  # 文本条件
print(df[df['cost'].str.strip('$').astype(float) > 615.0])  # 数值条件 | 验证df.cost 与 df['cost'] 是否相同
print(df[df['cost'].str.replace('$', '').astype(float) > 1000.0])  # 数值条件 效果同上
print(df[df.cost.str.strip('$').astype(float) > 1000.0])  # 数值条件 效果同上 || 验证replace和strip函数对比
print(df[df['cost'].str.strip('$').astype(float) >= 750])  # 数值条件 效果同上 || 验证数值可以不用加小数点

df.set_index('supplier_name', inplace=True)
print(df[df.index == 'Supplier X'])  # 指定索引 即原数据中的supplier_name || 必须要设置索引才能运行成功！

# 组合条件
print(df[(df.cost.str.strip('$').astype(float) > 1000.0) & (df.part_number.astype(str) == '3321')])  # and关系
print(df[(df.invoice_number == '001-1001') | (df.part_number > 7000)])  # or 关系
print(df[df.part_number.astype(str) == '3321'])  # part_number数据类型为int64 转化为文本 直接.str报错
print(df[df['part_number'].astype(str) == '3321'])  # 单条件 效果同上
print(df[df['part_number'].astype(str) == '3321'].loc[df.invoice_number == '920-4805'])  # 多重筛选

print()
print('1.3 Pandas快速入门')
print('\t1.3.8 排序')
print()

# 文本类型 排序
print(df.sort_values(by='part_number'))  # 按part_number列数据升序排列
print(df.sort_values(by='part_number', ascending=False))  # 降序

# 处理cost列 转文本为数值类型后 排序
df['cost'] = df.cost.str.strip('$').astype(float)  # 转换数值类型 并赋值
print(df.sort_values(by='cost'))  # 升序
print(df.sort_values(by='cost', ascending=False))  # 降序
# 多字段排序
print(df.sort_values(['cost', 'part_number'], ascending=[True, False]))  # cost升序 part_number降序

print()
print('1.3 Pandas快速入门')
print('\t1.3.9 分组聚合')
print()

print(df.groupby('cost').sum())  # 运行成功
# print(df.groupby('cost').mean()) # 运行报错
print(df.groupby('supplier_name').sum())  # 运行成功
# print(df.groupby('supplier_name').mean()) # 运行报错
print(df.groupby('supplier_name').count())  # 运行成功
print(df.groupby('purchase_date').count())  # 运行成功 || 测试各个字段均可运行成功 | 分组字段好像成为索引列了
# 不同的列不同的计算方法
# 按supplier_name列分组 统计
print(df.groupby('supplier_name').agg({'cost': sum,
                                       'cost': 'count',
                                       'cost': 'mean',
                                       'cost': max
                                       }))  # 运行正常|| 仅显示计算最后一行 即'cost':max

# 仅显示2列计算列 mean列和max列
print(df.groupby('supplier_name').agg({'cost': sum,
                                       'cost': 'count',
                                       'cost': 'mean',
                                       'part_number': max}))

# 按照supplier_name分组 4列显示
# supplier_name列分组列，cost列按照最大值，part_number列显示汇总值，supplier_name列显示计算列
print(df.groupby('supplier_name').agg({'invoice_number': max,
                                       'part_number': sum,
                                       'purchase_date': 'count',
                                       'cost': 'mean'
                                       # 添加后 运行报错 TypeError: Could not convert 001-1001001-1001001-1001001-1001 to numeric
                                       }))

print()
print('1.3 Pandas快速入门')
print('\t1.3.10 数据转换')
print()

print(df.groupby('supplier_name').sum())
# 对聚合后的数据翻转 类似Excel中的数据转置
print(df.groupby('supplier_name').sum().T)
print()
print(df.groupby('supplier_name').count())
print(df.groupby('supplier_name').count().T)
print()
print(df.groupby('supplier_name').max())
print(df.groupby('supplier_name').max().T)
print()
# 类似数据透视表 多字段分组 || 但不完全一样
print(df.groupby('supplier_name').sum().stack())
# 在上行代码的基础上 进行 数据转置
print(df.groupby('supplier_name').sum().unstack())
# unstack_out = df.sort_values(by='cost')
# print(stack_out)
# print(unstack_out)


print()
print('1.3 Pandas快速入门')
print('\t1.3.11 增加列')
print()
# update20231222
# 增加一个固定值的列
df['one'] = 1
# 增加汇总列 || 多次增加相同的字段 以最后一次出现的为准 如仅显示df['total'] = df.cost * 2 的结果
df['total'] = df.cost + df.part_number
df['total'] = df.cost * 2
# 将计算得到的结果赋值给新列
df['total'] = df.loc[:, 'part_number':'cost'].apply(lambda x: sum(x), axis=1)
# 可以把所有为数字的列相加 | 运行报错 TypeError: can only concatenate str (not "int") to str
# df['total'] = df.sum(axis=1)
# 增加平均值列
df['avg'] = df.total / 4

print()
print('1.3 Pandas快速入门')
print('\t1.3.12 统计分析')
print()
# update20231222


print(df.max())  # 返回每一列的最大值
print(df.min())  # 返回每一列的最小值
print(df.count())  # 返回每一列中非空值的个数

# 下列几行代码，仅取数值列，即df = df.iloc[:,[2,3]]
# 否则运行报错 报错：ValueError: could not convert string to float: 'Supplier X'
df = df.iloc[:, [2, 3]]
print(df.mean())  # 返回所有列的均值 | 仅取数值列，df = df.iloc[:,[2,3]]否则运行报错
print(df.mean(1))  # 返回所有行的均值 | 仅取数值列，df = df.iloc[:,[2,3]]否则运行报错
print(df.corr())  # 返回列与列之间的相关系数 |
print(df.median())  # 返回每一列的中位数 | 报错 TypeError: could not convert string to float: 'Supplier X'
print(df.std())  # 返回每一列的标准差 | TypeError: could not convert string to float: 'Supplier X'
print()
print(df.var())  # 返回每一列的方差
print()
print(df.mode())

print()
print('1.3 Pandas快速入门')
print('\t1.3.13 绘图')
print()
# update20231222
df.set_index('supplier_name', inplace=True)

print(df['part_number'].plot())
df['part_number'].plot()
df.loc['Supplier X', 'part_number':'cost'].plot()  # 折线图
df.loc['Supplier X', 'part_number':'cost'].plot.bar()  # 柱状图
df.loc['Supplier X', 'part_number':'cost'].plot.barh()  # 横向柱状图
df.groupby('supplier_name').count().T.plot()
print(df.groupby('supplier_name').count().T.plot())
print(df.groupby('supplier_name').count().cost.plot.pie())  # 饼图

# dft文件设置
# print(dft)
dft.set_index('name', inplace=True)
dft['Q1'].plot()
print(dft)
print(dft.loc['Rick', 'Q1':'Q4'])
dft.loc['Rick', 'Q1':'Q4'].plot()
dft.loc['Rick', 'Q1':'Q4'].plot.bar()
dft.loc['Rick', 'Q1':'Q4'].plot.barh()

# 必须设置name列为索引 否则运行后报错
print(dft.groupby('team').sum().T)
dft.groupby('team').sum().T.plot()  # 各Team四个季度总成绩趋势
dft.groupby('team').count().Q1.plot.pie()  # 饼图 各组人数对比

plt.show()
# ------------------------------------------
print()
print('1.3 Pandas快速入门')
print('\t1.3.14 导出')
print()
# update20231225
# 导出Excel或csv文件

output_file = 'E:/bat/output_files/pandas_out_20230814047.csv'
df.to_csv(output_file, index=False)  # 导出csv文件
df.to_excel('E:/bat/output_files/pandas_out_20231225013.xlsx', index=False)  # 导出xlsx文件
df.to_excel('E:/bat/output_files/pandas_out_20231225012.xlsx', index='supplier_name')  # 导出xlsx文件 效果同上

print()
print('2.3 Numpy')
print('\t2.3.3 创建数据')
print()
# update20231227

import numpy as np

lx = np.array([1, 2, 3])

# 一维数组
print(np.array([1, 2, 3]))  # out：[1 2 3]
print(np.array((1, 2, 3)))  # 同上

# 二维数组
print(np.array(((1, 2), (2, 1))))
print(np.array(([1, 2], [2, 1])))  # 同上
'''
out:
[[1 2]
 [2 1]]
'''

# 常见的数据生成函数
# 生成10个数字,不包括10，步长为1
print(np.arange(10))  # 输出 [0 1 2 3 4 5 6 7 8 9]
print(np.arange(3, 10, 0.1))  # 从3到9.9，步长为0.1

# 生成均匀的5个值，False不包括终值3.0 True包括终值3.0
print(np.linspace(2.0, 3.0, num=5, endpoint=False))  # 输出[2.  2.2 2.4 2.6 2.8]
print(np.linspace(2.0, 3.0, num=5, endpoint=True))  # 输出[2.   2.25 2.5  2.75 3.  ]

# 返回1个6*4的随机数组 ， 浮点型
print(np.random.randn(6, 4))  # 输出 可能 -10 =< x < 10

# 指定范围、指定形状的数组，整形
print(np.random.randint(2, 9, size=(3, 3)))  # 个人测试
print(np.random.randint(3, 7, size=(2, 4)))  # 生成1个2x4的数组，整数型，元素值在3~7之间整数值 含3 不含7
print(type(np.random.randint(3, 7, size=(2, 4))))  # <class 'numpy.ndarray'>
'''
输出：
[[6 6 6 3]
 [4 5 5 3]]
'''

# 创建值为0的数组
print(np.zeros(6))  # 6个浮点0.  输出：[0. 0. 0. 0. 0. 0.]
print(np.zeros((5, 6), dtype=int))  # 5x6 整数型0
print(np.zeros((5, 6)))  # 5x6 浮点数0.
print(np.ones(4))  # 输出：[1. 1. 1. 1.]
print(np.ones((3, 3), dtype=int))  # 输出一个3x3的数组 元素值为整数1
print(np.empty(4))  # 输出：[1. 1. 1. 1.]

# 创建一份和目标结构相同的0值数组
print(np.zeros_like(np.arange(6)))  # 输出：[0 0 0 0 0 0]
print(np.ones_like(np.arange(6)))  # 输出：[1 1 1 1 1 1]
print(np.empty_like(np.arange(6)))  # 输出：[0 1 2 3 4 5]

print()
print('2.3 Numpy')
print('\t2.3.4 数据类型')
print()
# update20231228

print(np.int64)
print(np.float32)  # <class 'numpy.float32'>
print(np.float64)  # <class 'numpy.float64'>
print(np.complex_)  # <class 'numpy.complex128'>
print(np.complex128)  # 同上
print(np.bool_)
print(np.object_)
print(np.str_)
print(np.unicode_)  # <class 'numpy.str_'>
print(np.NaN)  # 输出：nan | np.float的子类型
print(np.NAN)  # 同上
print(np.nan)  # 同上

print()
print('2.3 Numpy')
print('\t2.3.5 数组信息')
print()
# update20231228

n = np.array([1, 2, 3])
m = np.array(((1, 2), (2, 1)))

print(n.shape)  # 数组的形状，返回值是1个元组 | out：(3,)
print(m.shape)  # out：(2, 2)
# print(n.shape = (4,1)) # 改变形状 | 运行失败
print(m.shape == (4, 1))  # out：False

a = m.reshape((1, 4))  # 改变原数组的形状，创建1个新数组
print(a)
print(n.dtype)  # 数据类型 | out：int32
print(m.dtype)  # 同上 | out：int32
print(n.ndim)  # 维度数 | out:1
print(m.ndim)  # 同上 | out:2
print(n.size)  # 元素数 | out:3
print(m.size)  # 元素数 | out:4
print(np.typeDict)  # np的所有数据类型

for i, j in np.typeDict.items():
    print(i, j)

print()
print('2.3 Numpy')
print('\t2.3.6 统计计算')
print()
# update20231228

# 两个数组间的操作采用行列式的运算规则
a = np.array([10, 20, 30, 40])
b = np.array([1, 2, 3, 4])
c = np.array(((2, 5, 4), (12, 13, 46)))

print(a[:3])  # 支持类似列表的切片 |out：[10 20 30]

print(a + b)  # 矩阵相加 | out:[11 22 33 44]
print(a - 1)  # out:[ 9 19 29 39]
print(np.sin(a))
print(4 * np.sin(a))

# 以下是一些数学函数的例子，还支持非常多的数学函数
print(a.max())  # out:40
print(a.min())  # out:10
print(a.sum())  # out:100
print(a.std())  # out:11.180339887498949
print(a.all())  # out:True
print(a.cumsum())  # out:[ 10  30  60 100]

'''
# print(b.sum(axis=1)) # 运行报错
出现这个错误的原因是因为在尝试对一个一维数组b使用axis=1进行求和操作。
一维数组没有第二个轴（axis 1），因为它只有一个维度（axis 0）。
numpy.AxisError表明你尝试访问的轴超出了数组的维度范围。
在这个例子中，数组b是一个一维数组，所以你只能沿着axis=0（即数组的唯一维度）进行操作。
解决这个问题的方法是，如果你想对数组b的所有元素求和，你应该使用axis=0或者不指定axis参数，因为默认就是沿着所有的轴求和。
'''
print(b.sum())  # out:10
print(b.sum(axis=0))  # out:10
print(c)
print(c.sum(axis=1))  # 多维可以指定方向 || out:[11 71]
print(c.sum(axis=0))  # out:[14 18 50]
print(c.sum())  # out:82

import numpy as np
import pandas as pd

print()
print('2.5 pandas生成数据')
print('\t2.5.2 创建数据')
print()
# update20231229

'''
使用pd.DataFrame()可以创建一个DataFrame，然后用df作为变量赋值给它。
df是指DataFrame，也已约定俗成，建议尽量使用。
'''
df = pd.DataFrame({'国家': ['中国', '美国', '日本'],
                   '地区': ['亚洲', '北美', '亚洲'],
                   '人口': [13.97, 3.28, 1.26],
                   'GDP': [14.34, 21.43, 5.08]
                   })
print(df)
'''
out:
   国家  地区     人口    GDP
0  中国  亚洲  13.97  14.34
1  美国  北美   3.28  21.43
2  日本  亚洲   1.26   5.08
'''

# df.set_index('国家',inplace=True)
# print(df['人口'])
# print(df[df.index == '中国'])
# print(df[df['地区'] == '亚洲'])

df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float'),
                    'D': np.array([3] * 4, dtype='int32'),  #
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'
                    })

print(df2)
print(df2.dtypes)

# 单独创建series：
ab = pd.Series([14.34, 21.43, 5.08], name='gdp')
# 指定index | 必须与data的长度一致，不指定默认从0开始
s = pd.Series([14.34, 21.43, 5.08], name='gdp', index=[4, 5, 6])
print(s)

# s.to_excel('E:/bat/output_files/pandas_out_20231226023.xlsx',index=True)
print(type(df))  # out:<class 'pandas.core.frame.DataFrame'>
print(type(s))  # out:<class 'pandas.core.series.Series'>

print()
print('2.5 pandas生成数据')
print('\t2.5.3 生成series')
print()

# series的创建方式如下：
# s = pd.Series(data,index=index)

# (1) 使用列表和元组
print(pd.Series(['a', 'b', 'c', 'd', 'e']))
print(pd.Series(('a', 'b', 'c', 'd', 'e')))  # 输出 同上

# (2) 使用ndarray
# 创建索引分别为'a','b','c','d','e' 的5个随机浮点数数组组成的series
sa = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
print(sa)
print(sa.index)  # 查看索引 | out：Index(['a', 'b', 'c', 'd', 'e'], dtype='object')
# 未指定索引
sb = pd.Series(np.random.randn(5))
print(sb)
print(sb.index)  # out:RangeIndex(start=0, stop=5, step=1)

# (3) 使用字典
print()
da = {'b': 1, 'a': 0, 'c': 2}
print(pd.Series(da))
print(pd.Series(da).index)

# 如果指定索引，则会按索引顺序，如有无法与索引对应的值，会产生缺失值
print()
print(pd.Series(da, index=['b', 'c', 'd', 'a']))

# (4) 使用标量 || 即定值
'''
对于一个具体的值，如果不指定索引，则其长度为1；
如果指定索引，则其长度为索引的数量，每个索引的值都是它
'''

print()
# 不指定索引
print(pd.Series(5.))
# 指定索引
print(pd.Series(5., index=['a', 'b', 'c', 'd', 'e']))

import numpy as np
import pandas as pd

print()
print('2.5 pandas生成数据')
print('\t2.5.4 生成dataframe')
print()
# update20231229

'''
使用pd.DataFrame()可以创建一个DataFrame，然后用df作为变量赋值给它。
df是指DataFrame，也已约定俗成，建议尽量使用。

dataframe是二维数据结构，数据以行与列的形式排列，表达一定的数据意义。
dataframe的形式类似于csv、excel和sql的结果表，有多个数据列，由多个series组成。
dataframe最基本的定义格式如下：
df = pd.DareFrame(data,index=None,columns=None)

'''

# 1、字典
# 1.1 未指定索引

d = {'国家': ['中国', '美国', '日本'],
     '人口': [13.97, 3.28, 1.26]
     }
df = pd.DataFrame(d)
print(df)
'''
out:
   国家     人口
0  中国  13.97
1  美国   3.28
2  日本   1.26
'''

# 1、字典 指定索引
print()
df = pd.DataFrame(d, index=['a', 'b', 'c'])
print(df)
'''
out:
   国家     人口
a  中国  13.97
b  美国   3.28
c  日本   1.26
'''

# 2、 series组成的字典
print()
d = {'x': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
     'y': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])
     }
df = pd.DataFrame(d)
print(df)

# 3、字典组成的列表
print()
# 定义一个字典列表
data = [{'x': 1, 'y': 2}, {'x': 3, 'y': 4, 'z': 5}]
# 生成DataFrame对象
print(pd.DataFrame(data))
# 指定索引
print()
print(pd.DataFrame(data, index=['a', 'b']))

# 4、series生成
# 一个series会生成只有1列的dataframe，示例如下：
print()
s = pd.Series(['a', 'b', 'c', 'd', 'e'])
print(pd.DataFrame(s))

# 5、其他方法
# 以下2种方法可以从字典和列表格式中取得数据
print()
# 从字典里生成
print(pd.DataFrame.from_dict({'国家': ['中国', '美国', '日本'],
                              '人口': [13.97, 3.28, 1.26]}))

# 从列表，元组，ndarray中生成
print(pd.DataFrame.from_records([('中国', '美国', '日本'), (13.97, 3.28, 1.26)]))
'''
out:
       0     1     2
0     中国    美国    日本
1  13.97  3.28  1.26
'''

# 列内容为1个字典
print()
print('列内容为1个字典')
# print(pd.json_normalize(df.col)) # 运行失败
# print(df.col.apply(pd.Series)) # 运行失败


import numpy as np
import pandas as pd

print()
print('2.6 pandas的数据类型')
print('\t2.6.1 数据类型查看')
print()
# update20240102

input_file = 'E:/bat/input_files/supplier_data.csv'

df = pd.read_csv(input_file, index_col=False)

df.columns = [heading.lower() for heading in df.columns.str.replace(' ', '_')]

print(df.columns)
# 查看指定列的数据类型
print(df.supplier_name.dtype)

print()
print('2.6 pandas的数据类型')
print('\t2.6.3 数据检测')
print()
s = pd.Series([14.34, 21.43, 5.08], name='gdp', index=[4, 5, 6])
print(s)
print()
'''
可以使用类型判断方法检测数据的类型是否与该方法中指定的类型一致，
如果一致，则返回True，注意传入的是一个Series：
'''
print(pd.api.types.is_bool_dtype(s))  # False
print(pd.api.types.is_categorical_dtype(s))  # False
print(pd.api.types.is_datetime64_dtype(s))  # False
print(pd.api.types.is_datetime64_any_dtype(s))  # False
print(pd.api.types.is_datetime64_ns_dtype(s))  # False
print(pd.api.types.is_float_dtype(s))  # True
print(pd.api.types.is_int64_dtype(s))
print(pd.api.types.is_numeric_dtype(s))  # True
print(pd.api.types.is_object_dtype(s))
print(pd.api.types.is_string_dtype(s))
print(pd.api.types.is_timedelta64_dtype(s))  # False

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
data = ('a,b,c\n1,Yes,2\n3,No,4')
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
cache_dates如果为True，则使用唯一的转换日期缓存来应用datetime转换。
解析重复的日期字符串，尤其是带有时区偏移的日期字符串时，
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
df = pd.read_csv(in_file, compression='gzip', encoding='ISO-8859-1', \
                 chunksize=chunk_size, usecols=[0], low_memory=False, skip_blank_lines=True, engine='c')

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


# -----------------------------------------------------------
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

path = 'E:/bat/output_files/pandas_read_csv_20240118.csv'

# 忽略特定的警告 | 当遇到 openpyxl 中的 UserWarning 类型的警告时，它们将不会被打印出来。
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

print()
print('------------------------------------------------------------')
print('\t3.3 读取Excel')
print('\t3.3.1 语法')
print()
'''
update20240122

data = 'a,b,c,1\n0,1,2,3\n1,2,3,5\n2,5,3,4'

pd.read_excel(io, sheet_name=0, header=0,
              names=None, index_col=None,
              usecols=None, squeeze=False,
              dtype=None, engine=None,
              converters=None, true_values=None,
              false_values=None, skiprows=None,
              nrows=None, na_values=None,
              keep_default_na=True, verbose=False,
              parse_dates=False, date_parser=None,
              thousands=None, comment=None, skipfooter=0,
              convert_float=True, mangle_dupe_cols=True, **kwds)
'''

print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.3.2 读取文件')
print()

# 如果文件与代码文件在同目录下
pd.read_excel('data.xls')
# 本地绝对路径：
pd.read_excel('E:/bat/input_files/sales_2013.xlsx')
# 使用网址 url
# pd.read_excel('https://www.gairuo.com/file/data/dataset/team.xlsx')


print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.3.3 读取指定sheet表')
print()

# str, int, list, or None, default 0

print('# 默认sheet')
df = pd.read_excel(input_file)
print(df)
print()

print('# 第二个 sheet')
df = pd.read_excel(input_file, sheet_name=1)
print(df)
print()

print('# 按 sheet 的名字')
df = pd.read_excel(input_file, sheet_name='march_2013')
print(df)
print()

print('# 修改标题列，取前五行')
df.columns = [column.lower() for column in df.columns.str.replace(' ', '_')]
print(df.head())
print()

print('# 取第一个、第二个、名为 march_2013 的，返回一个 df 组成的字典')
dfs = pd.read_excel(input_file, sheet_name=[0, 1, "march_2013"])
print(dfs)
print()

print('# 所有的 sheet')
dfs = pd.read_excel(input_file, sheet_name=None)  # 所有的 sheet
print(dfs)
print()
print('# 读取时按 sheet 名')
print(dfs['february_2013'])
print()

print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.3.4 表头 header')
print()

# int, list of int, default 0

print('# 不设表头')  # || 原表头成为第一行数据，表头成为阿拉伯数字索引
df = pd.read_excel(input_file, header=None)
print(df)
print()

print('# 第3行为表头')  # 输出显示跳过前2行
df = pd.read_excel(input_file, header=2)
print(df)
print()

print('# 两层表头，多层索引')
df = pd.read_excel(input_file, header=[0, 1])
print(df)
print()
# df = pd.read_excel('tmp.xlsx', header=None)  # 不设表头
# pd.read_excel('tmp.xlsx', header=2)  # 第3行为表头
# pd.read_excel('tmp.xlsx', header=[0, 1])  # 两层表头，多层索引

print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.3.5 列名 names')
print()
'''
用names指定列名，也就是表头的名称，如不指定，默认为表头的名称。

如指定部分列名，则输出列名与对应列不一致
'''
# array-like, default None
c_list = ['Customer ID', 'Customer Name', 'Invoice Number', 'Sale Amount', 'Purchase Date']

print('# 指定列名')
df = pd.read_excel(input_file, names=['Customer ID', 'Customer Name'])
print(df)
print()
'''
out:
                                  Customer ID Customer Name
1234 John Smith         100-0002         1200    2013-01-01
2345 Mary Harrison      100-0003         1425    2013-01-06
3456 Lucy Gomez         100-0004         1390    2013-01-11
4567 Rupert Jones       100-0005         1257    2013-01-18
5678 Jenny Walters      100-0006         1725    2013-01-24
6789 Samantha Donaldson 100-0007         1995    2013-01-31
'''

df = pd.read_excel(input_file, names=['Customer ID', 'Customer Name', 'Invoice Number', 'Sale Amount', 'Purchase Date'])
print(df)
print(df.dtypes)
print()

print('# 传入列表变量')
df = pd.read_excel(input_file, names=c_list)
print(df)
print()

print('# 没有表头，需要设置为 None')
df = pd.read_excel(input_file, header=None, names=None)
print(df)

print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.3.6 索引列 index_col')
print()

# int, list of int, default None
print('# 指定第一列为索引')
df = pd.read_excel(input_file, index_col=0)
print(df)
print()

print('# 前两列，多层索引')
df = pd.read_excel(input_file, index_col=[0, 1])
print(df)
print()

print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.3.7 使用列 usecols')
print()

print('# 取 A 和 B 两列')
df = pd.read_excel(input_file, usecols='A,B')
print(df)
print()

print('# 取 A 和 B 两列')
# 大写
df = pd.read_excel(input_file, usecols='A:D')
print(df)
print()
# 小写 | 运行成功
df = pd.read_excel(input_file, usecols='a:e')
print(df)
print()

print('# 取 a列和b列，再加c到e列')
df = pd.read_excel(input_file, usecols='a,b,c:e')
print(df)
print()

print('# 取前两列')
df = pd.read_excel(input_file, usecols=[0, 1])
print(df)
print()

print('# 取指定列名的列')
df = pd.read_excel(input_file, usecols=['Customer ID', 'Sale Amount'])
print(df)
print()

print('# 表头包含 m 的')  # 大小写敏感
df = pd.read_excel(input_file, usecols=lambda x: 'm' in x)
print(df)
print()

# print('# 返回序列 squeezebool') # squeezebool=True 报错！
# df = pd.read_excel(input_file,usecols='a')
# print(df.dtypes)
# print()

print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.3.8 数据类型 dtype')
print()
'''数据类型，如果不传则自动推断。如果被 converters 处理则不生效。'''
df = pd.read_excel(input_file)
print(df.dtypes)
print()

print('# 所有数据均为此数据类型')  # | 但是字符串列、日期列转换为浮点类型 会报错
df = pd.read_excel(input_file, usecols=['Customer ID', 'Sale Amount'], dtype=np.float64)
print(df.dtypes)  # ValueError: Unable to convert column Customer Name to type float64 (sheet: 0)
print()

print('# 指定字段的类型')
df = pd.read_excel(input_file, dtype={'Customer ID': float, 'Sale Amount': str})
print(df.dtypes)
print()

print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.3.9 处理引擎 engine')
print()

'''
可接受的参数值是 “xlrd”, “openpyxl” 或者 “odf”，如果文件不是缓冲或路径，就需要指定，用于处理 excel 使用的引擎，三方库。
如果未指定engine参数，pandas会尝试根据文件扩展名自动选择合适的引擎。
xlrd需要2.0.1及以上版本 ，  openpyxl及xlrd需要单独安装！
'''

print('# openpyxl: 用于读取.xlsx文件')  # 这是Excel 2007及以后版本的文件格式。
df = pd.read_excel(input_file, engine='openpyxl')
print(df)
print()

# xlrd: 早期被广泛使用来读取.xls和.xlsx文件，但从xlrd 2.0.0版本开始，不再支持.xlsx文件。
print('# xlrd: 用于读取.xls文件')
df = pd.read_excel(xls_file, engine='xlrd')
print(df)
print()

# pyxlsb: 用于读取Excel的二进制文件格式.xlsb。
df = pd.read_excel('tmp.xlsb', engine='pyxlsb')
print(df)
print()

print('# 未指定引擎，pandas自动选择合适的引擎')
df = pd.read_excel(xls_file)
print(df)
print()

# odf: 用于读取OpenDocument格式的.ods文件，这是LibreOffice和OpenOffice使用的格式。
df = pd.read_excel('tmp.ods', engine='odf')
print(df)
print()

print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.3.10 列数据处理 converters')
print()

'''
对列的数据进行转换，列名与函数组成的字典。key 可以是列名或者列的序号。
'''


# dict, default None
def foo(p):
    return p + 's'


print('# Customer Name 应用函数, Sale Amount 使用 lambda')
df = pd.read_excel(input_file, converters={'Customer Name': foo, 'Sale Amount': lambda x: x * 2})
print(df)
'''
out:
   Customer ID        Customer Name Invoice Number  Sale Amount Purchase Date
0         1234          John Smiths       100-0002         2400    2013-01-01
1         2345       Mary Harrisons       100-0003         2850    2013-01-06
2         3456          Lucy Gomezs       100-0004         2780    2013-01-11
3         4567        Rupert Joness       100-0005         2514    2013-01-18
4         5678       Jenny Walterss       100-0006         3450    2013-01-24
5         6789  Samantha Donaldsons       100-0007         3990    2013-01-31
'''

print('# 使用列索引')
df = pd.read_excel(input_file, converters={1: foo, 3: lambda x: x * 0.5})
print(df)
print()
'''
out:
   Customer ID        Customer Name Invoice Number  Sale Amount Purchase Date
0         1234          John Smiths       100-0002        600.0    2013-01-01
1         2345       Mary Harrisons       100-0003        712.5    2013-01-06
2         3456          Lucy Gomezs       100-0004        695.0    2013-01-11
3         4567        Rupert Joness       100-0005        628.5    2013-01-18
4         5678       Jenny Walterss       100-0006        862.5    2013-01-24
5         6789  Samantha Donaldsons       100-0007        997.5    2013-01-31
'''

print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.4.2 数据输出 Excel')
print()

'''

'''

out_file = 'E:/bat/output_files/pandas_read_excel_20240124036.xlsx'

df = pd.read_excel(input_file)
df.columns = [column.lower() for column in df.columns.str.replace(' ', '_')]
print(df)
print()

# 导入文件到指定路径
# df.to_excel('E:/bat/output_files/pandas_read_excel_20240124032.xlsx',index=False)

# 指定sheet名，不要索引
df.to_excel(out_file, sheet_name='out_data', index=False)

# 指定索引名，不合并单元格
# index_label='label' # 该参数 指定第0列为索引列 名为label
df.to_excel(out_file, index_label='label', merge_cells=False)  # 不合并单元格
df.to_excel(out_file, index_label='label', merge_cells=True)  # 合并单元格 ||  和上面一个没啥区别啊？
print()

# 创建一个具有多层次索引的DataFrame
# df = pd.DataFrame({
#     'Value': [1, 2, 3, 4]
# }, index=[['A', 'A', 'B', 'B'], [1, 2, 1, 2]])

# 测试 [1, 1, 2, 2] 该列 没有合并单元格
df = pd.DataFrame({
    'Value': [1, 2, 3, 4]
}, index=[['A', 'A', 'B', 'B'], [1, 1, 2, 2]])

print(df)
# 将DataFrame写入Excel，合并相同的索引值
df.to_excel(out_file, index_label='label', merge_cells=True)

# 将DataFrame写入Excel，不合并相同的索引值
df.to_excel('E:/bat/output_files/pandas_read_excel_20240124037.xlsx', index_label='label', merge_cells=False)

print('---------------------------------------------------------------------')
print('# 多个数据导出到同一工作簿 不同sheet中')
out_file = 'E:/bat/output_files/pandas_read_excel_20240125046.xlsx'

df = pd.read_excel(input_file)
df.columns = [column.lower() for column in df.columns.str.replace(' ', '_')]
print(df)
print()

df1 = pd.read_excel(input_file, usecols='a:c')
print(df1)
print()

df2 = pd.read_excel(input_file)
# print(df2)
# print()
out_third = df2[df2['Customer ID'] >= 4567]
print(out_third)
print()
# print(df2[df2['Customer Name'].str.startswith('J')])
# print()

# 多个数据导出到同一工作簿 不同sheet中
with pd.ExcelWriter(out_file) as writer:
    df.to_excel(writer, sheet_name='first', index=False)
    df1.to_excel(writer, sheet_name='second', index=False)
    out_third.to_excel(writer, sheet_name='third', index=False)

print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.4.3 数据输出 导出引擎')
print()

'''
在pandas中，当使用to_excel()方法将DataFrame写入到Excel文件时，
可以选择不同的引擎来处理写入操作。常用的引擎包括openpyxl、xlsxwriter、xlwt等。

openpyxl:
用于读写.xlsx文件，支持Excel 2010及以上版本的文件格式。
支持创建图表和修改Excel文件的高级功能，如设置单元格样式、过滤器、条件格式等。
df.to_excel('path_to_file.xlsx', sheet_name='Sheet1', engine='openpyxl')

xlsxwriter:
仅用于写入.xlsx文件，提供了丰富的格式化选项，如单元格格式、图表、图像插入等。
通常用于需要高度定制化的Excel报告生成。
writer = pd.ExcelWriter('path_to_file.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()

xlwt:
用于写入.xls文件，支持老版本的Excel文件格式（Excel 97-2003）。
不支持.xlsx文件格式，功能上比openpyxl和xlsxwriter受限。
writer = pd.ExcelWriter('path_to_file.xls', engine='xlwt')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()
'''

# 指定操作引擎
df.to_excel('path_to_file.xlsx', sheet_name='Sheet1', engine='xlsxwriter')  # 在'engine'参数中设置ExcelWriter使用的引擎
writer = pd.ExcelWriter('path_to_file.xlsx', engine='xlsxwriter')
df.to_excel(writer)
writer.save()

# 设置系统引擎
'''
当你在to_excel()方法中指定engine参数时，你是在为单次操作选择一个特定的引擎。
这种方式适用于你只想在特定情况下使用某个引擎，而不改变全局默认设置。
'''
from pandas import options  # noqa: E402

options.io.excel.xlsx.writer = 'xlsxwriter'
df.to_excel('path_to_file.xlsx', sheet_name='Sheet1')

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
df = pd.read_excel(team_file, index_col='name')
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
print(df[df.index == 'Rick'])  # 单索引 取值
print()

print('# 读取时 设置多个索引 指定阿拉伯数字')
df = pd.read_excel(team_file, index_col=[0, 1])
print(df)
print()
print(df.index)
print(df[df.index == ('Rick', 'B')])  # 多索引 取值
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
df = df.set_index(['name'])  # 返回新的 DataFrame 并赋值给 df
print(df)
print()

print('# 使用inplace参数')
df = pd.read_excel(team_file)
df.set_index(['name', 'team'], inplace=True)  # 直接在原始 DataFrame 上设置索引
print(df)
print()

# df.set_index(['name', 'team']) # 设置两层索引
# df.set_index([df.name.str[0],'name']) # 将姓名的第一个字母和姓名设置为索引

print('# 将姓名的第一个字母和姓名设置为索引')

df = pd.read_excel(team_file)
df = df.set_index([df.name.str[0], 'name'])  # 将姓名的第一个字母和姓名设置为索引
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
df = df.set_index([df.name.str[0]])  # 将姓名的第一个字母设置为索引
print(df)
print()

print(df.index)  # Index(['L', 'A', 'A', 'E', 'O', 'R'], dtype='object', name='name')
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

s = pd.Series([i for i in range(2, 8, 1)])
df.set_index(s, inplace=True)
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
s = pd.Series([i for i in range(2, 8, 1)], name='row_no')  # Series命名！
df.set_index([s, 'name'], inplace=True)
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
s = pd.Series([i for i in range(2, 8, 1)], name='row_no')  # Series命名！
df.set_index([s, s ** 2], inplace=True)
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
df = df.set_index('name', drop=False)  # 保留原列
print(df)
print()

df = pd.read_excel(team_file)
df = df.set_index('name', drop=True)  # 不保留原列
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
df = df.set_index('name', append=True)  # 保留原来的索引
print(df)
print()

df = pd.read_excel(team_file)
print()
df = df.set_index('name', append=False)  # 不保留原来的索引
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
df = pd.read_excel(team_file, index_col='name')
print(df)
print()

print('# 清除索引：df.reset_index()')
df = df.reset_index()
# df.reset_index(inplace=True) # 测试效果同上！
print(df)
print()

print('# 清除索引：相当于什么也没做')
df = pd.read_excel(team_file)
df = df.set_index('name').reset_index()  # 相当于什么也没做
# df.set_index('name').reset_index(inplace=True) # 效果同上
print(df)
print()

print('# 删除原索引：name列没了')
df = pd.read_excel(team_file)
# df.set_index('name').reset_index(drop=True,inplace=True) # name列还在
df = df.set_index('name').reset_index(drop=True)  # name列没了
# df = df.set_index('name',inplace=True).reset_index(drop=True) # 报错！
print(df)
print()

print('# name 一级索引取消')
df = pd.read_excel(team_file)
df = df.set_index(['name', 'team']).reset_index(level=0)  # name 索引取消
# df = df.set_index(['name','team']).reset_index(level=1) # team索引取消
print(df)
print()

print('# 使用 层级索引名')
df = pd.read_excel(team_file, index_col='name')
df = df.reset_index(level='name')  # name索引取消
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

print(df.index)  # out:Index(['Liver', 'Arry', 'Ack', 'Eorge', 'Oah', 'Rick'], dtype='object', name='name')
print(df.columns)  # out: Index(['team', 'Q1', 'Q2', 'Q3', 'Q4'], dtype='object')

print()
print('1、df.index.name')
print(df.index.name)  # 名称 out:name
print(df.columns.name)  # out:None

print()
print('2、df.index.array')
print(df.index.array)  # array数组
'''
out:
<PandasArray>
['Liver', 'Arry', 'Ack', 'Eorge', 'Oah', 'Rick']
Length: 6, dtype: object
'''

print(df.columns.array)  # array数组
'''
out:
<PandasArray>
['team', 'Q1', 'Q2', 'Q3', 'Q4']
Length: 5, dtype: object
'''

print()
print('3、df.index.dtype')
print(df.index.dtype)  # 数据类型  out：object
print(df.columns.dtype)  # 数据类型  out：object

print()
print('4、df.index.shape')  # 形状
print(df.index.shape)  # out：(6,)
print(df.columns.shape)  # out：(5,)

print()
print('5、df.index.size')  # 元素数量
print(df.index.size)  # out：6
print(df.columns.size)  # out：5

print()
print('6、df.index.values')  # array数组
print(df.index.values)  # out：['Liver' 'Arry' 'Ack' 'Eorge' 'Oah' 'Rick']
print(df.columns.values)  # out：['team' 'Q1' 'Q2' 'Q3' 'Q4']

# 其他，不常用

print()
print('7、df.index.empty')  # 是否为空
print(df.index.empty)  # out：False
print(df.columns.empty)  # out：False

print()
print('8、df.index.is_unique')  # 是否不重复
print(df.index.is_unique)  # out：True
print(df.columns.is_unique)  # out：True

print()
print('9、df.index.names')  # 名称列表
print(df.index.names)  # out：['name']
print(df.columns.names)  # out：[None]

print()
print('10、df.index._is_all_dates')  # 是否全是日期时间
print(df.index._is_all_dates)  # out：False
print(df.columns._is_all_dates)  # out：False

print()
print('11、df.index.has_duplicates')  # 是否有重复值
print(df.index.has_duplicates)  # out：False
print(df.columns.has_duplicates)  # out：False

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
df = pd.read_excel(team_file, index_col='Q1')  # Q1 name
print(df)
print()

# 常用方法
print()
print('# 转换类型')
print(df.index)
print(df.index.astype('int'))  # 转换类型
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
print(df.index.rename('number'))  # 修改索引名称
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
df = pd.read_excel(team_file, index_col='name')  # Q1 name
print(df)
print()

print('修改前：')
print(df.index)
print(df.index.name)

print('修改后：')
df.rename_axis('student_name', inplace=True)
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
df = pd.read_excel(team_file, index_col='name')  # Q1 name
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
replacements = {l1: l2 for l1, l2 in zip(list1, list2)}
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
df.rename(lambda x: 't_' + x, axis=1)
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
print('\t4.3 统计计算')
print('\t4.3.1 描述统计')
print()
# update20240227
'''
Pandas可以对Series与DataFrame进行快速的描述性统计，如求和、平均数、最大值、方差等，
这些是最基础也最实用的统计方法。
对于DataFrame，这些统计方法会按列进行计算，最终产出一个以列名为索引、以计算值为值的Series。

df.describe()会返回一个有多行的所有数字列的统计表，每一行对应一个统计指标，
有总数、平均数、标准差、最小值、四分位数、最大值等，这个表对我们初步了解数据很有帮助。

以下是一些具体的使用方法举例：

'''
df1 = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df1)
print()

print('df.describe() 读取 xlsx文件')
print(df1.describe())
print()

'''
如果没有数字，则会输出与字符相关的统计数据，
如数量、不重复值数、最大值（字符按首字母顺序）等。示例如下。
'''
print('# df.describe() 文本类型数据 描述统计')
s = pd.Series(['a', 'b', 'c', 'c']).describe()
print(s)
print()
'''
out:
count     4 # 非空值
unique    3 # 不重复值
top       c # 最频繁出现的值
freq      2 # 最频繁出现的值出现的频率 2次
dtype: object
'''

print('# df.describe() 时间数据 描述统计')
df = pd.date_range('2023-01-01', '2023-05-01')
s = pd.Series(df, name='day_code')
# print(s)
print(s.describe())
print()
'''
out:
# df.describe() 时间数据 描述统计
count                    121
mean     2023-03-02 00:00:00
min      2023-01-01 00:00:00
25%      2023-01-31 00:00:00
50%      2023-03-02 00:00:00
75%      2023-04-01 00:00:00
max      2023-05-01 00:00:00
Name: day_code, dtype: object
'''

# 还可以自己指定分位数（一般情况下，默认值包含中位数），指定和排除数据类型：
print('# 指定分位数')
print(df1.describe(percentiles=[.05, .25, .75, .95]))
print()
'''
out:
               Q1         Q2         Q3          Q4
count    6.000000   6.000000   6.000000    6.000000
mean    73.333333  60.333333  51.333333   78.166667
std     24.792472  31.582696  30.428057   15.625833
min     36.000000  21.000000  18.000000   57.000000
5%      41.250000  25.000000  19.500000   58.750000
25%     59.000000  40.000000  27.250000   67.500000
50%     77.000000  54.500000  49.000000   81.000000
75%     92.000000  87.000000  68.500000   85.500000
95%     98.250000  98.250000  90.500000   96.500000
max    100.000000  99.000000  97.000000  100.000000
'''

print('# 指定or排除数据类型 ')
# 创建一个包含数值和对象类型列的DataFrame
data = {
    'A': [1, 2, 3, 4],
    'B': [10.0, 20.0, 30.0, 40.0],
    'C': ['foo', 'bar', 'foo', 'bar']
}
df = pd.DataFrame(data)
print(df.describe())
print()

# 使用include参数同时描述数值和对象类型的列
description = df.describe(include=['object', 'number'])  # np.object_
print(description)
print()

# 使用exclude参数排除对象类型的列
description = df.describe(exclude=['object'])
print(description)
'''
out:
              A          B
count  4.000000   4.000000
mean   2.500000  25.000000
std    1.290994  12.909944
min    1.000000  10.000000
25%    1.750000  17.500000
50%    2.500000  25.000000
75%    3.250000  32.500000
max    4.000000  40.000000

               A          B    C
count   4.000000   4.000000    4
unique       NaN        NaN    2
top          NaN        NaN  foo
freq         NaN        NaN    2
mean    2.500000  25.000000  NaN
std     1.290994  12.909944  NaN
min     1.000000  10.000000  NaN
25%     1.750000  17.500000  NaN
50%     2.500000  25.000000  NaN
75%     3.250000  32.500000  NaN
max     4.000000  40.000000  NaN

              A          B
count  4.000000   4.000000
mean   2.500000  25.000000
std    1.290994  12.909944
min    1.000000  10.000000
25%    1.750000  17.500000
50%    2.500000  25.000000
75%    3.250000  32.500000
max    4.000000  40.000000

'''

print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.3 统计计算')
print('\t4.3.2 数学统计')
print()
# update20240227
'''
Pandas支持常用的数学统计方法，如平均数、中位数、众数、方差等，
还可以结合NumPy使用其更加丰富的统计功能。
我们先来使用mean()计算一下平均数，DataFrame使用统计函数后会生成一个Series，
这个Series的索引为每个数字类型列的列名，值为此列的平均数。如果DataFrame没有任何数字类型列，则会报错。

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file, usecols=[2, 3, 4, 5])  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print('df.mean() 读取 xlsx文件')
print(df.mean())
print()
'''
如果文件中有任意一列为非数值列 会报错！
out:
Q1    73.333333
Q2    60.333333
Q3    51.333333
Q4    78.166667
dtype: float64
'''

# Series应用数学统计函数一般会给出一个数字定值，直接计算出这一列的统计值：
print(df.Q1.mean())  # out:73.33333333333333
print(df.Q2.mean())  # out:60.333333333333336
print()

# 如果我们希望按行计算平均数，即数据集中每个学生Q1到Q4的成绩的平均数，可以传入axis参数，列传index或0，行传columns或1：
print('# df.mean(axis) axis参数')
print(df.mean(axis='columns'))
print(df.mean(axis=1))  # 效果同上
print(df.mean(1))  # 效果同上
print(df.mean(axis='index'))
print(df.mean(axis=0))  # 效果同上
print(df.mean(0))  # 效果同上
print()
'''
out:
0    49.50
1    41.75
2    54.75
3    84.50
4    65.25
5    99.00
dtype: float64
Q1    73.333333
Q2    60.333333
Q3    51.333333
Q4    78.166667
dtype: float64
'''

# 创建文本列为索引列，计算每行平均值，只看前3条
df = pd.read_excel(team_file)
df = df.set_index(['name', 'team'])  # 文本类型全部设置索引 单一列设置会报错
print(df)
print()
print(df.mean(1).head(3))
'''
out:
name   team
Liver  E       49.50
Arry   C       41.75
Ack    A       54.75
dtype: float64
'''

print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.3 统计计算')
print('\t4.3.3 统计函数')
print()
# update20240227
'''
上文我们介绍了平均数mean，Pandas提供了非常多的数学统计方法，如下：

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file, usecols=[2, 3, 4, 5])  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

df.mean()  # 返回所有列的均值
df.mean(1)  # 返回所有行的均值，下同
df.corr()  # 返回列与列之间的相关系数
df.count()  # 返回每一列中的非空值的个数
df.max()  # 返回每一列的最大值
df.min()  # 返回每一列的最小值
df.abs()  # 绝对值
df.median()  # 返回每一列的中位数
df.std()  # 返回每一列的标准差，贝塞尔校正的样本标准偏差
df.var()  # 无偏方差
df.sem()  # 平均值的标准误差
df.mode()  # 众数
df.prod()  # 连乘
df.mad()  # 平均绝对偏差
df.cumprod()  # 累积连乘，累乘
df.cumsum(axis=0)  # 累积连加，累加
df.nunique()  # 去重数量，不同值的量
df.idxmax()  # 每列最大值的索引名
df.idxmin()  # 每列最小值的索引名
df.cummax()  # 累积最大值
df.cummin()  # 累积最小值
df.skew()  # 样本偏度（第三阶）
df.kurt()  # 样本峰度（第四阶）
df.quantile()  # 样本分位数（不同 % 的值）

print()
print('# Pandas还提供了一些特殊的用法：')
# 很多支持指定行列（默认是axis=0列）等参数
df.mean(1)  # 按行计算
# 很多函数均支持
df.sum(0, skipna=False)  # 不除缺失数据
# 很多函数均支持
df.sum(level='blooded')  # 索引级别
df.sum(level=0)
# 执行加法操作所需的最小有效值数
df.sum(min_count=1)

print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.3 统计计算')
print('\t4.3.4 非统计计算')
print()
# update20240228
'''
非统计性计算，如去重、格式化等。接下来我们将介绍一些数据的加工处理方法。

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name' ,usecols=[2,3,4,5]
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

# 创建一个简单的DataFrame
data = {
    'Q1': [1.12345, 2.34567, 3.56789],
    'Q2': [4.12345, 5.34567, 0.0],
    'Q3': [np.nan, 7.0, np.nan]
}
df = pd.DataFrame(data)

# 创建一个含有重复值的Series
s = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])

print('# 返回所有列all()值的Series')
print(df.all())
# Q1    True
# Q2    False (因为有一个值为0.0)
# Q3    False (因为有NaN值)
# dtype: bool

print('# 返回任何列中至少有一个非零元素的Series')
print(df.any())
# Q1    True
# Q2    True
# Q3    True (因为有一个非NaN的值)
# dtype: bool

print('# 四舍五入到指定的小数位')
print(df.round(2))
#     Q1    Q2   Q3
# 0  1.12  4.12  NaN
# 1  2.35  5.35  7.0
# 2  3.57  0.00  NaN

print('# 四舍五入，为不同的列指定不同的小数位')
print(df.round({'Q1': 2, 'Q2': 0}))
#     Q1   Q2   Q3
# 0  1.12  4.0  NaN
# 1  2.35  5.0  7.0
# 2  3.57  0.0  NaN

print('# 四舍五入到最接近的10位')
print(df.round(-1))
#     Q1    Q2   Q3
# 0  0.0  0.0  NaN
# 1  0.0  10.0  10.0
# 2  0.0  0.0  NaN

print('# 每个列的去重值的数量')
print(df.nunique())
# Q1    3
# Q2    2
# Q3    1
# dtype: int64

print('# Series的去重值的数量')
print(s.nunique())
# 4

print('# 值的真假值替换，NaN为True，其他为False')
print(df.isna())
#       Q1     Q2     Q3
# 0  False  False   True
# 1  False  False  False
# 2  False  False   True

print('# 与上相反，NaN为False，其他为True')
print(df.notna())
#      Q1    Q2     Q3
# 0  True  True  False
# 1  True  True   True
# 2  True  True  False


# 以下可以传一个值或者另一个DataFrame，对数据进行广播方式计算，返回计算后的DataFrame：

# 创建两个简单的DataFrame
df1 = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

df2 = pd.DataFrame({
    'A': [10, 20, 30],
    'B': [40, 50, 60]
})

print(df1)
print()
print(df2)

print()
print('# 对df1中的每个元素加1 或者使用add函数')
print(df1 + 1)
# 或者使用add函数
print(df1.add(1))
# 结果:
#    A  B
# 0  2  5
# 1  3  6
# 2  4  7

print()
print('# 对df1中的每个元素减1')
print(df1.sub(1))
# 结果:
#    A  B
# 0  0  3
# 1  1  4
# 2  2  5

print()
print('# 对df1中的每个元素乘以2')
print(df1.mul(2))
# 结果:
#    A   B
# 0  2   8
# 1  4  10
# 2  6  12

print()
print('# 对df1中的每个元素除以2')
print(df1.div(2))
# 结果:
#      A    B
# 0  0.5  2.0
# 1  1.0  2.5
# 2  1.5  3.0

print()
print('# 对df1中的每个元素模2（取余数）')
print(df1.mod(2))
# 结果:
#    A  B
# 0  1  0
# 1  0  1
# 2  1  0

print()
print('# 对df1中的每个元素的指数幂（这里以2为例）')
print(df1.pow(2))
# 结果:
#    A   B
# 0  1  16
# 1  4  25
# 2  9  36

print()
print('# 对df1和df2进行矩阵乘法（点乘）')
# 注意：为了矩阵乘法的正确性，df1的列数必须与df2的行数相同
print(df1)
print(df2.transpose())
print()
print(df1.dot(df2.transpose()))
# 结果:
#     0    1    2
# 0  170  370  570
# 1  220  490  760
# 2  270  610  950
'''
矩阵的乘法计算过程：
请注意，矩阵乘法（点乘）df.dot(df2)要求第一个DataFrame的列数与第二个DataFrame的行数相同。
在上面的例子中，我们使用df2.transpose()来转置df2，使得其行数与df1的列数相同，从而可以进行矩阵乘法。
如果没有转置，直接使用df1.dot(df2)将会因为维度不匹配而导致错误。

对于矩阵乘法，我们按行取df1的每一行，按列取df2.transpose()的每一列，然后将两者对应元素相乘，并将乘积相加。
计算第一个元素（即170）的过程是这样的：

df1的第0行与df2.transpose()的第0列的点乘：
(1 * 10) + (4 * 40) = 10 + 160 = 170

数学上通常写作：C[i, j] = Σ (A[i, k] * B[k, j])
df1的第0行与df2.transpose()的第0列的点乘：(1 * 10) + (4 * 40) = 10 + 160 = 170
df1的第0行与df2.transpose()的第1列的点乘：(1 * 20) + (4 * 50) = 20 + 200 = 220
df1的第0行与df2.transpose()的第2列的点乘：(1 * 30) + (4 * 60) = 30 + 240 = 270
。。。。
'''

print('以下是Series专有的一些函数：')
# 不重复的值及数量
s.value_counts()
s.value_counts(normalize=True)  # 重复值的频率
s.value_counts(sort=False)  # 不按频率排序
s.unique()  # 去重的值 array
s.is_unique  # 是否有重复
# 最大最小值
s.nlargest()  # 最大的前5个
s.nlargest(15)  # 最大的前15个
s.nsmallest()  # 最小的前5个
s.nsmallest(15)  # 最小的前15个
s.pct_change()  # 计算与前一行的变化百分比
s.pct_change(periods=2)  # 前两行
s1.cov(s2)  # 两个序列的协方差

# 特别要掌握的是value_counts()和unique()，因为它们的使用频率非常高。
'''
实际业务：
pct_change() 在金融分析中经常被用来计算股票价格、投资回报率等的百分比变化。
比如，分析一支股票连续几天的涨跌幅，或者计算投资组合的日收益率。

cov() 通常用于投资组合管理中，分析不同资产之间的价格变动关系，协助判断资产配置是否能够实现风险分散。
例如，如果两只股票的协方差为正，表明它们往往同涨同跌，投资者可能会考虑添加一些协方差为负的资产以平衡投资组合风险。
协方差也是在计算相关系数和多变量统计分析中的一个重要步骤。
'''

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
print('\t4.4 位置计算')
print('\t4.4.1 位置差值diff')
print()
# update20240228
'''
本节介绍几个经常到用的位置计算操作。diff()和shift()经常用来计算数据的增量变化，rank()用来生成数据的整体排名。

df.diff()可以做位移差操作，经常用来计算一个序列数据中上一个数据和下一个数据之间的差值，如增量研究。
默认被减的数列下移一位，原数据在同位置上对移动后的数据相减，得到一个新的序列，
第一位由于被减数下移，没有数据，所以结果为NaN。可以传入一个数值来规定移动多少位，负数代表移动方向相反。
Series类型如果是非数字，会报错，
DataFrame会对所有数字列移动计算，同时不允许有非数字类型列。

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

s = pd.Series([9, 4, 6, 7, 9])
print(s)
print()
print('# 后一个值与前一个值的差值')
print(s.diff())
print()
# 0    NaN
# 1   -5.0
# 2    2.0
# 3    1.0
# 4    2.0
# dtype: float64


print('# 后方向，移动2位求差值')
print(s.diff(-2))
# 0    3.0
# 1   -3.0
# 2   -3.0
# 3    NaN
# 4    NaN
# dtype: float64

print('# 对于DataFrame，还可以传入axis=1进行左右移动：')
# 只筛选4个季度的5条数据
df1 = df.iloc[:5, 2:6]
print(df1)
print(df1.diff(1, axis=1))
# 上下方向移动 差值
print(df1.diff(1, axis=0))
print(df1.diff(1))  # 效果同上
print(df1.diff())  # 效果同上
print(df1.diff(2))  # 从上向下 间隔2位相减
print(df1.diff(-2))  # 从下向上 间隔2位相减

#    Q1  Q2  Q3  Q4
# 0 NaN -68   3  40
# 1 NaN   1   0  20
# 2 NaN   3 -42  66
# 3 NaN   3 -25   7
# 4 NaN -16  12  25

#      Q1    Q2    Q3    Q4
# 0   NaN   NaN   NaN   NaN
# 1 -53.0  16.0  13.0  -7.0
# 2  21.0  23.0 -19.0  27.0
# 3  36.0  36.0  53.0  -6.0
# 4 -28.0 -47.0 -10.0   8.0

print()
# 计算间隔为2的差分
print(df1.diff(periods=2))  # 效果同 df1.diff(2)
# 输出：
#      Q1    Q2    Q3    Q4
# 0   NaN   NaN   NaN   NaN
# 1   NaN   NaN   NaN   NaN
# 2 -32.0  39.0  -6.0  20.0
# 3  57.0  59.0  34.0  21.0
# 4   8.0 -11.0  43.0   2.0

# 如果dataframe中有非数值类型，运行报错。测试如下
# print(df.diff()) # TypeError: unsupported operand type(s) for -: 'str' and 'str'


print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.4 位置计算')
print('\t4.4.2 位置移动shift')
print()
# update20240229
'''
本节介绍几个经常用到的位置计算操作。diff()和shift()经常用来计算数据的增量变化，rank()用来生成数据的整体排名。

shift()可以对数据进行移位，不做任何计算，也支持上下左右移
动，移动后目标位置的类型无法接收的为NaN。

'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name' ,usecols=[2,3,4,5]
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print('# 整体下移1行，最顶的1行为NaN')
print(df.shift())
#     name  team    Q1    Q2    Q3    Q4
# 0   None  None   NaN   NaN   NaN   NaN
# 1  Liver     E  89.0  21.0  24.0  64.0
# 2   Arry     C  36.0  37.0  37.0  57.0
# 3    Ack     A  57.0  60.0  18.0  84.0
# 4  Eorge     C  93.0  96.0  71.0  78.0
# 5    Oah     D  65.0  49.0  61.0  86.0

print()
print('# 整体下移3行，最顶的3行为NaN')
print(df.shift(3))
#     name  team    Q1    Q2    Q3    Q4
# 0   None  None   NaN   NaN   NaN   NaN
# 1   None  None   NaN   NaN   NaN   NaN
# 2   None  None   NaN   NaN   NaN   NaN
# 3  Liver     E  89.0  21.0  24.0  64.0
# 4   Arry     C  36.0  37.0  37.0  57.0
# 5    Ack     A  57.0  60.0  18.0  84.0

print()
print('# 整体上移一行，最底的一行为NaN')
print(df.Q1.head())
print(df.Q1.head().shift(-1))
# 0    36.0
# 1    57.0
# 2    93.0
# 3    65.0
# 4     NaN
# Name: Q1, dtype: float64

print()
print('# 向右移动1位')
print(df.shift(axis=1))
#    name   team Q1   Q2  Q3  Q4
# 0  None  Liver  E   89  21  24
# 1  None   Arry  C   36  37  37
# 2  None    Ack  A   57  60  18
# 3  None  Eorge  C   93  96  71
# 4  None    Oah  D   65  49  61
# 5  None   Rick  B  100  99  97
print()
print('# 向右移动3位')
print(df.shift(3, axis=1))  # 向右移动3位
#    name  team    Q1     Q2 Q3   Q4
# 0  None  None  None  Liver  E   89
# 1  None  None  None   Arry  C   36
# 2  None  None  None    Ack  A   57
# 3  None  None  None  Eorge  C   93
# 4  None  None  None    Oah  D   65
# 5  None  None  None   Rick  B  100
print()
print('# 向左移动1位')
print(df.shift(-1, axis=1))
#   name  team  Q1  Q2   Q3  Q4
# 0    E    89  21  24   64 NaN
# 1    C    36  37  37   57 NaN
# 2    A    57  60  18   84 NaN
# 3    C    93  96  71   78 NaN
# 4    D    65  49  61   86 NaN
# 5    B   100  99  97  100 NaN
print()
print('# 实现了df.Q1.diff()')
# print(df.Q1,df.Q1.shift())
print(df.Q1 - df.Q1.shift())
# 0     NaN
# 1   -53.0
# 2    21.0
# 3    36.0
# 4   -28.0
# 5    35.0
# Name: Q1, dtype: float64


print()
print('------------------------------------------------------------')
print('第4章 pandas基础操作')
print('\t4.4 位置计算')
print('\t4.4.3 位置序号rank()')
print()
# update20240229
'''
本节介绍几个经常用到的位置计算操作。diff()和shift()经常用来计算数据的增量变化，rank()用来生成数据的整体排名。

rank()可以生成数据的排序值替换掉原来的数据值，它支持对所有类型数据进行排序，
如英文会按字母顺序。使用rank()的典型例子有学生的成绩表，给出排名：


'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name' ,usecols=[2,3,4,5]
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print('# 排名，将值变了序号')
# 数值从小到大排序 文本按照abc依次排序，首字母都为a，看第二个字母abc排序
print(df.rank())
print(df.head(3).rank())  # 前3行排序
#    name  team   Q1   Q2   Q3   Q4
# 0   4.0   6.0  4.0  1.0  2.0  2.0
# 1   2.0   3.5  1.0  2.0  3.0  1.0
# 2   1.0   1.0  2.0  4.0  1.0  4.0
# 3   3.0   3.5  5.0  5.0  5.0  3.0
# 4   5.0   5.0  3.0  3.0  4.0  5.0
# 5   6.0   2.0  6.0  6.0  6.0  6.0
#    name  team   Q1   Q2   Q3   Q4
# 0   3.0   3.0  3.0  1.0  2.0  2.0
# 1   2.0   2.0  1.0  2.0  3.0  1.0
# 2   1.0   1.0  2.0  3.0  1.0  3.0

print()
print('# 横向排名')
print(df.loc[:, 'Q1':'Q4'].head().rank(axis=1))  # 同一行排名 数值和文本类型不能同一行 否则报错！
#     Q1   Q2   Q3   Q4
# 0  4.0  1.0  2.0  3.0
# 1  1.0  2.5  2.5  4.0
# 2  2.0  3.0  1.0  4.0
# 3  3.0  4.0  1.0  2.0
# 4  3.0  1.0  2.0  4.0

print()
print('# 参数pct=True将序数转换成0~1的数')  # 适用文本类型
print(df.rank(pct=True).head(3).round(2))
print(df.loc[:, 'Q1':'Q4'].head(3).rank(pct=True, axis=1).round(2))  #
#    name  team    Q1    Q2    Q3    Q4
# 0  0.67  1.00  0.67  0.17  0.33  0.33
# 1  0.33  0.58  0.17  0.33  0.50  0.17
# 2  0.17  0.17  0.33  0.67  0.17  0.67
#      Q1    Q2    Q3    Q4
# 0  1.00  0.25  0.50  0.75
# 1  0.25  0.62  0.62  1.00
# 2  0.50  0.75  0.25  1.00

print()
print('--------method参数----------')
# 创建一个包含重复值的Series
s = pd.Series([7, 3.5, 3.5, 1, np.nan])

# 使用不同的method参数进行排名
print()
print('average：序号的平均值，如并列第1名，则按二次元计算（1+2）/2，都显示1.5，下个数据的值为3。')
print(s.rank(method='average'))  # 平均值
# 0    4.0
# 1    2.5
# 2    2.5
# 3    1.0
# 4    NaN
# dtype: float64


print()
print('min：最小的序数，如并列第1名，则都显示1，下个数据为3。')
print(s.rank(method='min'))  # 最小值
# 0    4.0
# 1    2.0
# 2    2.0
# 3    1.0
# 4    NaN
# dtype: float64


print()
print('max：最大的序数，如并列第1名，则都显示1，下个数据为2。')
print(s.rank(method='max'))  # 最大值
# 0    4.0
# 1    3.0
# 2    3.0
# 3    1.0
# 4    NaN
# dtype: float64

print()
print('first：如并列第1名，则按出现顺序分配排名')
print(s.rank(method='first'))  # 索引顺序
# 输出：
# 0    4.0
# 1    2.0
# 2    3.0
# 3    1.0
# 4    NaN
# dtype: float64

print()
print('\\ndense：并列排名相同，但下一个不同值的排名加1')
print(s.rank(method='dense'))  # 紧密排名
# 输出：
# 0    3.0
# 1    2.0
# 2    2.0
# 3    1.0
# 4    NaN
# dtype: float64

# 处理空值的排名
print(s.rank(method='min', na_option='bottom'))  # 空值放在最后
print(s.rank(method='min', na_option='top'))  # 空值放在前面
# out：
# 0    4.0
# 1    2.0
# 2    2.0
# 3    1.0
# 4    5.0
# dtype: float64
# 0    5.0
# 1    3.0
# 2    3.0
# 3    2.0
# 4    1.0
# dtype: float64


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
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print('# dataframe单列逻辑运算')  # 输出一个布尔值组成的series
print(df.Q1 > 36)
# 0     True
# 1    False
# 2     True
# 3     True
# 4     True
# 5     True
# Name: Q1, dtype: bool
print()
print('# 索引==1')  # 输出一个array类型数组
print(df.index == 1)
# [False  True False False False False]
print(type(df.index == 1))  # <class 'numpy.ndarray'>

print()
print('')
print(df.head(2).loc[:, 'Q1':'Q4'] > 60)  # 只取数字部分，否则会因字符无大于运算而报错
#       Q1     Q2     Q3     Q4
# 0   True  False  False   True
# 1  False  False  False  False

print('# ~ & | 运算符号')
print(df.loc[(df.Q1 < 60) & (df['team'] == 'C')])  # Q1成绩小于60分，并且是C组成员
#    name team  Q1  Q2  Q3  Q4
# 1  Arry    C  36  37  37  57
print(df.loc[~(df.Q1 < 60) & (df['team'] == 'C')])  # Q1成绩不小于60分，并且是C组成员
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print('# 是切片（[]）的一些逻辑筛选的示例')  # 输出一个布尔值组成的series

print()
print('# Q1等于36')
print(df[df['Q1'] == 36])  # Q1等于36
#    name team  Q1  Q2  Q3  Q4
# 1  Arry    C  36  37  37  57
print()
print('# Q1不等于36')
print(df[~(df['Q1'] == 36)])  # Q1不等于36
#     name team   Q1  Q2  Q3   Q4
# 0  Liver    E   89  21  24   64
# 2    Ack    A   57  60  18   84
# 3  Eorge    C   93  96  71   78
# 4    Oah    D   65  49  61   86
# 5   Rick    B  100  99  97  100
print()
print(df[df.name == 'Rick'])  # 姓名为rick
print(df[df.Q1 > df.Q2])

print()
print('# 以下是.loc[]和.lic[]的一些示例：')
print(df.loc[df['Q1'] > 90, 'Q1'])  # Q1大于90，仅显示Q1列
print(df.loc[df['Q1'] > 90, 'Q1':])  # Q1大于90，显示Q1及其后面所有的列
print(df.loc[(df.Q1 > 80) & (df.Q2 < 30)])  # and关系
print(df.loc[(df.Q1 > 90) | (df.Q2 < 30)])  # or关系
print(df.loc[df.Q1 == 100])  # 等于100
print(df.loc[df['Q1'] == 100])  # 等于100
'''
需要注意的是在进行或（|）、与（&）、非（~）运算时，各个独立逻辑表达式需要用括号括起来。
'''

print()
print('# any() 和 all() 函数')
print('# Q1、Q2成绩全为超过80分的')
print(df.loc[:, ['Q1', 'Q2']])
print((df.loc[:, ['Q1', 'Q2']] > 80))
# print(df[(df.loc[:,['Q1','Q2']] > 80)])
print((df.loc[:, ['Q1', 'Q2']] > 80).all(axis=1))
print((df.loc[:, ['Q1', 'Q2']] > 80).all(axis=0))

print()
# print('# Q1、Q2成绩全为超过80分的')
print(df[(df.loc[:, ['Q1', 'Q2']] > 80).all(axis=1)])
#     name team   Q1  Q2  Q3   Q4
# 3  Eorge    C   93  96  71   78
# 5   Rick    B  100  99  97  100
print()
print('# Q1、Q2成绩至少有一个超过80分的')
print(df[(df.loc[:, ['Q1', 'Q2']] > 80).any(axis=1)])  # any(1) 会报错！
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print()
print('# 查询最大索引的值')
print(df.Q1[lambda s: max(s.index)])  # 值为100
print()
print('# 计算最大值')
print(max(df.Q1.index))  # 最大索引值 为 5

print()
# print(df[5]) # 报错！
print(df.Q1[5])  # 100
print(df.Q1[df.index == 5])
# 5    100
# Name: Q1, dtype: int64
print(df.index == 5)  # [False False False False False  True]

print()
print('# 下面是一些lambda示例：')
print(df[lambda df: df['Q1'] == 100])  # Q1为100的
#    name team   Q1  Q2  Q3   Q4
# 5  Rick    B  100  99  97  100
print(df[df['Q1'] == 100])  # 结果同上

print()
print(df.loc[lambda df: df.Q1 == 100, 'Q1':'Q2'])  # Q1为100的,进显示'Q1','Q2' 列
#     Q1  Q2
# 5  100  99

print()
print('# 由真假值组成的序列')
print(df.loc[:, lambda df: df.columns.str.len() == 4])  # 由真假值组成的序列
#     name team
# 0  Liver    E
# 1   Arry    C
# ....
print()
print(df.loc[:, lambda df: [i for i in df.columns if 'Q' in i]])
#     Q1  Q2  Q3   Q4
# 0   89  21  24   64
# 1   36  37  37   57
# ....
print()
print(df.iloc[:3, lambda df: df.columns.str.len() == 2])
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
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
print(df[df.Q1.ne(60)])  # 不等于60  !=
print(df[df.Q1.le(57)])  # 小于等于57 <=
print(df[df.Q1.lt(57)])  # 小于57 <
print(df[df.Q1.ge(93)])  # 大于等于93 >=
print(df[df.Q1.gt(93)])  # 大于93 >
print(df.loc[df.Q1.gt(80) & df.Q2.lt(90)])  # Q1大于80 且 Q2小于90
print(df.loc[(df.Q1.gt(80)) & (df.Q2.lt(90))])  # 同上

print()
'''
这些函数可以传入一个定值、数列、布尔序列、Series或DataFrame，来与原数据比较。
另外还有一个. isin()函数，用于判断数据是否包含指定内容。可以传入一个列表，原数据只需要满足其中一个存在即可；
也可以传入一个字典，键为列名，值为需要匹配的值，以实现按列个性化匹配存在值。
'''
print('# eq() 字符串测试')
print(df[df.name.eq('Rick')])  # 字符串也行
#    name team   Q1  Q2  Q3   Q4
# 5  Rick    B  100  99  97  100

print()
print('# isin')
print(df[df.team.isin(['A', 'B'])])  # 包含A、B两组的
#    name team   Q1  Q2  Q3   Q4
# 2   Ack    A   57  60  18   84
# 5  Rick    B  100  99  97  100
print(df[df.isin({'team': ['C', 'D'], 'Q1': [36, 96]})])  # 复杂查询 其他值为NaN
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print()
print('# df.query()')
print(df.query('Q1>Q2>90'))  # 直接写类型SQL where语句
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
print(df.query('Q1<50 & Q2>30 and Q3>20')) # & == and
print(df.query('(Q1<50) & (Q2>30) and (Q3>20)'))  # 同上
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
print(df.query('Q1 > @a+4'))  # 支持传入变量，如大于平均分4分的行
# print(df.query('Q1 > a + 10'))  # 直接引用a 会报错！name 'a' is not defined
print(df.query('Q1 > `Q2` + @a'))
print(df.query('Q1 > Q2 + @a'))  # 同上！

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


print(df.query('Q1 > 90').loc[:, 'Q1':'Q4'])  # 筛选行列

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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print()

print()
print('# df.filter(items)')
print(df.filter(items=['Q1', 'Q2']))  # 选择2列
#     Q1  Q2
# 0   89  21
# ...

print()
print('# df.filter(regex)')  # 正则参数
print(df.filter(regex='Q', axis=1))  # 列名包含Q的列
#     Q1  Q2  Q3   Q4
# 0   89  21  24   64
# ...
print(df.filter(regex='e$', axis=1))  # 以e结尾的列
#     name
# 0  Liver
# ...
print()
print(df.filter(regex='1$', axis=0))  # 正则，索引名以1结尾

print()
print('# 模糊匹配 like')
print(df.filter(like='2', axis=0))  # 索引中有2的
#   name team  Q1  Q2  Q3  Q4
# 2  Ack    A  57  60  18  84
# ...
# 索引中以2开头的，列名有Q的
print(df.filter(regex='^2', axis=0).filter(like='Q', axis=1))
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
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
print(df.select_dtypes(include=['int64']))  # 只选择int64型数据
print(df.select_dtypes(include=['int']))  # 同上
print(df.select_dtypes(include=['number']))  # 同上
# print(df.dtypes) # object int64
#     Q1  Q2  Q3   Q4
# 0   89  21  24   64
# ...
print()
print('exclude')
print(df.select_dtypes(exclude=['number']))  # 排除数字类型数据
#     name team
# 0  Liver    E
# ...

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
print('\t5.2 数据类型转换')
print('\t5.2.1 推断类型')
print()
# update20240315
'''
在开始数据分析前，我们需要为数据分配好合适的类型，这样才能够高效地处理数据。
不同的数据类型适用于不同的处理方法。之前的章节中介绍过，加载数据时可以指定数据各列的类型：

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print(df.dtypes)
print()

# TODO 加载数据时指定数据类型
print()
print('对指定字段分别指定 数据类型')
df1 = pd.read_excel(team_file, dtype={'name': 'string', 'Q1': 'string'})
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
df2 = pd.read_excel(team_file, dtype='string')
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
print(df.infer_objects())  # 推断后的DataFrame
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
print(df.convert_dtypes())  # 推断后的DataFrame
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 按大体类型推定')
m = ['1', 2, 3]
s = pd.to_numeric(m)  # 转成数字
print(s)  # [1 2 3]
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
print(pd.to_datetime(m, errors='coerce'))  # 错误处理
# DatetimeIndex(['NaT', '1970-01-01 00:00:00.000000002',
#                '1970-01-01 00:00:00.000000003'],
#               dtype='datetime64[ns]', freq=None)

print("errors='ignore'")
print(pd.to_numeric(m, errors='ignore'))
# print(pd.to_numeric(m,errors='ignore').dtype)
# [1 2 3]
# int64

print()
# print(pd.to_numeric(m,errors='coerce').fillna(0)) # 兜底填充
# AttributeError: 'numpy.ndarray' object has no attribute 'fillna'

# 错误处理，如果解析失败则使用 NaN，并使用 fillna() 填充 NaN 为 0
# 需要先将 numpy.ndarray 转换为 pandas.Series 对象
print(pd.Series(pd.to_numeric(m, errors='coerce')).fillna(0))  # 运行正常
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
print(pd.to_numeric(m, downcast='integer'))  # 至少为有符号int数据类型
print(pd.to_numeric(m, downcast='signed'))  # 同上
# [1 2 3]
print(pd.to_numeric(m, downcast='unsigned'))  # 至少为无符号int数据类型 [1 2 3] ||它适用于所有数据都是非负的情况。
print(pd.to_numeric(m, downcast='float'))  # 至少为float浮点类型 [1. 2. 3.]

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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
print(df.dtypes)
print()

print(df.Q1.astype('int32').dtypes)  # int32
print(df.astype({'Q1': 'int32', 'Q2': 'int16'}).dtypes)
# name    object
# team    object
# Q1       int32
# Q2       int16
# Q3       int64
# Q4       int64
# dtype: object

print()
print(df.index.dtype)  # int64  # 索引类型转换
print(df.index.astype('int32'))  # 所有数据转换为int32 | Index([0, 1, 2, 3, 4, 5], dtype='int32')
print(df.astype({'Q1': 'int16'}).dtypes)  # 指定字段转指定类型

s = pd.Series([1.2, 3.4, 2.56, 2.456, 4, -0.12])  # {'amount':[1.2,3.4,2.56,2.456,4,-0.12]}
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
print(s.astype('float32', copy=False))  # 不与原数据关联
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
s = pd.Series(['23.45%', 89.23])
da = pd.DataFrame({'num': ['23.45%']})
print(s)
print(da.dtypes)
# data.rate.apply(lambda x: x.replace('%', '')).astype('float')/100
print(da.num.apply(lambda x: x.replace('%', '')))
print(da.num.apply(lambda x: x.replace('%', '')).astype('float'))
print(da.num.apply(lambda x: x.replace('%', '')).astype('float') / 100)
# 0    23.45
# Name: num, dtype: object
# 0    23.45
# Name: num, dtype: float64
# 0    0.2345
# Name: num, dtype: float64


db = pd.DataFrame({'num': ['23.45%', 89.23]})
# AttributeError: 'float' object has no attribute 'replace'
db = db.astype('string')  # 如果不转换文本类型 下一步会报错！
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
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
print(t.astype('datetime64[ns]'))  # 同上

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
print('\t5.3 数据排序')
print('\t5.3.1 索引排序')
print()
# update20240318
'''
df.sort_index()实现按索引排序，默认以从小到大的升序方式排列。如希望按降序排序，传入ascending=False：

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

# TODO 索引降序
print('索引降序')
print(df.sort_index())  # 默认按 行索引 排序
print(df.sort_index(ascending=True))  # 同上
print(df.sort_index(ascending=False))  # 索引降序
#     name team   Q1  Q2  Q3   Q4
# 5   Rick    B  100  99  97  100
# 4    Oah    D   65  49  61   86
# ...

print()
print('在列索引方向上排序')
print(df.sort_index(axis=1, ascending=False))
#   team   name   Q4  Q3  Q2   Q1
# 0    E  Liver   64  24  21   89
# ...

print()
print('更多方法')
# 创建一个无序的 Series 对象
s = pd.Series([3, 1, 4, 1, 5], index=[5, 3, 4, 1, 2])

# 创建一个无序的 DataFrame 对象
df = pd.DataFrame({
    'team': ['B', 'A', 'D', 'C'],
    'score': [85, 90, 78, 88]
}, index=[2, 3, 1, 4])

# print(s)
# print(df)
print()

print('# 对 Series 对象升序排序')
s_sorted = s.sort_index()
print(s_sorted)
print()
# 1    1
# 2    5
# 3    1
# 4    4
# 5    3
# dtype: int64

print('# 对 DataFrame 对象升序排序')
df_sorted = df.sort_index()
print(df_sorted)
print()
#   team  score
# 1    D     78
# 2    B     85
# 3    A     90
# 4    C     88

print('# 只对 DataFrame 的某一列按索引排序')
team_sorted = df.team.sort_index()
print(team_sorted)
print()
# 1    D
# 2    B
# 3    A
# 4    C
# Name: team, dtype: object

print('# 对 Series 对象降序排序')
s_sorted_desc = s.sort_index(ascending=False)
print(s_sorted_desc)
print()
# 5    3
# 4    4
# 3    1
# 2    5
# 1    1
# dtype: int64

print('# 对原 Series 对象就地排序')
s.sort_index(inplace=True)
print(s)
print()
# 1    1
# 2    5
# 3    1
# 4    4
# 5    3
# dtype: int64

print('# 对 Series 对象排序并重置索引')
s_sorted_reset = s.sort_index(ignore_index=True)
print(s_sorted_reset)
print()
# 0    1
# 1    5
# 2    1
# 3    4
# 4    3
# dtype: int64


print("sort_values() 使用 na_position='first' 参数排序并将空值放在前面")

# 创建一个包含空值的 Series 对象
s_with_na = pd.Series([np.nan, 1, 3, np.nan, 2])
print(s_with_na)
# 对 Series 对象排序并将空值放在前面
s_sorted_na_first = s_with_na.sort_values(na_position='first')
print(s_sorted_na_first)
print()
# 0    NaN
# 3    NaN
# 1    1.0
# 4    2.0
# 2    3.0
# dtype: float64
s_sorted_na_first = s_with_na.sort_values()
print(s_sorted_na_first)
print()
# 1    1.0
# 4    2.0
# 2    3.0
# 0    NaN
# 3    NaN
# dtype: float64

print()
print('sort_index() 参数level=0 和 sort_remaining=False')
# 创建一个无序的 Series 对象
s = pd.Series([3, 1, 4, 1, 5], index=[5, 3, 4, 1, 2])

# 创建一个无序的 DataFrame 对象
df = pd.DataFrame({
    'team': ['B', 'A', 'D', 'C'],
    'score': [85, 90, 78, 88]
}, index=[2, 3, 1, 4])

# print(s)
# print(df)
print()

# 创建一个多层索引的 Series 对象
s_multi_index = pd.Series([3, 1, 4, 1, 5],
                          index=pd.MultiIndex.from_tuples([(1, 'c'), (1, 'b'), (1, 'a'), (2, 'a'), (2, 'b')]))
print(s_multi_index)
# 1  c    3
#    b    1
#    a    4
# 2  a    1
#    b    5
# dtype: int64
# 对 Series 对象的第一级索引排序
s_sorted_level = s_multi_index.sort_index(level=0)
print(s_sorted_level)
# 1  a    4
#    b    1
#    c    3
# 2  a    1
#    b    5
# dtype: int64

# 在多层索引中使用 level 和 sort_remaining=False 参数
# 对 Series 对象的第二级索引排序，但不排序其余部分
s_sorted_level_no_remaining = s_multi_index.sort_index(level=1, sort_remaining=False)
print(s_sorted_level_no_remaining)
# 1  a    4
# 2  a    1
# 1  b    1
# 2  b    5
# 1  c    3
# dtype: int64

s_sorted_level_no_remaining = s_multi_index.sort_index(level=1, sort_remaining=True)
print(s_sorted_level_no_remaining)
# 1  a    4
# 2  a    1
# 1  b    1
# 2  b    5
# 1  c    3
# dtype: int64

'''
。sort_index 方法用于根据索引排序，level 参数用于指定多级索引的哪个级别进行排序，
而 sort_remaining 参数用于控制是否对除了指定级别之外的其他级别的索引也进行排序。

当 sort_remaining=True（默认值）时，会在对指定索引级别排序后，对剩余的索引级别按照字典顺序进行排序。
当 sort_remaining=False 时，只会对指定的索引级别进行排序，而不对剩余的索引级别进行排序。
对于 sort_values 方法，它并没有 sort_remaining 参数，它只用于根据数据值进行排序，并可以通过 na_position 参数控制 NaN 值的位置。
'''

print()
print('df.reindex()指定自己定义顺序的索引，实现行和列的顺序重新定义：')
df1 = pd.DataFrame({
    'A': [1, 2, 4],
    'B': [3, 5, 6]
}, index=['a', 'b', 'c'])
print(df1)
#    A  B
# a  1  3
# b  2  5
# c  4  6

print()
print('# 按要求重新指定索引顺序')
print(df1.reindex(['c', 'b', 'a']))
#    A  B
# c  4  6
# b  2  5
# a  1  3

print()
print('# 指定列顺序')
print(df1.reindex(['B', 'A'], axis=1))
#    B  A
# a  3  1
# b  5  2
# c  6  4


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.3 数据排序')
print('\t5.3.2 数值排序')
print()
# update20240318
'''
数据值的排序主要使用sort_values()，数字按大小顺序，字符按字母顺序。Series和DataFrame都支持此方法：

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print(df.Q1.sort_values())  # 单列按照数值 升序排序
# 1     36
# 2     57
# 4     65
# 0     89
# 3     93
# 5    100
# Name: Q1, dtype: int64

print()
print('# DataFrame需要传入一个或多个排序的列名：')
print(df.sort_values('Q4'))
#     name team   Q1  Q2  Q3   Q4
# 1   Arry    C   36  37  37   57
# 0  Liver    E   89  21  24   64
# 3  Eorge    C   93  96  71   78
# 2    Ack    A   57  60  18   84
# 4    Oah    D   65  49  61   86
# 5   Rick    B  100  99  97  100

# TODO 默认排序是升序，但可以指定排序方式，下例先按team升序排列，如遇到相同的team再按name降序排列。
print()
print(df.sort_values(['team', 'name'], ascending=[True, False]))
#     name team   Q1  Q2  Q3   Q4
# 2    Ack    A   57  60  18   84
# 5   Rick    B  100  99  97  100
# 3  Eorge    C   93  96  71   78
# 1   Arry    C   36  37  37   57
# 4    Oah    D   65  49  61   86
# 0  Liver    E   89  21  24   64

print()
print('# 其它常用方法')
# s.sort_values(ascending=False) # 降序
# s.sort_values(inplace=True) # 修改生效
# s.sort_values(na_position='first') # 空值在前
# # df按指定字段排列
# df.sort_values(by=['team'])
# df.sort_values('Q1')
# # 按多个字段，先排team，在同team内再看Q1
# df.sort_values(by=['team', 'Q1'])
# # 全降序
# df.sort_values(by=['team', 'Q1'], ascending=False)
# # 对应指定team升Q1降
# df.sort_values(by=['team', 'Q1'], ascending=[True, False])
# 索引重新0-(n-1)排
print(df.sort_values('team', ignore_index=True))
#     name team   Q1  Q2  Q3   Q4
# 0    Ack    A   57  60  18   84
# 1   Rick    B  100  99  97  100
# 2   Arry    C   36  37  37   57
# 3  Eorge    C   93  96  71   78
# 4    Oah    D   65  49  61   86
# 5  Liver    E   89  21  24   64


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.3 数据排序')
print('\t5.3.3 混合排序')
print()
# update20240320
'''
有时候需要用索引和数据值混合排序。下例中假如name是索引，我们需要先按team排名，再按索引排名：

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

# df.set_index('name',inplace=True) # 设置那么列为索引
df1 = df.set_index('name')
# print(df1)
df1.index.names = ['s_name']  # 给索引起名
print(df1.sort_values(by=['s_name', 'team']))  # 排序 sort_index() 无 by= 参数；sort_values()有 by= 参数
#        team   Q1  Q2  Q3   Q4
# s_name
# Ack       A   57  60  18   84
# Arry      C   36  37  37   57
# Eorge     C   93  96  71   78
# Liver     E   89  21  24   64
# Oah       D   65  49  61   86
# Rick      B  100  99  97  100
'''以下方法也可以实现上述需求，不过要注意顺序：'''

print()
# 设置索引，按team排序，再按索引排序
print(df.set_index('name').sort_values('team').sort_index())  # 结果同上

'''看结果 应该是先按照team排序后 在按照index排序 实际还是以name索引列排序为准'''
# print()
# print(df.set_index('name').sort_values('team'))
# print()
# print(df.set_index('name').sort_index()) # 结果等于print(df.set_index('name').sort_values('team').sort_index())
print()
print('# 按姓名排序后取出排名后的索引列表')
print(df.name.sort_values().index)  # Index([2, 1, 3, 0, 4, 5], dtype='int64')

print()
print('# 将新的索引应用到数据中')
print(df.reindex(df.name.sort_values().index))  # 将新的索引应用到数据中
#     name team   Q1  Q2  Q3   Q4
# 2    Ack    A   57  60  18   84
# 1   Arry    C   36  37  37   57
# 3  Eorge    C   93  96  71   78
# 0  Liver    E   89  21  24   64
# 4    Oah    D   65  49  61   86
# 5   Rick    B  100  99  97  100


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.3 数据排序')
print('\t5.3.4 按值大小排序')
print()
# update20240320
'''
nsmallest()和nlargest()用来实现数字列的排序，并可指定返回的个数：

以下是一些具体的使用方法举例：

'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 先按Q1最小在前，如果相同，Q2小的在前')
print(df.nsmallest(5, ['Q1', 'Q2']))  # 取前5行数据
#     name team  Q1  Q2  Q3  Q4
# 1   Arry    C  36  37  37  57
# 2    Ack    A  57  60  18  84
# 4    Oah    D  65  49  61  86
# 0  Liver    E  89  21  24  64
# 3  Eorge    C  93  96  71  78

'''仅支持数字类型的排序。下面是几个其他示例：'''

print()
s = pd.Series([9, 4, 5, 5, 2, 3, 0], name='num')
print(s)
print(s.nsmallest(3))  # 最小的3个
print(s.nlargest(3))  # 最大的3个

print()
print('指定列')
print(df.nlargest(3, 'Q1'))  # Q1列最大的前3行数据
print(df.nlargest(5, ['Q1', 'Q2']))  # Q1列最大的前5行数据 如果有相同值，看Q2列数据较大的
print(df.nsmallest(5, ['Q1', 'Q2']))  # Q1列最小的前5行数据 如果有相同值，看Q2列数据较小的

'''
5.3.5 小结
本节介绍了索引的排序、数值的排序以及索引和数值混合的排序方法。
在实际需求中，更加复杂的排序可能需要通过计算增加辅助列来实现。
'''

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
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.5 高级过滤')
print('\t5.5.1 df.where()')
print()
# update20240408
'''
本节介绍几个非常好用的数据过滤输出方法，它们经常用在一些复杂的数据处理过程中。
df.where()和df.mask()通过给定的条件对原数据是否满足条件进行筛选，最终返回与原数据形状相同的数据。
为了方便讲解，我们仅取我们的数据集的数字部分，即只有Q1到Q4列：
'''
# 小节注释
'''
df.where()中可以传入一个布尔表达式、布尔值的Series/DataFrame、序列或者可调用的对象，
然后与原数据做对比，返回一个行索引与列索引与原数据相同的数据，且在满足条件的位置保留原值，
在不满足条件的位置填充NaN。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

# 只保留数字类型列
df1 = df.select_dtypes(include='number')
print(df1)

print()
print('# 数值大于70')
print(df1.where(df1 > 70))  # 不满足条件的填充NaN ,满足条件的 保留原值
#       Q1    Q2    Q3     Q4
# 0   89.0   NaN   NaN    NaN
# 1    NaN   NaN   NaN    NaN
# 2    NaN   NaN   NaN   84.0
# 3   93.0  96.0  71.0   78.0
# 4    NaN   NaN   NaN   86.0
# 5  100.0  99.0  97.0  100.0

print()
print('# 传入一个可调用对象，这里我们用lambda：')
# Q1列大于50
print(df1.where(lambda d: d.Q1 > 50))  # 条件不满足 整行值全部填充为NaN
#       Q1    Q2    Q3     Q4
# 0   89.0  21.0  24.0   64.0
# 1    NaN   NaN   NaN    NaN
# 2   57.0  60.0  18.0   84.0
# 3   93.0  96.0  71.0   78.0
# 4   65.0  49.0  61.0   86.0
# 5  100.0  99.0  97.0  100.0

print()
print('# 条件为一个布尔值的Series：')
# 传入一个布尔值Series，前三个为真
print(df1.Q1.where(pd.Series([True] * 3)))
# print(pd.Series([True]*3))
# 0    89.0
# 1    36.0
# 2    57.0
# 3     NaN
# 4     NaN
# 5     NaN
# Name: Q1, dtype: float64

'''上例中不满足条件的都返回为NaN，我们可以指定一个值或者算法来替换NaN：'''

print()
print('# 大于等于60分的显示成绩，小于的显示“不及格”')
print(df1.where(df1 >= 60, '不及格'))
#     Q1   Q2   Q3   Q4
# 0   89  不及格  不及格   64
# 1  不及格  不及格  不及格  不及格
# 2  不及格   60  不及格   84
# 3   93   96   71   78
# 4   65  不及格   61   86
# 5  100   99   97  100

print()
print('# 给定一个算法，df为偶数时显示原值减去20后的相反数')

# 定义一个数是否为偶数的表达式 c
c = df1 % 2 == 0
# 传入c, 为偶数时显示原值减去20后的相反数
print(df1.where(~c, -(df1 - 20)))
#    Q1  Q2  Q3  Q4
# 0  89  21  -4 -44
# 1 -16  37  37  57
# 2  57 -40   2 -64
# 3  93 -76  71 -58
# 4  65  49  61 -66
# 5 -80  99  97 -80

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.5 高级过滤')
print('\t5.5.2 np.where()')
print()
# update20240408
'''
本节介绍几个非常好用的数据过滤输出方法，它们经常用在一些复杂的数据处理过程中。
df.where()和df.mask()通过给定的条件对原数据是否满足条件进行筛选，最终返回与原数据形状相同的数据。
为了方便讲解，我们仅取我们的数据集的数字部分，即只有Q1到Q4列：
'''
# 小节注释
'''
np.where()是NumPy的一个功能，虽然不是Pandas提供的，但可以弥补df.where()的不足，所以有必要一起介绍。
df.where()方法可以将满足条件的值筛选出来，将不满足的值替换为另一个值，但无法对满足条件的值进行替换，
而np.where()就实现了这种功能，达到SQL中if（条件，条件为真的值，条件为假的值）的效果。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

# 只保留数字类型列
df1 = df.select_dtypes(include='number')
print(df1)

print()
print('# 小于60分为不及格')
# np.where()返回的是一个二维array：
print(np.where(df1 >= 60, '合格', '不合格'))
# array([0, 0, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5], dtype=int64), array([0, 3, 1, 3, 0, 1, 2, 3, 0, 2, 3, 0, 1, 2, 3], dtype=int64))
# [['合格' '不合格' '不合格' '合格']
#  ['不合格' '不合格' '不合格' '不合格']
#  ['不合格' '合格' '不合格' '合格']
#  ['合格' '合格' '合格' '合格']
#  ['合格' '不合格' '合格' '合格']
#  ['合格' '合格' '合格' '合格']]

# 让df.where()中的条件为假，从而应用np.where()的计算结果
print(df1.where(df1 == 9999999, np.where(df1 >= 60, '合格', '不合格')))
#     Q1   Q2   Q3   Q4
# 0   合格  不合格  不合格   合格
# 1  不合格  不合格  不合格  不合格
# 2  不合格   合格  不合格   合格
# 3   合格   合格   合格   合格
# 4   合格  不合格   合格   合格
# 5   合格   合格   合格   合格

print()
print('# 包含是、否结果的Series')
'''下例是np.where()对一个Series（d.avg为计算出来的虚拟列）进行判断，返回一个包含是、否结果的Series。'''
print(df1.assign(avg=df1.mean(1))
      .assign(及格=lambda d: np.where(d.avg >= 50, '是', '否')))
#     Q1  Q2  Q3   Q4    avg 及格
# 0   89  21  24   64  49.50  否
# 1   36  37  37   57  41.75  否
# 2   57  60  18   84  54.75  是
# 3   93  96  71   78  84.50  是
# 4   65  49  61   86  65.25  是
# 5  100  99  97  100  99.00  是

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.5 高级过滤')
print('\t5.5.3 df.mask()')
print()
# update20240409
'''
本节介绍几个非常好用的数据过滤输出方法，它们经常用在一些复杂的数据处理过程中。
df.where()和df.mask()通过给定的条件对原数据是否满足条件进行筛选，最终返回与原数据形状相同的数据。
为了方便讲解，我们仅取我们的数据集的数字部分，即只有Q1到Q4列：
'''
# 小节注释
'''
df.mask()的用法和df.where()基本相同，唯一的区别是df.mask()将满足条件的位置填充为NaN。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

# 只保留数字类型列
df1 = df.select_dtypes(include='number')
print(df1)

print()
print('# 小于60分 保留原值，否则NaN填充')
print(df1.mask(df1 >= 60))
#      Q1    Q2    Q3    Q4
# 0   NaN  21.0  24.0   NaN
# 1  36.0  37.0  37.0  57.0
# 2  57.0   NaN  18.0   NaN
# 3   NaN   NaN   NaN   NaN
# 4   NaN  49.0   NaN   NaN
# 5   NaN   NaN   NaN   NaN

print()
print('# 对满足条件的位置指定填充值')
s = df1.Q1
# print(s)
print(df1.Q1.mask(s > 80, '优秀'))
# 0    优秀
# 1    36
# 2    57
# 3    优秀
# 4    65
# 5    优秀
# Name: Q1, dtype: object
print()
print('# 通过数据筛选返回布尔序列')
# df.mask()和df.where()还可以通过数据筛选返回布尔序列：
# 返回布尔序列，符合条件的行值为True
print((df.where((df.team == 'A') & (df.Q1 > 30)) == df).Q1)
# 0    False
# 1    False
# 2     True
# 3    False
# 4    False
# 5    False
# Name: Q1, dtype: bool

print()
# 返回布尔序列，符合条件的行值为False
print((df.mask((df.team == 'A') & (df.Q1 > 30)) == df).Q1)
# 0     True
# 1     True
# 2    False
# 3     True
# 4     True
# 5     True
# Name: Q1, dtype: bool


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
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.6 数据迭代')
print('\t5.6.1 迭代Series')
print()
# update20240410
'''
数据迭代和数据遍历都是按照某种顺序逐个对数据进行访问和操作，
在Python中大多由for语句来引导。Pandas中的迭代操作可以将数据按行或者按列遍历，
我们可以进行更加细化、个性化的数据处理。
'''
# 小节注释
'''
Series本身是一个可迭代对象，Series df.name.values返回array结构数据可用于迭代，
不过可直接对Series使用for语句来遍历它的值：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print()
print('# 迭代指定的列')

print(df.name)

for i in df['name']:
    print(i)

for i, n, q in zip(df.index, df.name, df.Q1):
    print(i, n, q)
# 0 Liver 89
# 1 Arry 36
# 2 Ack 57
# 3 Eorge 93
# 4 Oah 65
# 5 Rick 100


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.6 数据迭代')
print('\t5.6.2 df.iterrows()')
print()
# update20240410
# 小节注释
'''
df.iterrows()生成一个可迭代对象，将DataFrame行作为（索引，行数据）组成的Series数据对进行迭代。
在for语句中需要两个变量来承接数据：一个为索引变量，即使索引在迭代中不会使用（这种情况可用useless作为变量名）；
另一个为数据变量，读取具体列时，可以使用字典的方法和对象属性的方法。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print()
print('# 迭代，使用name、Q1数据')
for index, row in df.iterrows():
    print(index, row['name'], row.Q1)
# 0 Liver 89
# 1 Arry 36
# 2 Ack 57
# 3 Eorge 93
# 4 Oah 65
# 5 Rick 100
'''df.iterrows()是最常用、最方便的按行迭代方法。'''

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.6 数据迭代')
print('\t5.6.3 df.itertuples()')
print()
# update20240410
# 小节注释
'''
df.itertuples()生成一个namedtuples类型数据，name默认名为Pandas，可以在参数中指定。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print()
print('# 迭代，使用name、Q1数据')
for row in df.itertuples():
    print(row)
# Pandas(Index=0, name='Liver', team='E', Q1=89, Q2=21, Q3=24, Q4=64)
# Pandas(Index=1, name='Arry', team='C', Q1=36, Q2=37, Q3=37, Q4=57)
# Pandas(Index=2, name='Ack', team='A', Q1=57, Q2=60, Q3=18, Q4=84)
# Pandas(Index=3, name='Eorge', team='C', Q1=93, Q2=96, Q3=71, Q4=78)
# Pandas(Index=4, name='Oah', team='D', Q1=65, Q2=49, Q3=61, Q4=86)
# Pandas(Index=5, name='Rick', team='B', Q1=100, Q2=99, Q3=97, Q4=100)

print()
print('# 以下是其它一些使用方法示例：')
# 不包含索引
for row in df.itertuples(index=False):
    print(row)
# Pandas(name='Liver', team='E', Q1=89, Q2=21, Q3=24, Q4=64)
# Pandas(name='Arry', team='C', Q1=36, Q2=37, Q3=37, Q4=57)
# ...

print()
# 自定义name
for row in df.itertuples(index=False, name='Gairuo'):
    print(row)
# Gairuo(name='Liver', team='E', Q1=89, Q2=21, Q3=24, Q4=64)
# Gairuo(name='Arry', team='C', Q1=36, Q2=37, Q3=37, Q4=57)
# ...

print()
# 使用数据
for row in df.itertuples():
    print(row.Index, row.name)
# 0 Liver
# 1 Arry
# 2 Ack
# 3 Eorge
# 4 Oah
# 5 Rick

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.6 数据迭代')
print('\t5.6.4 df.items()')
print()
# update20240410
# 小节注释
'''
df.items()和df.iteritems()功能相同，它迭代时返回一个（列名，本列的Series结构数据），实现对列的迭代：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print()
print('# 迭代，使用name、Q1数据')

for label, ser in df.items():
    print(label)
    print(ser[:3], end='\n\n')

# name
# 0    Liver
# 1     Arry
# 2      Ack
# Name: name, dtype: object
#
# team
# 0    E
# 1    C
# 2    A
# Name: team, dtype: object
# ........


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.6 数据迭代')
print('\t5.6.5 按列迭代')
print()
# update20240410
# 小节注释
'''
除了df.items()，如需要迭代一个DataFrame的列，可以直接对DataFrame迭代，会循环得到列名：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 直接对DataFrame迭代')

for column in df:
    print(column)
# name
# team
# Q1
# Q2
# Q3
# Q4

print()
print('# 再利用df [列名]的方法迭代列：')

for column in df:
    print(df[column])

print()
print('# 可对每个列的内容进行迭代')

for column in df:
    for i in df[column]:
        print(i)
# Liver
# Arry
# Ack
# Eorge
# Oah
# Rick
# E
# C
# ...

print()
print('# 可以迭代指定列')

for i in df.name:
    print(i)

print()
print('# 只迭代想要的列')
l = ['name', 'Q1']
cols = df.columns.intersection(l)
for col in cols:
    print(col)

# name
# Q1

'''
与df.iterrows()相比，df.itertuples()运行速度会更快一些，推荐在数据量庞大的情况下优先使用。
'''

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.7 函数应用')
print('\t5.7.1 pipe()')
print()
# update20240412
'''
我们知道，函数可以让复杂的常用操作模块化，既能在需要使用时直接调用，达到复用的目的，也能简化代码。
Pandas提供了几个常用的调用函数的方法。
pipe()：应用在整个DataFrame或Series上。
apply()：应用在DataFrame的行或列中，默认为列。
applymap()：应用在DataFrame的每个元素中。
map()：应用在Series或DataFrame的一列的每个元素中。
'''
# 小节注释
'''
Pandas提供的pipe()叫作管道方法，它可以让我们写的分析过程标准化、流水线化，达到复用目标，
它也是最近非常流行的链式方法的重要代表。
DataFrame和Series都支持pipe()方法。pipe()的语法结构为df.pipe(<函数名>, <传给函数的参数列表或字典>)。
它将DataFrame或Series作为函数的第一个参数（见图5-1），
可以根据需求返回自己定义的任意类型数据。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# pipe()可以将复杂的调用简化:')

df1 = df.select_dtypes(include='number')


# print(df1)

def h(df1):
    # 假设 h 函数将所有值增加 1
    return df1 + 1


# print(h(df1)) # 测试正常

def g(df1, arg1):
    # g 函数 将 所有值乘以 arg1
    return df1 * arg1


# print(g(df1,2)) # 测试正常

def f(df1, arg2, arg3):
    # f 函数 将所有值 加上 arg2 和 arg3 的和
    return df1 + arg2 + arg3


# print(f(df1,-80,1)) # 测试正常

df2 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
# print(df2)

print()
print('# 不使用 pipe,直接嵌套调用函数')
result1 = f(g(h(df2), arg1=2), arg2=3, arg3=4)
print(result1)
#     A   B
# 0  11  17
# 1  13  19
# 2  15  21

print('# 使用 pipe 连接函数')
result2 = (df2.pipe(h)
           .pipe(g, arg1=2)
           .pipe(f, arg2=3, arg3=4)
           )

print(result2)
#     A   B
# 0  11  17
# 1  13  19
# 2  15  21


print()
print('# 实际案例 2：')


# 定义一个函数，给所有季度的成绩加n，然后增加平均数
# 其中n中要加的值为必传参数
def add_mean(rdf, n):
    # pass
    df3 = rdf.copy()
    df3 = df3.loc[:, 'Q1':'Q4'].applymap(lambda x: x + n)
    df3['avg'] = df3.loc[:, 'Q1':'Q4'].mean(1)
    return df3


print(df.pipe(add_mean, 100))
#     Q1   Q2   Q3   Q4     avg
# 0  189  121  124  164  149.50
# 1  136  137  137  157  141.75
# 2  157  160  118  184  154.75
# 3  193  196  171  178  184.50
# 4  165  149  161  186  165.25
# 5  200  199  197  200  199.00

print()
print('# 使用lambda')
# 筛选出Q1大于等于80且Q2大于等于90的数据
# df.pipe(lambda df_, x, y: df_[(df_.Q1 >= x) & (df_.Q2 >= y)], 80, 90)

result3 = df.pipe(lambda df_, x, y: df_[(df_.Q1 >= x) & (df_.Q2 >= y)], 80, 90)
print(result3)
#     name team   Q1  Q2  Q3   Q4
# 3  Eorge    C   93  96  71   78
# 5   Rick    B  100  99  97  100

print()
print(df.loc[(df.loc[:, 'Q2'] >= 90) & (df.loc[:, 'Q1'] >= 80)])  # 结果同上

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.7 函数应用')
print('\t5.7.2 apply()')
print()
# update20240412
# 小节注释
'''
apply()可以对DataFrame按行和列（默认）进行函数处理，也支持Series。
如果是Series，逐个传入具体值，DataFrame逐行或逐列传入，

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 将name全部变为小写:')
# print(df.name)
print(df.name.apply(lambda x: x.lower()))
# 0    liver
# 1     arry
# 2      ack
# 3    eorge
# 4      oah
# 5     rick
# Name: name, dtype: object
print()
print('# 案例二：DataFrame例子')
print('# 去掉一个最高分和一个最低分再算出平均分')


# s = df.Q1

def my_mean(s):
    max_min_ser = pd.Series([-s.max(), -s.min()])
    return s._append(max_min_ser).sum() / (s.count() - 2)


# 对数字列应用函数
print(df.select_dtypes(include='number').apply(my_mean))
# print(my_mean(s))
# Q1    76.00
# Q2    60.50
# Q3    48.25
# Q4    78.00
# dtype: float64

print()
print('# 同样的算法以学生为维度计算')
print(df.set_index('name')
      .select_dtypes(include='number')
      .apply(my_mean, axis=1)
      )  # 横向计算
# name
# Liver    44.0
# Arry     37.0
# Ack      58.5
# Eorge    85.5
# Oah      63.0
# Rick     99.5
# dtype: float64

print()
print('# 判断一个值是否在另一个类似列表的列中')

df11 = pd.DataFrame({'s': [1, 2, 3, 6],
                     's_list': [[1, 2], [2, 3], [3, 4], [4, 5]]})

# print(df11)

bool_series = df11.apply(lambda d: d.s in d.s_list, axis=1)
print(bool_series)
# 0    True
# 1    True
# 2    True
# 3    False
# dtype: bool

print()
print('# 将布尔序列转换为 0 和 1 序列')
int_series = bool_series.astype(int)
print(int_series)
# 0    1
# 1    1
# 2    1
# 3    0
# dtype: int32

# 它常被用来与NumPy库中的np.where()方法配合使用，如下例：
print()
print('# 函数，将大于90分的数字标记为good')
fun = lambda x: np.where(x.team == 'A' and x.Q1 > 30, 'good', 'other')  #
print(df.apply(fun, axis=1))
# 0    other
# 1    other
# 2     good
# 3    other
# 4    other
# 5    other
# dtype: object

print('# 结果同上')
print(df.apply(lambda x: x.team == 'A' and x.Q1 > 30, axis=1)
      .map({True: 'good', False: 'other'}))

# df.apply(lambda x: 'good' if x.team=='A' and x.Q1>90 else '', axis=1)
# print(df.apply(lambda x: 'good' if x.team == 'A' and x.Q1>30 else '',axis=1)) # 逻辑同上

# result = df.apply(lambda x: 'good' if x.team == 'A' and x.Q1>30 else '',axis=1)
# result = df.apply(lambda x: True if x.team == 'A' and x.Q1>30 else False,axis=1)
# print(df.where(result).dropna()) # 不符合填充空值 删除空值！

print()
print('小节')
# 总结一下，apply()可以应用的函数类型如下：
# df.apply(fun) # 自定义
# df.apply(max) # Python内置函数
# df.apply(lambda x: x*2) # lambda
# df.apply(np.mean) # NumPy等其他库的函数
# df.apply(pd.Series.first_valid_index) # Pandas自己的函数

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.7 函数应用')
print('\t5.7.3 applymap()')
print()
# update20240416

# 小节注释
'''
df.applymap()可实现元素级函数应用，即对DataFrame中所有的元素（不包含索引）应用函数处理

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 计算数据的长度:')


# 使用lambda时，变量是指每一个具体的值。
def mylen(x):
    return len(str(x))


print(df.applymap(lambda x: mylen(x)))  # 应用函数
#    name  team  Q1  Q2  Q3  Q4
# 0     5     1   2   2   2   2
# 1     4     1   2   2   2   2
# 2     3     1   2   2   2   2
# 3     5     1   2   2   2   2
# 4     3     1   2   2   2   2
# 5     4     1   3   2   2   3

print()
print(df.applymap(mylen))  # 结果同上

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.7 函数应用')
print('\t5.7.4 map()')
print()
# update20240416
# 小节注释
'''
map()根据输入对应关系映射值返回最终数据，用于Series对象或DataFrame对象的一列。
传入的值可以是一个字典，键为原数据值，值为替换后的值。
可以传入一个函数（参数为Series的每个值），
还可以传入一个字符格式化表达式来格式化数据内容。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 枚举替换:')
print()
print(df.team.map({'A': '一班', 'B': '二班', 'C': '三班', 'D': '四班'}))  # 枚举替换
# 0    NaN
# 1     三班
# 2     一班
# 3     三班
# 4     四班
# 5     二班
# Name: team, dtype: object

print()
print(df.team.map('I am a {}'.format))
# 0    I am a E
# 1    I am a C
# 2    I am a A
# 3    I am a C
# 4    I am a D
# 5    I am a B
# Name: team, dtype: object

# na_action='ignore' 参数指定如果遇到 NaN 值，则忽略它，不对其应用任何操作。
print(df.team.map('I am a {}'.format, na_action='ignore'))  # 结果同上

# t = pd.Series({'six':6.,'seven':7.})
# s.map(t)
print(t)


# 应用函数
def f(x):
    return len(str(x))


print(df['name'].map(f))
# 0    5
# 1    4
# 2    3
# 3    5
# 4    3
# 5    4
# Name: name, dtype: int64


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.7 函数应用')
print('\t5.7.5 agg()')
print()
# update20240416
# 小节注释
'''
agg()一般用于使用指定轴上的一项或多项操作进行汇总，
可以传入一个函数或函数的字符，还可以用列表的形式传入多个函数。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 每列的最大值')
print(df.agg('max'))
# name    Rick
# team       E
# Q1       100
# Q2        99
# Q3        97
# Q4       100
# dtype: object

print()
print('# 将所有列聚合产生sum和min两行')
print(df.agg(['sum', 'min']))
#                          name    team   Q1   Q2   Q3   Q4
# sum  LiverArryAckEorgeOahRick  ECACDB  440  362  308  469
# min                       Ack       A   36   21   18   57

print()
print('# 序列多个聚合')
print(df.agg({'Q1': ['sum', 'min'], 'Q2': ['min', 'max']}))
#         Q1    Q2
# sum  440.0   NaN
# min   36.0  21.0
# max    NaN  99.0

print()
print('# 分组后聚合')
print(df.groupby('team').agg('max'))
#        name   Q1  Q2  Q3   Q4
# team
# A       Ack   57  60  18   84
# B      Rick  100  99  97  100
# C     Eorge   93  96  71   78
# D       Oah   65  49  61   86
# E     Liver   89  21  24   64
print()
print(df.Q1.agg(['sum', 'mean']))
# sum     440.000000
# mean     73.333333
# Name: Q1, dtype: float64

'''
另外，agg()还支持传入函数的位置参数和关键字参数，支持每个列
分别用不同的方法聚合，支持指定轴的方向。
'''

print()
print('# 每列使用不同的方法进行聚合')
print(df.agg(a=('Q1', max),
             b=('Q2', 'min'),
             c=('Q3', np.mean),
             d=('Q4', lambda s: s.sum() + 1)
             ))
#       Q1    Q2         Q3     Q4
# a  100.0   NaN        NaN    NaN
# b    NaN  21.0        NaN    NaN
# c    NaN   NaN  51.333333    NaN
# d    NaN   NaN        NaN  470.0
print()
print(df.groupby('name').agg(a=('Q1', max),
                             b=('Q2', 'min'),
                             c=('Q3', np.mean),
                             d=('Q4', lambda s: s.sum() + 1)
                             ))

#          a   b     c    d
# name
# Ack     57  60  18.0   85
# Arry    36  37  37.0   58
# Eorge   93  96  71.0   79
# Liver   89  21  24.0   65
# Oah     65  49  61.0   87
# Rick   100  99  97.0  101

print()
print('# 按行聚合')
print(df.loc[:, 'Q1':].agg('mean', axis='columns'))
# 0    49.50
# 1    41.75
# 2    54.75
# 3    84.50
# 4    65.25
# 5    99.00
# dtype: float64
print(df.loc[:, 'Q1':].agg('mean'))
# Q1    73.333333
# Q2    60.333333
# Q3    51.333333
# Q4    78.166667
# dtype: float64

print()
print('# 利用pd.Series.add方法对所有数据加分，other是add方法的参数')

print(df.loc[:, 'Q1':].agg(pd.Series.add, other=10))
#     Q1   Q2   Q3   Q4
# 0   99   31   34   74
# 1   46   47   47   67
# 2   67   70   28   94
# 3  103  106   81   88
# 4   75   59   71   96
# 5  110  109  107  110

'''agg()的用法整体上与apply()极为相似。'''

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.7 函数应用')
print('\t5.7.6 transform()')
print()
# update20240417

# 小节注释
'''
DataFrame或Series自身调用函数并返回一个与自身长度相同的数据。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 应用匿名函数')
print(df.transform(lambda x: x * 2))
#          name team   Q1   Q2   Q3   Q4
# 0  LiverLiver   EE  178   42   48  128
# 1    ArryArry   CC   72   74   74  114
# 2      AckAck   AA  114  120   36  168
# 3  EorgeEorge   CC  186  192  142  156
# 4      OahOah   DD  130   98  122  172
# 5    RickRick   BB  200  198  194  200

print()
print('# 调用多个函数')
# print(df.transform([np.sqrt,np.exp])) # 报错！
print(df.select_dtypes('number').transform([np.sqrt, np.exp]))  # 运行正常
#           Q1                      Q2  ...            Q3         Q4
#         sqrt           exp      sqrt  ...           exp       sqrt           exp
# 0   9.433981  4.489613e+38  4.582576  ...  2.648912e+10   8.000000  6.235149e+27
# 1   6.000000  4.311232e+15  6.082763  ...  1.171914e+16   7.549834  5.685720e+24
# 2   7.549834  5.685720e+24  7.745967  ...  6.565997e+07   9.165151  3.025077e+36
# 3   9.643651  2.451246e+40  9.797959  ...  6.837671e+30   8.831761  7.498417e+33
# 4   8.062258  1.694889e+28  7.000000  ...  3.104298e+26   9.273618  2.235247e+37
# 5  10.000000  2.688117e+43  9.949874  ...  1.338335e+42  10.000000  2.688117e+43

print()
print('# 调用多个函数 例2')
print(df.select_dtypes('number').transform([np.abs, lambda x: x + 1]))
#         Q1                Q2                Q3                Q4
#   absolute <lambda> absolute <lambda> absolute <lambda> absolute <lambda>
# 0       89       90       21       22       24       25       64       65
# 1       36       37       37       38       37       38       57       58
# 2       57       58       60       61       18       19       84       85
# 3       93       94       96       97       71       72       78       79
# 4       65       66       49       50       61       62       86       87
# 5      100      101       99      100       97       98      100      101

print()
print('# 调用函数 例3')
print(df.select_dtypes('number').transform({'abs'}))
#    abs abs abs  abs
# 0   89  21  24   64
# 1   36  37  37   57
# 2   57  60  18   84
# 3   93  96  71   78
# 4   65  49  61   86
# 5  100  99  97  100

print()
print('# 调用函数 例4 lambda x:x.abs()')
print(df.select_dtypes('number').transform(lambda x: x.abs()))  # 结果同上！

print()
print('# 对比2个操作')
# transform sum 每行都有
print(df.groupby('team').sum())
#            name   Q1   Q2   Q3   Q4
# team
# A           Ack   57   60   18   84
# B          Rick  100   99   97  100
# C     ArryEorge  129  133  108  135
# D           Oah   65   49   61   86
# E         Liver   89   21   24   64
print(df.groupby('team').transform(sum))
#         name   Q1   Q2   Q3   Q4
# 0      Liver   89   21   24   64
# 1  ArryEorge  129  133  108  135
# 2        Ack   57   60   18   84
# 3  ArryEorge  129  133  108  135
# 4        Oah   65   49   61   86
# 5       Rick  100   99   97  100
'''
分组后，直接使用计算函数并按分组显示合计数据。
使用transform()调用计算函数，返回的是原数据的结构，
但在指定位置上显示聚合计算后的结果，这样方便我们了解数据所在组的情况。
'''

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.7 函数应用')
print('\t5.7.7 copy()')
print()
# update20240417
# 小节注释
'''
类似于Python中copy()函数，df.copy()方法可以返回一个新对象，
这个新对象与原对象没有关系。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 应用匿名函数')

s = pd.Series([1, 2], index=['a', 'b'])
s_1 = s
s_copy = s.copy()
print(s_1 is s)  # True
print(s_copy is s)  # False

# print(s)
# print(s_1)
# print(s_copy)


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
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
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
print(df.groupby('team').agg({'Q1': 'sum',
                              'Q2': 'count',  # 必须加引号 否则报错
                              'Q3': 'mean',  # 必须加引号 否则报错
                              'Q4': 'max'
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
print(df.groupby('team').agg({'Q1': [sum, 'std', 'max'],  #
                              'Q2': 'count',
                              'Q3': 'mean',
                              'Q4': 'max'}))
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 对Series df.Q1按team分组，求和')
# 对Series也可以使用分组聚合，但相对来说场景比较少。
print(df.Q1.groupby(df.team).sum())  #
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
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
grouped2 = df.groupby(['name', 'team'])
# print(grouped2)
# <pandas.core.groupby.generic.DataFrameGroupBy object at 0x0000020EDD0EC2B0>

# ValueError: must supply a tuple to get_group with multiple grouping keys
# print(grouped2.get_group(['Arry','C'])) # 运行失败

print(grouped2.get_group(('Arry', 'C')))  # 必须使用元组才运行成功！
#    name team  Q1  Q2  Q3  Q4
# 1  Arry    C  36  37  37  57

print()
print('# 按行分组')
grouped3 = df.groupby('team', axis='columns')
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 索引值是否为偶数，分成两组')
print(df.groupby(lambda x: x % 2 == 0).sum())
#                 name team   Q1   Q2   Q3   Q4
# False  ArryEorgeRick  CCB  229  232  205  235
# True     LiverAckOah  EAD  211  130  103  234
print(df.groupby(df.index % 2 == 0).sum())  # 结果同上

print()
print('# 以下为按索引值是否大于等于4为标准分为True和False两组：')
print(df.groupby(lambda x: x >= 4).sum())
#                     name  team   Q1   Q2   Q3   Q4
# False  LiverArryAckEorge  ECAC  275  214  150  283
# True             OahRick    DB  165  148  158  186
print(df.groupby(df.index >= 4).sum())  # 结果同上

print()
print('# 列名包含Q的分成一组')
print(df.groupby(lambda x: 'Q' in x, axis=1).sum())
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
print(df.groupby(df.team.isin(['A', 'B'])).sum())
#                     name  team   Q1   Q2   Q3   Q4
# team
# False  LiverArryEorgeOah  ECCD  283  203  193  285
# True             AckRick    AB  157  159  115  184

print()
print('# 按姓名第一个字母和第二个字母分组')
# df.groupby([df.name.str[0], df.name.str[1]])
print(df.groupby([df.name.str[0], df.name.str[1]]).sum())
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
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

print(df.groupby(['team', df.name.apply(get_letter_type)]).sum())
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 使用pipe调用分组函数')
print(df.pipe(pd.DataFrame.groupby, 'team').sum())
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
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


data = {'date': pd.date_range(start='2024-04-22 00:00:00', periods=10, freq='15s'),
        'value': np.random.randint(0, 100, size=10)
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
grouped = df1.groupby(pd.Grouper(key='date', freq='60s'))
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 取消分组列为索引！')
print(df.groupby('team', as_index=False).sum())
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 不对索引排序')
print(df.groupby('team', sort=False, as_index=False).sum())
#   team       name   Q1   Q2   Q3   Q4
# 0    E      Liver   89   21   24   64
# 1    C  ArryEorge  129  133  108  135
# 2    A        Ack   57   60   18   84
# 3    D        Oah   65   49   61   86
# 4    B       Rick  100   99   97  100


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
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.3 分组对象的操作')
print('\t6.3.2 迭代分组')
print()
# update20240423
'''
上一节完成了分组对象的创建，分组对象包含数据的分组情况，
接下来就来对分组对象进行操作，获取其相关信息，为最后的数据聚合统计打好基础。
'''
# 小节注释
'''
分组对象的groups方法会生成一个字典（其实是Pandas定义的PrettyDict），
这个字典包含分组的名称和分组的内容索引列表，然后我们可以使用字典的.keys()方法取出分组名称：
▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

# 分组，为了方便案例介绍，删去name列，分组后全为数字
grouped = df.drop('name', axis=1).groupby('team')
# print(grouped.sum())
#        Q1   Q2   Q3   Q4
# team
# A      57   60   18   84
# B     100   99   97  100
# C     129  133  108  135
# D      65   49   61   86
# E      89   21   24   64

print()
print('# 查看分组内容')
print(df.groupby('team').groups)
# {'A': [2], 'B': [5], 'C': [1, 3], 'D': [4], 'E': [0]}  列表值为行索引值

print()
print('# 查看分组名')
print(df.groupby('team').groups.keys())
# dict_keys(['A', 'B', 'C', 'D', 'E'])
# print(df.groupby('team') .groups.items())

print()
print('# 多层索引，可以使用元组（）')
grouped2 = df.groupby(['team', df.name.str[0]])
# print(grouped2.get_group(('B','A'))) # 报错 因为没有这种组合 所以会报错！
print(grouped2.get_group(('B', 'R')))
#    name team   Q1  Q2  Q3   Q4
# 5  Rick    B  100  99  97  100

print()
print('# 获取分组字典数据')
'''grouped.indices返回一个字典，其键为组名，值为本组索引的array格式，可以实现对单分组数据的选取：'''
print(grouped.indices)
# {'A': array([2], dtype=int64), 'B': array([5], dtype=int64), 'C': array([1, 3], dtype=int64), 'D': array([4], dtype=int64), 'E': array([0], dtype=int64)}
print()
print('# 选择A组')
print(grouped.indices['A'])
# [2]


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.3 分组对象的操作')
print('\t6.3.2 迭代分组')
print()
# update20240423
# 小节注释
'''
我们对分组对象grouped进行迭代，看每个元素是什么数据类型：
▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

# 分组，为了方便案例介绍，删去name列，分组后全为数字
grouped = df.drop('name', axis=1).groupby('team')
# print(grouped.sum())
#        Q1   Q2   Q3   Q4
# team
# A      57   60   18   84
# B     100   99   97  100
# C     129  133  108  135
# D      65   49   61   86
# E      89   21   24   64

print()
print('# 迭代')

for g in grouped:
    print(type(g))

# <class 'tuple'>
# <class 'tuple'>
# <class 'tuple'>
# <class 'tuple'>
# <class 'tuple'>

print('# 迭代元素的数据类型')
for name, group in grouped:
    print(type(name))
    print(type(group))

# <class 'str'>
# <class 'pandas.core.frame.DataFrame'>
# <class 'str'>
# <class 'pandas.core.frame.DataFrame'>
# <class 'str'>
# <class 'pandas.core.frame.DataFrame'>
# <class 'str'>
# <class 'pandas.core.frame.DataFrame'>
# <class 'str'>
# <class 'pandas.core.frame.DataFrame'>

# for name,group in grouped:
#     print(name)
#     print(group)
'''因此我们可以通过以上方式迭代分组对象。'''

for name, group in grouped:
    print(group)

#   team  Q1  Q2  Q3  Q4
# 2    A  57  60  18  84
#   team   Q1  Q2  Q3   Q4
# 5    B  100  99  97  100
#   team  Q1  Q2  Q3  Q4
# 1    C  36  37  37  57
# 3    C  93  96  71  78
#   team  Q1  Q2  Q3  Q4
# 4    D  65  49  61  86
#   team  Q1  Q2  Q3  Q4
# 0    E  89  21  24  64


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.3 分组对象的操作')
print('\t6.3.3 选择列')
print()
# update20240423

# 小节注释
'''
我们对分组对象grouped进行迭代，看每个元素是什么数据类型：
▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

# 分组，为了方便案例介绍，删去name列，分组后全为数字
grouped = df.drop('name', axis=1).groupby('team')
# print(grouped.sum())
#        Q1   Q2   Q3   Q4
# team
# A      57   60   18   84
# B     100   99   97  100
# C     129  133  108  135
# D      65   49   61   86
# E      89   21   24   64

print()
print('# 选择分组后的某一列')
print(grouped.Q1)
# <pandas.core.groupby.generic.SeriesGroupBy object at 0x000001A66458BB50>
print(grouped['Q1'])  # 结果同上

print()
print('# 选择多列')
print(grouped[['Q1', 'Q2']])
# <pandas.core.groupby.generic.DataFrameGroupBy object at 0x0000024301F8B0D0>

print('# 对多列进行聚合运算')
print(grouped[['Q1', 'Q2']].sum())
#        Q1   Q2
# team
# A      57   60
# B     100   99
# C     129  133
# D      65   49
# E      89   21

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.3 分组对象的操作')
print('\t6.3.4 应用函数apply()')
print()
# update20240423
# 小节注释
'''
分组对象使用apply()调用一个函数，传入的是DataFrame，
返回一个经过函数计算后的DataFrame、Series或标量，然后再把数据组合。
▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 将所有的元素乘以2')
print(df.groupby('team').apply(lambda x: x * 2))
#               name team   Q1   Q2   Q3   Q4
# team
# A    2      AckAck   AA  114  120   36  168
# B    5    RickRick   BB  200  198  194  200
# C    1    ArryArry   CC   72   74   74  114
#      3  EorgeEorge   CC  186  192  142  156
# D    4      OahOah   DD  130   98  122  172
# E    0  LiverLiver   EE  178   42   48  128

# print(df.groupby('team',as_index=False).apply(lambda x:x * 2))
#            name team   Q1   Q2   Q3   Q4
# 0 2      AckAck   AA  114  120   36  168
# 1 5    RickRick   BB  200  198  194  200
# 2 1    ArryArry   CC   72   74   74  114
#   3  EorgeEorge   CC  186  192  142  156
# 3 4      OahOah   DD  130   98  122  172
# 4 0  LiverLiver   EE  178   42   48  128

print()
print('# 按分组将一列输出位列表')
print(df.groupby('team').apply(lambda x: x['name'].to_list()))

# team
# A            [Ack]
# B           [Rick]
# C    [Arry, Eorge]
# D            [Oah]
# E          [Liver]
# dtype: object

print()
print('# 查看某个组')
print(df.groupby('team').apply(lambda x: x['name'].to_list()).C)
# ['Arry', 'Eorge']

# 调用函数，实现每组Q1成绩最高的前三个：
print()
print('# 各组Q1（为参数）成绩最高的前三个')


def first_3(df_, c):
    return df_[c].sort_values(ascending=False).head(1)


# 调用函数
print(df.set_index('name').groupby('team').apply(first_3, 'Q1'))
'''
本df数据量少，所以看不出效果。
如果数据足够多 会看到如下：
team  name 
A     Ack       57
      Rick     100
      Eorge     93
B     Rick     100
      Rick     100
      Eorge     93
C     Eorge     93
      Arry      36
      Rick     100
D     Oah       65
      Rick     100
      Eorge     93
E     Liver     89
Name: Q1, dtype: int64
'''

# 通过设置group_keys，可以使分组字段不作为索引
print(df.set_index('name')
      .groupby('team', group_keys=False)
      .apply(first_3, 'Q1')
      )

print()
# 传入一个Series，映射系列不同的聚合统计算法：
print(df.groupby('team')
      .apply(lambda x: pd.Series({
    'Q1_sum': x['Q1'].sum(),
    'Q1_max': x['Q1'].max(),
    'Q2_mean': x['Q2'].mean(),
    'Q4_prodsum': (x['Q4'] * x['Q4']).sum()
}

)))

#       Q1_sum  Q1_max  Q2_mean  Q4_prodsum
# team
# A       57.0    57.0     60.0      7056.0
# B      100.0   100.0     99.0     10000.0
# C      129.0    93.0     66.5      9333.0
# D       65.0    65.0     49.0      7396.0
# E       89.0    89.0     21.0      4096.0

print()


def f_mi(x):
    d = []
    d.append(x['Q1'].sum())
    d.append(x['Q2'].max())
    d.append(x['Q3'].mean())
    d.append((x['Q4'] * x['Q4']).sum())
    return pd.Series(d, index=[['Q1', 'Q2', 'Q3', 'Q4'], ['sum', 'max', 'mean', 'prodsum']])


print(df.groupby('team').apply(f_mi))  # 同比上述代码 数值相同 列标签不同

#          Q1    Q2    Q3       Q4
#         sum   max  mean  prodsum
# team
# A      57.0  60.0  18.0   7056.0
# B     100.0  99.0  97.0  10000.0
# C     129.0  96.0  54.0   9333.0
# D      65.0  49.0  61.0   7396.0
# E      89.0  21.0  24.0   4096.0


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.3 分组对象的操作')
print('\t6.3.5 管道方法pipe()')
print()
# update20240424
# 小节注释
'''
类似于DataFrame的管道方法，分组对象的管道方法是接收之前的分组对象，
将同组的所有数据应用在方法中，最后返回的是经过函数处理过的返回数据格式。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 每组最大值和最小值之和')
print(df.groupby('team').pipe(lambda x: x.max() + x.min()))
#             name   Q1   Q2   Q3   Q4
# team
# A         AckAck  114  120   36  168
# B       RickRick  200  198  194  200
# C      EorgeArry  129  133  108  135
# D         OahOah  130   98  122  172
# E     LiverLiver  178   42   48  128

print()
print('# 定义了A组和B组平均值的差值')


# 原函数 报错！TypeError: Could not convert ['A'] to numeric
# def mean_diff(x):
#     return x.get_group('A').mean() - x.get_group('B').mean()

# 修改代码
def mean_diff(x):
    a_mean = x.get_group('A').select_dtypes(include=np.number).mean()
    b_mean = x.get_group('B').select_dtypes(include=np.number).mean()
    return a_mean - b_mean


df1 = df.drop(['name'], axis=1)
print(df1)
# 使用函数
print(df1.groupby('team').pipe(mean_diff))
# Q1   -43.0
# Q2   -39.0
# Q3   -79.0
# Q4   -16.0
# dtype: float64


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.3 分组对象的操作')
print('\t6.3.6 转换方法transform()')
print()
# update20240424
# 小节注释
'''
transform()类似于agg()，但与agg()不同的是它返回的是一个与原始数据相同形状的DataFrame，
会将每个数据原来的值一一替换成统计后的值。例如按组计算平均成绩，
那么返回的新DataFrame中每个学生的成绩就是它所在组的平均成绩。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 将所有数据替换成分组中的平均成绩')
print(df.drop(['name'], axis=1).groupby('team').transform(np.mean))
#       Q1    Q2    Q3     Q4
# 0   89.0  21.0  24.0   64.0
# 1   64.5  66.5  54.0   67.5
# 2   57.0  60.0  18.0   84.0
# 3   64.5  66.5  54.0   67.5
# 4   65.0  49.0  61.0   86.0
# 5  100.0  99.0  97.0  100.0

print()
print('# 其它方法')
print(df.groupby('team').transform(max))
#     name   Q1  Q2  Q3   Q4
# 0  Liver   89  21  24   64
# 1  Eorge   93  96  71   78
# 2    Ack   57  60  18   84
# 3  Eorge   93  96  71   78
# 4    Oah   65  49  61   86
# 5   Rick  100  99  97  100

print(df.drop(['name'], axis=1).groupby('team').transform(np.std))  # 标准差
#           Q1       Q2         Q3         Q4
# 0        NaN      NaN        NaN        NaN
# 1  40.305087  41.7193  24.041631  14.849242
# 2        NaN      NaN        NaN        NaN
# 3  40.305087  41.7193  24.041631  14.849242
# 4        NaN      NaN        NaN        NaN
# 5        NaN      NaN        NaN        NaN

print()
print('使用函数，和上一个学生的差值（没有处理姓名列）')
print(df.groupby('team').transform(lambda x: x.shift(-1)))


#     name    Q1    Q2    Q3    Q4
# 0   None   NaN   NaN   NaN   NaN
# 1  Eorge  93.0  96.0  71.0  78.0
# 2   None   NaN   NaN   NaN   NaN
# 3   None   NaN   NaN   NaN   NaN
# 4   None   NaN   NaN   NaN   NaN
# 5   None   NaN   NaN   NaN   NaN

def score(gb):
    return (gb - gb.mean()) / gb.std() * 10


# 调用
grouped = df.drop(['name'], axis=1).groupby('team')
# print(grouped)
print(grouped.transform(score))
#          Q1        Q2        Q3        Q4
# 0       NaN       NaN       NaN       NaN
# 1 -7.071068 -7.071068 -7.071068 -7.071068
# 2       NaN       NaN       NaN       NaN
# 3  7.071068  7.071068  7.071068  7.071068
# 4       NaN       NaN       NaN       NaN
# 5       NaN       NaN       NaN       NaN

print()
# 也可以用它来进行按组筛选：
print('# Q1成绩大于60的组的所有成员')
print(df[df.drop(['name'], axis=1).groupby('team').transform('mean').Q1 > 60])
#     name team   Q1  Q2  Q3   Q4
# 0  Liver    E   89  21  24   64
# 1   Arry    C   36  37  37   57
# 3  Eorge    C   93  96  71   78
# 4    Oah    D   65  49  61   86
# 5   Rick    B  100  99  97  100

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.3 分组对象的操作')
print('\t6.3.7 筛选方法filter()')
print()
# update20240424
'''
上一节完成了分组对象的创建，分组对象包含数据的分组情况，
接下来就来对分组对象进行操作，获取其相关信息，为最后的数据聚合统计打好基础。
'''
# 小节注释
'''
使用filter()对组作为整体进行筛选，如果满足条件，则整个组会被显示。
传入它调用函数中的默认变量为每个分组的DataFrame，
经过计算，最终返回一个布尔值（不是布尔序列），为真的DataFrame全部显示。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 每组4个季度的平均分的平均分为本组的总平均分')
print(df.drop(['name'], axis=1).groupby('team').mean().mean(1))
# team
# A    54.750
# B    99.000
# C    63.125
# D    65.250
# E    49.500
# dtype: float64

print()
print('# 筛选出所在组总平均分大于51的成员')
print(df.drop(['name'], axis=1).groupby('team').filter(lambda x: x.select_dtypes(include='number').mean(1).mean() > 51))
#   team   Q1  Q2  Q3   Q4
# 1    C   36  37  37   57
# 2    A   57  60  18   84
# 3    C   93  96  71   78
# 4    D   65  49  61   86
# 5    B  100  99  97  100

print()
print('# Q1成绩至少有一个大于92的组')
print(df.drop(['name'], axis=1).groupby('team').filter(lambda x: (x['Q1'] > 92).any()))
#   team   Q1  Q2  Q3   Q4
# 1    C   36  37  37   57
# 3    C   93  96  71   78
# 5    B  100  99  97  100
print()
print('# Q1成绩全部大于92的组')
print(df.drop(['name'], axis=1).groupby('team').filter(lambda x: (x['Q1'] > 92).all()))
#   team   Q1  Q2  Q3   Q4
# 5    B  100  99  97  100


print()
# print(df.groupby('team').first(2))
print(df.groupby('team').rank())

'''
# 其它功能！ 

# 所有成员平均成绩大于60的组
df.groupby(['team']).filter(lambda x: (x.mean() >= 60).all())
# Q1所有成员成绩之和超过1060的组
df.groupby('team').filter(lambda g: g.Q1.sum() > 1060)

df.groupby('team').first() # 组内第一个
df.groupby('team').last() # 组内最后一个
df.groupby('team').ngroups # 5（分组数）
df.groupby('team').ngroup() # 分组序号
grouped.backfill()
grouped.bfill()
df.groupby('team').head() # 每组显示前5个
grouped.tail(1) # 每组最后一个
grouped.rank() # 排序值
grouped.fillna(0)
grouped.indices() # 组名:索引序列组成的字典
# 分组中的第几个值

gp.nth(1) # 第一个
gp.nth(-1) # 最后一个
gp.nth([-2, -1])
# 第n个非空项
gp.nth(0, dropna='all')
gp.nth(0, dropna='any')
df.groupby('team').shift(-1) # 组内移动
grouped.tshift(1) # 按时间周期移动
df.groupby('team').any()
df.groupby('team').all()
df.groupby('team').rank() # 在组内的排名
# 仅 SeriesGroupBy 可用
df.groupby("team").Q1.nlargest(2) # 每组最大的两个
df.groupby("team").Q1.nsmallest(2) # 每组最小的两个
df.groupby("team").Q1.nunique() # 每组去重数量
df.groupby("team").Q1.unique() # 每组去重值
df.groupby("team").Q1.value_counts() # 每组去重值及数量
df.groupby("team").Q1.is_monotonic_increasing # 每组值是否单调递增
df.groupby("team").Q1.is_monotonic_decreasing # 每组值是否单调递减
# 仅 DataFrameGroupBy 可用
df.groupby("team").corrwith(df2) # 相关性

'''

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
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.4 聚合统计')
print('\t6.4.1 描述统计')
print()
# update20240425
'''
本节主要介绍对分组完的数据的统计工作，这是分组聚合的最后一步。
通过最终数据的输出，可以观察到业务的变化情况，体现数据的价值。

'''
# 小节注释
'''
分组对象如同df.describe()，也支持.describe()，用来对数据的总体进行描述：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 描述统计')
# print(df.groupby('team').Q1.describe())
#       count   mean        std    min     25%    50%     75%    max
# team
# A       1.0   57.0        NaN   57.0   57.00   57.0   57.00   57.0
# B       1.0  100.0        NaN  100.0  100.00  100.0  100.00  100.0
# C       2.0   64.5  40.305087   36.0   50.25   64.5   78.75   93.0
# D       1.0   65.0        NaN   65.0   65.00   65.0   65.00   65.0
# E       1.0   89.0        NaN   89.0   89.00   89.0   89.00   89.0

print(df.groupby('team').describe())
#         Q1                           ...      Q4
#      count   mean        std    min  ...     25%    50%     75%    max
# team                                 ...
# A      1.0   57.0        NaN   57.0  ...   84.00   84.0   84.00   84.0
# B      1.0  100.0        NaN  100.0  ...  100.00  100.0  100.00  100.0
# C      2.0   64.5  40.305087   36.0  ...   62.25   67.5   72.75   78.0
# D      1.0   65.0        NaN   65.0  ...   86.00   86.0   86.00   86.0
# E      1.0   89.0        NaN   89.0  ...   64.00   64.0   64.00   64.0
#
# [5 rows x 32 columns]

print()
print('# 由于列过多，我们进行转置')
print(df.groupby('team').describe().T)
# 由于列过多，我们进行转置
# team         A      B          C     D     E
# Q1 count   1.0    1.0   2.000000   1.0   1.0
#    mean   57.0  100.0  64.500000  65.0  89.0
# ...
# Q4 count   1.0    1.0   2.000000   1.0   1.0
#    mean   84.0  100.0  67.500000  86.0  64.0
#    std     NaN    NaN  14.849242   NaN   NaN
#    min    84.0  100.0  57.000000  86.0  64.0
#    25%    84.0  100.0  62.250000  86.0  64.0
#    50%    84.0  100.0  67.500000  86.0  64.0
#    75%    84.0  100.0  72.750000  86.0  64.0
#    max    84.0  100.0  78.000000  86.0  64.0

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.4 聚合统计')
print('\t6.4.2 统计函数')
print()
# update20240426
# 小节注释
'''
对分组对象直接使用统计函数，对分组内的所有数据进行此计算，最终以DataFrame形式显示数据。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 各组平均数')
grouped = df.drop('name', axis=1).groupby('team')
print(grouped.mean())
#          Q1    Q2    Q3     Q4
# team
# A      57.0  60.0  18.0   84.0
# B     100.0  99.0  97.0  100.0
# C      64.5  66.5  54.0   67.5
# D      65.0  49.0  61.0   86.0
# E      89.0  21.0  24.0   64.0

print()
print('# 其它统计')

print(df.groupby('team').size())
# team
# A    1
# B    1
# C    2
# D    1
# E    1
# dtype: int64

print()
print(df.drop('name', axis=1).groupby('team').prod())
#          Q1    Q2    Q3     Q4
# team
# A      57.0  60.0  18.0   84.0
# B     100.0  99.0  97.0  100.0
# C      64.5  66.5  54.0   67.5
# D      65.0  49.0  61.0   86.0
# E      89.0  21.0  24.0   64.0

# df.groupby('team').describe() # 描述性统计
# df.groupby('team').sum() # 求和
# df.groupby('team').count() # 每组数量，不包括缺失值
# df.groupby('team').max() # 求最大值
# df.groupby('team').min() # 求最小值
# df.groupby('team').size() # 分组数量
# df.groupby('team').mean() # 平均值
# df.groupby('team').median() # 中位数
# df.groupby('team').std() # 标准差
# df.groupby('team').var() # 方差
# grouped.corr() # 相关性系数
# grouped.sem() # 标准误差
# grouped.prod() # 乘积
# grouped.cummax() # 每组的累计最大值
# grouped.cumsum() # 累加
# grouped.mad() # 平均绝对偏差


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.4 聚合统计')
print('\t6.4.3 聚合方法agg()')
print()
# update20240426
# 小节注释
'''
对分组对象直接使用统计函数，对分组内的所有数据进行此计算，最终以DataFrame形式显示数据。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 所有列使用一个计算方法')
grouped = df.drop('name', axis=1).groupby('team')
# print(grouped.mean())
print(df.groupby('team').aggregate(sum))
print(df.groupby('team').agg(sum))  # 结果同上

print(grouped.agg(np.size))
#       Q1  Q2  Q3  Q4
# team
# A      1   1   1   1
# B      1   1   1   1
# C      2   2   2   2
# D      1   1   1   1
# E      1   1   1   1
print(grouped['Q1'].agg(np.mean))
# team
# A     57.0
# B    100.0
# C     64.5
# D     65.0
# E     89.0
# Name: Q1, dtype: float64

'''我们使用它主要是为了实现一个字段使用多种统计方法，不同字段使用不同方法：'''

print()
print('# 每个字段使用多个计算方法')
print(grouped[['Q1', 'Q3']].agg([np.sum, np.mean, np.std]))
#        Q1                     Q3
#       sum   mean        std  sum  mean        std
# team
# A      57   57.0        NaN   18  18.0        NaN
# B     100  100.0        NaN   97  97.0        NaN
# C     129   64.5  40.305087  108  54.0  24.041631
# D      65   65.0        NaN   61  61.0        NaN
# E      89   89.0        NaN   24  24.0        NaN

print()
# 不同列使用不同计算方法，且一个列用多个计算方法
print(df.groupby('team').agg({'Q1': ['min', 'max'], 'Q2': 'sum'}))
#        Q1        Q2
#       min  max  sum
# team
# A      57   57   60
# B     100  100   99
# C      36   93  133
# D      65   65   49
# E      89   89   21

# 类似于我们之前学过的增加新列的方法df.assign()，agg()可以指定新列的名字：
print()
print('# 相同列，不同统计函数')
print(df.groupby('team').Q1.agg(Mean='mean', Sum='sum'))
#        Mean  Sum
# team
# A      57.0   57
# B     100.0  100
# C      64.5  129
# D      65.0   65
# E      89.0   89

print()
print('# 不同列 不同统计函数')
print(df.groupby('team').agg(Mean=('Q1', 'mean'), Sum=('Q2', 'sum')))
#        Mean  Sum
# team
# A      57.0   60
# B     100.0   99
# C      64.5  133
# D      65.0   49
# E      89.0   21

print()
print(df.groupby('team').agg(
    Q1_max=pd.NamedAgg(column='Q1', aggfunc='max'),
    Q2_min=pd.NamedAgg(column='Q2', aggfunc='min')
))
#       Q1_max  Q2_min
# team
# A         57      60
# B        100      99
# C         93      37
# D         65      49
# E         89      21

print()
print('# 如果列名不是有效的Python变量格式，则可以用以下方法：')
print(df.groupby('team').agg(**{
    '1_max': pd.NamedAgg(column='Q1', aggfunc='max')
}))

#       1_max
# team
# A        57
# B       100
# C        93
# D        65
# E        89

print()
print('# 聚合结果使用函数')


# lambda函数，所有方法都可以使用
def max_min(x):
    return x.max() - x.min()


# 定义函数
print(df.groupby('team').Q1.agg(Mean='mean',
                                Sum='sum',
                                Diff=lambda x: x.max() - x.min(),
                                Max_min=max_min)
      )
#        Mean  Sum  Diff  Max_min
# team
# A      57.0   57     0        0
# B     100.0  100     0        0
# C      64.5  129    57       57
# D      65.0   65     0        0
# E      89.0   89     0        0


print(df.groupby('team')[['Q1', 'Q2']].agg(
    Mean_Q1=('Q1', 'mean'),
    Sum_Q1=('Q1', 'sum'),
    Diff_Q1=('Q1', lambda x: x.max() - x.min()),
    Max_min_Q1=('Q1', max_min),
    Mean_Q2=('Q2', 'mean'),
    Sum_Q2=('Q2', 'sum'),
    Diff_Q2=('Q2', lambda x: x.max() - x.min()),
    Max_min_Q2=('Q2', max_min)
))

# print(df.groupby('team').agg(Mean='mean',
#                           Sum='sum',
#                           Diff=lambda x:x.max() - x.min(),
#                           Max_min=max_min)
# )

# print(df.groupby('team').agg(max_min))  报错！


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.4 聚合统计')
print('\t6.4.4 时序重采样方法resample()')
print()
# update20240428

# 小节注释
'''
针对时间序列数据，resample()将分组后的时间索引按周期进行聚合统计。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

idx = pd.date_range('1/1/2024', periods=100, freq='T')
df2 = pd.DataFrame(data={'a': [0, 1] * 50, 'b': 1},
                   index=idx)
print(df2)
'''
                     a  b
2024-01-01 00:00:00  0  1
2024-01-01 00:01:00  1  1
2024-01-01 00:02:00  0  1
2024-01-01 00:03:00  1  1
2024-01-01 00:04:00  0  1
...                 .. ..
2024-01-01 01:35:00  1  1
2024-01-01 01:36:00  0  1
2024-01-01 01:37:00  1  1
2024-01-01 01:38:00  0  1
2024-01-01 01:39:00  1  1

[100 rows x 2 columns]
'''

print()
print('# 每20分钟聚合一次')
print(df2.groupby('a').resample('20T').sum())
'''
                        a   b
a                            
0 2024-01-01 00:00:00   0  10
  2024-01-01 00:20:00   0  10
  2024-01-01 00:40:00   0  10
  2024-01-01 01:00:00   0  10
  2024-01-01 01:20:00   0  10
1 2024-01-01 00:00:00  10  10
  2024-01-01 00:20:00  10  10
  2024-01-01 00:40:00  10  10
  2024-01-01 01:00:00  10  10
  2024-01-01 01:20:00  10  10
'''

print()
print('# 其他案例')
print('# 三个周期一聚合（一分钟一个周期）')
print(df2.groupby('a').resample('3T').sum())  # [67 rows x 2 columns]

print()
print('# 30秒一分组')
print(df2.groupby('a').resample('30S').sum())  # [394 rows x 2 columns]

print()
print('# 每月')
print(df2.groupby('a').resample('M').sum())
#                a   b
# a
# 0 2024-01-31   0  50
# 1 2024-01-31  50  50

print()
print('# 以右边时间点为标识')
print(df2.groupby('a').resample('3T', closed='right').sum())  # [67 rows x 2 columns]

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.4 聚合统计')
print('\t6.4.5 组内头尾值')
print()
# update20240428

# 小节注释
'''
在一个组内，如果希望取第一个值和最后一个值，可以使用以下方法。
当然，定义第一个和最后一个是你需要事先完成的工作。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print(df.groupby('team').first())
#        name   Q1  Q2  Q3   Q4
# team
# A       Ack   57  60  18   84
# B      Rick  100  99  97  100
# C      Arry   36  37  37   57
# D       Oah   65  49  61   86
# E     Liver   89  21  24   64

print()
print(df.groupby('team').last())
#        name   Q1  Q2  Q3   Q4
# team
# A       Ack   57  60  18   84
# B      Rick  100  99  97  100
# C     Eorge   93  96  71   78
# D       Oah   65  49  61   86
# E     Liver   89  21  24   64


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.4 聚合统计')
print('\t6.4.6 组内分位数')
print()
# update20240428

# 小节注释
'''
在一个组内，如果希望取第一个值和最后一个值，可以使用以下方法。
当然，定义第一个和最后一个是你需要事先完成的工作。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 二分位数，即中位数')
print(df.drop('name', axis=1).groupby('team').median())
#          Q1    Q2    Q3     Q4
# team
# A      57.0  60.0  18.0   84.0
# B     100.0  99.0  97.0  100.0
# C      64.5  66.5  54.0   67.5
# D      65.0  49.0  61.0   86.0
# E      89.0  21.0  24.0   64.0

print(df.drop('name', axis=1).groupby('team').quantile())  # 结果同上
print(df.drop('name', axis=1).groupby('team').quantile(0.5))  # 结果同上

print()
print('# 三分位数 、 四分位数')
print(df.drop('name', axis=1).groupby('team').quantile(0.33))  # 三分位数
#           Q1     Q2     Q3      Q4
# team
# A      57.00  60.00  18.00   84.00
# B     100.00  99.00  97.00  100.00
# C      54.81  56.47  48.22   63.93
# D      65.00  49.00  61.00   86.00
# E      89.00  21.00  24.00   64.00

print(df.drop('name', axis=1).groupby('team').quantile(0.25))  # 四分位数
#           Q1     Q2    Q3      Q4
# team
# A      57.00  60.00  18.0   84.00
# B     100.00  99.00  97.0  100.00
# C      50.25  51.75  45.5   62.25
# D      65.00  49.00  61.0   86.00
# E      89.00  21.00  24.0   64.00

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.4 聚合统计')
print('\t6.4.7 组内差值')
print()
# update20240428
# 小节注释
'''
和DataFrame的diff()一样，分组对象的diff()方法会在组内进行前后数据的差值计算，并以原DataFrame形状返回数据：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('#  grouped为全数字列，计算在组内的前后差值')
grouped = df.drop('name', axis=1).groupby('team')
print(grouped.diff())

#      Q1    Q2    Q3    Q4
# 0   NaN   NaN   NaN   NaN
# 1   NaN   NaN   NaN   NaN
# 2   NaN   NaN   NaN   NaN
# 3  57.0  59.0  34.0  21.0
# 4   NaN   NaN   NaN   NaN
# 5   NaN   NaN   NaN   NaN

'''
6.4.8 小结
本节介绍的功能是将分组的结果最终统计并展示出来。
我们需要掌握常见的数学统计函数，另外也可以使用NumPy的大量统计方法。
特别是要熟练使用agg()方法，它功能强大，显示功能完备，是在我们今后的
数据分析中最后的数据分组聚合工具。
'''

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.5 数据分箱')
print('\t6.5.1 定界分箱pd.cut()')
print()
# update20240429
'''
数据分箱（data binning，也称为离散组合或数据分桶）是一种数据预处理技术，
它将原始数据分成几个小区间，即bin（小箱子），是一种量子化的形式。数据分箱
可以最大限度减小观察误差的影响。落入给定区间的原始数据值被代表该区间的值（通常是中心值）替换。
然后将其替换为针对该区间计算的常规值。这具有平滑输入数据的作用，
并且在小数据集的情况下还可以减少过拟合。

Pandas主要基于以两个函数实现连续数据的离散化处理。
pandas.cut：根据指定分界点对连续数据进行分箱处理。
pandas.qcut：根据指定区间数量对连续数据进行等宽分箱处理。
所谓等宽，指的是每个区间中的数据量是相同的。

'''
# 小节注释
'''
pd.cut()可以指定区间将数字进行划分。
以下例子中的0、60、100三个值将数据划分成两个区间，从而将及格或者不及格分数进行划分

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 将Q1成绩换60分及以上、60分以下进行分类')

print(pd.cut(df.Q1, bins=[0, 60, 100]))
# 0    (60, 100]
# 1      (0, 60]
# 2      (0, 60]
# 3    (60, 100]
# 4    (60, 100]
# 5    (60, 100]
# Name: Q1, dtype: category
# Categories (2, interval[int64, right]): [(0, 60] < (60, 100]]

# 将分箱结果应用到groupby分组中：
# Series使用
print(df.Q1.groupby(pd.cut(df.Q1, bins=[0, 60, 100])).count())
# Q1
# (0, 60]      2
# (60, 100]    4
# Name: Q1, dtype: int64

print()
print('# Dataframe使用')
print(df.groupby(pd.cut(df.Q1, bins=[0, 60, 100])).count())
#            name  team  Q1  Q2  Q3  Q4
# Q1
# (0, 60]       2     2   2   2   2   2
# (60, 100]     4     4   4   4   4   4

print()
print('# 其它参数示例')
print('# 不显示区间，使用数字作为每个箱子的标签，形式如0，1，2，n等')
print(pd.cut(df.Q1, bins=[0, 60, 100], labels=False))
# 0    1
# 1    0
# 2    0
# 3    1
# 4    1
# 5    1
# Name: Q1, dtype: int64

print()
print('# 指定标签名')
print(pd.cut(df.Q1, bins=[0, 60, 100], labels=['不及格', '及格', ]))
# 0     及格
# 1    不及格
# 2    不及格
# 3     及格
# 4     及格
# 5     及格
# Name: Q1, dtype: category
# Categories (2, object): ['不及格' < '及格']

# print(pd.cut(df.Q1,bins=[0,60,100],labels=['不及格','及格'])) # 结果同上


# print(df.groupby(pd.cut(df.Q1,bins=[0,60,100],labels=['不及格','及格'])).sum())
#                   name  team   Q1   Q2   Q3   Q4
# Q1
# 不及格            ArryAck    CA   93   97   55  141
# 及格   LiverEorgeOahRick  ECDB  347  265  253  328

print()
print('# 包含最低部分')
print(pd.cut(df.Q1, bins=[0, 60, 100], include_lowest=True))
# 0     (60.0, 100.0]
# 1    (-0.001, 60.0]
# 2    (-0.001, 60.0]
# 3     (60.0, 100.0]
# 4     (60.0, 100.0]
# 5     (60.0, 100.0]
# Name: Q1, dtype: category
# Categories (2, interval[float64, right]): [(-0.001, 60.0] < (60.0, 100.0]]

print()
print('# 是否为右闭区间，下例为[89, 100)')
print(pd.cut(df.Q1, bins=[0, 89, 100], right=False))
# 0    [89.0, 100.0)
# 1      [0.0, 89.0)
# 2      [0.0, 89.0)
# 3    [89.0, 100.0)
# 4      [0.0, 89.0)
# 5              NaN
# Name: Q1, dtype: category
# Categories (2, interval[int64, left]): [[0, 89) < [89, 100)]


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.5 数据分箱')
print('\t6.5.2 等宽分箱pd.qcut()')
print()
# update20240429
# 小节注释
'''
pd.qcut()可以指定所分区间的数量，Pandas会自动进行分箱：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 按Q1成绩分为两组')
print(pd.qcut(df.Q1, q=2))
# 0     (77.0, 100.0]
# 1    (35.999, 77.0]
# 2    (35.999, 77.0]
# 3     (77.0, 100.0]
# 4    (35.999, 77.0]
# 5     (77.0, 100.0]
# Name: Q1, dtype: category
# Categories (2, interval[float64, right]): [(35.999, 77.0] < (77.0, 100.0]]

print()
print('# 查看分组区间')
print(pd.qcut(df.Q1, q=2).unique())
# [(77.0, 100.0], (35.999, 77.0]]
# Categories (2, interval[float64, right]): [(35.999, 77.0] < (77.0, 100.0]]

print()
print('# 应用到分组中：')
# Series使用
print(df.Q1.groupby(pd.qcut(df.Q1, q=2)).count())
# Q1
# (35.999, 77.0]    3
# (77.0, 100.0]     3
# Name: Q1, dtype: int64

print()
# DataFrame使用
print(df.groupby(pd.qcut(df.Q1, q=2)).max())
#                 name team   Q1  Q2  Q3   Q4
# Q1
# (35.999, 77.0]   Oah    D   65  60  61   86
# (77.0, 100.0]   Rick    E  100  99  97  100

print()
print('# 其它参数如下：')
print(pd.qcut(range(5), 4))
# [(-0.001, 1.0], (-0.001, 1.0], (1.0, 2.0], (2.0, 3.0], (3.0, 4.0]]
# Categories (4, interval[float64, right]): [(-0.001, 1.0] < (1.0, 2.0] < (2.0, 3.0] < (3.0, 4.0]]
print(pd.qcut(range(5), 4, labels=False))
# [0 0 1 2 3]

print()
print('# 指定标签名')
print(pd.qcut(range(5), 3, labels=["good", "medium", "bad"]))
# ['good', 'good', 'medium', 'bad', 'bad']
# Categories (3, object): ['good' < 'medium' < 'bad']

print()
print('# 返回箱子标签 array([1. , 51.5, 98.]))')
print(pd.qcut(df.Q1, q=2, retbins=True))
# (0     (77.0, 100.0]
# 1    (35.999, 77.0]
# 2    (35.999, 77.0]
# 3     (77.0, 100.0]
# 4    (35.999, 77.0]
# 5     (77.0, 100.0]
# Name: Q1, dtype: category
# Categories (2, interval[float64, right]): [(35.999, 77.0] < (77.0, 100.0]], array([ 36.,  77., 100.])

print()
print('# 分箱位小数位数')
print(pd.qcut(df.Q1, q=2, precision=3))
# 0     (77.0, 100.0]
# 1    (35.999, 77.0]
# 2    (35.999, 77.0]
# 3     (77.0, 100.0]
# 4    (35.999, 77.0]
# 5     (77.0, 100.0]
# Name: Q1, dtype: category
# Categories (2, interval[float64, right]): [(35.999, 77.0] < (77.0, 100.0]]

print()
print('# 排名分3个层次')
print(pd.qcut(df.Q1.rank(method='first'), 3))
# 0    (2.667, 4.333]
# 1    (0.999, 2.667]
# 2    (0.999, 2.667]
# 3      (4.333, 6.0]
# 4    (2.667, 4.333]
# 5      (4.333, 6.0]
# Name: Q1, dtype: category
# Categories (3, interval[float64, right]): [(0.999, 2.667] < (2.667, 4.333] < (4.333, 6.0]]

'''
6.5.3 小结
本节介绍的分箱也是一种数据分组方式，经常用在数据建模、机器
学习中，与传统的分组相比，它更适合离散数据。
'''

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.6 分组可视化')
print('\t6.6.1 绘图方法plot()')
print()
# update20240429
'''
Pandas为我们提供了几个简单的、与分组相关的可视化方法，
这些可视化方法能够提高我们对聚合数据的视觉直观认识，本节就来做一些介绍。

'''
# 小节注释
'''
在1.3节中我们介绍过plot()方法，它能够为我们绘制出我们想要的常见图形。
数据分组对象也支持plot()，不过它以分组对象中每个DataFrame或Series为对象，绘制出所有分组的图形。
默认情况下，它绘制的是折线图，示例代码如下。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 分组，设置索引为name')
grouped = df.set_index('name').groupby('team')
# 绘制图形
# 折线图
grouped.plot(kind='bar')  # 默认 kind='line'
plt.show()
# print(plt.show())
'''
还可以通过plot.x()或者plot(kind='x')的形式调用其他形状的图形，
比如：
plot.line：折线图
plot.pie：饼图
plot.bar：柱状图
plot.hist：直方图
plot.box：箱形图
plot.area：面积图
plot.scatter：散点图
plot.hexbin：六边形分箱图
plot()可传入丰富的参数来控制图形的样式，可参阅第16章。

'''

print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.6 分组可视化')
print('\t6.6.2 直方图hist()')
print()
# update20240429

# 小节注释
'''
分组对象的hist()可以绘制出每个分组的直方图矩阵，每个矩阵为一个分组：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 分组，设置索引为name')
grouped = df.set_index('name').groupby('team')
# 绘制图形
grouped.hist()
plt.show()
# print()


print()
print('------------------------------------------------------------')
print('第6章 pandas分组聚合')
print('\t6.6 分组可视化')
print('\t6.6.3 箱线图boxplot()')
print()
# update20240430
'''
Pandas为我们提供了几个简单的、与分组相关的可视化方法，
这些可视化方法能够提高我们对聚合数据的视觉直观认识，本节就来做一些介绍。

'''
# 小节注释
'''
分组的boxplot()方法绘制出每个组的箱线图。
箱线图展示了各个字段的最大值、最小值、分位数等信息，为我们展示了数据的大体形象，代码如下。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 分组箱线图')
grouped = df.set_index('name').groupby('team')
# 绘制图形
# grouped.boxplot()
# grouped.boxplot(figsize=(15,12)) # 比默认的图片要大几倍
# plt.show()

print()
print('# Dataframe 分组箱线图')
df.boxplot(by='team', figsize=(15, 10))
plt.show()
'''以上代码会按team分组并返回箱线图'''

'''
6.6.4 小结
本节介绍了关于分组的可视化方法，它们会将一个分组对象中的各组数据进行分别展示，
便于我们比较，从不同角度发现数据的变化规律，从而得出分析结果。

这些操作都是在分拆应用之后进行的，合并后数据的可视化并没有什么特殊的，
第16章将对数据可视化进行统一讲解

'''

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
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

# 语法结构
df._append(self, other, ignore_index=False, verify_integrity=False, sort=False)

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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 代码示例')
df1 = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
print(df1)

df2 = pd.DataFrame({'x': [5, 6], 'y': [7, 8]})

# 追加合并
print(df1._append(df2))
#    x  y
# 0  1  3
# 1  2  4
# 0  5  7
# 1  6  8
print(df1._append(df2, ignore_index=True, verify_integrity=True, sort=True))
#    x  y
# 0  1  3
# 1  2  4
# 2  5  7
# 3  6  8

print()
print('# 追加多个数据')
print(df1._append([df2, df2, df2]))
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 代码示例')
df1 = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
print(df1)

df2 = pd.DataFrame({'x': [5, 6], 'y': [7, 8]})

df3 = pd.DataFrame({'y': [5, 6], 'z': [7, 8]})

# 追加合并
print(df1._append(df3))
#      x  y    z
# 0  1.0  3  NaN
# 1  2.0  4  NaN
# 0  NaN  5  7.0
# 1  NaN  6  8.0
print(df1._append(df3, ignore_index=True))
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 忽略索引')
df1 = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
print(df1)

df2 = pd.DataFrame({'x': [5, 6], 'y': [7, 8]})

df3 = pd.DataFrame({'y': [5, 6], 'z': [7, 8]})

# 追加合并
print(df1._append(df3))
#      x  y    z
# 0  1.0  3  NaN
# 1  2.0  4  NaN
# 0  NaN  5  7.0
# 1  NaN  6  8.0
print(df1._append(df3, ignore_index=True))
#      x  y    z
# 0  1.0  3  NaN
# 1  2.0  4  NaN
# 2  NaN  5  7.0
# 3  NaN  6  8.0

'''或者，可以根据自己的需要重新设置索引。对索引的操作前面介绍过。'''
print()
print('# 修改索引')
df4 = df1._append(df3, ignore_index=True)
print(df4.set_axis(['a', 'b', 'c', 'd'], axis='index'))

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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 验证内容是否重复（包括索引值）')
df1 = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
# print(df1)
df2 = pd.DataFrame({'x': [5, 6], 'y': [7, 8]})

df3 = pd.DataFrame({'y': [5, 6], 'z': [7, 8]})

# 追加合并
print(df1._append(df2, ignore_index=False, verify_integrity=True))  # 报错！
print(df1._append(df2, ignore_index=True, verify_integrity=True))  # 忽略索引，则运行正常

df4 = pd.DataFrame({'y': [4, 5], 'z': [7, 8]}, index=['a', 'b'])  # 行索引值不同，数值相同，运行正常。
print(df1._append(df4, verify_integrity=True))
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
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

lily = pd.Series(['lily', 'C', 55, 56, 57, 58],
                 index=['name', 'team', 'Q1', 'Q2', 'Q3', 'Q4'])

# print(lily)
df = df._append(lily, ignore_index=True)
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
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
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

lily = {'name': 'lily', 'team': 'C', 'Q1': 55, 'Q2': 56, 'Q3': 57, 'Q4': 58}

# print(lily)
print()
df = df._append(lily, ignore_index=True)
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
E:/bat/input_files/team.xlsx
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
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

print()
print('------------------------------------------------------------')
print('第8章 Pandas多层索引')
print('\t8.1 概述')
print('\t8.1.1 什么是多层索引')
print()
# update20240509
'''
截至目前，我们处理的数据基本都是一列索引和一行表头，但在实际业务中会存在有多层索引的情况。
多层索引（Hierarchical indexing）又叫多级索引，
它为一些非常复杂的数据分析和操作（特别是处理高维数据）提供了方法。
从本质上讲，它使你可以在Series（一维）和DataFrame（二维）等较低维度的数据结构中存储和处理更高维度的数据。

'''
# 小节注释
'''
本节介绍多层数据的一些基本概念和使用场景，以及如何创建多层索引数据。
理解了多层数据的基本概念和使用场景，我们就能更好地应用它的特性来解决实际数据分析中的问题。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

'''看了图 个人感觉 就是一个 有合并同类项的 表格'''

print()
print('------------------------------------------------------------')
print('第8章 Pandas多层索引')
print('\t8.1 概述')
print('\t8.1.2 通过分组产生多层索引')
print()
# update20240509

# 小节注释
'''
在之前讲过的数据分组案例中，多个分组条件会产生多层索引的情况，如：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print('# 按团队分组，各团队中平均成绩及格的人数')
# 行多层索引
print(df.groupby(['team', df.select_dtypes('number').mean(1) > 60]).count())
#             name  Q1  Q2  Q3  Q4
# team
# A    False     1   1   1   1   1
# B    True      1   1   1   1   1
# C    False     1   1   1   1   1
#      True      1   1   1   1   1
# D    True      1   1   1   1   1
# E    False     1   1   1   1   1

print()
print('# 列 多层索引')
print(df.groupby('team').agg({'Q1': ['max', 'min'], 'Q2': ['sum', 'count']}))
#        Q1        Q2
#       max  min  sum count
# team
# A      57   57   60     1
# B     100  100   99     1
# C      93   36  133     2
# D      65   65   49     1
# E      89   89   21     1

print()
print('# 行列多层索引！')
print(df.groupby(['team', df.select_dtypes('number').mean(1) > 60]).agg({'Q1': ['max', 'min'], 'Q2': ['sum', 'count']}))
#              Q1       Q2
#             max  min sum count
# team
# A    False   57   57  60     1
# B    True   100  100  99     1
# C    False   36   36  37     1
#      True    93   93  96     1
# D    True    65   65  49     1
# E    False   89   89  21     1

'''
这样就清晰地表达了业务意义。在处理复杂数据时常常会出现多层索引，
相当于我们对Excel同样值的表头进行了合并单元格。
'''

print()
print('------------------------------------------------------------')
print('第8章 Pandas多层索引')
print('\t8.1 概述')
print('\t8.1.3 由序列创建多层索引')
print()
# update20240509

# 小节注释
'''
MultiIndex对象是Pandas标准Index的子类，由它来表示多层索引业务。
可以将MultiIndex视为一个元组对序列，其中每个元组对都是唯一的。可以通过以下方式生成一个索引对象。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print('# 序列数据')
# 定义一个序列
arrays = [[1, 1, 2, 2], ['A', 'A', 'B', 'B']]
# 生成多层索引
index = pd.MultiIndex.from_arrays(arrays, names=('class', 'team'))
print(index)
# MultiIndex([(1, 'A'),
#             (1, 'A'),
#             (2, 'B'),
#             (2, 'B')],
#            names=['class', 'team'])

print()
print('# 指定的索引是多层索引')
print(pd.DataFrame([{'Q1': 60, 'Q2': 70}], index=index))
#             Q1  Q2
# class team
# 1     A     60  70
#       A     60  70
# 2     B     60  70
#       B     60  70

# 个人测试
# print(pd.DataFrame([{'Q1':[50,60],'Q2':[70,80]}],index=index))
# print(pd.DataFrame({'Q1':[30,40,50,60],'Q2':[70,80,90,100]},index=index))

print()
print('------------------------------------------------------------')
print('第8章 Pandas多层索引')
print('\t8.1 概述')
print('\t8.1.4 由元组创建多层索引')
print()
# update20240509

# 小节注释
'''
可以使用pd.MultiIndex.from_tuples()将由元组组成的序列转换为多层索引

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print('# 定义一个两层的序列')
# 定义一个序列
arrays = [[1, 1, 2, 2], ['A', 'B', 'A', 'B']]
# 转换为元组
tuples = list(zip(*arrays))  # *意思是解包 zip(*arrays)类似zip(arrays[0],arrays[1])
print(tuples)
# [(1, 'A'), (1, 'B'), (2, 'A'), (2, 'B')]
# 将元组转换为多层索引对象
index = pd.MultiIndex.from_tuples(tuples, names=['class', 'team'])
#  使用多层索引对象
print(pd.Series(np.random.randn(4), index=index))
# 1      A       0.868282
#        B       0.721167
# 2      A       2.634965
#        B      -1.010468
# dtype: float64

print()
print('------------------------------------------------------------')
print('第8章 Pandas多层索引')
print('\t8.1 概述')
print('\t8.1.5 可迭代对象的笛卡儿积')
print()
# update20240509

# 小节注释
'''
使用上述方法时我们要将所有层的所有值都写出来，
而pd.MultiIndex.from_product()可以做笛卡儿积计算，将所有情况排列组合出来，如：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print('# 生成多层索引对象')
_class = [1, 2]
team = ['A', 'B']
index = pd.MultiIndex.from_product([_class, team],
                                   names=['class', 'team'])

# Series应用多层索引对象
print(pd.Series(np.random.randn(4), index=index))
# class  team
# 1      A      -0.211555
#        B      -1.193187
# 2      A      -1.333777
#        B       0.729419
# dtype: float64


print()
print('------------------------------------------------------------')
print('第8章 Pandas多层索引')
print('\t8.1 概述')
print('\t8.1.6 将DataFrame转为多层索引对象')
print()
# update20240509

# 小节注释
'''
pd.MultiIndex.from_frame()可以将DataFrame的数据转换为多层索引对象，如：

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print('# 生成多层索引对象')

df_i = pd.DataFrame([['1', 'A'], ['1', 'B'], ['2', 'B'], ['2', 'A']],
                    columns=['class', 'team'])

print(df_i)
#   class team
# 0     1    A
# 1     1    B
# 2     2    B
# 3     2    A
# 将DataFrame中的数据转换成多层索引对象
index = pd.MultiIndex.from_frame(df_i)
# print(index)
# 应用多层对象
print(pd.Series(np.random.randn(4), index=index))
# class  team
# 1      A      -0.234935
#        B       1.176749
# 2      B       0.132324
#        A      -0.774019
# dtype: float64

'''
8.1.7 小结
多层索引最为常见的业务场景是数据分组聚合，它一般会产生多层索引的数据。
本节介绍了什么是多层索引、多层索引的业务意义，以及如果创建多层索引，如何将多层索引应用到DataFrame和Series上。
'''

print()
print('------------------------------------------------------------')
print('第8章 Pandas多层索引')
print('\t8.2 多层索引操作')
print('\t8.2.1 生成数据')
print()
# update20240509

# 小节注释
'''
索引的常规操作也适用于多层索引，但多层索引还有一些特定的操作需要我们熟练掌握，以便更加灵活地运用它。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file)  # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print('# 典型的多层索引数据的生成过程')
# 索引
index_arrays = [[1, 1, 2, 2], ['男', '女', '男', '女']]
# 列名
columns_arrays = [['2019', '2019', '2020', '2020'],
                  ['上半年', '下半年', '上半年', '下半年']]

# 索引转换为多层
index = pd.MultiIndex.from_arrays(index_arrays,
                                  names=('班级', '性别'))

# 列名转换为多层
columns = pd.MultiIndex.from_arrays(columns_arrays,
                                    names=('年份', '学期'))
# 应用到Dataframe中
df = pd.DataFrame([(88, 99, 88, 99), (77, 88, 97, 98),
                   (67, 89, 54, 78), (34, 67, 89, 54)],
                  columns=columns, index=index)

# df.to_excel('E:/bat/output_files/pandas_out_20240509043.xlsx') # 默认 index=True || ,index=False 会报错！

print(df)
# 年份    2019     2020
# 学期     上半年 上半年  上半年 上半年
# 班级 性别
# 1  男    88  99   88  99
#    女    77  88   97  98
# 2  男    67  89   54  78
#    女    34  67   89  54

'''在行和列上可以分别定义多层索引。'''

print('\t8.2.2 索引信息')

print()
print(df.index)
'''
MultiIndex([(1, '男'),
            (1, '女'),
            (2, '男'),
            (2, '女')],
           names=['班级', '性别'])
'''

print(df.columns)
'''
MultiIndex([('2019', '上半年'),
            ('2019', '上半年'),
            ('2020', '上半年'),
            ('2020', '上半年')],
           names=['年份', '学期'])
'''

print()
print('# 查看行、列索引的名称')
print(df.index.names)
print(df.columns.names)
# ['班级', '性别']
# ['年份', '学期']


print()
print('\t8.2.3 查看层级')
'''多层索引由于层级较多，在数据分析时需要查看它共有多少个层级。'''
print(df.index.nlevels)  # 行层级数
# 2
print(df.index.levels)  # 行的层级
# [[1, 2], ['女', '男']]
print(df.columns.nlevels)  # 列层级数
# 2
print(df.columns.levels)  # 列的层级
# [['2019', '2020'], ['上半年', '下半年']]

print(df[['2019', '2020']].index.levels)  # 筛选后的层级
# [[1, 2], ['女', '男']]

# print()
# print(df['2019'])


print()
print('\t8.2.4 索引内容')
'''可以取指定层级的索引内容，也可以按索引名取索引内容：'''
print('# 获取索引第2层内容')
print(df.index.get_level_values(1))
# Index(['男', '女', '男', '女'], dtype='object', name='性别')

print('# 获取列索引第1层内容')
print(df.columns.get_level_values(0))
# Index(['2019', '2019', '2020', '2020'], dtype='object', name='年份')

print()
print('#  按索引名称取索引内容')
print(df.index.get_level_values('班级'))
# Index([1, 1, 2, 2], dtype='int64', name='班级')
print(df.columns.get_level_values('年份'))
# Index(['2019', '2019', '2020', '2020'], dtype='object', name='年份')


print()
print('\t8.2.5 排序')
'''多层索引可以根据需要实现较为复杂的排序操作'''
print('# 使用索引名可进行排序，可以指定具体的列')

print(df.sort_values(by=['性别', ('2020', '下半年')]))  # 必须是个元组
# 年份    2019     2020
# 学期     上半年 下半年  上半年 下半年
# 班级 性别
# 2  女    34  67   89  54
# 1  女    77  88   97  98
# 2  男    67  89   54  78
# 1  男    88  99   88  99

# print(df.sort_values(by=[('2','性别'),'下半年']))  # 也报错！
print(df.index.reorder_levels([1, 0]))  # 等级顺序，互换
# MultiIndex([('男', 1),
#             ('女', 1),
#             ('男', 2),
#             ('女', 2)],
#            names=['性别', '班级'])

# 个人测试 替换原df的行索引 成功
df.index = df.index.reorder_levels([1, 0])
print(df)
# 年份    2019     2020
# 学期     上半年 下半年  上半年 下半年
# 性别 班级
# 男  1    88  99   88  99
# 女  1    77  88   97  98
# 男  2    67  89   54  78
# 女  2    34  67   89  54

print()
print(df.index.sortlevel(level=0, ascending=True))  # 按指定级别排序
# (MultiIndex([('女', 1),
#             ('女', 2),
#             ('男', 1),
#             ('男', 2)],
#            names=['性别', '班级']), array([1, 3, 0, 2], dtype=int64))

print(df.index.sortlevel(level=1, ascending=True))  # 个人测试 按照班级顺序排列
# (MultiIndex([('女', 1),
#             ('男', 1),
#             ('女', 2),
#             ('男', 2)],
#            names=['性别', '班级']), array([1, 0, 3, 2], dtype=int64))

print()
print(df.index.reindex(df.index[::-1]))  # 更换顺序，或者指定一个顺序
# (MultiIndex([('女', 2),
#             ('男', 2),
#             ('女', 1),
#             ('男', 1)],
#            names=['性别', '班级']), array([3, 2, 1, 0], dtype=int64))

# str_list = ['a','b','c','d']
# print(str_list[::-1])


print()
print('\t8.2.6 其他操作')
'''以下是一些其他操作'''
print(df.index.to_numpy())  # 生成一个笛卡儿积的元组对序列
# [('男', 1) ('女', 1) ('男', 2) ('女', 2)]

print(df.index.remove_unused_levels())  # 返回没有使用的层级 | 搞不懂
# MultiIndex([('男', 1),
#             ('女', 1),
#             ('男', 2),
#             ('女', 2)],
#            names=['性别', '班级'])

print(df.swaplevel(0, 1))  # 交换索引
# 年份    2019     2020
# 学期     上半年 下半年  上半年 下半年
# 班级 性别
# 1  男    88  99   88  99
#    女    77  88   97  98
# 2  男    67  89   54  78
#    女    34  67   89  54
# 个人测试
# print(df.columns.swaplevel(0,1)) # 测试能交换 返回列索引

print()
# TODO print(df.to_frame()) # # 转为DataFrame  运行报错

# 个人测试 有效
# 假设df是一个Series
# df = pd.Series([1, 2, 3, 4],name='num')
# print(df)
#
# # 将Series转换为DataFrame
# df_df = df.to_frame()
#
# # 打印转换后的DataFrame
# print(df_df)

'''
在Pandas中，to_frame()方法用于将Series对象转换为DataFrame。
如果尝试在一个已经为DataFrame类型的对象上调用此方法，会引发错误。
解决这个问题的关键是确认您正在操作的对象类型，并根据其类型选择正确的操作。
如果对象是Series，使用to_frame()是合适的；
如果对象是DataFrame，则不需要使用此方法。
'''

print()
print('# 删除指定等级')
print(df.index.droplevel(0))
# Index([1, 1, 2, 2], dtype='int64', name='班级')

# TODO 个人测试 有效
# df.index = df.index.droplevel(0)
# print(df)
# 年份 2019     2020
# 学期  上半年 下半年  上半年 下半年
# 班级
# 1    88  99   88  99
# 1    77  88   97  98
# 2    67  89   54  78
# 2    34  67   89  54

print()
print('# 返回索引的位置')
print(df.index.get_locs(('女', 2)))  # [3]
print(df.index.get_loc(('女', 2)))  # 3
# [3]

'''
8.2.7 小结
多层索引的基础操作与普通索引的操作一样。
本节介绍了多层索引的一些特殊操作，如查看索引的信息、索引内容、索引的层级及排序等。
在Pandas中，大多数方法都针对多层索引进行了适配，可传入类似level的参数对指定层级进行操作。
'''

print()
print('------------------------------------------------------------')
print('第8章 Pandas多层索引')
print('\t8.3 数据查询')
print('\t8.3.1 查询行')
print()
# update20240510
'''
截至目前，我们处理的数据基本都是一列索引和一行表头，但在实际业务中会存在有多层索引的情况。
多层索引（Hierarchical indexing）又叫多级索引，
它为一些非常复杂的数据分析和操作（特别是处理高维数据）提供了方法。
从本质上讲，它使你可以在Series（一维）和DataFrame（二维）等较低维度的数据结构中存储和处理更高维度的数据。

'''
# 小节注释
'''
多层索引组成的数据相对复杂，在确定需求后我们要清晰判断是哪个层级下的数据，
并充分运用本节的内容进行各角度的数据筛选。
需要注意的是，如果行或列中有一个是单层索引，那么与之前介绍过的单层索引一样操作。
本节中的行和列全是多层索引。

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print('# 测试读取 含有多重索引的excel')  # 测试成功！
df = pd.read_excel(multindex_file, header=[0, 1], index_col=[0, 1])
print(df.index)
# print(df)

print()
print('# 典型的多层索引数据的生成过程')
# 索引
index_arrays = [[1, 1, 2, 2], ['男', '女', '男', '女']]
# 列名
columns_arrays = [['2019', '2019', '2020', '2020'],
                  ['上半年', '下半年', '上半年', '下半年']]

# 索引转换为多层
index = pd.MultiIndex.from_arrays(index_arrays,
                                  names=('班级', '性别'))

# 列名转换为多层
columns = pd.MultiIndex.from_arrays(columns_arrays,
                                    names=('年份', '学期'))
# 应用到Dataframe中
df = pd.DataFrame([(88, 99, 88, 99), (77, 88, 97, 98),
                   (67, 89, 54, 78), (34, 67, 89, 54)],
                  columns=columns, index=index)

print(df)

print()
print('# 查看一班的数据')
print(df.loc[1])
# 年份 2019     2020
# 学期  上半年 下半年  上半年 下半年
# 性别
# 男    88  99   88  99
# 女    77  88   97  98

print(df.loc[1:2])  # 查询1班和2班的数据

'''如果我们要同时根据一二级索引查询，可以将需要查询的索引条件组成一个元组：'''
print()
print(df.loc[(1, '男')])

# 个人测试 输出转换为dataframe 存入文件不用使用to_frame()  有效果！
# df1 = df.loc[(1,'男')]
# print(df1.to_frame())
# df1.to_excel('E:/bat/output_files/pandas_out_20240510052.xlsx')


print()
print('\t8.3.2 查询列')
'''
查询列时，可以直接用切片选择需要查询的列，使用元组指定相关的层级数据：
'''

print(df['2020'])  # 整个一级索引下
# 学期     上半年  下半年
# 班级 性别
# 1  男    88   99
#    女    97   98
# 2  男    54   78
#    女    89   54
print()
print(df[('2020', '上半年')])  # # 指定二级索引
print(df['2020']['上半年'])  # 结果同上

print()
print('\t8.3.3 行列查询')
'''
行列查询和单层索引一样，指定层内容也用元组表示。slice(None)可以在元组中占位，表示本层所有内容：
'''
print(df.loc[(1, '男'), '2020'])  # 只显示2020年1班男生
# 学期
# 上半年    88
# 下半年    99
# Name: (1, 男), dtype: int64

# df1 = df.loc[(1,'男'),'2020']  || 文件不显示年份
# df1.to_excel('E:/bat/output_files/pandas_out_20240510053.xlsx')

print()
print('# 只看下半年')
print(df.loc[:, (slice(None), '下半年')])  #
# 年份    2019 2020
# 学期     下半年  下半年
# 班级 性别
# 1  男    99   99
#    女    88   98
# 2  男    89   78
#    女    67   54

# TODO 个人测试
# df1 = df.loc[:,(slice(None),'下半年')]  # || 文件如输出显示
# df1.to_excel('E:/bat/output_files/pandas_out_20240510053.xlsx')

print()
print('# 只看女生')
print(df.loc[(slice(None), '女'), :])
# 年份    2019     2020
# 学期     上半年 下半年  上半年 下半年
# 班级 性别
# 1  女    77  88   97  98
# 2  女    34  67   89  54

print('# 只看一班')
print(df.loc[1, (slice(None)), :])
# 年份 2019     2020
# 学期  上半年 下半年  上半年 下半年
# 性别
# 男    88  99   88  99
# 女    77  88   97  98

print()
print(df.loc[(1, slice(None)), :])  # 逻辑同上 显示班级
# 年份    2019     2020
# 学期     上半年 下半年  上半年 下半年
# 班级 性别
# 1  男    88  99   88  99
#    女    77  88   97  98

print()
print('# 只看2020年的数据')
print(df.loc[:, ('2020', slice(None))])
# 年份    2020
# 学期     上半年 下半年
# 班级 性别
# 1  男    88  99
#    女    97  98
# 2  男    54  78
#    女    89  54

print()
print('\t8.3.4 条件查询')
'''
按照一定的条件查询数据，和单层索引的数据查询一样，不过在选择列上要按多层的规则.
'''

print(df[df[('2020', '上半年')] > 80])
# 年份    2019     2020
# 学期     上半年 下半年  上半年 下半年
# 班级 性别
# 1  男    88  99   88  99
#    女    77  88   97  98
# 2  女    34  67   89  54
print(df[df.loc[:, ('2020', '上半年')] > 80])  # 结果同上！

print()
print('\t8.3.5 用pd.IndexSlice索引数据')
'''
pd.IndexSlice可以创建一个切片对象，轻松执行复杂的索引切片操作：
'''
# idx = pd.IndexSlice
# idx[0] # 0
# idx[:] # slice(None, None, None)
# idx[0,'x'] # (0, 'x')
# idx[0:3] # slice(0, 3, None)
# idx[0.1:1.5] # slice(0.1, 1.5, None)
# idx[0:5,'x':'y'] # (slice(0, 5, None), slice('x', 'y', None))

# 应用在查询中：
idx = pd.IndexSlice
print(df.loc[idx[:, ['男']], :])  # 只显示男生
# 年份    2019     2020
# 学期     上半年 下半年  上半年 下半年
# 班级 性别
# 1  男    88  99   88  99
# 2  男    67  89   54  78
print(df.loc[:, idx[:, ['上半年']]])  # 只显示上半年
# 年份    2019 2020
# 学期     上半年  上半年
# 班级 性别
# 1  男    88   88
#    女    77   97
# 2  男    67   54
#    女    34   89

print()
print('\t8.3.6 df.xs()')
''' 使用df.xs()方法采用索引内容作为参数来选择多层索引数据中特定级别的数据：'''

print(df.xs((1, '男')))  # 1班男生
# 年份    学期
# 2019  上半年    88
#       下半年    99
# 2020  上半年    88
#       下半年    99
# Name: (1, 男), dtype: int64

print()
print(df.xs('2020', axis=1))  # 2020年

print()
print(df.xs('男', level=1))  # 所有男生

# print()
# print(df.xs(1)) # 个人测试 1班 运行成功  print(df.xs('男')) 报错！

print()
print('参数drop_level=0')
print(df.xs(1, drop_level=0))  # 在返回的结果中保留层级
# 年份    2019     2020
# 学期     上半年 下半年  上半年 下半年
# 班级 性别
# 1  男    88  99   88  99
#    女    77  88   97  98

print(df.xs(1))  # 无参数 drop_level=false  或 drop_level=0
# 年份 2019     2020
# 学期  上半年 下半年  上半年 下半年
# 性别
# 男    88  99   88  99
# 女    77  88   97  98

'''
xs 只能选择索引的一个层级，如果需要进行更复杂的选择，可能需要使用 loc 或 iloc 等其他方法。

在实际业务中，xs 方法特别适用于需要快速选择特定层级数据的场景。
例如，在金融数据分析中，可能需要选择某一特定日期或某一特定证券的数据，这时 xs 就非常有用。
然而，对于更加复杂的数据选择任务，可能需要使用其他更加灵活的方法。

8.3.7 小结
本节介绍了多层索引的数据查询操作，这些操作让我们可以方便地对于复杂的多层数据按需求进行查询。
和单层索引数据一样，多层索引数据也可以使用切片、loc、iloc等操作，只是需要用元组表达出层级。

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
multindex_file = 'E:/bat/input_files/pandas_out_20240509042.xlsx'

path = 'E:/bat/output_files/pandas_read_csv_20240118.csv'

# TODO 忽略特定的警告 | 当遇到 openpyxl 中的 UserWarning 类型的警告时，它们将不会被打印出来。
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

print()
print('------------------------------------------------------------')
print('第9章 Pandas数据重塑与透视')
print('\t9.1 数据查询')
print('\t9.1.1 整理透视')
print()
# update20240513
'''
数据透视表，顾名思义，是指它有“透视”数据的能力，可以找出大量复杂无关的数据的内在关系，
将数据转化为有意义、有价值的信息，从而看到它所代表的事物的规律和本质。

数据透视是最常用的数据汇总工具，Excel中经常会做数据透视，它可以根据一个或多个指定的维度来聚合数据。
Pandas也提供了数据透视函数来实现这些功能。
'''
# 小节注释
'''
要实现基础的透视操作，可以使用df.pivot()返回按给定的索引、列值重新组织整理后的DataFrame。
df.pivot()有3个参数:

index：作为新DataFrame的索引，取分组去重的值；如果不传入，则取现有索引。
columns：作为新DataFrame的列，取去重的值，当列和索引的组合有多个值时会报错，需要使用pd.pivot_table()进行操作。
values：作为新DataFrame的值，如果指定多个，会形成多层索引；如果不指定，会默认为所有剩余的列。

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print('------------------------------------------------------------')
print('\t9.1.2 整理透视操作')
print()
'''

'''
# 构造数据
df = pd.DataFrame({'A': ['a1', 'a1', 'a2', 'a2', 'a3', 'a3'],
                   'B': ['b1', 'b2', 'b3', 'b1', 'b2', 'b3'],
                   'C': ['c1', 'c2', 'c3', 'c4', 'c5', 'c6'],
                   'D': ['d1', 'd2', 'd3', 'd4', 'd5', 'd6'], })
# 展示数据
# print(df)
#     A   B   C   D
# 0  a1  b1  c1  d1
# 1  a1  b2  c2  d2
# 2  a2  b3  c3  d3
# 3  a2  b1  c4  d4
# 4  a3  b2  c5  d5
# 5  a3  b3  c6  d6

print('# 透视，指定索引、列、值')
print(df.pivot(index='A', columns='B', values='C'))
# B    b1   b2   b3
# A
# a1   c1   c2  NaN
# a2   c4  NaN   c3
# a3  NaN   c5   c6

'''
上面的代码取A列的去重值作为索引，取B列的去重值作为列，取C列内容作为具体数据值。
两个轴交叉处的取值方法是原表中A与B对应C列的值，如果无值则显示NaN。
'''

print()
print('# 不指定值内容')
print(df.pivot(index='A', columns='B'))
#       C              D
# B    b1   b2   b3   b1   b2   b3
# A
# a1   c1   c2  NaN   d1   d2  NaN
# a2   c4  NaN   c3   d4  NaN   d3
# a3  NaN   c5   c6  NaN   d5   d6

'''其效果和以下代码相同：'''
print(df.pivot(index='A', columns='B', values=['C', 'D']))  # 结果同上

print()
print('------------------------------------------------------------')
print('\t9.1.3 聚合透视')

'''
df.pivot()只是对原数据的结构、显示形式做了变换，在实现业务中，
往往还需要在数据透视过程中对值进行计算，这时候就要用到pd.pivot_table()。
它可以实现类似Excel那样的高级数据透视功能。

pd.pivot_table()有以下几个关键参数。
    data：要透视的DataFrame对象。
    index：在数据透视表索引上进行分组的列。
    columns：在数据透视表列上进行分组的列。
    values：要聚合的一列或多列。
    aggfunc：用于聚合的函数，默认是平均数mean。
    fill_value：透视会以空值填充值。
    margins：是否增加汇总行列。
'''

print()
print('------------------------------------------------------------')
print('\t9.1.4 聚合透视操作')

df = pd.DataFrame({
    'A': ['a1', 'a1', 'a1', 'a2', 'a2', 'a2'],
    'B': ['b2', 'b2', 'b1', 'b1', 'b1', 'b1'],
    'C': ['c1', 'c1', 'c2', 'c2', 'c1', 'c1'],
    'D': [1, 2, 3, 4, 5, 6],
})

print(df)
#     A   B   C  D
# 0  a1  b2  c1  1
# 1  a1  b2  c1  2
# 2  a1  b1  c2  3
# 3  a2  b1  c2  4
# 4  a2  b1  c1  5
# 5  a2  b1  c1  6

# 报错！ValueError: Index contains duplicate entries, cannot reshape
# print(df.pivot(index='A',columns='B',values='D'))

'''
问题解释
例如，在原始数据中，当 A=a1 和 B=b2 时，D 有两个值（1 和 2），这导致了重复，因为 pivot 期望每个 A-B 组合只对应一个 D 值。
'''

print()
print('# pd.pivot_table()透视')
print(pd.pivot_table(df, index='A', columns='B', values='D'))
# B    b1   b2
# A
# a1  3.0  1.5
# a2  5.0  NaN

'''需要将这些重复数据按一定的算法计算出来，pd.pivot_table()默认的算法是取平均值。'''

# 验证数据
# 筛选a2和b1的数据
print(df.loc[(df.A == 'a2') & (df.B == 'b1')])
# 对D求平均值
print(df.loc[(df.A == 'a2') & (df.B == 'b1')].D.mean())
# 5.0

print()
print('------------------------------------------------------------')
print('\t9.1.5 聚合透视高级操作')
print('# 高级聚合')
print(pd.pivot_table(df, index=['A', 'B'],  # 指定多个索引
                     columns='C',  # 指定列
                     values='D',  # 指定数据值
                     fill_value=0,  # 将聚合为空的值填充为0
                     aggfunc=np.sum,  # 指定聚合方法为求和 默认aggfunc='mean'  aggfunc='sum'/np.sum 结果相同
                     margins=True  # 增加行列汇总
                     ))

# C       c1  c2  All
# A   B
# a1  b1   0   3    3
#     b2   3   0    3
# a2  b1  11   4   15
# All     14   7   21

print()
print('# 使用多个聚合计算')
print(pd.pivot_table(df, index=['A', 'B'],
                     columns=['C'],
                     values='D',
                     aggfunc=[np.mean, np.sum]
                     ))

#       mean        sum
# C       c1   c2    c1   c2
# A  B
# a1 b1  NaN  3.0   NaN  3.0
#    b2  1.5  NaN   3.0  NaN
# a2 b1  5.5  4.0  11.0  4.0

print()
print('# 为各列分别指定计算方法')
df = pd.DataFrame({
    'A': ['a1', 'a1', 'a1', 'a2', 'a2', 'a2'],
    'B': ['b2', 'b2', 'b1', 'b1', 'b1', 'b1'],
    'C': ['c1', 'c1', 'c2', 'c2', 'c1', 'c1'],
    'D': [1, 2, 3, 4, 5, 6],
    'E': [9, 8, 7, 6, 5, 4],
})

print(pd.pivot_table(df,
                     index=['A', 'B'],
                     columns=['C'],
                     aggfunc={'D': np.mean, 'E': np.sum},
                     # fill_value=0
                     ))

#          D          E
# C       c1   c2    c1   c2
# A  B
# a1 b1  NaN  3.0   NaN  7.0
#    b2  1.5  NaN  17.0  NaN
# a2 b1  5.5  4.0   9.0  6.0

'''最终形成的数据，D列取平均，E列取和。'''

'''
9.1.6 小结
本节介绍了Pandas如何进行透视操作。df.pivot()是对数据进行整理，变换显示方式，
而pd.pivot_table()会在整理的基础上对重复的数据进行相应的计算。
Pandas透视表功能和Excel类似，但Pandas对数据的聚合方法更加灵活，
能够实现更加复杂的需求，特别是在处理庞大的数据时更能展现威力。
'''

print()
print('------------------------------------------------------------')
print('第9章 Pandas数据重塑与透视')
print('\t9.2 数据堆叠')
print('\t9.2.1 理解堆叠')
print()
# update20240514
'''
数据的堆叠也是数据处理的一种常见方法。在多层索引的数据中，通常为了方便查看、对比，会将所有数据呈现在一列中；
相反，对于层级过多的数据，我们可以实施解堆操作，让它呈现多列的状态。
'''
# 小节注释
'''
数据堆叠可以简单理解成将多列数据转为一列数据（见图9-3）。
如果原始数据有多个数据列，堆叠（stack）的过程表示将这些数据列的所有数据表全部旋转到行上；
类似地，解堆（unstack）的过程表示将在行上的索引旋转到列上。
解堆是堆叠的相反操作。


堆叠和解堆的本质如下。
堆叠：“透视”某个级别的（可能是多层的）列标签，返回带有索引的DataFrame，
      该索引带有一个新的行标签，这个新标签在原有索引的最右边。

解堆：将（可能是多层的）行索引的某个级别“透视”到列轴，从
     而生成具有新的最里面的列标签级别的重构的DataFrame。
     堆叠过程将数据集的列转行，解堆过程为行转列。
     上例中，原始数据集索引有两层，堆叠过程就是将最左的列转到最内层的行上，
     解堆是将最内层的行转移到最内层的列索引中。

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print()
print('------------------------------------------------------------')
print('\t9.2.2 堆叠操作df.stack()')

df = pd.DataFrame({
    'A': ['a1', 'a1', 'a2', 'a2'],
    'B': ['b1', 'b2', 'b1', 'b2'],
    'C': [1, 2, 3, 4],
    'D': [5, 6, 7, 8],
    'E': [5, 6, 7, 8],
})

# 设置多层索引
df.set_index(['A', 'B'], inplace=True)
print(df)
#        C  D  E
# A  B
# a1 b1  1  5  5
#    b2  2  6  6
# a2 b1  3  7  7
#    b2  4  8  8

print()
print('# 进行堆叠')
print(df.stack())
# A   B
# a1  b1  C    1
#         D    5
#         E    5
#     b2  C    2
#         D    6
#         E    6
# a2  b1  C    3
#         D    7
#         E    7
#     b2  C    4
#         D    8
#         E    8
# dtype: int64

# df.stack().to_excel('E:/bat/output_files/pandas_out_20240514023.xlsx')

# 查看类型
# print(type(df.stack())) # <class 'pandas.core.series.Series'>
'''我们看到生成了一个Series，所有的列都透视在了多层索引的新增列中。'''

print()
print('------------------------------------------------------------')
print('\t9.2.3 解堆操作df.unstack()')
print('# 将原来的数据堆叠并赋值给s')

s = df.stack()
# print(s)
print('# 操作解堆')
print(s.unstack())
#        C  D  E
# A  B
# a1 b1  1  5  5
#    b2  2  6  6
# a2 b1  3  7  7
#    b2  4  8  8

'''解堆后生成的是一个DataFrame。'''

'''
9.2.4 小结
数据的堆叠和解堆分别用来解决数据的展开和收缩问题。
堆叠让数据变成一维数据，可以让我们从不同维度来观察和使用数据。
解堆和堆叠互为相反操作。
'''

print()
print('------------------------------------------------------------')
print('第9章 Pandas数据重塑与透视')
print('\t9.3 交叉表')
print('\t9.3.1 基本语法')
print()
# update20240514
'''
交叉表（cross tabulation）是一个很有用的分析工具，是用于统计分组频率的特殊透视表。
简单来说，交叉表就是将两列或多列中不重复的元素组成一个新的DataFrame，新数据的行和列交叉部分的值为其组合在原数据中的数量。
'''
# 小节注释
'''
交叉表的基本语法如下：
    # 基本语法
    pd.crosstab(index, columns, values=None, rownames=None,
                colnames=None, aggfunc=None, margins=False,
                margins_name: str = 'All', dropna: bool = True,
                normalize=False)

参数说明如下。
    index：传入列，如df['A']，作为新数据的索引。
    columns：传入列，作为新数据的列，新数据的列为此列的去重值。
    values：可选，传入列，根据此列的数值进行计算，计算方法取aggfunc参数指定的方法，此时aggfunc为必传。
    aggfunc：函数，values列计算使用的计算方法。
    rownames：新数据和行名，一个序列，默认值为None，必须与传递的行数、组数匹配。
    colnames：新数据和列名，一个序列，默认值为None；如果传递，则必须与传递的列数、组数匹配。
    margins：布尔值，默认值为False，添加行/列边距（小计）。
    normalize：布尔值，{'all'，'index'，'columns'}或{0,1}，默认值为False。通过将所有值除以值的总和进行归一化。

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print()
print('------------------------------------------------------------')
print('\t9.3.2 生成交叉表')

# 原数据
df = pd.DataFrame({
    'A': ['a1', 'a1', 'a2', 'a2', 'a1'],
    'B': ['b2', 'b1', 'b2', 'b2', 'b1'],
    'C': [1, 2, 3, 4, 5],
})

# 生成交叉表
print(pd.crosstab(df['A'], df['B']))
'''
B   b1  b2
A         
a1   2   1
a2   0   2
'''

'''
在上例中，A、B两列进行交叉，A有不重复的值a1和a2，B有不重
复的值b1和b2。交叉后组成了新的数据，行和列的索引分别是a1、a2和
b1、b2，它们交叉位上对应的值为此组合的数量，如a1、b1组合有两
个，所以它们的交叉位上的值为2；没有a2、b1组合，所以对应的值为
0。
'''

print()
print('# 对分类数据做交叉')
one = pd.Categorical(['a', 'b'], categories=['a', 'b', 'c'])
two = pd.Categorical(['d', 'e'], categories=['d', 'e', 'f'])
print(pd.crosstab(one, two))
# col_0  d  e
# row_0
# a      1  0
# b      0  1

print()
# print(one)
# print(two)
# ['a', 'b']
# Categories (3, object): ['a', 'b', 'c']
# ['d', 'e']
# Categories (3, object): ['d', 'e', 'f']


print()
print('------------------------------------------------------------')
print('\t9.3.3 归一化')

'''
normalize参数可以帮助我们实现数据归一化，算法为对应值除以所
有值的总和，让数据处于0～1的范围，方便我们观察此位置上的数据在
全体中的地位。
'''

print(pd.crosstab(df['A'], df['B']))
# B   b1  b2
# A
# a1   2   1
# a2   0   2

print('#  交叉表，归一化')
print(pd.crosstab(df['A'], df['B'], normalize=True))
# B    b1   b2
# A
# a1  0.4  0.2
# a2  0.0  0.4

print('交叉表，按列归一化')
print(pd.crosstab(df['A'], df['B'], normalize='columns'))
# B    b1        b2
# A
# a1  1.0  0.333333
# a2  0.0  0.666667
print('交叉表，按行归一化')
print(pd.crosstab(df['A'], df['B'], normalize='index'))
# B         b1        b2
# A
# a1  0.666667  0.333333
# a2  0.000000  1.000000

print()
print('------------------------------------------------------------')
print('\t9.3.4 指定聚合方法')

'''
用aggfunc指定聚合方法对values指定的列进行计算：
'''

print('# 交叉表，按C列的和进行求和聚合')
print(pd.crosstab(df['A'], df['B'], values=df['C'], aggfunc=np.sum))
# B    b1   b2
# A
# a1  7.0  1.0
# a2  NaN  7.0

print()
print('------------------------------------------------------------')
print('\t9.3.5 汇总')

'''
margins=True可以增加行和列的汇总，按照行列方向对数据求和，
类似margins_name='total'可以定义这个汇总行和列的名称：
'''

print('# 交叉表，增加汇总')
print(pd.crosstab(df['A'], df['B'],
                  values=df['C'],
                  aggfunc=np.sum,
                  margins=True,
                  margins_name='total',  # 定义汇总行列名称
                  ))
# B       b1   b2  total
# A
# a1     7.0  1.0      8
# a2     NaN  7.0      7
# total  7.0  8.0     15

'''
9.3.6 小结
交叉表将原始数据中的两个列铺开，形成这两列所有不重复值的交叉位，
在交叉位上填充这个值在原始数据中的组合数。
交叉表可以帮助我们了解标签数据的构成情况。
'''

print()
print('------------------------------------------------------------')
print('第9章 Pandas数据重塑与透视')
print('\t9.4 数据转置df.T')
print('\t9.4.1 理解转置')
print()
# update20240515
'''
数据的转置是指将数据的行与列进行互换，它会使数据的形状和逻辑发生变化。
Pandas提供了非常便捷的df.T操作来进行转置。
'''
# 小节注释
'''
在数据处理分析过程中，为了充分利用行列的关系表达，
我们需要将原数据的行与列进行互换。转置其实是将数据沿着左上与右下形成的对角线进行翻转，

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print()
print('------------------------------------------------------------')
print('\t9.4.2 转置操作')

'''df.T属性是df.transpose()方法的简写形式，我们只要记住.T就可以快速进行转置操作。'''
# 原数据
df = pd.DataFrame({
    'A': ['a1', 'a2', 'a3', 'a4', 'a5'],
    'B': ['b1', 'b2', 'b3', 'b4', 'b5'],
    'C': [1, 2, 3, 4, 5],
})

# 转置
print(df.T)
#     0   1   2   3   4
# A  a1  a2  a3  a4  a5
# B  b1  b2  b3  b4  b5
# C   1   2   3   4   5

'''
我们观察到，a1和5这两个值始终在左上和右下，
以它们连成的对角线作为折线对数据进行了翻转，让列成为了行，行成为了列。
Series也支持转置，不过它返回的是它自己，没有变化。
'''

print()
print('------------------------------------------------------------')
print('\t9.4.3 类型变化')
'''一般情况下数据转置后，列的数据类型会发生变化：'''
print('# 原始数据的数据类型')
print(df.dtypes)
# A    object
# B    object
# C     int64
# dtype: object

print('# 转置后的数据类型')
print(df.T.dtypes)
# 0    object
# 1    object
# 2    object
# 3    object
# 4    object
# dtype: object

'''这是因为，数据结构带来的巨大变化会让数据类型重新得到确定。'''

print()
print('------------------------------------------------------------')
print('\t9.4.4 轴交换df.swapaxes()')

'''
Pandas提供了df.swapaxes(axis1, axis2, copy=True)方法，
该方法用来进行轴（行列）交换，如果是进行行列交换，它就相当于df.T。
'''
print(df)
print(df.swapaxes('index', 'columns'))  # 行列交换，相当于df.T
print(df.swapaxes('columns', 'index'))  # 同上

print('copy=True')
print(df.swapaxes('index', 'columns', copy=True))  # 结果同上
#     0   1   2   3   4
# A  a1  a2  a3  a4  a5
# B  b1  b2  b3  b4  b5
# C   1   2   3   4   5

print('# 无变化')
print(df.swapaxes('index', 'index'))
#     A   B  C
# 0  a1  b1  1
# 1  a2  b2  2
# 2  a3  b3  3
# 3  a4  b4  4
# 4  a5  b5  5

'''
9.4.5 小结
转置操作让我们能够以另一个角度看数据。随着行列的交换，
数据的意义也会发生变化，让数据的内涵表达增添了一种可能性。
'''

print()
print('------------------------------------------------------------')
print('第9章 Pandas数据重塑与透视')
print('\t9.5 数据融合')
print('\t9.5.1 基本语法')
print()
# update20240515
'''
数据融合df.melt()是df.pivot()的逆操作函数，简单来说，
它是将指定的列铺开，放到行上名为variable（可指定）、值为value（可指定）列。
'''
# 小节注释
'''
具体的语法结构如下：
    pd.melt(frame: pandas.core.frame.DataFrame,
            id_vars=None, value_vars=None,
            var_name='variable', value_name='value',
            col_level=None)

参数说明如下:
    id_vars：tuple、list或ndarray（可选），用作标识变量的列。
    value_vars：tuple、list或ndarray（可选），要取消透视的列。如果未指定，则使用未设置为id_vars的所有列。
    var_name：scalar，用于“变量”列的名称。如果为None，则使用frame.columns.name或“variable”。
    value_name：scalar，默认为“value”，用于“value”列的名称。
    col_level：int或str（可选），如果列是多层索引，则使用此级别来融合。
▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print()
print('------------------------------------------------------------')
print('\t9.5.2 融合操作')

# 原数据
df = pd.DataFrame({
    'A': ['a1', 'a2', 'a3', 'a4', 'a5'],
    'B': ['b1', 'b2', 'b3', 'b4', 'b5'],
    'C': [1, 2, 3, 4, 5],
})

# 数据融合
print(pd.melt(df))
# variable value
# 0         A    a1
# 1         A    a2
# ...
# 13        C     4
# 14        C     5
'''以上操作将A、B、C三列的标识和值展开，一列为标签，默认列名variable，另一列为值，默认列名为value。'''

print()
print('------------------------------------------------------------')
print('\t9.5.3 标识和值')
'''
数据融合时，可以指定标识，下例指定A、B两列作为融合操作后的标识，保持不变，对其余列（C列）展开数据：
'''

print('# 数据融合，指定标识')
print(pd.melt(df, id_vars=['A', 'B']))
#     A   B variable  value
# 0  a1  b1        C      1
# 1  a2  b2        C      2
# 2  a3  b3        C      3
# 3  a4  b4        C      4
# 4  a5  b5        C      5

print('# 数据融合，指定值列')
print(pd.melt(df, value_vars=['B', 'C']))
'''
  variable value
0        B    b1
1        B    b2
2        B    b3
3        B    b4
4        B    b5
5        C     1
6        C     2
7        C     3
8        C     4
9        C     5
'''

# print()
# print(pd.melt(df,id_vars=['C','B']))

'''如果想保留标识，可以使用id_vars同时指定。'''

print()
print('------------------------------------------------------------')
print('\t9.5.4 指定名称')

'''标识的名称和值的名称默认分别是variable和value，我们可以指定它们的名称：'''

print('# 指定标识和值列的名称')
print(pd.melt(df, id_vars=['A'], value_vars=['B'],
              var_name='B_lable', value_name='B_value'
              ))
'''
    A B_lable B_value
0  a1       B      b1
1  a2       B      b2
2  a3       B      b3
3  a4       B      b4
4  a5       B      b5
'''

'''
9.5.5 小结
数据融合让数据无限度地铺开，它是透视的反向操作。
另外，对于原始数据是多层索引的，可以使用col_level=0参数指定融合时的层级。

'''

print()
print('------------------------------------------------------------')
print('第9章 Pandas数据重塑与透视')
print('\t9.6 虚拟变量')
print('\t9.6.1 基本语法')
print()
# update20240515
'''
虚拟变量（Dummy Variable）又称虚设变量、名义变量或哑变量，
是一个用来反映质的属性的人工变量，是量化了的自变量，通常取值为0或1，常被用于one-hot特征提取。
'''
# 小节注释
'''
简单来说，生成虚拟变量的方法pd.get_dummies()是将一列或多列的去重值作为新表的列，
每列的值由0和1组成：如果原来位置的值与列名相同，则在新表中该位置的值为1，否则为0。
这样就形成了一个由0和1组成的特征矩阵。


# 基本语法
pd.get_dummies(data, 
                prefix=None,
                prefix_sep='_', 
                dummy_na=False,
                columns=None, 
                sparse=False,
                drop_first=False, 
                dtype=None)

参数如下：
data：被操作的数据，DataFrame或者Series。
prefix：新列的前缀。
prefix_sep：新列前缀的连接符。

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print()

print()
print('------------------------------------------------------------')
print('\t9.6.2 生成虚拟变量')

# 原数据
df = pd.DataFrame({
    'a': list('adcb'),
    'b': list('fehg'),
    'a1': range(4),
    'b1': range(4, 8),
})

print(df)
# print(df.dtypes)
#    a  b  a1  b1
# 0  a  f   0   4
# 1  d  e   1   5
# 2  c  h   2   6
# 3  b  g   3   7

print('# 生成虚拟变量')
print(pd.get_dummies(df.a))
#        a      b      c      d
# 0   True  False  False  False
# 1  False  False  False   True
# 2  False  False   True  False
# 3  False   True  False  False

'''
分析一下执行结果，只关注df.a列：a列共有a、b、c、d四个值，
故新数据有此四列；
    索引和列名交叉的位置如果为1，说明此索引位上的值为列名，为0则表示不为列名。
    例如，在索引2上，1在c列，对比原数据发现df.loc[2, ‘a’]为c。
'''

print()
print('------------------------------------------------------------')
print('\t9.6.3 列前缀')

'''
有的原数据的部分列（如下例中的a1列）值为纯数字，为了方便使用，
需要给生成虚拟变量的列名增加一个前缀，用prefix来定义：
'''

print(pd.get_dummies(df.a1, prefix='a1'))
#     a1_0   a1_1   a1_2   a1_3
# 0   True  False  False  False
# 1  False   True  False  False
# 2  False  False   True  False
# 3  False  False  False   True

print()
print('------------------------------------------------------------')
print('\t9.6.4 从DataFrame生成')

'''可以直接对DataFrame生成虚拟变量，会将所有非数字列生成虚拟变量（数字列保持不变）：'''

print(pd.get_dummies(df))
#    a1  b1    a_a    a_b    a_c    a_d    b_e    b_f    b_g    b_h
# 0   0   4   True  False  False  False  False   True  False  False
# 1   1   5  False  False  False   True   True  False  False  False
# 2   2   6  False  False   True  False  False  False  False   True
# 3   3   7  False   True  False  False  False  False   True  False

print('# 只指定一列：')
# 只生成b列的虚拟变量
print(pd.get_dummies(df, columns=['b']))
#    a  a1  b1    b_e    b_f    b_g    b_h
# 0  a   0   4  False   True  False  False
# 1  d   1   5   True  False  False  False
# 2  c   2   6  False  False  False   True
# 3  b   3   7  False  False   True  False


print()
print('个人测试')
print(pd.get_dummies(df))
print(pd.get_dummies(df, drop_first=True))  # 删除非数字列的第一个
print(pd.get_dummies(df, drop_first=True, dtype=float))  # 值的显示格式  默认是布尔值
#    a1  b1  a_b  a_c  a_d  b_f  b_g  b_h
# 0   0   4  0.0  0.0  0.0  1.0  0.0  0.0
# 1   1   5  0.0  0.0  1.0  0.0  0.0  0.0
# 2   2   6  0.0  1.0  0.0  0.0  0.0  1.0
# 3   3   7  1.0  0.0  0.0  0.0  1.0  0.0

'''
9.6.5 小结
虚拟变量生成操作将数据进行变形，形成一个索引与值（变形后成为列）的二维矩阵，
在对应交叉位上用1表示有此值，0表示无此值。虚拟变量经常用于与特征提取相关的机器学习场景。
'''

print()
print('------------------------------------------------------------')
print('第9章 Pandas数据重塑与透视')
print('\t9.7 因子化')
print('\t9.7.1 基本方法')
print()
# update20240515
'''
因子化是指将一个存在大量重复值的一维数据解析成枚举值的过程，这样可以方便我们进行分辨。
factorize既可以用作顶层函数pd.factorize()，也可以用作Series.factorize()和Index.factorize()方法。
'''
# 小节注释
'''
对数据因子化后返回两个值，一个是因子化后的编码列表，另一个是原数据的去重值列表：

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print('# 数据')
data = ['b', 'b', 'a', 'c', 'b']

print('# 因子化')
codes, uniques = pd.factorize(data)

# 编码
print(codes)  # [0 0 1 2 0]
# 去重值
print(uniques)  # ['b' 'a' 'c']

'''
上例中，我们将列表data进行了因子化，返回一个由两个元素组成的元组，我们用codes和uniques分别来承接这个元组的元素。
codes：数字编码表，将第一个元素编为0，其后依次用1、2等表示，但遇到相同元素使用相同编码。
       这个编码表的长度与原始数据相同，用编码对原始数据进行一一映射。
uniques：去重值，就是因子。

以上数据都是可迭代的array类型，需要注意的是codes和uniques是我们定义的变量名，
强烈推荐大家这么命名，这样代码会变得容易理解。

对Series操作后唯一值将生成一个index对象：
'''
print('示例2')
cat = pd.Series(['a', 'a', 'c'])
codes, uniques = pd.factorize(cat)
print(codes)  # [0 0 1]
print(uniques)  # Index(['a', 'c'], dtype='object')

print()
print('------------------------------------------------------------')
print('\t9.7.2 排序')

'''
使用sort=True参数后将对唯一值进行排序，编码列表将继续与原值保持对应关系，但从值的大小上将体现出顺序。
'''
codes, uniques = pd.factorize(['b', 'b', 'a', 'c', 'b', ], sort=True)
print(codes)
print(uniques)
# [1 1 0 2 1]
# ['a' 'b' 'c']

# 非排序
codes, uniques = pd.factorize(['b', 'b', 'a', 'c', 'b', ])
# [0 0 1 2 0]
# ['b' 'a' 'c']

print()
print('------------------------------------------------------------')
print('\t9.7.3 缺失值')
'''缺失值不会出现在唯一值列表中'''

codes, uniques = pd.factorize(['b', None, 'a', 'c', 'b', ])
print(codes)
print(uniques)
# [ 0 -1  1  2  0]
# ['b' 'a' 'c']

print()
print('------------------------------------------------------------')
print('\t9.7.4 枚举类型')
'''Pandas的枚举类型数据（Categorical）也可以使用此方法：'''

cat = pd.Categorical(['a', 'a', 'c', ], categories=['a', 'b', 'c'])
codes, uniques = pd.factorize(cat)
# print(cat)
print(codes)
print(uniques)
# [0 0 1]
# ['a', 'c']
# Categories (3, object): ['a', 'b', 'c']

'''
9.7.5 小结
简单来说，因子化方法pd.factorize()做了两件事：
    一是对数据进行数字编码，二是对数据进行去重。
    在大序列数据中，因子化能帮助我们抽取数据特征，将数据变成类别数据再进行分析。
'''

print()
print('------------------------------------------------------------')
print('第9章 Pandas数据重塑与透视')
print('\t9.8 爆炸列表')
print('\t9.8.1 基本功能')
print()
# update20240515
'''
爆炸这个词非常形象，在这里是指将类似列表的每个元素转换为一行，
索引值是相同的。这在我们处理一些需要展示的数据时非常有用。
'''
# 小节注释
'''
下面的两行数据中有类似列表的值，我们将它们炸开后，它们排好了队，但是依然使用原来的索引：

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print('# 原始数据')
s = pd.Series([[1, 2, 3], 'foo', [], [3, 4]])
print(s)
# 0    [1, 2, 3]
# 1          foo
# 2           []
# 3       [3, 4]
# dtype: object

print()
print('# 爆炸列')
print(s.explode())
# 0      1
# 0      2
# 0      3
# 1    foo
# 2    NaN
# 3      3
# 3      4
# dtype: object

'''
注意观察会发现，每行列表中的元素都独自占用了一行，
而索引保持不变，空列表值变成了NaN，非列表的元素没有变化。
这就是爆炸的作用，像一颗炸弹一样，将列表打散。
'''
print()
print('------------------------------------------------------------')
print('\t9.8.2 DataFrame的爆炸')
'''DataFrame可以对指定列实施爆炸'''

df = pd.DataFrame({'A': [[1, 2, 3], 'foo', [], [3, 4]], 'B': range(4)})
print(df)
#            A  B
# 0  [1, 2, 3]  0
# 1        foo  1
# 2         []  2
# 3     [3, 4]  3

print('# 爆炸指定列')
print(df.explode('A'))
#      A  B
# 0    1  0
# 0    2  0
# 0    3  0
# 1  foo  1
# 2  NaN  2
# 3    3  3
# 3    4  3

'''在DataFrame中爆炸指定列后，其他列的值会保持不变。'''

print()
print('------------------------------------------------------------')
print('\t9.8.3 非列表格式')

'''对于不是列表但具有列表特质的数据，我们也可以在处理之后让其完成爆炸，如下面的数据：'''

# 原数据
df = pd.DataFrame([{'var1': 'a,b,c', 'var2': 1},
                   {'var1': 'd,e,f', 'var2': 2},
                   ])

print(df)
#     var1  var2
# 0  a,b,c     1
# 1  d,e,f     2

'''var1列的数据虽然是按逗号隔开的，但它不是列表，这时候我们先将其处理成列表，再实施爆炸：'''

print()
# print(df.explode('var1')) # 成功运行，但结果同上 无爆炸

print('# 使用指定同名列的方式对列进行修改')
df = df.assign(var1=df.var1.str.split(','))
print(df)
#         var1  var2
# 0  [a, b, c]     1
# 1  [d, e, f]     2
print(df.explode('var1'))
#   var1  var2
# 0    a     1
# 0    b     1
# 0    c     1
# 1    d     2
# 1    e     2
# 1    f     2

'''
9.8.4 小结
在实际的数据处理和办公环境中，经常会遇到用指定符号隔开的数据，我们分析之前要先进行数据的二维化。
Pandas提供的s.explode()列表爆炸功能可以方便地实施这一操作，我们需要灵活掌握。

9.9 本章小结
本章介绍了Pandas提供的数据变换操作，通过这些数据变换，可以观察数据的另一面，
探究数据反映出的深层次的业务意义。

另外一些操作（如虚拟变量、因子化等）可帮助我们提取出数据的特征，
为下一步数据建模、数据分析、机器学习等操作打下良好基础。
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
multindex_file = 'E:/bat/input_files/pandas_out_20240509042.xlsx'

path = 'E:/bat/output_files/pandas_read_csv_20240118.csv'
# df1.to_excel('E:/bat/output_files/pandas_out_20240510053.xlsx')

# TODO 忽略特定的警告 | 当遇到 openpyxl 中的 UserWarning 类型的警告时，它们将不会被打印出来。
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

print()
print('------------------------------------------------------------')
print('第10章 Pandas数据清洗')
print('\t10.1 缺失值的认定')
print('\t10.1.1 缺失值类型')
print()
# update20240517
'''
数据清洗是数据分析的一个重要步骤，关系到数据的质量，而数据的质量又关系到数据分析的效果。
数据清洗一般包括缺失值填充、冗余数据删除、数据格式化、异常值处理、逻辑错误数据检测、数据一致性校验、重复值过滤、数据质量评估等。
Pandas提供了一系列操作方法帮助我们轻松完成这些操作。
'''
# 小节注释
'''
一般使用特殊类型NaN代表缺失值，可以用NumPy定义为np.NaN或np.nan。
在Pandas 1.0以后的版本中，实验性地使用标量pd.NA来代表。

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print('# 原始数据')
df = pd.DataFrame({
    'A': ['a1', 'a1', 'a2', 'a2'],
    'B': ['b1', 'b2', None, 'b2'],
    'C': [1, 2, 3, 4],
    'D': [5, 6, None, 8],
    'E': [5, None, 7, 8],
})

print(df)
#     A     B  C    D    E
# 0  a1    b1  1  5.0  5.0
# 1  a1    b2  2  6.0  NaN
# 2  a2  None  3  NaN  7.0
# 3  a2    b2  4  8.0  8.0

'''以上数据中，2B、2D、1E为缺失值。如果想把正负无穷也当作缺失值，可以通过以下全局配置来设定：'''

# 将无穷值设置为缺失值
# print()
# pd.options.mode.use_inf_as_na = True
# pd.options.mode.use_inf_as_na = True
# print(df)

print()
print('------------------------------------------------------------')
print('\t10.1.2 缺失值判断')

'''
df.isna()及其别名df.isnull()是Pandas中判断缺失值的主要方法。
对整个数据进行缺失值判断，True为缺失：
'''

# 检测缺失值
print(df.isna())
#        A      B      C      D      E
# 0  False  False  False  False  False
# 1  False  False  False  False   True
# 2  False   True  False   True  False
# 3  False  False  False  False  False

print()
print('# 检测指定列的缺失值')
print(df.D.isna())
# 0    False
# 1    False
# 2     True
# 3    False
# Name: D, dtype: bool

print()
print('# 检测非缺失值')
print(df.notna())
#       A      B     C      D      E
# 0  True   True  True   True   True
# 1  True   True  True   True  False
# 2  True  False  True  False   True
# 3  True   True  True   True   True

print()
print('# 检测某列非缺失值')
print(df.D.notna())
# 0     True
# 1     True
# 2    False
# 3     True
# Name: D, dtype: bool

print()
print('------------------------------------------------------------')
print('\t10.1.3 缺失值统计')

'''如果需要统计一个数据中有多少个缺失值，可利用sum计算，计算时将False当作0、将True当作1的特性：'''

print('# 布尔值的求和')
print(pd.Series([True, True, False]).sum())  # 2

# 如果需要计算数据中的缺失值情况，可以使用以下方法：
print()
print('# 每列有多少个缺失值')
print(df.isnull().sum())  # isna() = isnull()
# A    0
# B    1
# C    0
# D    1
# E    1

print()
print('# 每行有多少个缺失值')
print(df.isnull().sum(1))
# 0    0
# 1    1
# 2    2
# 3    0

print()
print('# 总共有多少个缺失值')
print(df.isna().sum().sum())  # 3

print()
print('------------------------------------------------------------')
print('\t10.1.4 缺失值筛选')

print(df)
print()
print(df.isna().any(axis=1))
# 0    False
# 1     True
# 2     True
# 3    False
# dtype: bool

print()
print('# 有缺失值的行')
print(df.loc[df.isna().any(axis=1)])  # 必须加参数axis=1
#     A     B  C    D    E
# 1  a1    b2  2  6.0  NaN
# 2  a2  None  3  NaN  7.0
print('# 有缺失值的列')
print(df.loc[:, df.isna().any()])
#       B    D    E
# 0    b1  5.0  5.0
# 1    b2  6.0  NaN
# 2  None  NaN  7.0
# 3    b2  8.0  8.0

'''如果要查询没有缺失值的行和列，可以对表达式取反'''
print()
print('# 没有缺失值的行')
print(df.loc[~(df.isna().any(axis=1))])
#     A   B  C    D    E
# 0  a1  b1  1  5.0  5.0
# 3  a2  b2  4  8.0  8.0
print('# 没有缺失值的列')
print(df.loc[:, ~(df.isna().any())])
#     A  C
# 0  a1  1
# 1  a1  2
# 2  a2  3
# 3  a2  4

print()
print('------------------------------------------------------------')
print('\t10.1.5 NA标量')
'''
Pandas 1.0以后的版本中引入了一个专门表示缺失值的标量pd.NA，
它代表空整数、空布尔、空字符，这个功能目前处于实验阶段。
pd.NA的目标是提供一个“缺失值”指示器，该指示器可以在各种数据类型中一致使用
（而不是np.nan、None或pd.NaT，具体取决于数据类型）。
'''

s = pd.Series([1, 2, None, 4], dtype='Int64')
print(s)
print(s[2])
# 0       1
# 1       2
# 2    <NA>
# 3       4
# dtype: Int64
# <NA>

print(s[2] is pd.NA)  # True

print(pd.isna(pd.NA))  # True

# 以下是pd.NA参与运算的一些逻辑示例：

print()
print('# 运算')
# 加法
print(pd.NA + 1)  # <NA>
# 乘法
print('a' * pd.NA)  # <NA>
print('a' + pd.NA)  # 同上
print(pd.NA ** 0)  # 1
print(1 ** pd.NA)  # 1
# 其它示例
print(pd.NA == 1)  # <NA>
print(pd.NA == pd.NA)  # <NA>
print(pd.NA < 2.5)  # <NA>

print()
print('------------------------------------------------------------')
print('\t10.1.6 时间数据中的缺失值')
'''
对于时间数据中的缺失值，Pandas提供了一个NaT来表示，并且NaT和NaN是兼容的：
'''
print('# 时间数据中的缺失值')
s = pd.Series([pd.Timestamp('20200101'), None, pd.Timestamp('20200103')])
# pd.Timestamp('20200103')])
print(s)
# 0   2020-01-01
# 1          NaT
# 2   2020-01-03
# dtype: datetime64[ns]

print()
print('------------------------------------------------------------')
print('\t10.1.7 整型数据中的缺失值')

'''由于NaN是浮点型，因此缺少一个整数的列可以转换为整型。'''

# print(df)
print(type(df.at[2, 'D']))
# <class 'numpy.float64'>
print(pd.Series([1, 2, np.nan, 4], dtype=pd.Int64Dtype()))
# 0       1
# 1       2
# 2    <NA>
# 3       4
# dtype: Int64

print()
print('------------------------------------------------------------')
print('\t10.1.8 插入缺失值')
'''如同修改数据一样，我们可以通过以下方式将缺失值插入数据中：'''
print('# 修改为缺失值')
df.loc[0] = None
df.loc[1] = np.nan
df.A = pd.NA
print(df)
#       A     B    C    D    E
# 0  <NA>  None  NaN  NaN  NaN
# 1  <NA>   NaN  NaN  NaN  NaN
# 2  <NA>  None  3.0  NaN  7.0
# 3  <NA>    b2  4.0  8.0  8.0

'''
10.1.9 小结
本节我们介绍了数据中的None、np.nan和pd.NA，它们都是缺失值的类型，对缺失值的识别和判定非常关键。
只有有效识别出数据的缺失部分，我们才能对这些缺失值进行处理。
'''

print()
print('------------------------------------------------------------')
print('第10章 Pandas数据清洗')
print('\t10.2 缺失值的操作')
print('\t10.2.1 缺失值填充')
print()
# update20240517
'''
对于缺失值，我们通常会根据业务需要进行修补，但对于缺失严重的数据，会直接将其删除。
本节将介绍如何对缺失值进行一些常规的操作。
'''
# 小节注释
'''
对于缺失值，我们常用的一个办法是利用一定的算法去填充它。
这样虽然不是特别准确，但对于较大的数据来说，不会对结果产生太大影响。
df.fillna(x)可以将缺失值填充为指定的值：

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print('# 原始数据')
df = pd.DataFrame({
    'A': ['a1', 'a1', 'a2', 'a2'],
    'B': ['b1', 'b2', None, 'b2'],
    'C': [1, 2, 3, 4],
    'D': [5, 6, None, 8],
    'E': [5, None, 7, 8],
})

print(df)
#     A     B  C    D    E
# 0  a1    b1  1  5.0  5.0
# 1  a1    b2  2  6.0  NaN
# 2  a2  None  3  NaN  7.0
# 3  a2    b2  4  8.0  8.0

'''以上数据中，2B、2D、1E为缺失值。如果想把正负无穷也当作缺失值，可以通过以下全局配置来设定：'''

print('# 将缺失值填充为0')
print(df.fillna(0))
#     A   B  C    D    E
# 0  a1  b1  1  5.0  5.0
# 1  a1  b2  2  6.0  0.0
# 2  a2   0  3  0.0  7.0
# 3  a2  b2  4  8.0  8.0

print('# 常用的方法还有以下几个：')
'''
# 填充为 0
df.fillna(0)
# 填充为指定字符
df.fillna('missing')
df.fillna('暂无')
df.fillna('待补充')
# 指定字段填充
df.one.fillna('暂无')
# 指定字段填充
df.one.fillna(0, inplace=True)
# 只替换第一个
df.fillna(0, limit=1)
# 将不同列的缺失值替换为不同的值
values = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
df.fillna(value=values)

有时候我们不能填入固定值，而要按照一定的方法填充。
df.fillna()提供了一个method参数，可以指定以下几个方法。
pad / ffill：向前填充，使用前一个有效值填充，
df.fillna(method='ffill')可以简写为df.ffill()。
bfill / backfill：向后填充，使用后一个有效值填充，
df.fillna(method='bfill')可以简写为df.bfill()。


'''
print(df.fillna(method='pad'))
#     A   B  C    D    E
# 0  a1  b1  1  5.0  5.0
# 1  a1  b2  2  6.0  5.0
# 2  a2  b2  3  6.0  7.0
# 3  a2  b2  4  8.0  8.0

'''除了取前后值，还可以取经过计算得到的值，比如常用的平均值填充法：'''

print()
print('# 填充列的平均值')

dff = df.fillna(0).drop(columns=['A', 'B']).mean()
print(dff)
# C    2.50
# D    4.75
# E    5.00
# dtype: float64

print(df.fillna(dff))
#     A     B  C     D    E
# 0  a1    b1  1  5.00  5.0
# 1  a1    b2  2  6.00  5.0
# 2  a2  None  3  4.75  7.0
# 3  a2    b2  4  8.00  8.0
print(df.fillna(dff.mean()))
#     A         B  C         D         E
# 0  a1        b1  1  5.000000  5.000000
# 1  a1        b2  2  6.000000  4.083333
# 2  a2  4.083333  3  4.083333  7.000000
# 3  a2        b2  4  8.000000  8.000000

print()
print('# 对指定列填充平均值')
# df.loc[:,['B','D']] = df.loc[:,['B','D']].fillna(dff.mean())
# print(df)
#     A         B  C         D    E
# 0  a1        b1  1  5.000000  5.0
# 1  a1        b2  2  6.000000  NaN
# 2  a2  4.083333  3  4.083333  7.0
# 3  a2        b2  4  8.000000  8.0

'''缺失值填充的另一个思路是使用替换方法df.replace()：'''
print()
print('# 将指定列的空值替换成指定值')
print(df.replace({'E': {np.nan: 100}}))

print()
print('------------------------------------------------------------')
print('\t10.2.2 插值填充')
'''
插值（interpolate）是离散函数拟合的重要方法，利用它可根据函数在有限个点处的取值状况，
估算出函数在其他点处的近似值。
Series和DataFrame对象都有interpolate()方法，默认情况下，该方法在缺失值处执行线性插值。
它利用数学方法来估计缺失点的值，对于较大的数据非常有用。
'''

s = pd.Series([0, 1, np.nan, 3])

# 插值填充
print(s.interpolate())
# 0    0.0
# 1    1.0
# 2    2.0
# 3    3.0
# dtype: float64

'''其中默认method ='linear'，即使用线性方法，认为数据呈一条直线。method方法指定的是插值的算法。'''

'''
如果你的数据增长速率越来越快，可以选择method='quadratic'二次插值；
如果数据集呈现出累计分布的样子，推荐选择method='pchip'；
如果需要填补默认值，以平滑绘图为目标，推荐选择method='akima'。

这些都需要你的环境中安装了SciPy库。
'''

print()
print(s.interpolate(method='akima'))

print()
print('------------------------------------------------------------')
print('\t10.2.3 缺失值删除')

print(df)
print('# 删除有缺失值的行')
print(df.dropna())
#     A   B  C    D    E
# 0  a1  b1  1  5.0  5.0
# 3  a2  b2  4  8.0  8.0

print('# 删除有缺失值的列')
print(df.dropna(axis=1))  # axis='columns'
#     A  C
# 0  a1  1
# 1  a1  2
# 2  a2  3
# 3  a2  4

'''
# 删除所有有缺失值的行
df.dropna()
# 删除所有有缺失值的列
df.dropna(axis='columns')
df.dropna(axis=1)
# 删除所有值都缺失的行
df.dropna(how='all')
# 删除至少有两个缺失值的行 / 保留非缺失值大于等于2的行
df.dropna(thresh=2)
# 指定判断缺失值的列范围
df.dropna(subset=['name', 'born'])
# 使删除的结果生效
df.dropna(inplace=True)
# 指定列的缺失值删除
df.col.dropna()

'''
print(df.dropna(thresh=4))  # 保留非缺失值大于等于4的行
#     A   B  C    D    E
# 0  a1  b1  1  5.0  5.0
# 1  a1  b2  2  6.0  NaN
# 3  a2  b2  4  8.0  8.0

print()
# print(df)
print(df.dropna(subset=['E']))  # 指定E列有空值行时 才删除该行
#     A     B  C    D    E
# 0  a1    b1  1  5.0  5.0
# 2  a2  None  3  NaN  7.0
# 3  a2    b2  4  8.0  8.0
# print(df.dropna(subset=['D','E']))

print()
print('# 指定列的缺失值删除')
print(df.E.dropna())
# 0    5.0
# 2    7.0
# 3    8.0
# Name: E, dtype: float64


print()
print('------------------------------------------------------------')
print('\t10.2.4 缺失值参与计算')
# 对所有列求和

print(df.drop(columns='B'))
df1 = df.drop(columns='B')  # 如果B列存在 运算会报错！
print(df1.sum())
# A    a1a1a2a2
# C          10
# D        19.0
# E        20.0
# dtype: object

'''加法会忽略缺失值，或者将其按0处理，再试试累加：'''
print(df.D.cumsum())
# 0     5.0
# 1    11.0
# 2     NaN
# 3    19.0
# Name: D, dtype: float64

'''cumsum()和cumprod()会忽略NA值，但值会保留在序列中，可以使用skipna=False跳过有缺失值的计算并返回缺失值：'''

print()
# print(df.D.cumprod()) # 累乘
print(df.D.cumsum(skipna=False))  # 累加，跳过空值
# 0     5.0
# 1    11.0
# 2     NaN
# 3     NaN
# Name: D, dtype: float64

print()
print('# 缺失值不计数')
print(df.count())
# A    4
# B    3
# C    4
# D    3
# E    3
# dtype: int64

'''
再看看缺失值在做聚合分组操作时的情况，如果聚合分组的列里有空值，则会自动忽略这些值（当它不存在）：
'''

print()
print(df.groupby('B').sum())
#        A  C     D    E
# B
# b1    a1  1   5.0  5.0
# b2  a1a2  6  14.0  8.0

print('# 聚合计入缺失值')
print(df.groupby('B', dropna=False).sum())
#         A  C     D    E
# B
# b1     a1  1   5.0  5.0
# b2   a1a2  6  14.0  8.0
# NaN    a2  3   0.0  7.0

# df.drop(columns='A',inplace=True)
# print(df)

'''
10.2.5 小结
本节介绍了缺失值的填充方法。
如果数据质量有瑕疵，在不影响分析结果的前提下，可以用固定值填充、插值填充。
对于质量较差的数据可以直接丢弃。
'''

print()
print('------------------------------------------------------------')
print('第10章 Pandas数据清洗')
print('\t10.3 数据替换')
print('\t10.3.1 指定值替换')
print()
# update20240520
'''
Pandas中数据替换的方法包含数值、文本、缺失值等替换，
经常用于数据清洗与整理、枚举转换、数据修正等情形。
Series和DataFrame中的replace()都提供了一种高效而灵活的方法。
'''
# 小节注释
'''
以下是在Series中将0替换为5：

▶ 以下是一些具体的使用方法举例：
'''
# df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
# print(df)
# print(df.dtypes)
print('# 原始数据')
# 以下是在Series中将0替换为5：
ser = pd.Series([0., 1., 2., 3., 4.])
print(ser)
print(ser.replace(0, 5))
# 0    5.0
# 1    1.0
# 2    2.0
# 3    3.0
# 4    4.0
# dtype: float64

print()
# 也可以批量替换：
print('# 一一对应进行替换')
# print(ser.replace([0,1,2,3,4],[4,3,2,1,0]))
# # 用字典映射对应替换值
# ser.replace({0: 10, 1: 100})
# # 将a列的0、b列中的5替换为100
# df.replace({'a': 0, 'b': 5}, 100)
# # 指定列里的替换规则
# df.replace({'a': {0: 100, 4: 400}})

print()
print('------------------------------------------------------------')
print('\t10.3.2 使用替换方式')

'''除了给定指定值进行替换，我们还可以指定一些替换的方法：'''
# 将 1，2，3 替换为它们前一个值
print(ser.replace([1, 2, 3], method='pad'))  # ffill是它同义词
# 将 1，2，3 替换为它们后一个值
print(ser.replace([1, 2, 3], method='bfill'))

'''
如果指定的要替换的值不存在，则不起作用，也不会报错。以上的替换也适用于字符类型数据。
'''

print()
print('------------------------------------------------------------')
print('\t10.3.3 字符替换')

'''
替换方法默认没有开启正则匹配模式，直接按原字符匹配替换，
如果遇到字符规则比较复杂的内容，可使用正则表达式进行匹配：
'''
df = pd.DataFrame({'A': ['bat', 'foo', 'bar', 'baz', 'foobar'], 'B': ['bat', 'foo', 'bar', 'xyz', 'foobar']})
print(df)
# 把bat替换为new，不使用正则表达式
print(df.replace(to_replace='bat', value='new'))
#        A      B
# 0     new     new
# 1     foo     foo
# 2     bar     bar
# 3     baz     xyz
# 4  foobar  foobar

print()
print('# 利用正则表达式将ba开头的值替换为new')
# df.replace(to_replace=r'^ba.$', value='new', regex=True)
print(df.replace(to_replace=r'^ba.$', value='new', regex=True))

print('# 如果多列规则不一，可以按以下格式对应传入')
print(df.replace({'A': r'^ba.$'}, {'A': 'new'}, regex=True))

# 测试不同列 不同修改方式，报错！
# print(df.replace({'A':r'^ba.$'},{'A':'new'},{'B':r'^ba.$'},{'B':'wocao'},regex=True))
print()
# 多个规则均替换为同样的值
# df.replace(regex=[r'^ba.$', 'foo'], value='new')
# 多个正则及对应的替换内容
# df.replace(regex={r'^ba.$': 'new', 'foo': 'xyz'})

print()
print('------------------------------------------------------------')
print('\t10.3.4 缺失值替换')

'''替换可以处理缺失值相关的问题，例如我们可以先将无效的值替换为nan，再进行缺失值处理：'''

d = {'a': list(range(4)),
     'b': list('ab..'),
     'c': ['a', 'b', np.nan, 'd']
     }

df = pd.DataFrame(d)
print(df)

print('# 将.替换为NaN')
# print(df.replace('.',np.nan))

print('# 使用正则表达式，将空格等替换为NaN')
print(df.replace(r'\s*\.\s*', np.nan, regex=True))  # 结果同上  将.替换为NaN
#    a    b    c
# 0  0    a    a
# 1  1    b    b
# 2  2  NaN  NaN
# 3  3  NaN    d

print()
# 对应替换，a换b，点换NaN
# print(df)
print(df.replace(['a', '.'], ['b', np.nan]))
#    a    b    c
# 0  0    b    b
# 1  1    b    b
# 2  2  NaN  NaN
# 3  3  NaN    d
print()
print(df.replace([r'\.', r'(a)'], ['dot', r'\1stuff'], regex=True))  # 点换dot，a换astuff
#    a       b       c
# 0  0  astuff  astuff
# 1  1       b       b
# 2  2     dot     NaN
# 3  3     dot       d


'''
# b中的点要替换，将b替换为NaN，可以多列
df.replace({'b': '.'}, {'b': np.nan})
# 使用正则表达式
df.replace({'b': r'\s*\.\s*'}, {'b': np.nan}, regex=True)
# b列的b值换为空
df.replace({'b': {'b': r''}}, regex=True)
# b列的点、空格等替换为NaN
df.replace(regex={'b': {r'\s*\.\s*': np.nan}})
# 在b列的点后加ty，即.ty
df.replace({'b': r'\s*(\.)\s*'},
{'b': r'\1ty'},
regex=True)
# 多个正则规则
df.replace([r'\s*\.\s*', r'a|b'], np.nan, regex=True)
# 用参数名传参
df.replace(regex=[r'\s*\.\s*', r'a|b'], value=np.nan)

'''

print()
print('------------------------------------------------------------')
print('\t10.3.5 数字替换')

'''将相关数字替换为缺失值：'''
df = pd.DataFrame(np.random.randn(10, 2))
# 生成一个布尔索引数组，长度与df的行数相同
mask = np.random.rand(df.shape[0]) > 0.5
print(mask)
df.loc[mask] = 1.5
print(df)
# print(df)

print(df.replace(1.5, None))

'''个人感觉没啥意义'''

print()
print('------------------------------------------------------------')
print('\t10.3.5 数据修剪')
'''
对于数据中存在的极端值，过大或者过小，可以使用df.clip(lower,upper)来修剪。
当数据大于upper时使用upper的值，小于lower时用lower的值，这和numpy.clip方法一样。
'''
# 包含极端值的数据
df = pd.DataFrame({'a': [-1, 2, 5], 'b': [6, 1, -3]})
# print(df)
print('# 修剪成最大为3，最小为0')
print(df.clip(0, 3))
#    a  b
# 0  0  3
# 1  2  1
# 2  3  0

# 按列指定下限和上限阈值进行修剪，如下例中数据按同索引位的c值和c对应值+1进行修剪
print()
c = pd.Series([-1, 1, 3])
print(df.clip(c, c + 1, axis=0))
#    a  b
# 0 -1  0
# 1  2  1
# 2  4  3

'''
10.3.7 小结
替换数据是数据清洗的一项很普遍的操作，同时也是修补数据的一种有效方法。
df.replace()方法功能强大，在本节中，我们了解了它实现定值替换、定列替换、广播替换、运算替换等功能。
'''

print()
print('------------------------------------------------------------')
print('第10章 Pandas数据清洗')
print('\t10.4 重复值及删除数据')
print('\t10.4.1 重复值识别')
print()
# update20240523
'''
数据在收集、处理过程中会产生重复值，包括行和列，既有完全重复，又有部分字段重复。
重复的数据会影响数据的质量，特别是在它们参与统计计算时。
本节介绍Pandas如何识别重复值、删除重复值，以及如何删除指定的数据。
'''
# 小节注释
'''
df.duplicated()是Pandas用来检测重复值的方法，语法为：：
# 检测重复值语法
df.duplicated(subset=None, keep='first')

它可以返回表示重复行的布尔值序列，默认为一行的所有内容，
subset可以指定列。keep参数用来确定要标记的重复值，可选的值有：
first：将除第一次出现的重复值标记为True，默认。
last：将除最后一次出现的重复值标记为True。
False：将所有重复值标记为True。
▶ 以下是一些具体的使用方法举例：
'''
df = pd.DataFrame({'A': ['x', 'x', 'z'],
                   'B': ['x', 'x', 'x'],
                   'C': [1, 1, 2],
                   })

print('# 全行检测，除第一次出现的外，重复的为True')

print(df.duplicated())
# 0    False
# 1     True
# 2    False
# dtype: bool

print('# 除最后一次出现的外，重复的为True')
print(df.duplicated(keep='last'))
# 0     True
# 1    False
# 2    False
# dtype: bool

print('# 所有重复的都为True')
print(df.duplicated(keep=False))
# 0     True
# 1     True
# 2    False
# dtype: bool
print('# 指定列检测')
print(df.duplicated(subset=['B'], keep=False))
# 0    True
# 1    True
# 2    True
# dtype: bool

'''重复值的检测可用于数据的查询和筛选，示例如下：'''
print(df[df.duplicated()])
#    A  B  C
# 1  x  x  1

print()
print('------------------------------------------------------------')
print('\t10.4.2 删除重复值')

'''
删除重复值的语法如下：
     df.drop_duplicates(subset=None,
                         keep='first',
                         inplace=False,
                         ignore_index=False)

参数说明如下。
subset：指定的标签或标签序列，仅删除这些列重复值，默认情况为所有列。
keep：确定要保留的重复值，有以下可选项。
     first：保留第一次出现的重复值，默认。
     last：保留最后一次出现的重复值。
     False：删除所有重复值。
inplace：是否生效。
ignore_index：如果为True，则重新分配自然索引（0，1，…，n–1）。

'''

print(df)
print('# 删除重复行')
print(df.drop_duplicates())
#    A  B  C
# 0  x  x  1
# 2  z  x  2

print('# 删除指定列')
print(df.drop_duplicates(subset=['A']))  # 结果同上
#    A  B  C
# 0  x  x  1
# 2  z  x  2

print('# 保留最后一个')
print(df.drop_duplicates(subset=['A'], keep='last'))
#    A  B  C
# 1  x  x  1
# 2  z  x  2

print('# 删除全部重复值')
print(df.drop_duplicates(subset=['A'], keep=False))
#    A  B  C
# 2  z  x  2

print()
print('------------------------------------------------------------')
print('\t10.4.3 删除数据')
'''
df.drop()通过指定标签名称和相应的轴，或直接给定索引或列名称来删除行或列。
使用多层索引时，可以通过指定级别来删除不同级别上的标签。

# 语法
df.drop(labels=None, axis=0,
          index=None, columns=None,
          level=None, inplace=False,
          errors='raise')

参数说明如下：
     labels：要删除的列或者行，如果要删除多个，传入列表。
     axis：轴的方向，0为行，1为列，默认为0。
     index：指定的一行或多行。
     column：指定的一列或多列。
     level：索引层级，将删除此层级。
     inplace：布尔值，是否生效。
     errors：ignore或者raise，默认为raise，如果为ignore，则容忍错误，仅删除现有标签。
'''

print(df)
print('# 删除指定行')
print(df.drop([0, 1]))
#    A  B  C
# 2  z  x  2

print('# 删除指定列')
print(df.drop(['B', 'C'], axis=1))
print(df.drop(columns=['B', 'C']))  # 结果同上
#    A
# 0  x
# 1  x
# 2  z

'''
10.4.4 小结
本节介绍了三个重要的数据清洗工具：
df.duplicated()能够识别出重复值，返回一个布尔序列，用于查询和筛选重复值；
df.drop_duplicates()可以直接删除指定的重复数据；
df.drop()能够灵活地按行或列删除指定的数据，可以通过计算得到异常值所在的列和行再执行删除。

'''

print()
print('------------------------------------------------------------')
print('第10章 Pandas数据清洗')
print('\t10.5 NumPy格式转换')
print('\t10.5.1 转换方法')
print()
# update20240523
'''
2.5节介绍过可以将一个NumPy数据转换为DataFrame或者Series数据。
在特征处理和数据建模中，很多库使用的是NumPy中的ndarray数据类型，
Pandas在对数据进行处理后，要将其应用到上述场景，就需要将类型转为NumPy的ndarray。
本节就来介绍一下如何将Pandas的数据类型转换为NumPy的类型。
'''
# 小节注释
'''
Pandas 0.24.0引入了两种从Pandas对象中获取NumPy数组的新方
法。
ds.to_numpy()：可以用在Index、Series和DataFrame对象；
s.array：为PandasArray，用在Index和Series，它封装了numpy.ndarray接口。

有了以上方法，不再推荐使用Pandas的values和as_matrix()。
上述这两个函数旨在提高API的一致性，是Pandas官方未来支持的方向，
values和as_matrix()虽然在近期的版本中不会被弃用，
但可能会在将来的某个版本中被取消，因此官方建议用户尽快迁移到较新的API。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.DataFrame({'A': ['x', 'x', 'z'],
                   'B': ['x', 'x', 'x'],
                   'C': [1, 1, 2],
                   })

print()
print('------------------------------------------------------------')
print('\t10.5.2 DataFrame转为ndarray')
'''df.values和df.to_numpy()返回的是一个array类型：'''

df = pd.read_excel(team_file)
print(df)

print(df.values)  # 不推荐
print(df.to_numpy())  # 推荐，结果同上
# [['Liver' 'E' 89 21 24 64]
#  ['Arry' 'C' 36 37 37 57]
#  ['Ack' 'A' 57 60 18 84]
#  ['Eorge' 'C' 93 96 71 78]
#  ['Oah' 'D' 65 49 61 86]
#  ['Rick' 'B' 100 99 97 100]]

print(type(df.to_numpy()))
# <class 'numpy.ndarray'>

print(df.to_numpy().dtype)  # object

print('# 转换指定的列')
print(df[['name', 'Q1']].to_numpy())
# [['Liver' 89]
#  ['Arry' 36]
#  ['Ack' 57]
#  ['Eorge' 93]
#  ['Oah' 65]
#  ['Rick' 100]]

print()
print('------------------------------------------------------------')
print('\t10.5.3 Series转为ndarray')

'''对Series使用s.values和s.to_numpy()返回的是一个array类型：'''
# df.Q1.values # 不推荐
# df.Q1.to_numpy()
print(df.Q1.to_numpy())
# [ 89  36  57  93  65 100]
print(type(df.Q1.to_numpy()))
# <class 'numpy.ndarray'>
print(df.Q1.to_numpy().dtype)
# int64
print(type(df.Q1.to_numpy().dtype))
# <class 'numpy.dtype[int64]'>
print(df.Q1.array)
# Length: 6, dtype: int64
print(type(df.Q1.array))
# <class 'pandas.core.arrays.numpy_.PandasArray'>

print()
print('------------------------------------------------------------')
print('\t10.5.4 df.to_records()')

'''可以使用to_records()方法，但是如果数据类型不是你想要的，则必须对它们进行一些处理。'''

# 转为NumPy record array
print(df.to_records())
# [(0, 'Liver', 'E',  89, 21, 24,  64) (1, 'Arry', 'C',  36, 37, 37,  57)
#  (2, 'Ack', 'A',  57, 60, 18,  84) (3, 'Eorge', 'C',  93, 96, 71,  78)
#  (4, 'Oah', 'D',  65, 49, 61,  86) (5, 'Rick', 'B', 100, 99, 97, 100)]
print(type(df.to_records()))
# <class 'numpy.recarray'>
print('# 转为array')
print(np.array(df.to_records()))  # 看起来结果和上述一致，但是数据类型不同
# [(0, 'Liver', 'E',  89, 21, 24,  64) (1, 'Arry', 'C',  36, 37, 37,  57)
#  (2, 'Ack', 'A',  57, 60, 18,  84) (3, 'Eorge', 'C',  93, 96, 71,  78)
#  (4, 'Oah', 'D',  65, 49, 61,  86) (5, 'Rick', 'B', 100, 99, 97, 100)]
print(type(np.array(df.to_records())))  # <class 'numpy.ndarray'>

'''上例中，to_records()将数据转为了NumPy的record array类型，然后再用NumPy的np.array读取一下，转为array类型。'''
print(df.to_records()[0])
print(np.array(df.to_records())[0])  # 结果同上
# (0, 'Liver', 'E', 89, 21, 24, 64)

print()
print('------------------------------------------------------------')
print('\t10.5.5 np.array读取')
'''可以用np.array直接读取DataFrame或者Series数据，最终也会转换为array类型：'''

print(np.array(df))  # Dataframe转
# [['Liver' 'E' 89 21 24 64]
#  ['Arry' 'C' 36 37 37 57]
#  ['Ack' 'A' 57 60 18 84]
#  ['Eorge' 'C' 93 96 71 78]
#  ['Oah' 'D' 65 49 61 86]
#  ['Rick' 'B' 100 99 97 100]]
print(np.array(df.Q1))  # 直接转
# [ 89  36  57  93  65 100]
print(np.array(df.Q1.array))  # PandasArray转  结果同上
print(np.array(df.to_records().view(type=np.matrix)))  # 转为矩阵
# [[(0, 'Liver', 'E',  89, 21, 24,  64) (1, 'Arry', 'C',  36, 37, 37,  57)
#   (2, 'Ack', 'A',  57, 60, 18,  84) (3, 'Eorge', 'C',  93, 96, 71,  78)
#   (4, 'Oah', 'D',  65, 49, 61,  86) (5, 'Rick', 'B', 100, 99, 97, 100)]]

'''
10.5.6 小结
本节介绍了如何将Pandas的两大数据类型DataFrame和Series转为NumPy的格式，推荐使用to_numpy()方法。
关于NumPy的更多操作可以访问笔者的NumPy在线教程，地址为https://www.gairuo.com/p/numpytutorial。

10.6 本章小结
数据清洗是我们获取到数据集后要做的第一件事，处理缺失数据和缺失值是数据清洗中最棘手的部分。
只有保证数据的高质量才有可能得出高质量的分析结论，
一些数据建模和机器学习的场景对数据质量有严格的要求，甚至不允许有缺失值。

本章介绍了在Pandas中缺失值的表示方法以及如何找到缺失值，
重复值的筛选方法以及如何对它们进行删除、替换和填充等操作。
完成这些工作，将得到一个高质量的数据集，为下一步数据分析做好准备。
'''

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

print()
print('------------------------------------------------------------')
print('第12章 Pandas分类数据')
print('\t12.1 分类数据')
# update20240529
'''
分类数据（categorical data）是按照现象的某种属性对其进行分类或分组而得到的反映事物类型的数据，又称定类数据。

分类数据的特点是有限性，分类数据固定且能够枚举，而且数据不会太多。
通过将数据定义为分析数据类型，可以压缩数据内存存储大小，加快计算速度，让业务指向更加清晰明了。
'''
# 小节注释
'''
分类数据是固定数量的值，在一列中表达数值具有某种属性、类型和特征。

例如，人口按照性别分为“男”和“女”，按照年龄段分为“少儿”“青年”“中年”“老年”，
按照职业分为“工人”“农民”“医生”“教师”等。其中，“男”“少儿”“农民”“医生”“教师”这些就是分类数据。

为了便于计算机处理，经常会用数字类型表示，
如用1表示“男性”，用0表示“女性”，用2表示“性别未知”，但这些数字之前没有数量意义上的大小、先后等关系。
Pandas提供的分类数据类型名称为category。

如同文本数据拥有.str.<method>访问器，类别数据也有.cat.
<method>格式的访问器，帮助我们便捷访问和操作分类数据。

▶ 以下是一些具体的使用方法举例：
'''
print()
print('------------------------------------------------------------')
print('\t12.1.1 创建分类数据')
'''构造和加载数据时，使用dtype="category"来指定数据类型：'''
# 构造数据
s = pd.Series(["x", "y", "z", "x"], dtype="category")
print(s)
# 0    x
# 1    y
# 2    z
# 3    x
# dtype: category
# Categories (3, object): ['x', 'y', 'z']
'''同样，创建DataFrame时也可以指定数据类型：'''
print('# 构造数据')
df = pd.DataFrame({'A': list('xyzz'), 'B': list('aabc')}, dtype="category")
print(df)
print(df.dtypes)
# A    category
# B    category
# dtype: object
print('# 查看指定列的数据类型')
print(df.B)
# 0    a
# 1    a
# 2    b
# 3    c
# Name: B, dtype: category
# Categories (3, object): ['a', 'b', 'c']
print()
'''在一定的情况下，会自动将数据类型创建为分类数据类型，如分箱操作：'''
print('# 生成分箱序列')
print(pd.Series(pd.cut(range(1, 10, 2), [0, 4, 6, 10])))
# 0     (0, 4]
# 1     (0, 4]
# 2     (4, 6]
# 3    (6, 10]
# 4    (6, 10]
# dtype: category
# Categories (3, interval[int64, right]): [(0, 4] < (4, 6] < (6, 10]]

print()
print('------------------------------------------------------------')
print('\t12.1.2 pd.Categorical()')

'''
pd.Categorical()用与数据分析语言R语言和S-plus（一种S语言的实现）中类似的形式来表示分类数据变量：
'''

print('# 分类数据')
print(pd.Categorical(["x", "y", "z", "x"], categories=["y", "z", "x"], ordered=True))
'''
['x', 'y', 'z', 'x']
Categories (3, object): ['y' < 'z' < 'x']
'''

print('# 构建 Series')
print(pd.Series(pd.Categorical(["x", "y", "z", "x"],
                               categories=["y", "z", "x"],
                               ordered=False)
                ))
# 0    x
# 1    y
# 2    z
# 3    x
# dtype: category
# Categories (3, object): ['y', 'z', 'x']

print()
print('------------------------------------------------------------')
print('\t12.1.3 CategoricalDtype对象')

'''
CategoricalDtype是Pandas的分类数据对象，它可以传入以下参数。
    categories：没有缺失值的不重复序列。
    ordered：布尔值，顺序的控制，默认是没有顺序的。

CategoricalDtype可以在Pandas指定dtype的任何地方，
例如pd.read_csv()、df.astype()或Series构造函数中。
分类数据默认是无序的，可以使用字符串category代替CategoricalDtype，
换句话说，dtype='category'等效于dtype=CategoricalDtype()。
'''

# 不知道介绍这玩意干嘛 实际应用看不出来有啥 不写了 用到时在过来看吧！

'''
12.1.4 类型转换
将数据类型转换为分类数据类型的最简单的方法是使用s.astype('category')，

12.1.5 小结
分类是Pandas为解决大量重复的有限值数据而增加的一个专门的数
据类型，它可以提高程序的处理速度，也能让代码更加简洁。本节介绍
了分类数据的创建、分类数据对象和类型转换，这些都需要大家掌握并
灵活运用。


12.2 分类的操作
分类数据的其他操作与其他数据类型的操作没有区别，可以参与数
据查询、数组聚合、透视、合并连接。本节介绍一些常用的操作，更加
复杂的功能可以查询Pandas官方文档。

12.2.1 修改分类
12.2.2 追加新分类
12.2.3 删除分类
12.2.4 顺序

12.2.5 小结
本节介绍了一些常用分类数据的修改、添加、删除和排序操作，在数据分析中对分类数据的调整场景比较少，
一般都是将原有数据解析转换为分类数据以提高分析效率。

12.3 本章小结
本章介绍的分类数据类型是Pandas的另一个数据分析利器，它让业务更加清晰，代码性能更为出色。

当我们遇到重复有限值时，尽量将其转换为数据类型，通过分类数据的优势和各项功能来提高数据分析的效率。
'''

print()
print('------------------------------------------------------------')
print('第13章 Pandas窗口计算')
print('\t13.1 窗口计算')
# update20240530
'''
第五部分 时序数据分析

时序数据即时间序列数据，是按一定的时间尺度及顺序记录的数据。
通过时序数据，我们可以发现样本的特征和发展变化规律，进而进行样本以外的预测。

本部分主要介绍Pandas中对于时间类型数据的处理和分析，
包括固定时间、时长、周期、时间偏移等的表示方法、查询、计算和格式处理，
以及时区转换、重采样、工作日和工作时间的处理方法。
此外，本部分还介绍了在时序数据处理中常用的窗口计算。

如果业务呈现周期性变化，就不能以最小数据单元进行分析了，而需要按照这个周期产生稳定的趋势数据再进行分析，
这就会用到窗口计算。
Pandas提供几种窗口函数，如移动窗口函数rolling()、扩展窗口函数expanding()和指数加权移动ewm()，
同时可在此基础上调用适合的统计函数，如求和、中位数、均值、协方差、方差、相关性等。
'''
# 小节注释
'''
本节介绍窗口计算的一些概念和原理，帮助大家理解什么是窗口计算，窗口计算是如何运作的，以及它有哪些实际用途。

▶ 以下是一些具体的使用方法举例：
'''
print()
print('------------------------------------------------------------')
print('\t13.1.1 理解窗口计算')

'''
所谓窗口，就是在一个数列中，选择一部分数据所形成的一个数据区间。
按照一定的规则产生很多窗口，对每个窗口施加计算得到的结果集成为一个新的数列，这个过程就是窗口计算。
'''

print()
print('------------------------------------------------------------')
print('\t13.1.2 移动窗口')

'''
移动窗口rolling()与groupby很像，但并没有固定的分组，而是创建了一个按一定移动位（如10天）移动的移动窗口对象。
我们再对每个对象进行统计操作。
一个数据会参与到多个窗口（集合、分组）中，而groupby中的一个值只能在一个组中。

'''

print()
print('------------------------------------------------------------')
print('\t13.1.3 扩展窗口')

'''
“扩展”（expanding）是从数据（大多情况下是时间）的起始处开始窗口，
增加窗口直到指定的大小。一般所有的数据都会参与所有窗口。

图13-3演示了一个典型的扩展窗口，它设置一个最小起始窗口，然后逐个向后扩展，实现类似累加的效果。
'''

print()
print('------------------------------------------------------------')
print('\t13.1.4 指数加权移动')
'''
在上述两个统计方法中，分组中的所有数值的权重都是一样的，
而指数加权移动（exponential weighted moving）对分组中的数据给予不同的权重，用于后面的计算中。

机器学习中的重要算法梯度下降法就是计算了梯度的指数加权平均数，
并以此来更新权重，这种方法的运行速度几乎总是快于标准的梯度下降算法。

Pandas提供了ewm()来实现指数加权移动，不过它在日常分析中使用较少，本书不做过多介绍。

'''

'''
13.1.5 小结
窗口计算在实际业务中有广泛的使用场景，特别是一些时序数据中，
如股票波动、气温及气候变化、生物信息研究、互联网用户行为分析等。

了解了以上基础概念，接下来我们就开始用Pandas实现这些操作。
'''

print()
print('------------------------------------------------------------')
print('第13章 Pandas窗口计算')
print('\t13.2 窗口计算')
# update20240530
'''
s.rolling()是移动窗口函数，此函数可以应用于一系列数据，指定参数window=n，并在其上调用适合的统计函数。
'''
# 小节注释
'''
本节介绍窗口计算的一些概念和原理，帮助大家理解什么是窗口计算，窗口计算是如何运作的，以及它有哪些实际用途。

▶ 以下是一些具体的使用方法举例：
'''
print()
print('------------------------------------------------------------')
print('\t13.2.1 理解窗口计算')

'''我们先使用s.rolling()做一下移动窗口操作：'''
s = pd.Series(range(1, 7))
print(s)

print('# 移动窗口')
print(s.rolling(2).sum())  # 当前行和上一行累加，上一行不存在 则结果为空值！
# 0     NaN
# 1     3.0
# 2     5.0
# 3     7.0
# 4     9.0
# 5    11.0
# dtype: float64

print()
print('------------------------------------------------------------')
print('\t13.2.2 基本语法')

'''
s.rolling()的语法如下：
# 语法
df.rolling(window, min_periods=None,
            center=False, win_type=None,
            on=None, axis=0, closed=None)

它支持以下参数。
window：必传，如果使用int，可以表示窗口的大小；如果是offset类型，表示时间数据中窗口按此时间偏移量设定大小。
min_periods：每个窗口的最小数据，小于此值窗口的输出值为NaN，offset情况下，默认为1。默认情况下此值取窗口的大小。
win_type：窗口的类型，默认为加权平均，支持非常丰富的窗口函数，
            如boxcar、triang、blackman、hamming、bartlett、
            parzen、bohman、blackmanharris、nuttall、barthann、
            kaiser(beta)、gaussian(std)、general_gaussian (power, width)、
            slepian (width)、exponential (tau)等。
            具体算法可参考SciPy库的官方文档：https://docs.scipy.org/doc/scipy/reference/signal.windows.html。
on：可选参数，对于DataFrame要作为窗口的列。
axis：计算的轴方向。
closed：窗口的开闭区间定义，支持'right'、'left、'both'或'neither'。对于offset类型，默认是左开右闭，默认为right。
'''

print()
print('------------------------------------------------------------')
print('\t13.2.3 移动窗口使用')

# 数据
df = pd.DataFrame(np.random.randn(30, 4),
                  index=pd.date_range('10/1/2020', periods=30),
                  columns=['A', 'B', 'C', 'D'])
# print(df)
#                    A         B         C         D
# 2020-10-01  1.718881 -1.175391 -1.563521  0.061129
# 2020-10-02  0.773936 -1.944398 -1.952258  1.135505
# 2020-10-03 -1.715162  0.122924  0.416170  0.557903
# ....
# 2020-10-30 -0.293076  0.301969 -0.148113 -0.600654
print('# 每两天一个窗口，求平均数')
print(df.rolling(2).mean())
#                    A         B         C         D
# 2020-10-01       NaN       NaN       NaN       NaN
# 2020-10-02 -0.928327  0.950067  0.941985  0.220288
# 2020-10-03 -0.572597  0.059254 -0.402167 -0.969063
# ...
# 2020-10-30 -0.278189  0.891837  1.601555 -0.751876

'''
我们使用时间偏移作为周期，2D代表两天，与上例相同，
不过，使用时间偏移的话，默认的最小观察数据为1，所以第一天也是有数据的，即它自身：
'''
print('# 每两天一个窗口，求平均数')
print(df.rolling('2D').mean())
#                    A         B         C         D
# 2020-10-01 -0.160455 -0.385995 -0.083510 -0.047950
# 2020-10-02 -0.318638 -0.087972 -0.175445 -0.411212
# 2020-10-03  0.126216  1.035055  0.136721 -0.140395
# ...
# 2020-10-30  0.218113 -0.214285 -0.141040  0.825379

'''如果只对一指定列进行窗口计算，可用以下两个方法之一：'''
print('# 仅对A列进行窗口计算')
print(df.rolling('2D')['A'].mean())
print(df.A.rolling('2D').mean())  # 同上

# 2020-10-01    0.751548
# 2020-10-02    0.128525
# ...
# 2020-10-30    1.228998
# Freq: D, Name: A, dtype: float64

'''使用窗口函数时可以指定窗口类型，如汉明（Hamming）窗：'''
print('# 使用窗口函数，汉明窗')
print(df.rolling(2, win_type='hamming').sum())

#                    A         B         C         D
# 2020-10-01       NaN       NaN       NaN       NaN
# 2020-10-02  0.043914  0.021920 -0.056777  0.015278
# ...
# 2020-10-30  0.027101 -0.119280 -0.009099 -0.060454

print()
print('------------------------------------------------------------')
print('\t13.2.4 统计方法')

'''
窗口主要支持以下统计方法。
count()：非空值数
sum()：值的总和
mean()：平均值
median()：数值的算术中位数
min()：最小值
max()：最大值
std()：贝塞尔校正的样本标准偏差
var()：无偏方差
skew()：样本偏斜度（三阶矩）
kurt()：峰度样本（四阶矩）
quantile()：样本分位数（百分位上的值）
cov()：无偏协方差（二进制）
corr()：关联（二进制）

'''

print()
print('------------------------------------------------------------')
print('\t13.2.5 agg()')

'''
使用agg()可以调用多个函数，多列使用不同函数或者一列使用多个函数，如对窗口中的不同列使用不同的计算方法：
'''

print('# 对窗口中的不同列使用不同的计算方法')
print(df.rolling('2D').agg({'A': sum, 'B': np.std}))

#                    A         B
# 2020-10-01 -0.224801       NaN
# 2020-10-02 -1.207768  2.235446
# ...
# 2020-10-30 -0.846298  1.523351

print('# 对同一列使用多个函数')
print(df.A.rolling('2D').agg({'A_sum': sum, 'B_std': np.std}))

#                A_sum     B_std
# 2020-10-01  0.467737       NaN
# 2020-10-02  0.220993  0.505215
# ...
# 2020-10-30  1.190788  1.187549

print()
print('------------------------------------------------------------')
print('\t13.2.6 apply()')

'''apply()可以在窗口上实现自定义函数，要求应用此函数后产生一个
单一值，因为窗口计算后每个窗口产生的也是唯一值：'''

print('# 对窗口求和再加1，最终求绝对值')

print(df.rolling('2D').apply(lambda x: abs(sum(x) + 1)))
#                    A         B         C         D
# 2020-10-01  0.785234  0.481048  0.424229  0.513090
# 2020-10-02  1.487099  0.234810  0.170226  0.649228
# ...
# 2020-10-30  1.051135  0.597912  0.288842  0.243163

print()
print('------------------------------------------------------------')
print('\t13.2.7 扩展窗口')

'''
s.expanding()是Pandas扩展窗口的实现函数，在使用和功能上简单很多，使用逻辑与s.rolling()一样。
rolling()窗口大小固定，移动计算，而expanding()只设最小可计算数量，不固定窗口大小，不断扩展进行计算，
示例代码如下。

个人理解： 类似SQL中的sum() over()  累加函数
'''

# 原始数据
s = pd.Series(range(1, 7))
# print(s)
print('# 扩展窗口操作')
print(s.expanding(2).sum())
# 0     NaN
# 1     3.0
# 2     6.0
# 3    10.0
# 4    15.0
# 5    21.0
# dtype: float64

'''实际上，当rolling()函数的窗口大小参数window为len(df)时，最终效果与expanding()是一样的。'''

'''
13.2.8 小结
移动窗口函数rolling()和扩展窗口函数expanding()十分类似，不同点仅限于窗口大小是否固定。
rolling()更为常用，它提供了更为丰富的参数，可以指定非常多的窗口函数来实现复杂的计算。

13.3 本章小结
SQL提供了窗口函数用于数据的读取计算，本章介绍的Pandas的
rolling()和expanding()正是来解决同样的问题的。窗口计算在一些时序数
据处理分析方法中使用非常广泛，另外在理论研究方面也有诸多应用。
'''

print()
print('------------------------------------------------------------')
print('第14章 Pandas时序数据')
print('\t14.1 固定时间')
# update20240603
'''
本章将全面介绍Pandas在时序数据处理中的方法，
主要有时间的概念、时间的计算机表示方法、时间的属性操作和时间格式转换；
时间之间的数学计算、时长的意义、时长与时间的加减操作；
时间偏移的概念、用途、表示方法；时间跨度的意义、表示方法、单位转换等。
'''
# 小节注释
'''
本节介绍一些关于时间的基础概念，帮助大家建立对时间的表示方式和计算方式的一个简单认知。

▶ 以下是一些具体的使用方法举例：
'''
print()
print('------------------------------------------------------------')
print('\t14.1.1 时间的表示')

'''
固定时间是指一个时间点，如2020年11月11日00:00:00。
固定时间是时序数据的基础，一个固定时间带有丰富的信息，
如年份、周几、几月、哪个季度，需要我们进行属性的读取。
'''

from datetime import datetime

print('# 当前时间')
print(datetime.now())  # 当前行和上一行累加，上一行不存在 则结果为空值！
# 2024-06-03 10:55:17.711633
print('# 指定时间')
print(datetime(2020, 11, 1, 19))
# 2020-11-01 19:00:00
print(datetime(year=2020, month=11, day=11))
# 2020-11-11 00:00:00

# 个人测试
# print(datetime(2024,5,20,00,00,00))
# print(datetime(year=2024,month=5,day=20,hour=5,minute=20,second=00))

print()
print('------------------------------------------------------------')
print('\t14.1.2 创建时间点')

'''
pd.Timestamp()是Pandas定义时间的主要函数，代替Python中的datetime.datetime对象。下面介绍它可以传入的内容。
'''
import datetime

print('# 至少需要年、月、日')
print(pd.Timestamp(datetime.datetime(2020, 6, 8)))
# 2020-06-08 00:00:00
print('# 指定时、分、秒')
print(pd.Timestamp(datetime.datetime(2020, 6, 8, 16, 17, 18)))
# 2020-06-08 16:17:18

'''指定时间字符串：'''
print()
print(pd.Timestamp('2012-05-01'))
print(type(pd.Timestamp('2012-05-01')))
# 2012-05-01 00:00:00
# <class 'pandas._libs.tslibs.timestamps.Timestamp'>
print(pd.Timestamp('2017-01-01T12'))
# 2017-01-01 12:00:00
print(pd.Timestamp('2017-01-01T12:05:59'))
# 2017-01-01 12:05:59

'''指定时间位置数字，可依次定义year、month、day、hour、
minute、second、microsecond：'''

print(pd.Timestamp(2020, 5, 1))  # 2020-05-01 00:00:00
print(pd.Timestamp(2017, 1, 1, 12))  # 2017-01-01 12:00:00
print(pd.Timestamp(year=2017, month=1, day=1, hour=12))  # 2017-01-01 12:00:00

print('# 解析时间戳：')
print(pd.Timestamp(1513393355.5, unit='s'))  # 单位为秒 2017-12-16 03:02:35.500000
# print(pd.Timestamp(1513393355.5,unit='h')) # 测试 结果同上 why？

'''用tz指定时区，需要记住的是北京时间值为Asia/Shanghai：'''
print(pd.Timestamp(1513393355, unit='s', tz='US/Pacific'))  # 2017-12-15 19:02:35-08:00
print(pd.Timestamp(1513393355, unit='s'))  # 2017-12-16 03:02:35
print(pd.Timestamp(1513393355, unit='s', tz='Asia/Shanghai'))  # 2017-12-16 11:02:35+08:00
print()
print(pd.Timestamp(datetime.datetime.now()))  # 2024-06-03 13:27:04.434208
print(pd.Timestamp(datetime.datetime.now(), unit='s'))  # 同上
print(pd.Timestamp(datetime.datetime.now(), unit='s', tz='Asia/Shanghai'))  # 2024-06-03 13:28:03.099779+08:00

print('获取到当前时间，从而可通过属性取到今天的日期、年份等信息：')
print(pd.Timestamp('today'))  # 2024-06-03 13:29:14.516242
print(pd.Timestamp('now'))  # 同上
print(pd.Timestamp('today').date())  # 只取日期 2024-06-03

print('通过当前时间计算出昨天、明天等信息：')
# # 昨天
print(pd.Timestamp('now') - pd.Timedelta(days=1))  # 2024-06-02 14:04:42.100899
print(pd.Timedelta(days=1))  # 1 days 00:00:00
# 明天
print(pd.Timestamp('now') + pd.Timedelta(days=1))  # 2024-06-04 14:06:26.799704
# 当月初，一日
print(pd.Timestamp('now').replace(day=1))  # 2024-06-01 14:07:12.629509

'''pd.to_datetime()也可以实现上述功能，不过根据语义，它常用在时间转换上。'''
print()
print(pd.to_datetime('now'))  # 2024-06-03 14:14:45.508085
'''
由于Pandas以纳秒粒度表示时间戳，因此可以使用64位整数表示的时间跨度限制为大约584年，
意味着能表示的时间范围有最早和早晚的限制：
'''
print(pd.Timestamp.min)  # 1677-09-21 00:12:43.145224193
print(pd.Timestamp.max)  # 2262-04-11 23:47:16.854775807

'''
不过，Pandas也给出一个解决方案：使用PeriodIndex来解决。PeriodIndex后面会介绍。
'''

print()
print('------------------------------------------------------------')
print('\t14.1.3 时间的属性')

'''
一个固定的时间包含丰富的属性，包括时间所在的年份、月份、周几，是否月初，在哪个季度等。
利用这些属性，我们可以进行时序数据的探索。
我们先定义一个当前时间：
'''
time = pd.Timestamp('now')

'''以下是丰富的时间属性：'''
print(time.tz)
'''
time.asm8 # 返回NumPy datetime64格式（以纳秒为单位）
# numpy.datetime64('2020-06-09T16:30:54.813664000') || # 2024-06-03T14:23:26.961411000
time.dayofweek # 1（周几，周一为0）
time.dayofyear # 161（一年的第几天）  # 155
time.days_in_month # 30（当月有多少天）
time.daysinmonth # 30（同上）
time.freqstr # None（周期字符）
time.is_leap_year # True（是否闰年，公历的）
time.is_month_end # False（是否当月最后一天）
time.is_month_start # False（是否当月第一天）
time.is_quarter_end # False（是否当季最后一天）
time.is_quarter_start # False（是否当季第一天）
time.is_year_end # 是否当年最后一天
time.is_year_start # 是否当年第一天
time.quarter # 2（当前季度数）
# 如指定，会返回类似<DstTzInfo 'Asia/Shanghai' CST+8:00:00 STD>
time.tz # None（当前时区别名）
time.week # 24（当年第几周）
time.weekofyear # 24（同上）
time.day # 9（日）
time.fold # 0
time.freq # None（频度周期）
time.hour # 16
time.microsecond # 890462
time.minute # 46
time.month # 6
time.nanosecond # 0
time.second # 59
time.tzinfo # None
time.value # 1591721219890462000
time.year # 2020
'''

print()
print('------------------------------------------------------------')
print('\t14.1.4 时间的方法')

'''
可以对时间进行时区转换、年份和月份替换等一系列操作。我们取当前时间，并指定时区为北京时间：
'''
time = pd.Timestamp('now', tz='Asia/Shanghai')
print('# 转换为指定时区')
print(time.astimezone('UTC'))
# 2024-06-03 06:32:04.555630+00:00
print('# 转换单位，向上舍入')
print(time.ceil('s'))  # 2024-06-03 14:34:19+08:00
print(time.ceil('ns'))  # 转为以纳秒为单位 || 2024-06-03 14:34:40.371953+08:00
print(time.ceil('d'))  # 保留日 || 2024-06-04 00:00:00+08:00
print(time.ceil('h'))  # 保留时 || 2024-06-03 15:00:00+08:00

print('# 转换单位，向下舍入')
print(time.floor('h'))  # 保留时 | 2024-06-03 14:00:00+08:00
print(time.floor(freq='H'))  # 结果同上
print(time.floor('5T'))  # 2024-06-03 14:40:00+08:00
print('# 类似四舍五入')
print(time.round('h'))  # 保留时 |2024-06-03 15:00:00+08:00
print('# 返回星期名')
print(time.day_name())  # Monday
print('# 月份名称')
print(time.month_name())  # June
print('# 将时间戳规范化为午夜，保留tz信息')
print(time.normalize())  # 2024-06-03 00:00:00+08:00

print()
print('# 将时间元素替换datetime.replace，可处理纳秒')
print(time.replace(year=2019))  # 年份换为2019年
# 2019-06-03 14:50:47.318516+08:00
print(time.replace(month=8))  # 月份换为8月
# 2024-08-03 14:51:22.882376+08:00

print()
print('# 转换为周期类型，将丢失时区')
print(time.to_period(freq='h'))  # 2024-06-03 14:00
print('# 转换为指定时区')
print(time.tz_convert('UTC'))  # 转为UTC时间
# print(time)
# 2024-06-03 06:53:15.079697+00:00
print('# 本地化时区转换')
time = pd.Timestamp('now')  # 必须增加 否则下行代码运行失败
# print(time)
print(time.tz_localize('Asia/Shanghai'))

'''
对一个已经具有时区信息的时间戳再次进行本地化（tz_localize），这是不被允许的.
对于已经具有时区信息的时间戳，你应该使用tz_convert来转换时区
'''

print()
print('------------------------------------------------------------')
print('\t14.1.5 时间缺失值')
'''对于时间的缺失值，有专门的NaT来表示：'''
print(pd.Timestamp(pd.NaT))  # NaT
print(pd.Timedelta(pd.NaT))  # NaT
print(pd.Period(pd.NaT))  # NaT

print('# 类似np.nan')
print(pd.NaT == pd.NaT)  # False

'''
NaT可以代表固定时间、时长、时间周期为空的情况，类似于
np.nan可以参与到时间的各种计算中：
'''
print(pd.NaT + pd.Timestamp('20201001'))  # NaT
print(pd.NaT + pd.Timedelta('2 days'))  # NaT

'''
14.1.6 小结
时间序列是由很多个按照一定频率的固定时间组织起来的。
Pandas借助NumPy的广播机制，对时间序列进行高效操作。
因此熟练掌握时间的表示方法和一些常用的操作是至关重要的。
'''

print()
print('------------------------------------------------------------')
print('第14章 Pandas时序数据')
print('\t14.2 时长数据')
# update20240603
'''
本章将全面介绍Pandas在时序数据处理中的方法，
主要有时间的概念、时间的计算机表示方法、时间的属性操作和时间格式转换；
时间之间的数学计算、时长的意义、时长与时间的加减操作；
时间偏移的概念、用途、表示方法；时间跨度的意义、表示方法、单位转换等。
'''
# 小节注释
'''
前面介绍了固定时间，如果两个固定时间相减会得到什么呢？时间差或者时长。
时间差代表一个时间长度，它与固定时间已经没有了关系，没有指定的开始时间和结束时间，

▶ 以下是一些具体的使用方法举例：
'''
print()
print('------------------------------------------------------------')
print('\t14.2.1 创建时间差')

'''
pd.Timedelta()对象表示时间差，也就是时长，以差异单位表示，
例如天、小时、分钟、秒等。它们可以是正数，也可以是负数。
'''

from datetime import datetime

print('# 两个固定时间相减')
print(pd.Timestamp('2020-11-01 15') - pd.Timestamp('2020-11-01 14'))
# 0 days 01:00:00
print(pd.Timestamp('2020-11-01 08') - pd.Timestamp('2020-11-02 08'))
# -1 days +00:00:00

'''按以下格式传入字符串：'''
print('# 一天')
print(pd.Timedelta('1 days'))  # 1 days 00:00:00
print(pd.Timedelta('1 days 00:00:00'))  # 1 days 00:00:00
print(pd.Timedelta('1 days 2 hours'))  # 1 days 02:00:00
print(pd.Timedelta('-1 days 2 min 3us'))  # -2 days +23:57:59.999997

'''用关键字参数指定时间：'''
print(pd.Timedelta(days=5, seconds=10))  # 5 days 00:00:10
print(pd.Timedelta(minutes=3, seconds=2))  # 0 days 00:03:02

print('# 可以将指定分钟转换为天和小时')
print(pd.Timedelta(minutes=3242))  # 2 days 06:02:00

print('使用带周期量的偏移量别名：')
# 一天
print(pd.Timedelta('1D'))  # 1 days 00:00:00
# 两周
print(pd.Timedelta('2W'))  # 14 days 00:00:00
# 一天零2小时3分钟4秒
# print(pd.Timedelta('1D 2H 3M 4S')) # 报错！
print(pd.Timedelta('1D 2H 3m 4S'))  # 1 days 02:03:04
print(pd.Timedelta('1D2H3m4S'))  # 同上

print('带单位的整型数字：')
# 一天
print(pd.Timedelta(1, unit='d'))  # 1 days 00:00:00
# 100秒
print(pd.Timedelta(100, unit='s'))  # 0 days 00:01:40
# 4周
print(pd.Timedelta(4, unit='w'))  # 28 days 00:00:00

'''使用Python内置的datetime.timedelta或者NumPy的np.timedelta64：'''

import datetime
import numpy as np

print('# 一天零10分钟')
print(datetime.timedelta(days=1, minutes=10))  # 1 day, 0:10:00
print(pd.Timedelta(datetime.timedelta(days=1, minutes=10)))  # 1 days 00:10:00
# 100纳秒
print(pd.Timedelta(np.timedelta64(100, 'ns')))  # 0 days 00:00:00.000000100

print('负值：')
print(pd.Timedelta('-1min'))  # -1 days +23:59:00
print('# 空值，缺失值')
print(pd.Timedelta('nan'))  # NaT
print(pd.Timedelta('nat'))  # NaT

print('# ISO 8601 Duration strings')
# 标准字符串（ISO 8601 Duration strings）：
print(pd.Timedelta('P0DT0H1M0S'))  # 0 days 00:01:00
print(pd.Timedelta('P0DT0H0M0.000000123S'))  # 0 days 00:00:00.000000123
'''
使用时间偏移对象DateOffsets (Day, Hour, Minute, Second, Milli,Micro, Nano)直接创建：
'''
print('# 两分钟')
print(pd.Timedelta(pd.offsets.Minute(2)))  # 0 days 00:02:00
print(pd.Timedelta(pd.offsets.Day(3)))  # 3 days 00:00:00

'''
另外，还有一个pd.to_timedelta()可以完成以上操作，
不过根据语义，它会用在时长类型的数据转换上。
'''
print()
print(pd.to_timedelta(pd.offsets.Day(3)))  # 3 days 00:00:00
print(pd.to_timedelta('15.5min'))  # 0 days 00:15:30
print(pd.to_timedelta(124524564574835))  # 1 days 10:35:24.564574835

'''如时间戳数据一样，时长数据的存储也有上下限：'''
print(pd.Timedelta.min)  # -106752 days +00:12:43.145224193
print(pd.Timedelta.max)  # 106751 days 23:47:16.854775807

'''如果想处理更大的时长数据，可以将其转换为一定单位的数字类型。'''

print()
print('------------------------------------------------------------')
print('\t14.2.2 时长的加减')

'''时长可以相加，多个时长累积为一个更长的时长：'''
print('# 一天与5个小时相加')
print(pd.Timedelta(pd.offsets.Day(1)) + pd.Timedelta(pd.offsets.Hour(5)))
# 1 days 05:00:00
print('# 一天与5个小时相减')
print(pd.Timedelta(pd.offsets.Day(1)) - pd.Timedelta(pd.offsets.Hour(5)))
# 0 days 19:00:00

'''固定时间与时长相加或相减会得到一个新的固定时间：'''
print('# 11月11日减去一天')
print(pd.Timestamp('2024-11-11') - pd.Timedelta(pd.offsets.Day(1)))
# 2024-11-10 00:00:00
print('# # 11月11日加3周')
print(pd.Timestamp('2024-11-11') + pd.Timedelta('3W'))  # 2024-12-02 00:00:00

'''不过，此类计算我们使用时间偏移来操作，后面会介绍。'''

print()
print('------------------------------------------------------------')
print('\t14.2.3 时长的属性')

'''
时长数据中我们可以解析出指定时间计数单位的值，
比如小时、秒等，这对我们进行数据计算非常有用。
'''

tdt = pd.Timedelta('10 days 9 min 3 sec')
print(tdt)  # 10 days 00:09:03
print(tdt.days)  # 10
print(tdt.seconds)  # 543
print((-tdt).days)  # -11
print(tdt.value)  # 864543000000000

'''
14.2.4 时长索引
时长数据可以作为索引（TimedeltaIndex），它使用的场景比较少，
例如在一项体育运动中，分别有2分钟完成、4分钟完成、5分钟完成三类。
时长数据可能是完成人数、平均身高等。

14.2.5 小结
时长是两个具体时间的差值，是一个绝对的时间数值，没有开始和结束时间。
时长数据使用场景较少，但是它是我们在后面理解时间偏移和周期时间的基础。
'''

print()
print('------------------------------------------------------------')
print('\t14.3.1 时序索引')

'''
在时间序列数据中，索引经常是时间类型，我们在操作数据时经常
会与时间类型索引打交道，本节将介绍如何查询和操作时间类型索引。

DatetimeIndex是时间索引对象，一般由to_datetime()或date_range()来创建：

'''

import datetime

day_code = pd.to_datetime(['11/1/2020',  # 类时间字符串
                           np.datetime64('2020-11-02'),  # NumPy的时间类型
                           datetime.datetime(2020, 11, 3)  # Python自带时间类型
                           ])

print(day_code)
# DatetimeIndex(['2020-11-01', '2020-11-02', '2020-11-03'], dtype='datetime64[ns]', freq=None)

'''
date_range()可以给定开始或者结束时间，并给定周期数据、周期频率，
会自动生成在此范围内的时间索引数据：
'''
print('date_range()')
# 默认频率为天
print(pd.date_range('2020-01-01', periods=10))
# DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
#                '2020-01-05', '2020-01-06', '2020-01-07', '2020-01-08',
#                '2020-01-09', '2020-01-10'],
#               dtype='datetime64[ns]', freq='D')
print(pd.date_range('2020-01-01', '2020-01-10'))  # 结果同上
print(pd.date_range(end='2020-01-10', periods=10))  # 结果同上

'''pd.bdate_range()生成数据可以跳过周六日，实现工作日的时间索引序列：'''
print('# 频率为工作日')
print(pd.bdate_range('2024-06-01', periods=10))
# DatetimeIndex(['2024-06-03', '2024-06-04', '2024-06-05', '2024-06-06',
#                '2024-06-07', '2024-06-10', '2024-06-11', '2024-06-12',
#                '2024-06-13', '2024-06-14'],
#               dtype='datetime64[ns]', freq='B')
'''个人ps：无法跳过端午节 || 好像有参数可以操作！ '''

print()
print('------------------------------------------------------------')
print('\t14.3.2 创建时序数据')

'''
创建包含时序的Series和DataFrame与创建普通的Series和DataFrame一样，
将时序索引序列作为索引或者将时间列转换为时间类型。
'''

print('# 生成时序索引')
tidx = pd.date_range('2024-01-01', periods=5)
# 应用时序索引
s = pd.Series(range(len(tidx)), index=tidx)
print(s)
# 2024-01-01    0
# 2024-01-02    1
# 2024-01-03    2
# 2024-01-04    3
# 2024-01-05    4
# Freq: D, dtype: int64

'''如果将其作为Series的内容，我们会看到序列的数据类型为datetime64[ns]：'''
print(pd.Series(tidx))
# 0   2024-01-01
# 1   2024-01-02
# 2   2024-01-03
# 3   2024-01-04
# 4   2024-01-05
# dtype: datetime64[ns]

print('创建DataFrame：')
# 索引
tidx = pd.date_range('2024-1-1', periods=5)
# print(tidx)
df = pd.DataFrame({'A': range(len(tidx)), 'B': range(len(tidx))[::-1]}, index=tidx)
print(df)
#             A  B
# 2024-01-01  0  4
# 2024-01-02  1  3
# 2024-01-03  2  2
# 2024-01-04  3  1
# 2024-01-05  4  0

df1 = pd.DataFrame({'A': list(range(len(tidx))), 'B': list(range(len(tidx)))[::-1]}, index=tidx)
print(df1)  # 结果同上

print()
print('------------------------------------------------------------')
print('\t14.3.3 数据访问')

'''首先创建时序索引数据。以下数据包含2020年和2021年，以小时为频率：'''

idx = pd.date_range('1/1/2020', '12/1/2021', freq='H')
ts = pd.Series(np.random.randn(len(idx)), index=idx)
print(ts)
# 2020-01-01 00:00:00   -0.732463
# 2020-01-01 01:00:00   -1.235969
# 2020-01-01 02:00:00   -1.571011
# 2020-01-01 03:00:00    0.479507
# 2020-01-01 04:00:00    1.058393
#                          ...
# 2021-11-30 20:00:00    1.729897
# 2021-11-30 21:00:00   -0.211982
# 2021-11-30 22:00:00   -0.439532
# 2021-11-30 23:00:00   -0.130613
# 2021-12-01 00:00:00    0.541320
# Freq: H, Length: 16801, dtype: float64

'''查询访问数据时，和 []、loc等的用法一样，可以按切片的操作对数据进行访问，如：'''
print('# 指定区间的')
print(ts[5:10])
# 指定区间的
# 2020-01-01 05:00:00    0.340714
# 2020-01-01 06:00:00    1.132463
# 2020-01-01 07:00:00   -1.769089
# 2020-01-01 08:00:00   -1.752932
# 2020-01-01 09:00:00    1.112932
# Freq: H, dtype: float64
print('# 只筛选2020年的')
print(ts['2020'])
# 2020-01-01 00:00:00   -0.053891
# 2020-01-01 01:00:00    0.347673
#                          ...
# 2020-12-31 22:00:00   -0.548344
# 2020-12-31 23:00:00    1.526979
# Freq: H, Length: 8784, dtype: float64

'''还支持传入时间字符和各种时间对象：'''
print('# 指定天，结果相同')
print(ts['11/30/2020'])
# 2020-11-30 00:00:00   -0.507145
# 2020-11-30 01:00:00   -0.638819
# ...
# 2020-11-30 22:00:00   -1.498252
# 2020-11-30 23:00:00   -0.475406
# Freq: H, dtype: float64
print(ts['2020-11-30'])  # 结果同上
print(ts['20201130'])  # 结果同上

print('# 指定时间点')
print(ts[datetime.datetime(2020, 11, 30)])  # 0.22997129354557833
# print(datetime.datetime(2020,11,30)) # 2020-11-30 00:00:00
print(ts[pd.Timestamp(2020, 11, 30)])  # 结果同上
print(ts[pd.Timestamp('2020-11-30')])  # 结果同上
print(ts[np.datetime64('2020-11-30')])  # 结果同上

'''也可以使用部分字符查询一定范围内的数据：'''
print('部分字符串查询')
print(ts['2021'])  # 查询整个2021年的
print(ts['2021-06'])  # 查询2021年6月的
print(ts['2021-6'])  # 查询2021年6月的
print(ts['2021-6':'2021-10'])  # 查询2021年6月到10月的
print(ts['2021-1':'2021-2-28 00:00:00'])  # 精确时间
print(ts['2020-01-15':'2020-01-15 12:30:00'])
print(ts.loc['2020-01-15'])

'''如果想知道序列的粒度，即频率，可以使用ts.resolution查看（以上数据的粒度为小时）：'''
print()
print('# 时间粒度（频率）')
print(ts.index.resolution)  # hour

'''df.truncate()作为一个专门对索引的截取工具，可以很好地应用在时序索引上：'''
print('# 给定开始时间和结束时间来截取部分时间')
print(ts.truncate(before='2020-11-10 11:20', after='2020-12'))
# 2020-11-10 12:00:00   -0.968447
# 2020-11-10 13:00:00    0.249492
# ...
# 2020-11-30 23:00:00    1.833575
# 2020-12-01 00:00:00    0.051057
# Freq: H, Length: 493, dtype: float64

print()
print('------------------------------------------------------------')
print('\t14.3.4 类型转换')

'''
由于时间格式样式比较多，很多情况下Pandas并不能自动将时序数据识别为时间类型，
所以我们在处理前的数据清洗过程中，需要专门对数据进行时间类型转换。

astype是最简单的时间转换方法，它只能针对相对标准的时间格式，如以下数据的数据类型是object：
'''

s = pd.Series(['2020-11-01 01:10', '2020-11-11 11:10', '2020-11-30 20:10'])
print(s)
# 0    2020-11-01 01:10
# 1    2020-11-11 11:10
# 2    2020-11-30 20:10
# dtype: object

print('# 转为时间类型')
print(s.astype('datetime64[ns]'))
# 0   2020-11-01 01:10:00
# 1   2020-11-11 11:10:00
# 2   2020-11-30 20:10:00
# dtype: datetime64[ns]
print('修改频率：')
# 转为时间类型，指定频率为天
# print(s.astype('datetime64[D]')) # 报错！
# print(s.astype('datetime64[ns]').astype('datetime[D]')) # error
print(pd.to_datetime(s))  # 同print(s.astype('datetime64[ns]'))
# print(pd.to_datetime(s).astype('datetime[D]')) # error
# TypeError: data type 'datetime[D]' not understood
print(s.astype('datetime64[ns]').dt.floor('D'))  # 去除日期时间部分
# 0   2020-11-01
# 1   2020-11-11
# 2   2020-11-30
# dtype: datetime64[ns]
print('# 转为时间类型，指定时区为北京时间')
# print(pd.to_datetime(s))
# print(pd.to_datetime(s).astype('datetime64[ns,Asia/Shanghai]')) # 书中示例error
print(pd.to_datetime(s).dt)  # <pandas.core.indexes.accessors.DatetimeProperties object at 0x00000139C76A46A0>
print(pd.to_datetime(s).dt.tz_localize('Asia/Shanghai'))
# 0   2020-11-01 01:10:00+08:00
# 1   2020-11-11 11:10:00+08:00
# 2   2020-11-30 20:10:00+08:00
# dtype: datetime64[ns, Asia/Shanghai]

'''pd.to_datetime()也可以转换时间类型：'''
print('# 转为时间类型')
print(pd.to_datetime(s))
# 0   2020-11-01 01:10:00
# 1   2020-11-11 11:10:00
# 2   2020-11-30 20:10:00
# dtype: datetime64[ns]

'''pd.to_datetime()还可以将多列组合成一个时间进行转换：'''
df = pd.DataFrame({'year': [2020, 2020, 2020],
                   'month': [10, 11, 12],
                   'day': [10, 11, 12]
                   })

print(df)
print('# 转为时间类型')
print(pd.to_datetime(df))
# 0   2020-10-10
# 1   2020-11-11
# 2   2020-12-12
# dtype: datetime64[ns]

'''对于Series，pd.to_datetime()会智能识别其时间格式并进行转换：'''

s = pd.Series(['2020-11-01 01:10', '2020-11-11 11:10', None])
print(pd.to_datetime(s))
# 0   2020-11-01 01:10:00
# 1   2020-11-11 11:10:00
# 2                   NaT
# dtype: datetime64[ns]

'''对于列表，pd.to_datetime()也会智能识别其时间格式并转换为时间序列索引：'''
print()
# print(pd.to_datetime(['2020/11/11','2020.12.12'])) # 报错
print(pd.to_datetime(['2020/11/11', '2020.12.12'], dayfirst=True, errors='coerce'))
# DatetimeIndex(['2020-11-11', 'NaT'], dtype='datetime64[ns]', freq=None)
# print(pd.to_datetime(['2020/11/11','2020.12.12'],format='%Y-%m-%d',dayfirst=True)) # 报错

print(pd.to_datetime(['2020/11/11'], format='%Y/%m/%d', dayfirst=True))  # ,'2020.12.12'
# DatetimeIndex(['2020-11-11'], dtype='datetime64[ns]', freq=None)
print(pd.to_datetime(['2020.12.12'], format='%Y.%m.%d', dayfirst=True))  # ,'2020.12.12'
# DatetimeIndex(['2020-12-12'], dtype='datetime64[ns]', freq=None)
print(pd.to_datetime(['2020.12.12', '2020_11_11'], format='%Y.%m.%d', errors='coerce'))
# DatetimeIndex(['2020-12-12', 'NaT'], dtype='datetime64[ns]', freq=None)

'''用pd.DatetimeIndex直接转为时间序列索引：'''
print('# 转为时间序列索引，自动推断频率')
print(pd.DatetimeIndex(['20201101', '20201102', '20201103'], freq='infer'))
# DatetimeIndex(['2020-11-01', '2020-11-02', '2020-11-03'], dtype='datetime64[ns]', freq='D')

print('''针对单个时间，用pd.Timestamp()转换为时间格式：''')
print(pd.to_datetime('2020/11/12'))  # 2020-11-12 00:00:00
print(pd.Timestamp('2020/11/12'))  # 同上！

print()
print('------------------------------------------------------------')
print('\t14.3.5 按格式转换')

'''
如果原数据的格式为不规范的时间格式数据，可以通过格式映射来将其转为时间数据：
'''
print('# 不规则格式转换时间')
print(pd.to_datetime('2020_11_11', format='%Y_%m_%d'))  # 2020-11-11 00:00:00

print('# 可以让系统自己推断时间格式')
# print(pd.to_datetime('20200101', infer_datetime_format=True, errors='ignore'))
# 2020-01-01 00:00:00
'''多个时间格式，无法推断'''
'''提示该参数已弃用！ 默认 infer_datetime_format=True'''
# print(pd.to_datetime(['2020.12.12','2020_11_11','20200101'], infer_datetime_format=True, errors='coerce'))
# DatetimeIndex(['2020-12-12', 'NaT', 'NaT'], dtype='datetime64[ns]', freq=None)
print('''# 将errors参数设置为coerce，将不会忽略错误，返回空值''')
print(pd.to_datetime('20200101', format='%Y%m%d', errors='coerce'))
# 2020-01-01 00:00:00
print('fasdfas')
# print(pd.to_datetime(df.day.astype(str),format='%m/%d/%Y')) # 报错！

print('其它')
print(pd.to_datetime('01-10-2020 00:00', format='%d-%m-%Y %H:%M'))  # 2020-10-01 00:00:00
print('# 对时间戳进行转换，需要给出时间单位，一般为秒')
print(pd.to_datetime(1490195805, unit='s'))  # 2017-03-22 15:16:45
print(pd.to_datetime(1490195805433502912, unit='ns'))  # 2017-03-22 15:16:45.433502912

print('可以将数字列表转换为时间：')
print(pd.to_datetime([10, 11, 12, 15], unit='D', origin=pd.Timestamp('2020-11-01')))
# DatetimeIndex(['2020-11-11', '2020-11-12', '2020-11-13', '2020-11-16'], dtype='datetime64[ns]', freq=None)
'''个人测试'''
s = pd.Series([10, 11, 12, 15])
print(pd.to_datetime(s, unit='D', origin=pd.Timestamp('2024-01-01')))
# 0   2024-01-11
# 1   2024-01-12
# 2   2024-01-13
# 3   2024-01-16
# dtype: datetime64[ns]

print()
print('------------------------------------------------------------')
print('\t14.3.6 时间访问器.dt')
'''
之前介绍过了文本访问器（.str）和分类访问器（.cat），
对时间Pandas也提供了一个时间访问器.dt.<method>，
用它可以以time.dt.xxx的形式来访问时间序列数据的属性和调用它们的方法，返回对应值的序列。
'''

s = pd.Series(pd.date_range('2020-11-01', periods=5, freq='d'))
# print(s)
print('# 各天是星期几')
print(s.dt.day_name())
# 0       Sunday
# 1       Monday
# 2      Tuesday
# 3    Wednesday
# 4     Thursday
# dtype: object
print('# 时间访问器操作')
print(s.dt.date)
# 0    2020-11-01
# 1    2020-11-02
# 2    2020-11-03
# 3    2020-11-04
# 4    2020-11-05
# dtype: object
print(s.dt.time)
print(s.dt.timetz)  # 同上
# 0    00:00:00
# 1    00:00:00
# 2    00:00:00
# 3    00:00:00
# 4    00:00:00
# dtype: object

print('# 以下为时间各成分的值')
print(s.dt.year)  # 年
print(s.dt.month)  # 月份
print(s.dt.day)  # 天
print(s.dt.hour)  # 小时
print(s.dt.minute)  # 分钟
print(s.dt.second)  # 秒
print(s.dt.microsecond)
print(s.dt.nanosecond)

print('# 以下为与周、月、年相关的属性')

s = pd.Series(pd.date_range('2024-06-01', periods=3, freq='d'))
print(s)
# print(s.dt.week) # 警告已作废
print(s.dt.isocalendar().week)  # 年内第几周
print(s.dt.dayofweek)  # 星期一属于0
print(s.dt.weekday)  # 同上
print(s.dt.dayofyear)  # 一年中的第几天 元旦是第一天
print(s.dt.quarter)  # 季度
print(s.dt.is_month_start)  # 是否月初第一天
print(s.dt.is_month_end)  # 是否月初最后一天
print(s.dt.is_year_start)  # 是否年初第一天
print(s.dt.is_year_end)  # 是否年初最后一天
print(s.dt.is_leap_year)  # 是否闰年
print(s.dt.daysinmonth)  # 当月天数
print(s.dt.days_in_month)  # 同上
print(s.dt.tz)  # None
print(s.dt.freq)  # D
print()
print('# 以下为转换方法')
print(s.dt.to_period)
# <bound method PandasDelegate._add_delegate_accessors.<locals>._create_delegator_method.<locals>.f of <pandas.core.indexes.accessors.DatetimeProperties object at 0x00000207DC173A30>>
print(s.dt.to_pydatetime)
# <bound method DatetimeProperties.to_pydatetime of <pandas.core.indexes.accessors.DatetimeProperties object at 0x0000019B9C511A30>>
print(s.dt.to_pydatetime())
# [datetime.datetime(2024, 1, 1, 0, 0) datetime.datetime(2024, 1, 2, 0, 0)
#  datetime.datetime(2024, 1, 3, 0, 0) datetime.datetime(2024, 1, 4, 0, 0)
#  datetime.datetime(2024, 1, 5, 0, 0) datetime.datetime(2024, 1, 6, 0, 0)
#  datetime.datetime(2024, 1, 7, 0, 0) datetime.datetime(2024, 1, 8, 0, 0)]
print(s.dt.tz_convert)
print(s.dt.strftime)
print(s.dt.normalize)
print(s.dt.tz_localize)

print()
print(s.dt.round(freq='D'))  # 类似四舍五入 返回年月日
print(s.dt.floor(freq='D'))  # 向下舍入为天
print(s.dt.ceil(freq='D'))  # 向上舍入为天

print(s.dt.month_name())  # 月份名称
print(s.dt.day_name())  # 星期几的名称
print(s.dt.day)  # 当月第几天

print()
print('dfsdf')
s = pd.Series(pd.date_range('2024-01-30', periods=3, freq='d'))
# print(s.dt.start_time) # 报错警告
# print(s.dt.end_time()) # 报错警告
# print(s.dt.total_seconds) # 报错警告

print('# 个别用法举例')
# 将时间转为UTC时间，再转为美国东部时间
print(s.dt.tz_localize('UTC').dt.tz_convert('US/Eastern'))
# 0   2024-01-29 19:00:00-05:00
# 1   2024-01-30 19:00:00-05:00
# 2   2024-01-31 19:00:00-05:00
# dtype: datetime64[ns, US/Eastern]
# 输出时间显示格式
print(s.dt.strftime('%Y/%m/%d'))
# 0    2024/01/30
# 1    2024/01/31
# 2    2024/02/01
# dtype: object

print()
print('------------------------------------------------------------')
print('\t14.3.7 时长数据访问器')

'''
时长数据也支持访问器，可以解析出时长的相关属性，最终产出一个结果序列：
'''
# print(np.arange(5)) # [0 1 2 3 4]
ts = pd.Series(pd.to_timedelta(np.arange(5), unit='hour'))
print(ts)
# 0   0 days 00:00:00
# 1   0 days 01:00:00
# 2   0 days 02:00:00
# 3   0 days 03:00:00
# 4   0 days 04:00:00
# dtype: timedelta64[ns]

print('# 计算秒数')
print(ts.dt.seconds)
# 0        0
# 1     3600
# 2     7200
# 3    10800
# 4    14400
# dtype: int32
print('# 转为Python时间格式')
print(ts.dt.to_pytimedelta())
# [datetime.timedelta(0) datetime.timedelta(seconds=3600)
#  datetime.timedelta(seconds=7200) datetime.timedelta(seconds=10800)
#  datetime.timedelta(seconds=14400)]

print()
print('------------------------------------------------------------')
print('\t14.3.8 时序数据移动')

rng = pd.date_range('2024-06-01', '2024-06-05')
ts = pd.Series(range(len(rng)), index=rng)
print(ts)
# 2024-06-01    0
# 2024-06-02    1
# 2024-06-03    2
# 2024-06-04    3
# 2024-06-05    4
# Freq: D, dtype: int64
print('# 向上移动一位')
print(ts.shift(-1))
# 2024-06-01    1.0
# 2024-06-02    2.0
# 2024-06-03    3.0
# 2024-06-04    4.0
# 2024-06-05    NaN
# Freq: D, dtype: float64

'''shift()方法接受freq频率参数，该参数可以接受DateOffset类或其他类似timedelta的对象，也可以接受偏移别名：'''
print('# 向上移动一个工作日，06-01~06-02是周六、日')
print(ts.shift(-1, freq='B'))
# 2024-05-31    0
# 2024-05-31    1
# 2024-05-31    2
# 2024-06-03    3
# 2024-06-04    4
# dtype: int64

print()
print('------------------------------------------------------------')
print('\t14.3.9 频率转换')

'''
更换时间频率是将时间序列由一个频率单位更换为另一个频率单位，
实现时间粒度的变化。更改频率的主要功能是asfreq()方法。
以下是一个频率为自然日的时间序列：
'''

rng = pd.date_range('2020-11-01', '2020-12-01')
ts = pd.Series(range(len(rng)), index=rng)
print(ts)
'''
2020-11-01     0
2020-11-02     1
...
2020-11-30    29
2020-12-01    30
Freq: D, dtype: int64
'''
# 我们将它的频率变更为更加细的粒度，会产生缺失值：
print('# 频率转为12小时')
print(ts.asfreq(pd.offsets.Hour(12)))
# 2020-11-01 00:00:00     0.0
# 2020-11-01 12:00:00     NaN
# 2020-11-02 00:00:00     1.0
# 2020-11-02 12:00:00     NaN
#                        ...
# 2020-11-30 00:00:00    29.0
# 2020-11-30 12:00:00     NaN
# 2020-12-01 00:00:00    30.0
# Freq: 12H, Length: 61, dtype: float64

'''对于缺失值可以用指定值或者指定方法进行填充：'''
print('# 对缺失值进行填充')
print(ts.asfreq(freq='12h', fill_value=0))
# 2020-11-01 00:00:00     0
# 2020-11-01 12:00:00     0
# 2020-11-02 00:00:00     1
# 2020-11-02 12:00:00     0
# 2020-11-03 00:00:00     2
#                        ..
# 2020-11-29 00:00:00    28
# 2020-11-29 12:00:00     0
# 2020-11-30 00:00:00    29
# 2020-11-30 12:00:00     0
# 2020-12-01 00:00:00    30
# Freq: 12H, Length: 61, dtype: int64

print('# 对产生的缺失值使用指定方法填充')
print(ts.asfreq(pd.offsets.Hour(12), method='pad'))
# 2020-11-01 00:00:00     0
# 2020-11-01 12:00:00     0
# 2020-11-02 00:00:00     1
# 2020-11-02 12:00:00     1
# 2020-11-03 00:00:00     2
#                        ..
# 2020-11-29 00:00:00    28
# 2020-11-29 12:00:00    28
# 2020-11-30 00:00:00    29
# 2020-11-30 12:00:00    29
# 2020-12-01 00:00:00    30
# Freq: 12H, Length: 61, dtype: int64

'''
14.3.10 小结
时序数据由若干个固定时间组成，这些固定时间的分布大多呈现出一定的周期性。
时序数据经常作为索引，它也有可能在数据列中。
在数据分析业务实践中大量用到时序数据，因此本节内容是数序数据分析的关键内容，也是操作频率最高的内容。

'''

print()
print('------------------------------------------------------------')
print('第14章 Pandas时序数据')
print('\t14.4 时间偏移')
# update20240607
'''

'''
# 小节注释
'''
DateOffset类似于时长Timedelta，但它使用日历中时间日期的规则，
而不是直接进行时间性质的算术计算，让时间更符合实际生活。

▶ 以下是一些具体的使用方法举例：
'''
print()
print('------------------------------------------------------------')
print('\t14.4.1 DateOffset对象')

'''
我们通过夏令时来理解DateOffset对象。有些地区使用夏令时，每
日偏移时间有可能是23或24小时，甚至25个小时。

'''
print('# 生成一个指定的时间，芬兰赫尔辛基时间执行夏令时')
t = pd.Timestamp('2016-10-30 00:00:00', tz='Europe/Helsinki')
print(t)
# 2016-10-30 00:00:00+03:00

print(t + pd.Timedelta(days=1))  # 增加一个自然天
# 2016-10-30 23:00:00+02:00
print(t + pd.DateOffset(days=1))  # 增加一个时间偏移天
# 2016-10-31 00:00:00+02:00

print('工作日')
# 定义一个日期
d = pd.Timestamp('2024-06-07')
print(d.day_name())  # Friday
print('# 定义2个工作日时间偏移变量')
two_business_days = 2 * pd.offsets.BDay()
# print(two_business_days) # <2 * BusinessDays>
print('# 增加两个工作日')
# print(two_business_days.apply(d)) # 报错！
print(two_business_days + d)  # 2024-06-11 00:00:00

print('# 取增加两个工作日后的星期')
print((d + two_business_days).day_name())  # 'Tuesday'

print()
print(d + pd.Timedelta(days=3))  # 增加一个自然天
# 2024-06-10 00:00:00
print(d + pd.DateOffset(days=3))  # 增加一个时间偏移天
# 2024-06-10 00:00:00

'''
我们发现，与时长Timedelta不同，时间偏移DateOffset不是数学意义上的增加或减少，
而是根据实际生活的日历对现有时间进行偏移。
时长可以独立存在，作为业务的一个数据指标，而时间偏移DateOffset的意义是找到一个时间起点并对它进行时间移动。
所有的日期偏移对象都在pandas.tseries.offsets下，
其中pandas.tseries.offsets.DateOffset是标准的日期范围时间偏移类型，它默认是一个日历日。

from pandas.tseries.offsets import DateOffset
ts = pd.Timestamp('2020-01-01 09:10:11')
ts + DateOffset(months=3)
# Timestamp('2020-04-01 09:10:11')
ts + DateOffset(hours=2)
# Timestamp('2020-01-01 11:10:11')
ts + DateOffset()
# Timestamp('2020-01-02 09:10:11')
'''

print()
print('------------------------------------------------------------')
print('\t14.4.2 偏移别名')
'''
DateOffset基本都支持频率字符串或偏移别名，传入freq参数，时间
偏移的子类、子对象都支持时间偏移的相关操作。有效的日期偏移及频
率字符串见表14-1。

很多 请见该章节！
'''

add_week = 1 * pd.offsets.Week()
print(pd.Timestamp('now') + add_week)  # 增加一周
# 2024-06-14 17:07:11.829099

print()
print('------------------------------------------------------------')
print('\t14.4.3 移动偏移')

'''Offset通过计算支持向前或向后偏移：'''

ts = pd.Timestamp('2020-06-06 00:00:00')
print(ts.day_name())  # Saturday

# 定义一个工作小时偏移，默认是周一到周五9～17点，我们从10点开始
offset = pd.offsets.BusinessHour(start='10:00')
# print(offset) # <BusinessHour: BH=10:00-17:00>
print('# 向前偏移一个工作小时，是一个周一，跳过了周日')
print(offset.rollforward(ts))  # 2020-06-08 10:00:00

print()
print('# 向前偏移至最近的工作日，小时也会增加')
print(ts + offset)  # 2020-06-08 11:00:00
print('# 向后偏移，会在周五下班前的一个小时')
print(offset.rollback(ts))  # 2020-06-05 17:00:00

print(ts - pd.offsets.Day(1))  # 昨日
# 2020-06-05 00:00:00
print(ts - pd.offsets.Day(2))  # 前日
# 2020-06-04 00:00:00
print(ts - pd.offsets.Week(weekday=0))  # # 2020-06-01 00:00:00 本周一
print(ts - pd.offsets.Week(weekday=0) - pd.offsets.Day(14))  # 2020-05-18 00:00:00
print((ts - pd.offsets.Week(weekday=0) - pd.offsets.Day(14)).day_name())  # Monday 上上周一

print()
print(ts - pd.offsets.MonthEnd())  # 2020-05-31 00:00:00
print(ts - pd.offsets.MonthBegin())  # 2020-06-01 00:00:00
print(ts - pd.offsets.MonthEnd() - pd.offsets.MonthBegin())  # 2020-05-01 00:00:00  # 上月一日
'''
时间偏移操作会保留小时和分钟，有时候我们不在意具体的时间，
可以使用normalize()进行标准化到午夜0点：
'''
print(offset.rollback(ts).normalize())  # 2020-06-05 00:00:00
print(pd.Timestamp('now').normalize())  # 2024-06-07 00:00:00

print()
print('------------------------------------------------------------')
print('\t14.4.4 应用偏移')
'''update20240611 已作废！ apply可以使偏移对象应用到一个时间上'''

ts = pd.Timestamp('2020-06-01 09:00')
# print(ts)
day = pd.offsets.Day()  # 定义偏移对象
# print(day.apply(ts)) # 将偏移对象应用到时间上
print(ts + day)  # 2020-06-02 09:00:00
print((ts + day).normalize())  # 2020-06-02 00:00:00

ts = pd.Timestamp('2020-06-01 22:00')
hour = pd.offsets.Hour()
print(ts + hour)  # 2020-06-01 23:00:00
print((ts + hour).normalize())  # 2020-06-01 00:00:00

print()
print('------------------------------------------------------------')
print('\t14.4.5 偏移参数')

'''
之前我们只偏移了偏移对象的一个单位，可以传入参数来偏移多个单位和对象中的其他单位：
'''

import datetime

d = datetime.datetime(2020, 6, 1, 9, 0)
# d = datetime.datetime(2024,6,6,9,0)
print(d)  # 2020-06-01 09:00:00

print(d + pd.offsets.Week())  # # 偏移一周
# 2020-06-08 09:00:00
print(d + pd.offsets.Week(weekday=4))  # # 偏移4周中的日期
# 2020-06-05 09:00:00 ??? 看不懂 逻辑如下注释中 看懂了

'''
pd.offsets.Week(weekday=4)：偏移对象的 weekday 参数指定目标日期为周五。(周一是0，所以周五是4)
偏移逻辑：从初始日期 d 开始，找到最近的周五并进行偏移。
结果：从2020年6月1日（星期一）偏移到2020年6月5日（星期五）。
'''

print('# 取一周第几天')
print((d + pd.offsets.Week(weekday=4)).weekday())  # 4
print(d - pd.offsets.Week())  # 向后一周

print('参数也支持标准化normalize：')
print(d + pd.offsets.Week(normalize=True))  # 2020-06-08 00:00:00
print(d - pd.offsets.Week(normalize=True))  # 2020-05-25 00:00:00

print('YearEnd支持用参数month指定月份：')
print(d + pd.offsets.YearEnd())  # 2020-12-31 09:00:00
print(d + pd.offsets.YearEnd(month=6))  # 2020-06-30 09:00:00

'''
不同的偏移对象支持不同的参数，可以通过代码编辑器的代码提示进行查询。
'''

print()
d = pd.Timestamp('2024-06-07')
print(d + pd.offsets.DateOffset(years=5))  # 2029-06-07 00:00:00 增加年份  +s 年月日时分秒类似
print(d + pd.offsets.DateOffset(year=5))  # 0005-06-07 00:00:00 # 替换年份 无s 年月日时分秒类似

print()
print('------------------------------------------------------------')
print('\t14.4.6 相关查询')

'''当使用日期作为索引的DataFrame时，此函数可以基于日期偏移量选择最后几行：'''

i = pd.date_range('2018-04-09', periods=4, freq='2D')
ts = pd.DataFrame({'A': [1, 2, 3, 4]}, index=i)
print(ts)
#             A
# 2018-04-09  1
# 2018-04-11  2
# 2018-04-13  3
# 2018-04-15  4
# 取最后3天，请注意，返回的是最近3天的数据
# 而不是数据集中最近3天的数据，因此未返回2018-04-11的数据
print(ts.last('3D'))
#             A
# 2018-04-13  3
# 2018-04-15  4
print('# 前3天')
print(ts.first('3D'))
#             A
# 2018-04-09  1
# 2018-04-11  2
'''可以用at_time()来指定时间：'''
print('# 指定时间')

i = pd.date_range('2018-04-09', periods=4, freq='12H')
ts = pd.DataFrame({'A': [1, 2, 3, 4]}, index=i)

print(ts.at_time('12:00'))
#                      A
# 2018-04-09 12:00:00  2
# 2018-04-10 12:00:00  4
'''用between_time()来指定时间区间：'''
i = pd.date_range('2018-04-09', periods=4, freq='15T')
ts = pd.DataFrame({'A': [1, 2, 3, 4]}, index=i)
print(ts)
print(ts.between_time('0:15', '0:45'))
#                      A
# 2018-04-09 00:15:00  2
# 2018-04-09 00:30:00  3
# 2018-04-09 00:45:00  4
print()
print('------------------------------------------------------------')
print('\t14.4.7 与时序的计算')
'''
可以对Series或DatetimeIndex时间索引序列应用时间偏移，
与其他时间序列数据一样，时间偏移后的数据一般会作为索引。

举例类似上述内容 不写了
'''

print()
print('------------------------------------------------------------')
print('\t14.4.8 锚定偏移')

i = pd.date_range('2024-06-01', periods=14, freq='D')
# ts = pd.DataFrame({'A':pd.range(14)[::-1]},index=i)
ts = pd.DataFrame({'A': range(14)}, index=i)
# print(ts)
# 别名     说明
# W-SUN   周(星期日)，同"W"
print()
# 生成一个以星期日为结束日的周时间序列
date_range = pd.date_range('2024-06-01', periods=14, freq='W-SUN')  # W-MON ~ W-SAT
print(date_range)
# DatetimeIndex(['2024-06-02', '2024-06-09', '2024-06-16', '2024-06-23',
#                '2024-06-30', '2024-07-07', '2024-07-14', '2024-07-21',
#                '2024-07-28', '2024-08-04', '2024-08-11', '2024-08-18',
#                '2024-08-25', '2024-09-01'],
#               dtype='datetime64[ns]', freq='W-SUN')

# 创建一个数据框，以日期序列为索引
data = pd.DataFrame({'A': range(14)}, index=date_range)
print(data)

print()
'''
Q-JAN 季，结束于1月
...
Q-NOV 季，结束于11月
'''
date_range = pd.date_range('2024-06-01', periods=14, freq='Q')  # 季，结束于12月，同“Q
print(date_range)
# DatetimeIndex(['2024-06-30', '2024-09-30', '2024-12-31', '2025-03-31',
#                '2025-06-30', '2025-09-30', '2025-12-31', '2026-03-31',
#                '2026-06-30', '2026-09-30', '2026-12-31', '2027-03-31',
#                '2027-06-30', '2027-09-30'],
#               dtype='datetime64[ns]', freq='Q-DEC')
date_range = pd.date_range('2024-06-01', periods=14, freq='Q-NOV')  # 季，结束于11月
print(date_range)

date_range = pd.date_range('2024-06-01', periods=14, freq='A')  # 年，结束于12月
print(date_range)
# DatetimeIndex(['2024-12-31', '2025-12-31', '2026-12-31', '2027-12-31',
#                '2028-12-31', '2029-12-31', '2030-12-31', '2031-12-31',
#                '2032-12-31', '2033-12-31', '2034-12-31', '2035-12-31',
#                '2036-12-31', '2037-12-31'],
#               dtype='datetime64[ns]', freq='A-DEC')

print(pd.Timestamp('2020-01-02') - pd.offsets.MonthBegin(n=4))  # 2019-10-01 00:00:00

print()
print('------------------------------------------------------------')
print('\t14.4.9 自定义工作时间')
'''
由于不同地区不同文化，工作时间和休息时间不尽相同。
在数据分析时需要考虑工作日、周末等文化差异带来的影响，
比如，埃及的周末是星期五和星期六。

可以向Cday或CustomBusinessDay类传入节假日参数来自定义一个工作日偏移对象：
'''
import datetime

weekmask_egypt = 'Sun Mon Tue Wed Thu'

# 定义出五一劳动节的日期，因为放假
holidays = ['2018-05-01',
            datetime.datetime(2019, 5, 1),
            np.datetime64('2020-05-01')
            ]

# 自定义工作日中传入休假日期，一个正常星期工作日的顺序
bday_egypt = pd.offsets.CustomBusinessDay(holidays=holidays,
                                          weekmask=weekmask_egypt)

# 指定一个日期
dt = datetime.datetime(2020, 4, 30)
# 偏移两个工作日，跳过了休假日
print(dt + 2 * bday_egypt)  # 2020-05-04 00:00:00
print(dt + bday_egypt)  # 2020-05-03 00:00:00 || 2号、3号 是周六日

print()
print('# 输出时序及星期几')
idx = pd.date_range(dt, periods=5, freq=bday_egypt)
print(pd.Series(idx.weekday + 1, index=idx))
# 2020-04-30    4
# 2020-05-03    7
# 2020-05-04    1
# 2020-05-05    2
# 2020-05-06    3
# Freq: C, dtype: int32

'''
BusinessHour表是开始和结束工作的小时时间，默认的工作时间是9:00—17:00，
与时间相加超过一个小时会移到下一个小时，超过一天会移动到下一个工作日。

'''
print('BusinessHour')
bh = pd.offsets.BusinessHour()
print(bh)  # <BusinessHour: BH=09:00-17:00>
print('# 2020-08-01是周六')
print(pd.Timestamp('2020-08-01 10:00').weekday())  # 5
print('# 增加一个工作小时')
# 跳过周末
print(pd.Timestamp('2020-08-01 10:00') + bh)  # 2020-08-03 10:00:00
print(pd.Timestamp('2020-07-31 10:00') + bh)  # 2020-07-31 11:00:00

# 一旦计算就等于上班了，等同于pd.Timestamp('2020-08-01 09:00') + bh
print(pd.Timestamp('2020-08-01 08:00') + bh)  # 2020-08-03 10:00:00

print('# 计算后已经下班了，就移到下一个工作小时（跳过周末）')
print(pd.Timestamp('2020-07-31 16:00') + bh)  # 2020-08-03 09:00:00
print(pd.Timestamp('2020-08-01 16:30') + bh)  # 2020-08-03 10:00:00

print('# 偏移两个工作小时')
print(pd.Timestamp('2024-06-19') + pd.offsets.BusinessHour(2))  # 2024-06-19 11:00:00

print('# 减去3个工作小时')
print(pd.Timestamp('2024-06-19 10:00') + pd.offsets.BusinessHour(-3))  # 2024-06-18 15:00:00
print(pd.Timestamp('now') + pd.offsets.BusinessHour(-3))  # 2024-06-19 12:58:44.417465

print()
'''
可以自定义开始和结束工作的时间，格式必须是hour:minute字符串，不支持秒、微秒、纳秒。
'''
print('# 11点开始上班')
# print(datetime.time(20,0,0)) # 20:00:00
bh = pd.offsets.BusinessHour(start='11:00', end=datetime.time(20, 0))
# print(bh) # <BusinessHour: BH=11:00-20:00>

print(pd.Timestamp('2024-06-19 13:00') + bh)  # 2024-06-19 14:00:00
print(pd.Timestamp('2024-06-19 09:00') + bh)  # 2024-06-19 12:00:00
print(pd.Timestamp('2024-06-19 18:00') + bh)  # 2024-06-19 19:00:00

print()
'''start时间晚于end时间表示夜班工作时间。此时，工作时间将从午夜延至第二天。'''
bh = pd.offsets.BusinessHour(start='17:00', end='09:00')
print(bh)  # <BusinessHour: BH=17:00-09:00>

print(pd.Timestamp('2024-06-19 17:00') + bh)  # 2024-06-19 18:00:00
print(pd.Timestamp('2024-06-19 23:00') + bh)  # 2024-06-20 00:00:00

'''
# 尽管2024年6月15日是周六，但因为工作时间从周五开始，因此也有效
'''
print(pd.Timestamp('2024-06-15 07:00') + bh)  # 2024-06-15 08:00:00
print(pd.Timestamp('2024-06-15 08:00') + bh)  # 2024-06-17 17:00:00

'''# 尽管2024年6月17日是周一，但因为工作时间从周日开始，超出了工作时间'''
print(pd.Timestamp('2024-06-17 04:00') + bh)  # 2024-06-17 18:00:00

'''
14.4.10 小结
时间偏移与时长的根本不同是它是真实的日历上的时间移动，在数据分析中时间偏移的意义是大于时长的。
另外，通过继承pandas.tseries.holiday.AbstractHolidayCalendar创建子类，
可以自定义假期日历，完成更为复杂的时间偏移操作，可浏览Pandas官方文档了解。
'''

print()
print('------------------------------------------------------------')
print('第14章 Pandas时序数据')
print('\t14.5 时间段')
# update20240621
'''

'''
# 小节注释
'''
Pandas中的Period()对象表示一个时间段，比如一年、一个月或一个季度。
与时间长度不同，它表示一个具体的时间区间，有时间起点和周期频率。

▶ 以下是一些具体的使用方法举例：
'''
print()
print('------------------------------------------------------------')
print('\t14.5.1 Period对象')

'''
我们来利用pd.Period()创建时间段对象：
'''

print('# 创建一个时间段（年）')
print(pd.Period('2020'))  # 2020
print(type(pd.Period('2020')))  # <class 'pandas._libs.tslibs.period.Period'>

print('# 创建一个时间段（季度）')
print(pd.Period('2020Q4'))  # 2020Q4
print(type(pd.Period('2020Q4')))  # <class 'pandas._libs.tslibs.period.Period'>

print('# 2020-01-01全天的时间段')
print(pd.Period(year=2020, freq='D'))  # 2020-01-01

print('# 一周')
print(pd.Period('20240621', freq='W'))  # 2024-06-17/2024-06-23 || (周日为23)

print('# 默认周期，对应到最细粒度——分钟')
print(pd.Period('2020-11-11 23:00'))  # 2020-11-11 23:00
print(type(pd.Period('2020-11-11 23:00')))  # <class 'pandas._libs.tslibs.period.Period'>

print('# 指定周期')
print(pd.Period('2020-11-11 23:00', 'D'))  # 2020-11-11
print(pd.Period('2020-11-11 23:00', freq='D'))  # 同上

print()
print('------------------------------------------------------------')
print('\t14.5.2 属性方法')

'''
一个时间段有开始和结束时间，可以用如下方法获取：
'''

print('# 定义时间段')
p = pd.Period('2020Q4')
print('# 开始与结束时间')
print(p.start_time)  # 2020-10-01 00:00:00
print(p.end_time)  # 2020-12-31 23:59:59.999999999

'''如果当前时间段不符合业务实际，可以转换频率：'''
print('# 将频率转换为天')
print(p.asfreq('D'))  # 2020-12-31
print(p.asfreq('D', how='start'))  # 2020-10-01

'''其他的属性方法如下：'''
print(p.freq)  # <QuarterEnd: startingMonth=12> | （时间偏移对象）
print(p.freqstr)  # Q-DEC | （时间偏移别名）
print(p.is_leap_year)  # True（是否闰年）
print(p.to_timestamp())  # 2020-10-01 00:00:00

'''# 以下日期取时间段内最后一天'''
print()
print(p.day)  # 31
print(p.dayofweek)  # 3(周四)
print(p.dayofyear)  # 366
print(p.hour)  # 0
print(p.week)  # 53
print(p.minute)  # 0
print(p.second)  # 0
print(p.month)  # 12
print(p.quarter)  # 4
print(p.qyear)  # 2020 | (财年)
print(p.year)  # 2020
print(p.days_in_month)  # 31 （当月第几天）
print(p.daysinmonth)  # 31（当月共多少天）
'''
strptime()：用于将日期时间字符串解析为datetime对象，适用于处理用户输入、读取文件等场景。
strftime()：用于将datetime对象格式化为字符串，适用于格式化输出、生成文件名等场景。
'''
print(p.strftime('%Y年%m月'))  # 2020年12月
from datetime import datetime

user_input = "2024-05-28"
date_object = datetime.strptime(user_input, "%Y-%m-%d")
print(date_object)  # 输出：2024-05-28 00:00:00
user_two = '2024/06/21'
# print(datetime.strptime(user_two,'%Y-%m-%d')) # ValueError: time data '2024/06/21' does not match format '%Y-%m-%d'
print(datetime.strptime(user_two, '%Y/%m/%d'))  # 2024-06-21 00:00:00

print()
print('------------------------------------------------------------')
print('\t14.5.3 时间段的计算')
'''
时间段可以做加减法，表示将此时间段前移或后移相应的单位：
'''
print('# 在2020Q4上增加一个周期')
print(pd.Period('2020Q4') + 1)  # 2021Q1
print('# 在2020Q4上减少一个周期')
print(pd.Period('2020Q4') - 1)  # 2020Q3

'''当然，时间段对象也可以和时间偏移对象做加减：'''
print('# 增加一小时')
print(pd.Period('20200105 15'))  # 2020-01-05 15:00
# print(pd.Period('2020010515')) # ValueError: year 2020010515 is out of range
print(pd.Period('20200105 15') + pd.offsets.Hour())  # 2020-01-05 16:00
print(pd.Period('20200105 15') + pd.offsets.Hour(1))  # 同上

print('# 增加10天')
print(pd.Period('20200101') + pd.offsets.Day(10))  # 2020-01-11

'''
如果偏移量频率与时间段不同，则其单位要大于时间段的频率，否则会报错：
'''
print()
print(pd.Period('20200101 14') + pd.offsets.Day(10))  # 2020-01-11 14:00
# print(pd.Period('20200101 14') + pd.offsets.Minute(10)) # ValueError: Cannot losslessly convert units
print(pd.Period('20200101 1432') + pd.offsets.Minute(10))  # 2020-01-01 14:42
print(pd.Period('2020 10'))  # 2020-10
# print(pd.Period('202010')) # pandas._libs.tslibs.parsing.DateParseError: month must be in 1..12: 202010
print(pd.Period('2020 10') + pd.offsets.MonthEnd(3))  # 2021-01
# print(pd.Period('2020 10') + pd.offsets.MonthBegin()) error
print(pd.Period('2020 10') - pd.offsets.MonthEnd(3))  # 2020-07

print('时间段也可以和时间差相加减：')
print(pd.Period('20200101 14') + pd.Timedelta('1 days'))  # 2020-01-02 14:00
# print(pd.Period('20200101 14') + pd.Timedelta('1 seconds')) # error
# pandas._libs.tslibs.period.IncompatibleFrequency: Input cannot be converted to Period(freq=H)

'''相同频率的时间段实例之差将返回它们之间的频率单位数'''
print()
print(pd.Period('20200101 14') - pd.Period('20200101 10'))  # <4 * Hours>
print(pd.Period('2020Q4') - pd.Period('2020Q1'))  # <3 * QuarterEnds: startingMonth=12>

print()
print('------------------------------------------------------------')
print('\t14.5.4 时间段索引')

'''
类似于时间范围pd.date_range()生成时序索引数据，
pd.period_range()可以生成时间段索引数据：
'''
print('# 生成时间段索引对象')
print(pd.period_range('2020-11-01 10:00', periods=10, freq='H'))
# PeriodIndex(['2020-11-01 10:00', '2020-11-01 11:00', '2020-11-01 12:00',
#              '2020-11-01 13:00', '2020-11-01 14:00', '2020-11-01 15:00',
#              '2020-11-01 16:00', '2020-11-01 17:00', '2020-11-01 18:00',
#              '2020-11-01 19:00'],
#             dtype='period[H]')

print('# 指定开始和结束时间')
print(pd.period_range('2020Q1', '2021Q4', freq='Q-NOV'))
'''
上例定义了一个从2020年第一季度到2021第四季度共8个季度的时间段，一年以11月为最后时间。

PeriodIndex(['2020Q1', '2020Q2', '2020Q3', '2020Q4', '2021Q1', '2021Q2',
             '2021Q3', '2021Q4'],
            dtype='period[Q-NOV]')
'''

print('# 通过传入时间段对象来定义')
print(pd.period_range(start=pd.Period('2020Q1', freq='Q'),
                      end=pd.Period('2021Q2', freq='Q'), freq='M'
                      ))
'''
PeriodIndex(['2020-03', '2020-04', '2020-05', '2020-06', '2020-07', '2020-08',
             '2020-09', '2020-10', '2020-11', '2020-12', '2021-01', '2021-02',
             '2021-03', '2021-04', '2021-05', '2021-06'],
            dtype='period[M]')
'''

print('时间段索引可以应用于数据中：')
print(pd.Series(pd.period_range('2020Q1', '2021Q4', freq='Q-NOV')))
# 0    2020Q1
# 1    2020Q2
# 2    2020Q3
# 3    2020Q4
# 4    2021Q1
# 5    2021Q2
# 6    2021Q3
# 7    2021Q4
# dtype: period[Q-NOV]

print(pd.Series(range(8), index=pd.period_range('2020Q1', '2021Q4', freq='Q-NOV')))
# 2020Q1    0
# 2020Q2    1
# 2020Q3    2
# 2020Q4    3
# 2021Q1    4
# 2021Q2    5
# 2021Q3    6
# 2021Q4    7
# Freq: Q-NOV, dtype: int64

print()
print('------------------------------------------------------------')
print('\t14.5.5 数据查询')

'''在数据查询时，支持切片操作：'''

s = pd.Series(1, index=pd.period_range('2020-10-01 10:00', '2021-10-01 10:00', freq='H'))
print(s)
'''
2020-10-01 10:00    1
2020-10-01 11:00    1
2020-10-01 12:00    1
2020-10-01 13:00    1
2020-10-01 14:00    1
                   ..
2021-10-01 06:00    1
2021-10-01 07:00    1
2021-10-01 08:00    1
2021-10-01 09:00    1
2021-10-01 10:00    1
Freq: H, Length: 8761, dtype: int64
'''

print(s['2020'])
'''
2020-10-01 10:00    1
2020-10-01 11:00    1
2020-10-01 12:00    1
2020-10-01 13:00    1
2020-10-01 14:00    1
                   ..
2020-12-31 19:00    1
2020-12-31 20:00    1
2020-12-31 21:00    1
2020-12-31 22:00    1
2020-12-31 23:00    1
Freq: H, Length: 2198, dtype: int64
'''

print('# 进行切片操作')
print(s['2020-10':'2020-11'])
'''
2020-10-01 10:00    1
2020-10-01 11:00    1
2020-10-01 12:00    1
2020-10-01 13:00    1
2020-10-01 14:00    1
                   ..
2020-11-30 19:00    1
2020-11-30 20:00    1
2020-11-30 21:00    1
2020-11-30 22:00    1
2020-11-30 23:00    1
Freq: H, Length: 1454, dtype: int64
'''

'''数据的查询方法与之前介绍过的时序查询一致。'''

print()
print('------------------------------------------------------------')
print('\t14.5.6 相关类型转换')

'''
astype()可以在几种数据之间自由转换，如DatetimeIndex转PeriodIndex：
'''

ts = pd.date_range('20201101', periods=100)
print(ts)
'''
DatetimeIndex(['2020-11-01', '2020-11-02', '2020-11-03', '2020-11-04',
                ...
               '2021-02-05', '2021-02-06', '2021-02-07', '2021-02-08'],
              dtype='datetime64[ns]', freq='D')
'''

print('# 转为PeriodIndex，频率为月')
print(ts.astype('period[M]'))
# PeriodIndex(['2020-11', '2020-11', '2020-11', '2020-11', '2020-11', '2020-11',
#              ...
#              '2021-01', '2021-01', '2021-02', '2021-02', '2021-02', '2021-02',
#              '2021-02', '2021-02', '2021-02', '2021-02'],
#             dtype='period[M]')

print('PeriodIndex转DatetimeIndex：')

ts = pd.period_range('2020-11', periods=100, freq='M')
print(ts)
'''
PeriodIndex(['2020-11', '2020-12', '2021-01', '2021-02', '2021-03', '2021-04',
             ...
             '2028-11', '2028-12', '2029-01', '2029-02'],
            dtype='period[M]')
'''

print('# 转为DatetimeIndex')
print(ts.astype('datetime64[ns]'))
'''
DatetimeIndex(['2020-11-01', '2020-12-01', '2021-01-01', '2021-02-01',
               '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01',
                ...
               '2028-07-01', '2028-08-01', '2028-09-01', '2028-10-01',
               '2028-11-01', '2028-12-01', '2029-01-01', '2029-02-01'],
              dtype='datetime64[ns]', freq='MS')
'''

print('# 频率从月转为季度')
print(ts.astype('period[Q]'))
'''
PeriodIndex(['2020Q4', '2020Q4', '2021Q1', '2021Q1', '2021Q1', '2021Q2',
             '2021Q2', '2021Q2', '2021Q3', '2021Q3', '2021Q3', '2021Q4',
             ........
             '2028Q2', '2028Q2', '2028Q3', '2028Q3', '2028Q3', '2028Q4',
             '2028Q4', '2028Q4', '2029Q1', '2029Q1'],
            dtype='period[Q-DEC]')
'''

'''
14.5.7 小结
时间段与时长和时间偏移不同的是，时间段有开始时间（当然也能推出结束时间）和长度，
在分析周期性发生的业务数据时，它会让你如鱼得水。
'''

print()
print('------------------------------------------------------------')
print('第14章 Pandas时序数据')
print('\t14.6 时间操作')
# update20240621
'''

'''
# 小节注释
'''
在前面几大时间类型的介绍中，我们需要进行转换、时间解析和输出格式化等操作，
本节就来介绍一些与之类似的通用时间操作和高级功能。

▶ 以下是一些具体的使用方法举例：
'''
print()
print('------------------------------------------------------------')
print('\t14.6.1 时区转换')

'''
Pandas使用pytz和dateutil库或标准库中的
datetime.timezone对象为使用不同时区的时间戳提供了丰富的支持。
可以通过以下方法查看所有时区及时区的字符名称：

'''

import pytz

# print(pytz.common_timezones) # out:长列表

# for i in pytz.common_timezones:  # 434种时区
#     print(i)

'''如果没有指定，时间一般是不带时区的：'''
ts = pd.date_range('11/11/2020 00:00', periods=10, freq='D')
print(ts.tz)  # None

'''进行简单的时区指定，中国通用的北京时区使用'Asia/Shanghai'定义：'''
print(pd.date_range('2020-01-01', periods=10, freq='D', tz='Asia/Shanghai'))
'''
DatetimeIndex(['2020-01-01 00:00:00+08:00', '2020-01-02 00:00:00+08:00',
               '2020-01-03 00:00:00+08:00', '2020-01-04 00:00:00+08:00',
               '2020-01-05 00:00:00+08:00', '2020-01-06 00:00:00+08:00',
               '2020-01-07 00:00:00+08:00', '2020-01-08 00:00:00+08:00',
               '2020-01-09 00:00:00+08:00', '2020-01-10 00:00:00+08:00'],
              dtype='datetime64[ns, Asia/Shanghai]', freq='D')
'''

print('简单指定时区的方法：')
print(pd.Timestamp('2020-01-01', tz='Asia/Shanghai'))  # 2020-01-01 00:00:00+08:00

print('以下是指定时区的更多方法：')
# 使用pytz支持
rng_pytz = pd.date_range('11/11/2020 00:00', periods=3, freq='D', tz='Europe/London')
print(rng_pytz.tz)  # Europe/London

# 使用dateutil支持
rng_dateutil = pd.date_range('11/11/2020 00:00', periods=3, freq='D')
# 转为伦敦所在的时区
rng_dateutil = rng_dateutil.tz_localize('dateutil/Europe/London')
print(rng_dateutil.tz)  # tzfile('Europe/Belfast')

print()
print('# 使用dateutil指定为UTC时间')
rng_utc = pd.date_range('11/11/2020 00:00', periods=3, freq='D', tz=dateutil.tz.tzutc())
print(rng_utc.tz)  # tzutc()

print()
print('# datetime.timezone')
rng_utc = pd.date_range('11/11/2020 00:00', periods=3, freq='D', tz=datetime.timezone.utc)
print(rng_utc.tz)  # UTC

'''从一个时区转换为另一个时区，使用tz_convert方法：'''
print('tz_convert')
print(rng_pytz.tz_convert('US/Eastern'))
'''
DatetimeIndex(['2020-11-10 19:00:00-05:00', '2020-11-11 19:00:00-05:00',
               '2020-11-12 19:00:00-05:00'],
              dtype='datetime64[ns, US/Eastern]', freq='D')
'''

print('其他方法：')
# 示例数据：UTC时间戳的交易记录
s_naive = pd.Series(pd.date_range('2024-05-27 00:00', periods=5, freq='H'))
print(s_naive)
# 0   2024-05-27 00:00:00
# 1   2024-05-27 01:00:00
# 2   2024-05-27 02:00:00
# 3   2024-05-27 03:00:00
# 4   2024-05-27 04:00:00
# dtype: datetime64[ns]

print('# 将时间戳本地化为UTC，然后转换为美国东部时间')
s_eastern = s_naive.dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
print(s_eastern)
# 0   2024-05-26 20:00:00-04:00
# 1   2024-05-26 21:00:00-04:00
# 2   2024-05-26 22:00:00-04:00
# 3   2024-05-26 23:00:00-04:00
# 4   2024-05-27 00:00:00-04:00
# dtype: datetime64[ns, US/Eastern]

# 直接转换为带有美国东部时间的时间戳
# s_naive = pd.Series(pd.date_range('2024-05-27 00:00', periods=5, freq='H'))
# s_eastern_2 = s_naive.astype('datetime64[ns, US/Eastern]')
# print(s_eastern_2)
'''该示例实际会 报错！'''

print('# 转换为不带时区信息的numpy数组')
# 示例数据：带有时区信息的时间戳
s_aware = pd.Series(pd.date_range('2024-05-27 00:00', periods=5, freq='H', tz='US/Eastern'))
s_no_tz = s_aware.to_numpy(dtype='datetime64[ns]')

print(s_no_tz)
# ['2024-05-27T04:00:00.000000000' '2024-05-27T05:00:00.000000000'
#  '2024-05-27T06:00:00.000000000' '2024-05-27T07:00:00.000000000'
#  '2024-05-27T08:00:00.000000000']

print()
print('------------------------------------------------------------')
print('\t14.6.2 时间的格式化')
# update20240625
'''
在数据格式解析、输出格式和格式转换过程中，需要用标识符来匹配日期元素的位置，
Pandas使用了Python的格式化符号系统，如：
'''

print('# 解析时间格式')
print(pd.to_datetime('2020*11*12', format='%Y*%m*%d'))  # 2020-11-12 00:00:00

print('# 输出的时间格式')
print(pd.Timestamp('now').strftime('%Y年%m月%d日'))  # 2024年06月25日

print(pd.Timestamp('now').strftime('%a'))  # Tue
print(pd.Timestamp('now').strftime('%A'))  # Tuesday
print(pd.Timestamp('now').strftime('%b'))  # Jun
print(pd.Timestamp('now').strftime('%B'))  # June
print(pd.Timestamp('now').strftime('%c'))  # Tue Jun 25 10:13:02 2024 |本地相应的日期表示和时间表示
print(pd.Timestamp('now').strftime('%j'))  # 177 | 年内的一天(001~366)

print(pd.Timestamp('now').strftime('%w'))  # 2  | 星期(0~6)，星期天为星期的开始
print(pd.Timestamp('now').strftime('%W'))  # 26 | 一年中的星期数(00~53)，星期一为星期的开始
print(pd.Timestamp('now').strftime('%U'))  # 25 | 一年中的星期数(00~53)，星期天为星期的开始

print(pd.Timestamp('now').strftime('%x'))  # 06/25/24 | 本地相应的日期表示
print(pd.Timestamp('now').strftime('%X'))  # 10:16:35 | 本地相应的时间表示
print(pd.Timestamp('now').strftime('%Z'))  # 空白 | 当前时区的名称
print(pd.Timestamp('now').strftime('%%'))  # % | %号本身

print(pd.Timestamp('now').strftime('%p'))  # AM | 本地 A.M.或 PM.的等价符

print()
print('------------------------------------------------------------')
print('\t14.6.3 时间重采样')

'''
Pandas可以对时序数据按不同的频率进行重采样操作，例如，原时序数据频率为分钟，
使用resample()可以按5分钟、15分钟、半小时等频率进行分组，然后完成聚合计算。
时间重采样在资金流水、金融交易等业务下非常常用。
'''

idx = pd.date_range('2020-01-01', periods=500, freq='Min')
ts = pd.Series(range(len(idx)), index=idx)
print(ts)
'''
2020-01-01 00:00:00      0
2020-01-01 00:01:00      1
                      ... 
2020-01-01 08:18:00    498
2020-01-01 08:19:00    499
Freq: T, Length: 500, dtype: int64
'''

print('# 每5分钟进行一次聚合')
print(ts.resample('5Min').sum())
'''
2020-01-01 00:00:00      10
2020-01-01 00:05:00      35
                       ... 
2020-01-01 08:10:00    2460
2020-01-01 08:15:00    2485
Freq: 5T, Length: 100, dtype: int64
'''

'''
重采样功能非常灵活，你可以指定许多不同的参数来控制频率转换和重采样操作。
通过类似于groupby聚合后的各种统计函数实现数据的分组聚合，
包括sum、mean、std、sem、max、min、mid、median、first、last、ohlc。
'''
print()
print(ts.resample('5Min').mean())  # 平均
'''
2020-01-01 00:00:00      2.0
2020-01-01 00:05:00      7.0
                       ...  
2020-01-01 08:10:00    492.0
2020-01-01 08:15:00    497.0
Freq: 5T, Length: 100, dtype: float64
'''
print(ts.resample('5Min').max())  # 最大值
'''
2020-01-01 00:00:00      4
2020-01-01 00:05:00      9
2020-01-01 00:10:00     14
2020-01-01 00:15:00     19
2020-01-01 00:20:00     24
                      ... 
2020-01-01 07:55:00    479
2020-01-01 08:00:00    484
2020-01-01 08:05:00    489
2020-01-01 08:10:00    494
2020-01-01 08:15:00    499
Freq: 5T, Length: 100, dtype: int64
'''

'''
其中ohlc是又叫美国线（Open-High-Low-Close chart，OHLCchart），
可以呈现类似股票的开盘价、最高价、最低价和收盘价：
'''
print('# 两小时频率的美国线')
print(ts.resample('2h').ohlc())
'''
                     open  high  low  close
2020-01-01 00:00:00     0   119    0    119
2020-01-01 02:00:00   120   239  120    239
2020-01-01 04:00:00   240   359  240    359
2020-01-01 06:00:00   360   479  360    479
2020-01-01 08:00:00   480   499  480    499
'''

print()
print('closed参数设置')
'''可以将closed参数设置为“left”或“right”，以指定开闭区间的哪一端：'''
print(ts.resample('2h').mean())
# 2020-01-01 00:00:00     59.5
# 2020-01-01 02:00:00    179.5
# 2020-01-01 04:00:00    299.5
# 2020-01-01 06:00:00    419.5
# 2020-01-01 08:00:00    489.5
# Freq: 2H, dtype: float64
print(ts.resample('2h', closed='left').mean())  # 结果同上
print(ts.resample('2h', closed='right').mean())
# 2019-12-31 22:00:00      0.0
# 2020-01-01 00:00:00     60.5
# 2020-01-01 02:00:00    180.5
# 2020-01-01 04:00:00    300.5
# 2020-01-01 06:00:00    420.5
# 2020-01-01 08:00:00    490.0
# Freq: 2H, dtype: float64

print()
print('# label参数')
'''使用label可以控制输出结果显示左还是右，但不像closed那样影响计算结果：'''
print(ts.resample('5Min').mean())  # 默认 label='left'
print(ts.resample('5Min', label='right').mean())  # 计算结果一样 可能是格式显示不一样吧 肉眼不大看出来

print()
print('------------------------------------------------------------')
print('\t14.6.4 上采样')

'''
上采样（upsampling）一般应用在图形图像学中，目的是放大图像。
由于原数据有限，放大图像后需要对缺失值进行内插值填充。

在时序数据中同样存在着类似的问题，上例中的数据频率是分钟，我们要对其按30秒重采样：
'''
print(ts.head(3).resample('30S').asfreq())
'''
2020-01-01 00:00:00    0.0
2020-01-01 00:00:30    NaN
2020-01-01 00:01:00    1.0
2020-01-01 00:01:30    NaN
2020-01-01 00:02:00    2.0
Freq: 30S, dtype: float64
'''

'''我们发现由于原数据粒度不够，出现了缺失值，这就需要用.ffill()和.bfill()来计算填充值：'''
print(ts.head(3).resample('30S').ffill())  # 向前填充
'''
2020-01-01 00:00:00    0
2020-01-01 00:00:30    0
2020-01-01 00:01:00    1
2020-01-01 00:01:30    1
2020-01-01 00:02:00    2
Freq: 30S, dtype: int64
'''
print(ts.head(3).resample('30S').bfill())  # 向后填充
'''
2020-01-01 00:00:00    0
2020-01-01 00:00:30    1
2020-01-01 00:01:00    1
2020-01-01 00:01:30    2
2020-01-01 00:02:00    2
Freq: 30S, dtype: int64
'''

print()
print('------------------------------------------------------------')
print('\t14.6.5 重采样聚合')

'''类似于agg API、groupby API和窗口方法API，重采样也适用于相关的统计聚合方法：'''

df = pd.DataFrame(np.random.randn(1000, 3),
                  index=pd.date_range('1/1/2020', freq='S', periods=1000),
                  columns=['A', 'B', 'C']
                  )
print(df)
'''
                            A         B         C
2020-01-01 00:00:00 -1.339569  0.392938 -0.329884
2020-01-01 00:00:01 -0.451720 -0.841686  1.594416
2020-01-01 00:00:02 -0.925979 -0.040797 -0.101541
2020-01-01 00:00:03  0.202807  1.097003 -0.153193
2020-01-01 00:00:04  0.804477 -0.703568 -0.159791
...                       ...       ...       ...
2020-01-01 00:16:35 -2.575767 -0.316534  0.295881
2020-01-01 00:16:36 -2.310698  0.571659 -0.301216
2020-01-01 00:16:37 -0.464210  0.208668 -1.025299
2020-01-01 00:16:38  0.275189 -1.978887  0.257300
2020-01-01 00:16:39  0.238310  0.276412  0.492659

[1000 rows x 3 columns]
'''

print('# 生成Resampler重采样对象')
r = df.resample('3T')
# print(r) DatetimeIndexResampler [freq=<3 * Minutes>, axis=0, closed=left, label=left, convention=start, origin=start_day]
print(r.mean())
'''
                            A         B         C
2020-01-01 00:00:00  0.029331  0.071450 -0.052328
2020-01-01 00:03:00  0.174249  0.033578  0.099816
2020-01-01 00:06:00  0.021705  0.113303  0.014374
2020-01-01 00:09:00 -0.026221  0.053145  0.082342
2020-01-01 00:12:00 -0.075547  0.100557 -0.097778
2020-01-01 00:15:00 -0.082372 -0.119609 -0.076147
'''

print()
print('有多个聚合方式：')
print(r['A'].agg([np.sum, np.mean, np.std]))
'''
                           sum      mean       std
2020-01-01 00:00:00  10.195274  0.056640  1.141293
2020-01-01 00:03:00 -11.834165 -0.065745  1.016519
2020-01-01 00:06:00  12.333678  0.068520  0.956461
2020-01-01 00:09:00  -3.122459 -0.017347  0.942420
2020-01-01 00:12:00  14.771384  0.082063  0.977905
2020-01-01 00:15:00  -3.185838 -0.031858  0.910143
'''

print('# 每个列')
print(r.agg([np.sum, np.mean]))
'''
                           sum      mean  ...        sum      mean
2020-01-01 00:00:00   4.317965  0.023989  ...   2.472293  0.013735
2020-01-01 00:03:00   8.653921  0.048077  ...   4.473018  0.024850
2020-01-01 00:06:00   7.267614  0.040376  ...  -0.635775 -0.003532
2020-01-01 00:09:00   5.423832  0.030132  ...   4.825570  0.026809
2020-01-01 00:12:00 -10.215428 -0.056752  ... -13.677265 -0.075985
2020-01-01 00:15:00  -4.046951 -0.040470  ...  -8.146950 -0.081469

[6 rows x 6 columns]
'''

print()
print(r.agg({'A': np.sum,
             'B': lambda x: np.std(x, ddof=1)}))
'''
                             A         B
2020-01-01 00:00:00  -3.465165  0.954057
2020-01-01 00:03:00   7.450780  1.125230
2020-01-01 00:06:00   5.596317  0.980065
2020-01-01 00:09:00   8.468608  1.050400
2020-01-01 00:12:00 -14.944876  0.983348
2020-01-01 00:15:00  -0.679696  0.944394
'''

print()
print('# 用字符指定')
print(r.agg({'A': 'sum', 'B': 'std'}))
'''
                             A         B
2020-01-01 00:00:00  12.566308  1.052866
2020-01-01 00:03:00  -9.757681  0.972359
2020-01-01 00:06:00  18.627415  1.000087
2020-01-01 00:09:00   2.906557  0.917702
2020-01-01 00:12:00  -2.211160  0.994550
2020-01-01 00:15:00  -5.333259  0.934275
'''

print(r.agg({'A': ['sum', 'std'], 'B': ['mean', 'std']}))
'''
                             A                   B          
                           sum       std      mean       std
2020-01-01 00:00:00 -12.735055  1.089689 -0.040202  1.023641
2020-01-01 00:03:00  15.675315  0.927174 -0.005197  1.023730
2020-01-01 00:06:00  22.353234  0.930815  0.017241  0.938834
2020-01-01 00:09:00   5.565829  0.907279  0.183779  0.980010
2020-01-01 00:12:00   2.016427  1.065811 -0.019264  1.027851
2020-01-01 00:15:00 -16.844252  0.832082 -0.031579  0.999082
'''

'''如果索引不是时间，可以指定采样的时间列：'''
print('# date是一个普通列')
# print(df.resample('M',on='date').sum()) # KeyError: 'The grouper name date is not found'
# print(df.resample('M', level='d').sum()) # 多层索引 | # ValueError: The level d is not valid

# 创建示例 DataFrame
data = {
    'date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
    'value': range(100)
}
df = pd.DataFrame(data)

# 确保 'date' 列存在
print('# date是一个普通列')
# print(df.head())  # 查看前几行数据，确保 'date' 列存在

# 使用 'date' 列进行重采样
resampled_df = df.resample('M', on='date').sum()

print(resampled_df)
#             value
# date
# 2024-01-31    465
# 2024-02-29   1305
# 2024-03-31   2325
# 2024-04-30    855


# 创建示例 DataFrame
data = {
    'date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
    'category': ['A'] * 50 + ['B'] * 50,
    'value': range(100)
}
df = pd.DataFrame(data)
print(df)
'''
         date category  value
0  2024-01-01        A      0
1  2024-01-02        A      1
2  2024-01-03        A      2
3  2024-01-04        A      3
4  2024-01-05        A      4
..        ...      ...    ...
95 2024-04-05        B     95
96 2024-04-06        B     96
97 2024-04-07        B     97
98 2024-04-08        B     98
99 2024-04-09        B     99

[100 rows x 3 columns]
'''

# 设置多层索引
df = df.set_index(['date', 'category'])
print(df)
'''
                     value
date       category       
2024-01-01 A             0
2024-01-02 A             1
2024-01-03 A             2
2024-01-04 A             3
2024-01-05 A             4
...                    ...
2024-04-05 B            95
2024-04-06 B            96
2024-04-07 B            97
2024-04-08 B            98
2024-04-09 B            99

[100 rows x 1 columns]
'''

# 使用 'date' 级别进行重采样
resampled_df = df.resample('M', level='date').sum()

# 查看重采样后的 DataFrame
print("\\nResampled DataFrame:")
print(resampled_df)
#             value
# date
# 2024-01-31    465
# 2024-02-29   1305
# 2024-03-31   2325
# 2024-04-30    855

print()
print('迭代采样对象：')
# r 是重采样对象
for name, group in r:
    print("Group:", name)
    print("-" * 20)
    print(group, end="\n\n")

'''输出结果为6组 前5组[180 rows x 3 columns] 最后一组[100 rows x 3 columns]'''

print()
print('------------------------------------------------------------')
print('\t14.6.6 时间类型间转换')

'''介绍一下不同时间概念之间的相互转换。to_period()将DatetimeIndex转换为PeriodIndex：'''

print('# 转换为时间周期')
p = pd.date_range('1/1/2020', periods=5)
print(p)
'''
DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
               '2020-01-05'],
              dtype='datetime64[ns]', freq='D')
'''
print(p.to_period())
'''
PeriodIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
             '2020-01-05'],
            dtype='period[D]')
'''

print()
print('# to_timestamp()')
'''to_timestamp()将默认周期的开始时间转换为DatetimeIndex：'''
pt = pd.period_range('1/1/2020', periods=5)
print(pt.to_timestamp())
'''
DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
               '2020-01-05'],
              dtype='datetime64[ns]', freq='D')
'''

print()
print('------------------------------------------------------------')
print('\t14.6.7 超出时间戳范围时间')
'''
在介绍时间表示方法时我们说到Pandas原生支持的时间范围大约在
1677年至2262年之间，那么如果分析数据不在这个区间怎么办呢？
可以使用PeriodIndex来进行计算，我们来测试一下：
'''

print('# 定义一个超限时间周期')
print(pd.period_range('1111-01-01', '8888-01-01', freq='D'))
'''
PeriodIndex(['1111-01-01', '1111-01-02', '1111-01-03', '1111-01-04',
             '1111-01-05', '1111-01-06', '1111-01-07', '1111-01-08',
             '1111-01-09', '1111-01-10',
             ...
             '8887-12-23', '8887-12-24', '8887-12-25', '8887-12-26',
             '8887-12-27', '8887-12-28', '8887-12-29', '8887-12-30',
             '8887-12-31', '8888-01-01'],
            dtype='period[D]', length=2840493)
'''

print()
'''可以正常计算和使用。还可以将时间以数字形式保存，在计算的时候再转换为周期数据：'''
print(pd.Series([123_1111, 2008_10_01, 8888_12_12]))
# 0     1231111
# 1    20081001
# 2    88881212
# dtype: int64

# 将整型转为时间周期类型
print(pd.Series([123_1111, 2008_10_01, 8888_12_12])
      .apply(lambda x: pd.Period(year=x // 10000,
                                 month=x // 100 % 100,
                                 day=x % 100,
                                 freq='D')))
'''
0    0123-11-11
1    2008-10-01
2    8888-12-12
dtype: period[D]
'''

# print(123_1111 // 100) # 整除
# print(123_1111 / 100) # 除数
# print(123_1111 % 100) # 取模

print()
print('------------------------------------------------------------')
print('\t14.6.8 区间间隔')
# update20240626

'''
pandas.Interval可以解决数字区间和时间区间的相关问题，
它实现一个名为Interval的不可变对象，该对象是一个有界的切片状间隔。
构建Interval对象的方法如下：
'''
print('# Interval对象构建')
print(pd.Interval(left=0, right=5, closed='right'))  # (0, 5]

print('# 4 是否在1～10之间')
print(4 in pd.Interval(1, 10))  # True
print(pd.Interval(1, 10))  # (1, 10]
print(pd.Interval(1, 10, closed='left'))  # [1, 10)

print('# 10 是否在1～9之间')
print(10 in pd.Interval(1, 10, closed='left'))  # False

# for i in pd.Interval(1,10):  # TypeError: 'pandas._libs.interval.Interval' object is not iterable
#     print(i)

print()
print('将区间转换为整数列表')
# 尝试将区间转换为整数列表 报错！
# print(list(pd.Interval(1,10))) # TypeError: 'pandas._libs.interval.Interval' object is not iterable

# openai建议：
# 创建一个区间
interval = pd.Interval(1, 10, closed='both')
print(interval)  # [1, 10]
# 手动创建区间内的整数列表
int_list = list(range(interval.left, interval.right + 1))

print(int_list)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# for i in int_list:
#     print(i)

'''
参数的定义如下。
left：定值，间隔的左边界。
right：定值，间隔的右边界。
closed：字符，可选right、left、both、neither，分别代表区间是
        在右侧、左侧、同时闭合、都不闭合。默认为right。
'''

print()
'''Interval可以对数字、固定时间、时长起作用，以下是构建数字类型间隔的方法和案例：'''
iv = pd.Interval(left=0, right=5)
print(iv)  # (0, 5]

print('# 可以检查元素是否属于它')
print(3.5 in iv)  # True
print(5.5 in iv)  # False

print('# 可以测试边界值')
# closed ='right'，所以0 < x <= 5
print(0 in iv)  # False
print(5 in iv)  # True
print(0.00001 in iv)  # True

print()
print('创建时间区间间隔：')
print('# 定义一个2020年的区间')
year_2020 = pd.Interval(pd.Timestamp('2020-01-01 00:00:00'),
                        pd.Timestamp('2021-01-01 00:00:00'),
                        closed='left')
print(year_2020)  # [2020-01-01, 2021-01-01)

# 检查指定时间是否
在2020年区间里
print(pd.Timestamp('2020-01-01 00:00:00') in year_2020)  # True

# 2020年时间区间的长度
print(year_2020.length)  # 366 days 00:00:00

print()
print('创建时长区间间隔：')
# 定义一个时长区间，3秒到1天

time_deltas = pd.Interval(pd.Timedelta('3 seconds'),
                          pd.Timedelta('1 days'),
                          closed='both'
                          )
print(time_deltas)  # [0 days 00:00:03, 1 days 00:00:00]

# 5分钟是否在时间区间里
print(pd.Timedelta('5 minutes') in time_deltas)  # True

# 时长区间长度
print(time_deltas.length)  # 0 days 23:59:57

print()
print('pd.Interval支持以下属性：')
# 区间闭合之处
print(iv.closed)  # right
# 检查间隔是否在左侧关闭
print(iv.closed_left)  # False
# 检查间隔是否在右侧关闭
print(iv.closed_right)  # True
# 间隔是否为空，表示该间隔不包含任何点
print(iv.is_empty)  # False
# 间隔的左边界
print(iv.left)  # 0
# 间隔的右边界
print(iv.right)  # 5
# 间隔的长度
print(iv.length)  # 5
# 间隔的中点
print(iv.mid)  # 2.5
# 间隔是否在左侧为开区间
print(iv.open_left)  # True
# 间隔是否在右侧为开区间
print(iv.open_right)  # False

print()
print(pd.Interval(0, 1, closed='right').is_empty)  # False
print('# 不包含任何点的间隔为空')
print(pd.Interval(0, 0, closed='right').is_empty)  # True
print(pd.Interval(0, 0, closed='left').is_empty)  # True
print(pd.Interval(0, 0, closed='neither').is_empty)  # True

print('# 包含单个点的间隔不为空')
print(pd.Interval(0, 0, closed='both').is_empty)  # False

'''
# 一个IntervalArray或IntervalIndex返回一个布尔ndarray
# 它在位置上指示Interval是否为空
'''
ivs = [pd.Interval(0, 0, closed='neither'),
       pd.Interval(1, 2, closed='neither')]
print(ivs)  # [Interval(0, 0, closed='neither'), Interval(1, 2, closed='neither')]
print(pd.arrays.IntervalArray(ivs))
'''
<IntervalArray>
[(0, 0), (1, 2)]
Length: 2, dtype: interval[int64, neither]
'''
print(pd.arrays.IntervalArray(ivs).is_empty)  # [ True False]

print('# 缺失值不为空')
ivs = [pd.Interval(0, 0, closed='neither'), np.nan]
print(ivs)  # [Interval(0, 0, closed='neither'), nan]
print(pd.IntervalIndex(ivs))  # IntervalIndex([(0.0, 0.0), nan], dtype='interval[float64, neither]')
print(pd.IntervalIndex(ivs).is_empty)  # [ True False]

print()
'''
pd.Interval.overlaps检查两个Interval对象是否重叠。
如果两个间隔至少共享一个公共点（包括封闭的端点），则它们重叠。
'''
i1 = pd.Interval(0, 2)
i2 = pd.Interval(1, 3)
print(i1.overlaps(i2))  # True

i3 = pd.Interval(4, 5)
print(i1.overlaps(i3))  # False

print('共享封闭端点的间隔重叠：')
i4 = pd.Interval(0, 1, closed='both')
i5 = pd.Interval(1, 2, closed='both')
print(i4.overlaps(i5))  # True

'''只有共同的开放端点的间隔不会重叠：'''
i6 = pd.Interval(1, 2, closed='neither')
print(i4.overlaps(i6))  # False

print()
'''
间隔对象能使用+和*与一个固定值进行计算，此操作将同时应用于对象的两个边界，
结果取决于绑定边界值数据的类型。
以下是边界值为数字的示例：
'''
print(iv)  # (0, 5]
shift_iv = iv + 3
print(shift_iv)  # (3, 8]
extended_iv = iv * 10.0
print(extended_iv)  # (0.0, 50.0]

'''
另外，Pandas还不支持两个区间的合并、取交集等操作，可以使用Python的第三方库portion来实现。

14.6.9 小结
本节主要介绍了在时间操作中的一些综合功能。
由于大多数据库开发规范要求存储时间的时区为UTC，因此我们拿到数据就需要将其转换为北京时间。

时区转换是数据清洗整理的一个必不可少的环节。
对数据的交付使用需要人性化的显示格式，时间的格式化让我们能够更好地阅读时间。
时间重采样让我们可以如同Pandas的groupby那样方便地聚合分组时间。

14.7 本章小结
时序数据是数据类型中一个非常庞大的类型，在我们生活中无处不在，学习数据分析是无法绕开时序数据的。
Pandas构建了多种时间数据类型，提供了多元的时间处理方法，为我们打造了一个适应各种时间场景的时序数据分析平台。

Pandas有关时间处理的更多强大功能有待我们进一步挖掘，也值得我们细细研究。
'''


import datetime

import dateutil.tz
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import glob
import os
# import jinja2

import matplotlib

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

'''
                第六部分
                 可视化
        可视化是数据分析的终点也是起点。
得益于生动呈现的可视化数据效果，我们能够跨越对于数据的认知鸿沟。
本部分主要介绍Pandas的样式功能如何让数据表格更有表现力，
Pandas的绘图功能如何让数据自己说话，如何定义不同类型的数据图形，
以及如何对图形中的线条、颜色、字体、背景等进行细节处理。
'''

print()
print('------------------------------------------------------------')
print('第15章 Pandas样式')
# update20240626
'''
Pandas的样式是一个被大多数人忽视的可视化方法，它不仅仅是美化数据、提高数据的可视化效果这么简单。
回忆一下，你在Excel中是不是经常对特定的数据加粗、标红、背景标黄？
这些操作就是为了让数据更加醒目清晰，突显数据的逻辑和特征。

在本章中，我们将介绍Pandas的一些内置样式，如何使用这些样式功能快速实现可视化效果，
如何自定义一些个性化的样式，还将介绍内容的格式化显示方法，
最后介绍样式的函数调用、复用、清除、带样式文件导出等操作。
'''
print('\t15.1 内置样式')
# 小节注释
'''
Pandas的样式在Jupyter Notebook和JupyterLab等代码编辑工具上获得了非常好的数据展示效果，让数据呈现更加专业，更加友好。
本节介绍Pandas样式并告诉大家一些它内置的、非常有用的样式功能。

▶ 以下是一些具体的使用方法举例：
'''
print()
print('------------------------------------------------------------')
print('\t15.1.1 样式功能')

'''
如同给Excel中的数据设置各种颜色、字体一样，Pandas提供的样式功能可实现：
    数值格式化，如千分号、小数位数、货币符号、日期格式、百分比等；
    凸显某些数据，对行、列或特定的值（如最大值、最小值）使用样式，如字体大小、黄色、背景；
    显示数据关系，如用颜色深浅代表数据大小；
    迷你条形图，如在一个百分比的格子里，用颜色比例表达占比；
    表达趋势，类似Excel中每行代表趋势变化的迷你走势图（sparkline）。

  我们发现，样式和可视化图形的区别是，数据图形化不关注具体数据内容，
  而样式则在保留具体内容的基础上进行修饰，让可读性更强。
  有时候两者有交叉共用的情况。


'''

print()
print('------------------------------------------------------------')
print('\t15.1.2 Styler对象')

'''
DataFrame有一个df.style Styler对象，用来生成数据的样式，样式是使用CSS来完成的。
如果你懂点CSS的知识会得心应手，不过也不用担心，CSS非常简单，基本就是一个字典，单词也是我们最常见的。

这里有个使用技巧，仅使用df.style就可以在Jupyter Notebook未给样式的情况下显示所有数据：
'''

# 读取数据
df = pd.read_excel(team_file)
print(df)
print(df.style)  # <pandas.io.formats.style.Styler object at 0x000001DF0829A250>
# 查看类型
print(type(df.style))  # <class 'pandas.io.formats.style.Styler'>

'''
Pandas提供了几个非常实用的内置样式，它们也是我们经常要使用的，
接下来将对这些提高计数效率的样式功能进行介绍。
'''

print()
print('------------------------------------------------------------')
print('\t15.1.3 空值高亮')

'''style.highlight_null()对为空的值高亮标示，增加背景颜色，使其更为醒目：'''
# 将一个值改为空
df.iloc[1, 1] = np.NaN
print(df)
print('# 将空值高亮，默认为红色背景')
print(df.head().style.highlight_null())

# 将空值高亮，默认为红色背景
styled_df = df.style.highlight_null()
# <pandas.io.formats.style.Styler object at 0x0000023C5DD9B250>

# 输出.xlsx文件 测试正常！
# styled_df.to_excel('E:/bat/output_files/pandas_out_20240626032.xlsx',index=False)
# df.head().style.highlight_null().to_excel('E:/bat/output_files/pandas_out_20240626034.xlsx',index=False)

print('可以指定颜色：')
# 使用颜色名
# blue_style_df = df.style.highlight_null(color='blue') # 指定蓝色
# blue_style_df.to_excel('E:/bat/output_files/pandas_out_20240626035.xlsx',index=False)

# 使用颜色值 | 使用十六进制颜色代码
# num_style_df = df.style.highlight_null(color='#ccc') # 指定灰色
# num_style_df.to_excel('E:/bat/output_files/pandas_out_20240626036.xlsx',index=False)

# 使用 RGB 颜色值 || 报错 pandas不支持rgb格式
# green_style_df = df.style.highlight_null(color='rgb(0, 255, 0)')
# green_style_df.to_excel('E:/bat/output_files/pandas_out_20240626037.xlsx',index=False)

'''
颜色名和颜色值与CSS中的颜色表示方法相同，可以用CSS颜色名
和CSS合法颜色值表示，相关内容将会在第16章详细介绍。
'''

print()
print('------------------------------------------------------------')
print('\t15.1.4 极值高亮')

'''
分别将最大值高亮显示，最小值高亮显示，二者同时高亮显示并指定颜色，
示例代码如下，效果分别如图15-2～图15-4所示。
'''
print('# 将最大值高亮，默认为黄色背景')

# print(df.select_dtypes(include='number'))
df_select = df.select_dtypes(include='number')
# max_df = df_select.style.highlight_max(color='yellow') # 仅输出数值列 非数值列没有显示！
# max_df.to_excel('E:/bat/output_files/pandas_out_20240626038_01.xlsx',index=False)

max_df = df.style.highlight_max(color='yellow', subset=df_select.columns)
max_df.to_excel('E:/bat/output_files/pandas_out_20240627041_001.xlsx', index=False)

print()
print('# 将最小值高亮')
# min_df = df.style.highlight_min(color='green')
# min_df.to_excel('E:/bat/output_files/pandas_out_20240626039.xlsx',index=False)
'''
# 测试成功！
min_df = df.style.highlight_min(color='green',subset=['Q1']) # 仅应用Q1列
min_df = df.style.highlight_min(color='green',subset=df.select_dtypes(include='number').columns) # 应用每个数值列 成功！
min_df = df.style.highlight_min(color='green',subset=df_select.columns,axis=1) # 应用每行数值最小值 成功！
min_df.to_excel('E:/bat/output_files/pandas_out_20240626039_004.xlsx',index=False)
'''

'''
# update20240627 openai提供的函数应用方法 测试成功！参考！

# 每个数值列中的最大值

# 定义一个函数来应用高亮
# highlight_max 函数会对每列的数据进行检查，如果数据是最大值，则返回相应的高亮样式。
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

# 选择数值列
numeric_cols = df.select_dtypes(include='number').columns

# 应用高亮到数值列
# 通过 apply 方法对数值列应用高亮函数。
styled_df = df.style.apply(highlight_max, subset=numeric_cols)
# styled_df = df.style.apply(highlight_max, subset=numeric_cols,axis=1) # axis=1 表示每行最大值应用高亮

# 保存到 Excel 文件中
output_path = 'E:/bat/output_files/pandas_out_20240626039.xlsx'
styled_df.to_excel(output_path, engine='openpyxl', index=False)

print(f'文件已成功保存到 {output_path}')

'''

"""
# 所有数值列中的最大值 显示高亮

# 选择数值列
numeric_df = df.select_dtypes(include='number')

# 找到所有数值列中的全局最大值
global_max = numeric_df.max().max()

# 定义一个函数来应用高亮
def highlight_global_max(val):
    color = 'background-color: yellow' if val == global_max else ''
    return color

# 应用高亮到数值列
styled_df = df.style.applymap(highlight_global_max, subset=numeric_df.columns)

# 保存到 Excel 文件中
output_path = 'E:/bat/output_files/pandas_out_20240626039_01.xlsx'
styled_df.to_excel(output_path, engine='openpyxl', index=False)

print(f'文件已成功保存到 {output_path}')

"""

print()
print('# 以上同时使用并指定颜色')

mi_mx_df = (df.style
            .highlight_max(color='lime', subset=df_select.columns)
            .highlight_min(subset=df_select.columns)  # 默认颜色为黄色
            )
mi_mx_df.to_excel('E:/bat/output_files/pandas_out_20240627041_002.xlsx', index=False)

print('指定行级')

mi_mx_df_row = (df.style
                .highlight_max(color='lime', subset=df_select.columns, axis=1)
                .highlight_min(subset=df_select.columns, axis=1)  # 默认颜色为黄色
                )
mi_mx_df_row.to_excel('E:/bat/output_files/pandas_out_20240627041_003.xlsx', index=False)

print()
print('也可以作用于指定行：')
# 只对Q1起作用
one_colmn_df = df.style.highlight_min(subset=['Q1'])
one_colmn_df.to_excel('E:/bat/output_files/pandas_out_20240627041_004.xlsx', index=False)

# 对Q1、Q2两列起作用
two_columns_df = df.style.highlight_max(subset=['Q1', 'Q2'])
two_columns_df.to_excel('E:/bat/output_files/pandas_out_20240627041_005.xlsx', index=False)

# 使用pd.IndexSlice索引器（和loc[]类似）
# 注意，数据是所有数据，算最小值的范围而不是全部
cross_df = df.style.highlight_max(subset=pd.IndexSlice[:3, ['Q1', 'Q3']])
cross_df.to_excel('E:/bat/output_files/pandas_out_20240627041_006.xlsx', index=False)

# 按行，只在这两列进行
intersec_df = df.style.highlight_min(axis=1, subset=['Q1', 'Q2'])
intersec_df.to_excel('E:/bat/output_files/pandas_out_20240627041_007.xlsx', index=False)

print()
print('------------------------------------------------------------')
print('\t15.1.5 背景渐变')

'''
根据数值的大小，背景颜色呈现梯度渐变，越深表示越大，越浅表示越小，
类似于Excel中的色阶样式。颜色指定为Matplotlib库的色系表（Matplotlib colormap）中的色系名，
可通过这个网址查看色系名和颜色示例：https://matplotlib.org/devdocs/gallery/color/colormap_reference.html。
background_gradient()会对数字按大小用背景色深浅表示，

示例代码如下，效果如图15-6所示。
'''

print('# 数字类型按列背景渐变')

df_select = df.select_dtypes('number')
# print(df_select)

# background_df = df.style.background_gradient(subset=df_select.columns,)
# 指定数值列
background_df = df.style.background_gradient(subset=df_select.columns, cmap='BuGn')
background_df.to_excel('E:/bat/output_files/pandas_out_20240627042_003.xlsx', index=False)

df.style.background_gradient()
print(df.style.background_gradient())  # <pandas.io.formats.style.Styler object at 0x0000013FF2032AF0>

print(df.style.background_gradient(subset=['Q1'], cmap='BuGn'))
# <pandas.io.formats.style.Styler object at 0x00000134A9592AF0>
print('指定具体一列')
back_df = df.style.background_gradient(subset=['Q1'], cmap='BuGn')
back_df.to_excel('E:/bat/output_files/pandas_out_20240627042_001.xlsx', index=False)

print()
print(pd.__version__)  # 2.1.4
print(matplotlib.__version__)  # 3.8.4

print()
print('# 低百分比和高百分比范围, 更换颜色时避免使用所有色域')
# high_df = df.style.background_gradient(subset=df_select.columns, low=0.6, high=0)
# high_df = df.style.background_gradient(subset=df_select.columns,cmap='viridis', low=0.6, high=0) # cmap参数
# high_df = df.style.background_gradient(subset=df_select.columns, low=0, high=1)  #
# high_df = df.style.background_gradient(subset=df_select.columns, low=1, high=0)  # 这个也是从低到高颜色加深 如上
# high_df = df.style.background_gradient(subset=df_select.columns, low=0.6, high=0)
# high_df = df.style.background_gradient(subset=df_select.columns, low=0, high=0.6)
high_df = df.style.background_gradient(subset=df_select.columns, low=0.4, high=0.6)
high_df = df.style.background_gradient(subset=df_select.columns, low=0.1, high=0.3)
print(high_df)
high_df.to_excel('E:/bat/output_files/pandas_out_20240701012_008.xlsx', index=False)
'''
默认 low 和 high（即 low=0 和 high=1）：颜色渐变从最小值到最大值均匀分布。
low=0.6 和 high=0：颜色渐变集中在数据的高值部分，低值部分几乎没有颜色。
low=0 和 high=0.4：颜色渐变集中在数据的低值部分，高值部分几乎没有颜色。
通过调整 low 和 high 参数，你可以控制颜色渐变的范围和集中区域，在数据可视化时突出显示特定部分的数据。

个人测试上述情况 感觉上述解释 不靠谱
除了颜色有些变化，感觉参数调整高低值后，还是按照数值从小到大 颜色渐深  
'''

print()
print('# 内容的颜色，取0～1（深色到浅色），方便凸显文本')
# text_df = df.style.background_gradient(text_color_threshold=0.5)
# text_df = df.style.background_gradient(text_color_threshold=0.9)  # 低值看不清值
text_df = df.style.background_gradient(text_color_threshold=0.1)  # 中高值看不清值 最高值可以看清
# text_df = df.style.background_gradient(text_color_threshold=0.6)
text_df.to_excel('E:/bat/output_files/pandas_out_20240701012_012.xlsx', index=False)

print()
print('# 颜色应用的取值范围，不在这个范围的不应用')
# vin_df = df.style.background_gradient(vmin=60, vmax=100)

# vin_df = df.style.background_gradient(vmin=30, vmax=50)  # 看起来颜色更深了 极低值好像没颜色。
# vin_df = df.style.background_gradient(vmin=0, vmax=50)  # 比上个看起来颜色更深。
vin_df = df.style.background_gradient(vmin=50, vmax=70)  # 颜色分明
vin_df.to_excel('E:/bat/output_files/pandas_out_20240701012_016.xlsx', index=False)

print()
print('# 链式方法使用样式')
vin_df = (df.style
          .background_gradient(subset=['Q1'], cmap='spring')  # 指定色系
          .background_gradient(subset=['Q2'], vmin=60, vmax=100)  # 指定应用值区间
          .background_gradient(subset=['Q3'], low=0.6, high=0)  # 高低百分比范围
          .background_gradient(subset=['Q4'], text_color_threshold=0.9)  # 文本色深
          )

vin_df.to_excel('E:/bat/output_files/pandas_out_20240701012_017.xlsx', index=False)

print()
print('------------------------------------------------------------')
print('\t15.1.6 条形图')

'''
条形图在表格里一般以横向柱状图的形式代表这个值的大小。
'''

print('# 显示Q4列的条形图')

df_select = df.select_dtypes('number')
# print(df_select)

# bar_df = df.style.bar(subset=['Q4'], vmin=50, vmax=100) # 输出没有效果！
bar_df = df.style.bar(subset=['Q4'])

bar_df.to_excel('E:/bat/output_files/pandas_out_20240701012_019.xlsx', index=False)

'''
Pandas的Styler对象使用background_gradient方法生成的渐变色样式可以在导出到Excel文件时保留，
因为该方法直接在Excel单元格中应用背景颜色。
而bar方法生成的条形图样式是通过CSS样式实现的，这些样式在导出到Excel文件时不会被保留，因此你无法在Excel文件中看到条形图。
'''

print('一些常用的参数及方法：')
'''
# 基本用法，默认对数字应用
df.style.bar()
# 指定应用范围
df.style.bar(subset=['Q1'])
# 定义颜色
df.style.bar(color='green')
df.style.bar(color='#ff11bb')
# 以行方向进行计算和展示
df.style.bar(axis=1)
# 样式在格中的占位百分比，0～100，100占满
df.style.bar(width=80)
# 对齐方式：
# 'left'：最小值开始
# 'zero'：0值在中间
# 'mid'：(max-min)/2 值在中间，负（正）值0在右（左）
df.style.bar(align='mid')
# 大小基准值
df.style.bar(vmin=60, vmax=100)


以下是一个综合示例，
# 以下是一个综合示例，代码效果如图15-9所示。
(df.head(10)
    .assign(avg=df.mean(axis=1, numeric_only=True)) # 增加平均值
    .assign(diff=lambda x: x.avg.diff()) # 和前一位同学的差值
    .style
    .bar(color='yellow', subset=['Q1'])
    .bar(subset=['avg'],width=90,align='mid',vmin=60, vmax=100,color='#5CADAD')
    .bar(subset=['diff'],color=['#ffe4e4','#bbf9ce'], # 上涨和下降的颜色
         # vmin=0, vmax=30, # 范围定为以0为基准的上下30 || 负数没有颜色
         vmin=-20, vmax=30, # 范围定为以0为基准的上下30 || 负数显示颜色
         align='zero') # 0 值居中
    )


15.1.7 小结
Pandas的内置样式也是我们在Excel操作中经常用到的功能，这些功能非常实用又方便操作，
希望大家在数据处理的最后环节不要忘记给数据增加样式。

'''

print('\t15.2 显示格式')
# update20240702
# 小节注释
'''
我们在最终输出数据以进行查看时，需要对数据进行相应的格式化，常见的如加货币符号、加百分号、增加千分位等，
目的是让计数更加场景化，明确列表一定的业务意义。
Styler.format是专门用来处理格式的方法。

▶ 以下是一些具体的使用方法举例：
'''
print()
print('------------------------------------------------------------')
print('\t15.2.1 语法结构')

'''
Styler.format的语法格式为：
# 语法格式
Styler.format(self, formatter,
                subset=None,
                na_rep: Union[str, NoneType]=None)

'''
# 读取数据
df = pd.read_excel(team_file)

'''
以上语法中的formatter可以是（str,callable, dict, None）中的任意一个，
一般是一个字典（由列名和格式组成），也可以是一个函数。
关于字符的格式化可参考Python的格式化字符串方法。
'''

print('# 给所有数据加一个方括号')
# 在jupyter 环境中 显示正常，输出文件没有显示
format_df = df.style.format("[{}]")
format_df.to_excel('E:/bat/output_files/pandas_out_20240702021_005.xlsx', index=False)

# 百分号
percent_df = df.style.format("{:.2%}", subset=df.select_dtypes('number').columns)
# percent_df.to_excel('E:/bat/output_files/pandas_out_20240702021_007.xlsx', index=False)
print(percent_df)

print()
print('------------------------------------------------------------')
print('\t15.2.2 常用方法')
'''
由于支持Python的字符串格式，Styler.format可以实现丰富多样的数据格式显示，以下为常用的格式方法：
'''

'''
# 百分号
df.style.format("{:.2%}")
# 指定列全变为大写
df.style.format({'name': str.upper})
# B，保留四位；D，两位小数并显示正负号
df.style.format({'B': "{:0<4.0f}", 'D': '{:+.2f}'})
# 应用lambda
df.style.format({"B": lambda x: "±{:.2f}".format(abs(x))})
# 缺失值的显示格式
df.style.format("{:.2%}", na_rep="-")
# 处理内置样式函数的缺失值
df.style.highlight_max().format(None, na_rep="-")
# 常用的格式
{'a': '¥{0:,.0f}', # 货币符号
'b': '{:%Y-%m}', # 年月
'c': '{:.2%}', # 百分号
'd': '{:,f}', # 千分位
'e': str.upper} # 大写
'''

print()
print('------------------------------------------------------------')
print('\t15.2.3 综合运用')
'''
显示格式可以多次设定，也可以与颜色相关样式一起使用。以下是一个综合的应用案例：
'''

# 链式方法使用格式
intergrative_df = (df
                   .assign(avg=df.select_dtypes('number').mean(axis=1, numeric_only=True) / 100)  # 增加平均值百分比
                   .assign(diff=lambda x: x.avg.diff())  # 与前一位同学的差值
                   .style
                   .format({'name': str.lower})
                   .format({'avg': "{:.2%}"})
                   .format({'diff': "¥{:.2f}"}, na_rep='-')
                   )
print(intergrative_df)

'''
15.2.4 小结
为数据增加常用的格式（如大小写、千分位符、百分号、正负号等），
既可以让数据表达更加直观清晰，也可以让数据的显示更加专业。
'''

print()
print('------------------------------------------------------------')
print('\t15.3.1 样式配置操作')

'''
.set_caption('xxx')给显示的表格数据增加一个标题，
以下代码的效果如图15-11所示。
'''
# 读取数据
df = pd.read_excel(team_file)

'''
以上语法中的formatter可以是（str,callable, dict, None）中的任意一个，
一般是一个字典（由列名和格式组成），也可以是一个函数。
关于字符的格式化可参考Python的格式化字符串方法。
'''
# 添加标题
title_df = df.head().style.set_caption('学生成绩表')
# print(title_df)
# 测试文件输出 还是没效果
# title_df.to_excel('E:/bat/output_files/pandas_out_20240703031_001.xlsx', index=False)

hide_df = df.style.hide(subset=[1, 2], axis=0)  # 隐藏指定行
# 输出文件没有隐藏效果
hide_df.to_excel('E:/bat/output_files/pandas_out_20240703031_002.xlsx', index=False)

print('# 隐藏指定行列')
# df.style.hide_columns(['Q1','Q2']) # AttributeError: 'Styler' object has no attribute 'hide_columns'

df.style.hide(axis=1)  # 不显示列名（列索引）
df.style.hide(axis='columns')  # 同上

# df.style.hide(axis=1, names = ['Q1','Q2']) # 输出没效果 == df  依然显示全部行列索引
# df.style.hide(axis=1, subset = ['Q1','Q2']) # 隐藏指定列
# df.style.hide(subset = ['Q1','Q2'], axis=1) # 隐藏指定列
# df.style.hide(subset = ['Q1','Q2'], axis=1, names=True) # 隐藏指定列
# df.style.hide(subset = ['Q1','Q2'], axis=1, names=False) # 隐藏指定列 || false设置后 依然显示隐藏列名

df.style.hide(subset=[1, 2], axis=0)  # 隐藏指定行

print()
print('------------------------------------------------------------')
print('\t15.3.2 表格CSS样式')

'''
我们知道，Pandas的样式是通过生成和修改输出的HTML让浏览器渲染而得到一定的显示效果的，
如果内置样式无法实现，可以通过直接指定HTML树节点上的CSS样式来实现复杂的功能。

因此，在理解以下功能时需要有一定的HTML和CSS等前端编程基础。
.set_properties()给单元格配置CSS样式，以下代码的效果如图15-14所示。
'''

print('# 将Q1列文字设为红色')

# 测试输出文件 单元格文字显示 红色 （标题没有颜色）
red_df = df.style.set_properties(subset=['Q1'], **{'color': 'red'})
red_df.to_excel('E:/bat/output_files/pandas_out_20240703031_003.xlsx', index=False)

print('# 一些其他示例')

# 测试输出文件 单元格文字显示 白色 （标题没有颜色）
# mmp 输出的文字 太白了 差点以为只有标题 没有内容
white_df = df.style.set_properties(color='white', align='right')
white_df.to_excel('E:/bat/output_files/pandas_out_20240703031_004.xlsx', index=False)

# 单元格背景色 黄色
white_df = df.style.set_properties(**{'background-color': 'yellow'})
white_df.to_excel('E:/bat/output_files/pandas_out_20240703031_005.xlsx', index=False)

# 字体增大
white_df = df.style.set_properties(**{'width': '100px', 'font-size': '18px'})
white_df.to_excel('E:/bat/output_files/pandas_out_20240703031_006.xlsx', index=False)

# border-color 边框颜色  | lawngreen 草坪绿
white_df = df.style.set_properties(**{'background-color': 'black',
                                      'color': 'lawngreen',
                                      'border-color': 'white'})
white_df.to_excel('E:/bat/output_files/pandas_out_20240703031_007.xlsx', index=False)

'''
.set_table_attributes()用于给<table>标签增加属性，可以随意给定属性名和属性值：
'''
white_df = df.style.set_table_attributes('class="pure-table"')
# 没看出来输出文件有什么变化
white_df.to_excel('E:/bat/output_files/pandas_out_20240703031_009.xlsx', index=False)

white_df = df.style.set_table_attributes('id="gairuo-table"')
print(white_df)
# 同样没看出来输出文件有什么变化
white_df.to_excel('E:/bat/output_files/pandas_out_20240703031_010.xlsx', index=False)

'''
set_table_styles()用于设置表格样式属性，用来实现极为复杂的显示功能。
可以带有选择器和props键的字典。
选择器selector的值是样式将应用此CSS样式的内容（自动以表的UUID为前缀），
props是由CSS样式属性和值组成的元组列表。
如下例：
'''

# 给所有的行（tr标签）的hover方法设置黄色背景
# 效果是当鼠标移动上去时整行背景变黄  jupyter中有效果！
new_df = df.style.set_table_styles([{'selector': 'tr:hover',
                                     'props': [('background-color', 'yellow')]
                                     }])
# 输出文件无效果！
new_df.to_excel('E:/bat/output_files/pandas_out_20240703031_012.xlsx', index=False)

print()
# 给所有的行（tr标签）的hover方法设置黄色背景
# 效果是当鼠标移动上去时整行背景变黄

print('# 为每个表格增加一个相同的符缀')
# 没看出来有啥不一样
df.style.set_uuid(9999)

# 加"gairuo"
# 同样没看出来有啥不一样
df.style.set_uuid('gairuo')

print()
print('------------------------------------------------------------')
print('\t15.3.3 应用函数')
# update20240704
'''
像Series和DataFrame一样，Styler也可以使用apply()和applymap()定义复杂的样式。
如用函数实现将最大值显示为红色：
'''


# 将最大值显示红色
def highlight_max(x):
    return ['color : red' if v == x.max() else '' for v in x]


print('# 应用函数')
func_style = df.style.apply(highlight_max)
print(func_style)
# 文件输出有颜色效果
func_style.to_excel('E:/bat/output_files/pandas_out_20240704041_001.xlsx', index=False)

print('# 按条件为整行加背景色（样式）')
# row.name 行数 ||
'''
len(row) 
len(row)返回当前行的长度，即该行中列的数量。
['background-color: red'] * len(row)生成一个长度与当前行相同的列表，每个元素都是字符串'background-color: red'。
例如，如果当前行有4个单元格，那么['background-color: red'] * 4就会生成['background-color: red', 'background-color: red', 
'background-color: red', 'background-color: red']。

这样做的目的是确保为每个单元格都应用相同的样式。
'''


def background_color_red(row):
    if row.name >= 5:
        return ['background-color: red'] * len(row)
    elif row.name >= 3:
        return ['background-color: yellow'] * len(row)
    return [''] * len(row)


# 应用函数
# df.style.apply(background_color, axis=1)

print('# 按照行数 奇偶性 填充颜色')


# 按照行数 奇偶性 填充颜色
# 浅灰：rgb(211, 211, 211)  中灰：rgb(169, 169, 169)  深灰：rgb(105, 105, 105)
# 十六进制值: 使用#AAAAAA表示中灰色。你可以根据需要调整颜色的深浅，例如#D3D3D3表示浅灰色，#696969表示深灰色。
def background_color(row):
    if row.name % 2 == 0:
        return ['background-color: #D3D3D3'] * len(row)
        # return ['background-color: rgb(211, 211, 211)'] * len(row)
    else:
        return ['background-color: #AAAAAA'] * len(row)


# 应用函数
df.style.apply(background_color, axis=1)

# 应用函数
print(df.style.apply(background_color, axis=1))
# 测试 支持单词 或者 16进制 输出  不支持rgb格式
df.style.apply(background_color, axis=1).to_excel('E:/bat/output_files/pandas_out_20240704041_004.xlsx', index=False)

print('# 简单的整行背景设置 || 修改后：')
# 测试正常！
df.style.apply(lambda x: ['background-color: yellow'] * len(x) if x.Q1 > 68 else [''] * len(x), axis=1)

# 定义函数，只对数字起作用，将大于90的值的背景设置为黄色
bg = lambda x: 'background-color: yellow' if type(x) == int and x > 90 else ''  # 应用函数
# df.style.applymap(bg)
'''
FutureWarning: Styler.applymap has been deprecated. Use Styler.map instead.
  df.style.applymap(bg)
'''
df.style.map(bg)  # 测试正常

print('subset可以限制应用的范围：')
# 指定列表（值大于0）加背景色
df.style.map(lambda x: 'background-color: grey' if x > 0 else '',
             subset=pd.IndexSlice[:, ['Q1', 'Q2']])

print('# 定义样式函数')


# 了将字体放大和将name列全小写
# 'font-size': '-200%' 字体缩小  'font-size': '200%' ：字体放大
def my_style(styler):
    return (styler.set_properties(**{'font-size': '-200%'})
            .format({'name': str.lower}))


# 应用管道方法
df.style.pipe(my_style)

print()
print('------------------------------------------------------------')
print('\t15.3.4 样式复用')
# update20240704
'''
可以将数据和样式应用到新表格中：
'''

print('# 将df的样式赋值给变量')

# 示例数据
data1 = {
    'A': [1, -2, 3, -4, 5],
    'B': [-1, 2, -3, 4, -5]
}
df = pd.DataFrame(data1)

data2 = {
    'C': [10, -20, 30, -40, 50],
    'D': [-10, 20, -30, 40, -50]
}
df2 = pd.DataFrame(data2)


# 定义函数，为负值添加红色背景
# def color_negative_red(val):
#     color = 'grey' if val < 0 else ''
#     return f'background-color: {color}'

# 定义函数，为负值添加灰色背景
def color_negative_red(val):
    if val < 0:
        return 'background-color: grey'
    else:
        return 'background-color: transparent'


# 将df的样式赋值给变量
style1 = df.style.map(color_negative_red)

# df2的样式为style2
style2 = df2.style

# style2使用style1的样式
style2.use(style1.export())

# 在Jupyter Notebook中显示结果
style2

# 输出到Excel文件 | 有效
# style2.to_excel('E:/bat/output_files/pandas_out_20240704041_006.xlsx', index=False)


print()
print('------------------------------------------------------------')
print('\t15.3.5 样式清除')
# update20240704
'''
df.style.clear()会返回None。如下清除所有样式：
'''

print('# 定义为一个变量')



print()
print('------------------------------------------------------------')
print('\t15.3.6 导出Excel')
# update20240705
'''
可以将样式生成HTML和导出Excel。
生成HTML可以用它来发邮件，做网页界面，生成Excel可以做二次处理或者传播。
样式导出Excel后会保留原来定义的大多数样式，方法如下：
'''

print('# 导出Excel')

# 定义为一个变量
print(df)

# 测试空值
# df.iloc[2, 2] = np.nan
# 测试浮点数
df['Q1'] = df['Q1'].astype(float)  # 转换类型 否则小数值插入整数列 会有报警
df.iloc[2, 2] = 3.1415926


def highlight_max(x):
    return ['color : red' if v == x.max() else '' for v in x]


dfs = df.loc[:, 'Q1':'Q4'].style.apply(highlight_max)

# df.style.to_excel('E:/bat/output_files/pandas_out_20240705051_001.xlsx', index=False)
# df.style.to_excel('E:/bat/output_files/pandas_out_20240705051_003.xlsx', index=False, engine='openpyxl')
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_002.xlsx', index=False)
print('# 指定标签页名称，sheet_name')
# df.style.to_excel('E:/bat/output_files/pandas_out_20240705051_004.xlsx', index=False, sheet_name='test')
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_005.xlsx', index=False, sheet_name='test')
print('# 指定缺失值的处理方式')
# df.style.to_excel('E:/bat/output_files/pandas_out_20240705051_006.xlsx', index=False, na_rep='-')
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_007.xlsx', index=False, na_rep='-')
print('# 浮点数字格式，下例将0.1234转为0.12')
# 测试 输出没有显示浮点数格式
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_008.xlsx', float_format="%.2f")
# 浮点数 转化格式 测试成功！
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_009.xlsx', float_format="%.2f")
print('# 只要这两列')
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_010.xlsx', columns=['Q1', 'Q3'])
print('# 不带表头、索引')
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_011.xlsx', header=False, index=False)
print('# 指定索引，多个值代表多层索引')
'''dfs没有team、name列'''
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_012.xlsx', index_label=['team', 'name'])
'''没看出来有啥 除了索引列名称为team 原name、team列均正常'''
# df.to_excel('E:/bat/output_files/pandas_out_20240705051_013.xlsx', index_label=['team', 'name'])
'''关闭默认索引，试试看|| 输出文件没变化， 猜测可能team、name列是索引'''
# df.to_excel('E:/bat/output_files/pandas_out_20240705051_014.xlsx', index_label=['team', 'name'], index=False)
print('# 从哪行取，从哪列取')
'''数据全部输出，数据向右平移2列，向下平移3行'''
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_015.xlsx', startrow=3, startcol=2)

'''
# 不合并单元格
dfs.to_excel('gairuo.xlsx', merge_cells=False)
# 指定编码格式
dfs.to_excel('gairuo.xlsx', encoding='utf-8')
# 无穷大表示法（Excel中没有无穷大的本机表示法）
dfs.to_excel('gairuo.xlsx', inf_rep='inf')
# 在错误日志中显示更多信息
dfs.to_excel('gairuo.xlsx', verbose=True)
'''

print('# 指定要冻结的最底行和最右列')
'''A、B列被冻结，但看不出最底行冻结效果'''
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_016.xlsx', freeze_panes=(0, 2))
'''Q1列被冻结'''
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_017.xlsx', freeze_panes=(0, 1), index=False)
'''
(1, 0) 表示冻结首行。
1 表示在第 1 行之前冻结，所以实际冻结的是第 1 行。
0 表示不冻结任何列。
'''
# 测试成功，首行首列被冻结！
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_018.xlsx', freeze_panes=(1, 1), index=False)

# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_019.xlsx', freeze_panes=(-1, 1), index=False)
'''UserWarning: Row number -1 must be >= 0
  warn("Row number %d must be >= 0" % row)'''

print()
print('------------------------------------------------------------')
print('\t15.3.7 生成HTML')
# update20240708

'''
Styler.render()可以输出样式的HTML代码，它可以传入以下参数：
head
cellstyle
body
uuid
precision
table_styles
caption
table_attributes
生成的HTML代码可用于网页显示、邮件正文内容等场景，方法如下：
'''


# 定义样式函数，为负值添加灰色背景
def color_negative_red(val):
    color = 'grey' if val > 90 else ''
    return f'background-color: {color}'


# 将样式应用到 DataFrame
styled_df = df.select_dtypes('number').style.map(color_negative_red)
# styled_df
print(styled_df)
# 生成 HTML 字符串
html = styled_df.to_html()
# html # 仅展示编码
# 使用 IPython.display.HTML 来展示
# HTML(html)
print(html)  # 生成HTML编码


'''
15.3.8 小结
本节介绍了Pandas样式的一些高级用法，这些是除了Pandas提供的内置方法外，
为有HTML和CSS基础的用户提供的超级功能，可以用它们来实现任何复杂的展示效果。
'''

import datetime

import dateutil.tz
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import glob
import os
import jinja2

import matplotlib

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

# 读取数据
df = pd.read_excel(team_file)

'''
                第六部分
                 可视化
        可视化是数据分析的终点也是起点。
得益于生动呈现的可视化数据效果，我们能够跨越对于数据的认知鸿沟。
本部分主要介绍Pandas的样式功能如何让数据表格更有表现力，
Pandas的绘图功能如何让数据自己说话，如何定义不同类型的数据图形，
以及如何对图形中的线条、颜色、字体、背景等进行细节处理。
'''

print()
print('------------------------------------------------------------')
print('第16章 Pandas可视化')
# update20240708
'''
一图胜千言，人类是视觉敏感的动物，大多数人无法在短时间内找到数据中所蕴含的规律和业务意义，
但可以通过图形快速了解数据的比例、分布、趋势等信息，因此，可视化势在必行。
Pandas的可视化图表（Chart）可以让数据直达我们的大脑，让数据自己说话。
'''

print()
print('------------------------------------------------------------')
print('\t16.1 plot()方法')
'''
Pandas提供的plot()方法可以快速方便地将Series和DataFrame中的数据进行可视化，
它是对matplotlib.axes.Axes.plot的封装。
代码执行后会生成一张可视化图形，并直接显示在Jupyter Notebook上。

除了plot()方法，本节还将介绍一些关于可视化及Python操作可视化
的背景知识。这些内容能帮助我们更好地理解和编写可视化逻辑代码。
'''

print()
print('------------------------------------------------------------')
print('\t16.1.1 plot()概述')
# update20240708
'''
plot默认是指折线图。折线图是最常用和最基础的可视化图形，足以满足我们日常80%的需求。示例如下：
'''

# DataFrame调用
df.plot()
# test_df.show()
print('# 显示图形')
# plt.show('E:/bat/output_files/pandas_out_20240708011_001.png') # error

# 保存图形
plt.savefig('E:/bat/output_files/pandas_out_20240708011_001.png')  # success
plt.savefig('E:/bat/output_files/pandas_out_20240708011_002.jpeg')  # success

# 显示图形
plt.show()  # success

'''
以上默认会生成折线图，x轴为索引，y轴为数据。
对于DataFrame，会将所有的数字列以多条折线的形式显示在图形中。

我们可以在plot后增加调用来使用其他图形（这些图形对数据结构有不同的要求，本章后面会逐个介绍）：
'''

df.plot.bar()  # 柱状图
plt.savefig('E:/bat/output_files/pandas_out_20240708011_003.jpeg')
plt.show()

print('其它函数')
# df.plot.line()  # 折线图
# df.plot.bar()  # 柱状图
# df.plot.barh()  # 横向柱状图（条形图）
# df.plot.hist()  # 直方图
# df.plot.box()  # 箱型图
# df.plot.kde()  # 核密度图估计图
# df.plot.density()  # 同df.plot.kde()
# df.plot.area()  # 面积图
'''
测试这3个函数报错，可能需要参数！
df.plot.pie() # 饼图
df.plot.scatter() # 散点图
df.plot.hexbin() # 六边形箱体图，或简称六边形图
'''

# plt.savefig('E:/bat/output_files/pandas_out_20240708011_003.jpeg')
plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.2 plot()基础方法')
# update20240708
'''
Series数据调用plot方法时，它的索引信息会排布在x轴上，y轴则是x轴上的索引对应的具体数据值。

示例代码如下，效果如图16-1所示。
'''

# DataFrame调用
# df.plot()
# test_df.show()
print('# series绘图')
# plt.show('E:/bat/output_files/pandas_out_20240708011_001.png') # error

# print(range(5))
# print(list(range(5)))
# print(list(range(5))+list(range(5)))
ts = pd.Series(list(range(5)) + list(range(5)),
               index=pd.date_range('1/1/2020', periods=10))
# print(ts)

ts.plot()
plt.show()

print('# DataFrame绘图')

'''
DataFrame调用plot时，x轴为DataFrame的索引，y轴上将显示其多列的多条折线数据。
'''
df = pd.DataFrame(np.random.randn(6, 4),
                  index=pd.date_range('1/1/2020', periods=6),
                  columns=list('ABCD')
                  )

df = abs(df)
print(df)

df.plot()

plt.show()

print('DataFrame在绘图时可以指定x轴和y轴的列')
df = pd.DataFrame(np.random.randn(50, 2),
                  columns=['B', 'C']).cumsum()

# print(df)
df['A'] = pd.Series(list(range(len(df))))
# print(df)

df.plot(x='A', y='B')  # 指定x和y轴的内容
plt.show()

# y轴指定两列
df.plot(x='A', y=['B', 'C'])
plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.3 图形类型')
# update20240709
'''
默认的plot()方法可以帮助我们快速绘制各种图形，接下来介绍在使用plot绘制不同图形时常用的一些参数。
df.plot()的kind参数，可以指定图形的类型：
'''

# df.plot(kind='bar')
# df.plot(kind='barh')
# df.plot(kind='hist')
# df.plot(kind='kde')
# df.plot(kind='density')
# df.plot(kind='box')
# df.plot(kind='area')
# df.plot(kind='scatter')  # error

# s = df['A']
# print(s)
# s.plot(kind='pie')
# plt.show()


print()
print('------------------------------------------------------------')
print('\t16.1.4 x轴和y轴')
# update20240709
'''
x、y参数可指定对应轴上的数据，常用在折线图、柱状图、面积图、散点图等图形上。
如果是Series，则索引是x轴，无须传入y轴的值。
'''

print('# 可以不写参数名，直接按位置传入')

# print(df)
# df.plot('name', 'Q1')
# df.plot.bar('name', ['Q1', 'Q2'])
# df.plot.bar(x='name', y='Q4')
# df.plot.area('name', ['Q1', 'Q2'])
# df.plot.scatter('name', 'Q3')
'''注意，散点图只允许有一个y值。'''

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.5 图形标题')
# update20240709
'''
用title参数来指定图形的标题，标题会显示在图形的顶部
'''

print('# 指定标题')

df.plot.bar(title='前五位学生成绩分布图')
'''对于中文出现乱码的情况，后文会给出解决方案。'''

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.6 字体大小')
# update20240709
'''
fontsize指定轴上的字体大小，单位是pt（磅）
'''

print('# 指定轴上的字体大小')

df.set_index('name')[:5].plot(fontsize=20)

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.7 线条样式')
# update20240709
'''
style可指定图的线条等样式，并组合使用
'''
df.set_index('name', inplace=True)

# df[0:5].plot(style=':')  # 虚线
# df.set_index('name')[0:5].plot(style=':')  # 虚线
# df.plot(style='-.')  # 虚实相间
# df.plot.line(style='-.')
# df.plot.bar(style='-.')  # 可以显示柱状图形，但是没有看出来啥效果
# df.plot(style='--')  # 长虚线
# df.plot(style='.')  # 点
# df.plot(style='-')  # 实线（默认）
# df.plot(style='*-')  # 实线，数值为星星
# df.plot(style='*--')  # 虚线，数值为星星
# df.plot(style='^-')  # 实线，数值为三角形

print('# 指定不同的线条样式')
# df.plot(style=[':', '--', '.-', '*-'])
df.plot(title='指定不同线条样式', style=[':', '--', '.-', '*-'])

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.8 背景辅助线')
# update20240709
'''
grid会给x方向和y方向增加背景辅助线
'''
df.set_index('name', inplace=True)

print('# 增加背景辅助线')
df.plot(style=[':', '-.'], grid=True)

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.9 图例')
# update20240709
'''
plot()默认会显示图例，传入参数legend=False可隐藏图例。
'''
df.set_index('name', inplace=True)

print('# 不显示图例')
# df.plot(style=[':', '-.'], grid=True, legend=False)

print('# 将图例倒排')
df.plot(legend='reverse')

# df.plot(color='#D0D0D0').legend(loc='best') # 自动选择最佳位置（默认）
# df.plot(color='#D0D0D0').legend(loc='lower left')  # 左下角
# df.plot(color='#D0D0D0').legend(loc='lower right')  # 右下角
df.plot(color='#D0D0D0').legend(loc='center right')  # 右下角

'''
'best'：自动选择最佳位置（默认）
'upper right'：右上角
'upper left'：左上角
'lower left'：左下角
'lower right'：右下角
'right'：右侧
'center left'：左侧中央
'center right'：右侧中央
'lower center'：底部中央
'upper center'：顶部中央
'center'：中央
'''

plt.show()

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.10 图形大小')
# update20240709
'''
figsize参数传入一个元组，可以指定图形的宽、高值（单位为英寸，1英寸约为0.0254米）：
'''
df.set_index('name', inplace=True)

print('# 定义图形大小')
# df.plot.bar(figsize=(10.5, 5))
# df.plot.bar(figsize=(10.5, 5), legend=False)

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.11 色系')
# update20240709
'''
colormap指定图形的配色，具体值可参考Matplotlib库的色系表。
'''
df.set_index('name', inplace=True)

print('# 指定色系')

df.plot.barh(colormap='rainbow')
plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.12 绘图引擎')
# update20240709
'''
backend 参数可以指定一个新的绘图引擎，默认使用的是Matplotlib。
'''
df.set_index('name', inplace=True)

# 使用bokeh
# import pandas_bokeh
#
# pandas_bokeh.output_notebook()  # Notebook展示
# df.head(10).plot.bar('name', ['Q1', 'Q2'], backend='pandas_bokeh')
'''搞了半天 总是出错，放弃！'''

df.plot()
plt.show()

print()
print('------------------------------------------------------------')
print('\t16.1.13 Matplotlib的其他参数')
# update20240710
'''
我们知道，Pandas的默认绘图引擎是Matplotlib。
在plot()中可以使用Matplotlib的一系列参数，示例代码如下：
'''
df.set_index('name', inplace=True)

# df.plot.line(color='k')  # 图的颜色
df.plot.bar(rot=45)  # 主轴上文字的方向度数

plt.show()

'''
更多参数和使用方法参见官网 https://matplotlib.org/stable/api/pyplot_summary.html 。
'''

print()
print('------------------------------------------------------------')
print('\t16.1.14 图形叠加')
# update20240710
'''
如果希望将两种类型的图形叠加在一起，可以将两个图形的绘图语
句组成一个元组或者列表，下例实现了将5位同学的第一季度成绩（柱
状图）与其4个季度的平均成绩叠加在一起，方便对比。
'''
df.set_index('name', inplace=True)

(
    df['Q1'].plot.bar(),
    df.select_dtypes('number').mean(1).plot(color='r')
)

plt.show()

'''
更多参数和使用方法参见官网 https://matplotlib.org/stable/api/pyplot_summary.html 。
'''

print()
print('------------------------------------------------------------')
print('\t16.1.15 颜色的表示')
# update20240710
'''
在可视化中颜色与CSS的表示方法相同，可以用CSS颜色名和CSS
合法颜色值表示。17种标准颜色的名称为：
aqua、black、blue、fuchsia、gray、green、lime、maroon、navy、olive、orange、purple、
red、silver、teal、white、yellow。
'''
df.set_index('name', inplace=True)

# df.plot(color='fuchsia') # 紫红色
# df.plot(color='olive') # 橄榄色
# df.plot(color='aqua') # 浅绿色
# df.plot(color='lime') # 石灰色 | 树叶绿？
# df.plot(color='gray')  # 灰色
# df.plot(color='teal')  # 蓝绿色
# df.plot(color='maroon')  # 栗色
# df.plot(color='navy')  # 海军蓝
# df.plot(color='purple')  # 紫色
# df.plot(color='silver')  # 银色
# df.plot(color='golden') # 非标准色
df.plot(color='#D0D0D0')

plt.show()

'''
更多16进制颜色参数和使用方法参见官网 https://gairuo.com/p/web-color 。
'''

print()
print('------------------------------------------------------------')
print('\t16.1.16 解决图形中的中文乱码问题')
# update20240710
'''
Pandas绘图依赖的Matplotlib库在安装初始化时会加载一个配置文
件，这个文件包含了将要用到的字体，而中文字体不在这个文件中，所
以会造成在绘图过程中图形中的中文显示为方框或乱码的情况。
'''
df.set_index('name', inplace=True)

print('# 临时方案')
# jupyter notebooks plt 图表配置
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (15.0, 8.0)  # 固定显示大小
plt.rcParams['font.family'] = ['sans-serif']  # 显示中文问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文问题
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

'''
16.1.17 小结
DataFrame和Series都支持用plot()方法快速生成各种常用图形，plot()的参数可以对图形完成精细化处理。
本节还介绍了可视化图形中颜色的表达方式和中文字符在图形中的兼容显示问题。
'''

print()
print('------------------------------------------------------------')
print('\t16.2 常用可视化图形')
'''
本节将介绍plot()方法适配的几个最为常用的图形绘制方法，
这些方法甚至不需要额外的参数就能快速将数据可视化，使用起来非常方便。
'''

print()
print('------------------------------------------------------------')
print('\t16.2.1 折线图plot.line')
# update20240711
'''
折线图（Line Chart）是用线条连接各数据点而成的图形，它能表达数据的走势，一般与时间相关。

plot的默认图形是折线图，因此对于折线图，可以省略df.plot.line()中的line()方法。
DataFrame可以直接调用plot来生成折线图，其中，x轴为索引，其他数字类型的列为y轴上的线条。
'''
# df.set_index('name', inplace=True)

print('# 折线图的使用')
# df.plot()
# df.plot.line()  # 全写方式

'''
基于以上逻辑，如果希望指定的列显示为x轴，可以先将其设为索引.
'''

'''
(
    df.head(10)
    .set_index('team')
    .sort_values(by='team')  # 排序有效
    .plot()
)
'''

# Series索引为x轴，值为y轴，有值为非数字时会报错：
(
    df.set_index('name')
    .head()
    .Q1  # Series
    .plot()
)

plt.show()

print('# Series索引为x轴，值为y轴，有值为非数字时会报错：')
# 测试非数字值

# df.iloc[3,3] = '文本'
df = pd.read_excel(team_file)
# df.iloc[3,2] = np.NaN # 运行显示一半

df.iloc[3, 2] = np.NaN  # 'text' | 文本直接报错！error
# df

(
    df.set_index('name')
    .head()
    .Q1  # Series
    .plot()
)

print('# 指定x轴和y轴')

# df.plot(x='name', y='Q1')
df.plot(x='name', y=['Q1', 'Q2'])  # 指定多条

plt.show()

print('# 显示子图')

# df.plot(x='name', y='Q1')
df.plot(x='name', y=['Q1', 'Q2'], subplots=True)  # 指定多条

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.2.2 饼图plot.pie')
# update20240716
'''
饼图（Pie Chart）可以表示不同分类的数据在总体中的占比情况，
将一个完整的圆形划分为若干个“小饼”，占比大小体现在弧度大小上，整个圆饼为数据的总量。

如果数据中有NaN值，会自动将其填充为0；
如果数据中有负值，则会引发ValueError错误。
'''
# df.set_index('name', inplace=True)

print('# 饼图的使用')

s = pd.Series(3 * np.random.rand(4),
              index=['a', 'b', 'c', 'd'],
              name='数列'
              )

print(s)

s.plot.pie(figsize=(6, 6))
plt.show()

# TODO random函数参考：
'''
np.random.rand和np.random.random生成均匀分布在[0, 1)的浮点数。
np.random.randn生成服从标准正态分布的浮点数。
np.random.randint生成指定范围内的整数。

np.random.rand(5) # 生成一个1维数组，包含5个[0, 1)之间的随机数
np.random.randn(2, 3) # 生成一个2x3的数组，包含标准正态分布的随机数
np.random.randint(low, high=None, size=None, dtype=int)
    low: 随机整数的下界（包括）。
    high: 随机整数的上界（不包括）。如果没有指定，则生成[0, low)之间的随机数。
    size: 输出随机数的形状。
    dtype: 输出数组的数据类型。
np.random.randint(0, 10, size=5) # 生成一个1维数组，包含5个[0, 10)之间的随机整数
np.random.randint(0, 10, size=(5,3)) # 生成一个2维数组，包含5行3列个[0, 10)之间的随机整数
np.random.randint(5, size=(5,3), dtype=int) # 生成一个2维数组，包含5行3列个[0, 5)之间的随机整数

np.random.random((2, 2)) # 生成一个2x2的数组，包含[0, 1)之间的随机数

np.random.choice(a, size=None, replace=True, p=None)
    参数：
    a: 生成随机数的数组或整数。
    size: 输出随机数的形状。
    replace: 是否允许重复抽样。
    p: 每个元素被抽中的概率。

# 从用户列表中随机抽取3个用户，允许重复
users = ['User1', 'User2', 'User3', 'User4', 'User5']
sample_users = np.random.choice(users, size=3, replace=True)
print(sample_users)

'''

print('# DataFrame需要指定y值')

df1 = pd.DataFrame(3 * np.random.rand(4, 2),
                   index=['a', 'b', 'c', 'd'],
                   columns=['x', 'y']
                   )

print(df1)

df1.plot.pie(y='x')

plt.show()

'''如果数据的总和小于1.0，则Matplotlib将绘制一个扇形.'''
# 实际Matplotlib会自动归一化处理数据 还是会显示饼图，扇形图显示不出来。
s2 = pd.Series([0.1] * 4,
               index=['a', 'b', 'c', 'd'],
               name='series2')

s2.plot.pie(figsize=(6, 6), autopct='%1.1f%%')

print('# 子图')
df1.plot.pie(subplots=True, figsize=(8, 4))

# 子图，不显示图例
df1.plot.pie(subplots=True, figsize=(8, 4), legend=False)

# 设定如下代码中的其他常用参数
s.plot.pie(labels=['AA', 'BB', 'CC', 'DD'],  # 标签，指定项目名称
           colors=['r', 'g', 'b', 'c'],  # 指定颜色
           autopct='%.2f',  # 数字格式
           fontsize=20,  # 字体大小
           figsize=(6, 6)  # 图大小
           )

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.2.3 柱状图plot.bar')
# update20240716
'''
柱状图（Bar Chart）又称条形图，使用与轴垂直的柱子，通过柱形的高低来表达数据的大小。
它适用于数据的对比，在整体中也能看到数据的变化趋势。
'''
# df.set_index('name', inplace=True)

df.plot.bar()
df.plot.barh()  # 横向
df[:5].plot.bar(x='name', y='Q4')  # 指定x、y轴 || name列不能设定为索引列
df[:5].plot.bar(x='name', y=['Q1', 'Q2'])  # 指定x、y轴
# 合并逻辑
(
    df.head()
    .set_index('name')  # 设为索引
    .plot
    .bar()
)

'''
Series索引为x轴，值为y轴，有值为非数字时会报错。与折线图一
样，如果数据中有负值，则0刻度会在x轴之上，不会在图形底边
'''
(
    df.assign(Q1=df.Q1 - 70)  # 让Q1产生部分负值
    .loc[:6]  # 取部分
    .set_index('name')  # 设为索引
    .plot
    .barh()
)

'''可以将同一索引的多个数据堆叠起来'''

(
    df.loc[:6]
    .set_index('name')
    .plot
    .bar(stacked=True)  # 柱状图，堆叠
)

# 柱状图，横向+堆叠
(
    df.loc[:6]
    .set_index('name')
    .plot
    .barh(stacked=True)  # 柱状图，横向+堆叠
)

# 柱状图 子图
(
    df.set_index('name')
    .plot
    .bar(subplots=True)
)

# 横向柱状图 子图
(
    df.set_index('name')
    .plot
    .barh(subplots=True)
)

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.2.4 直方图plot.hist')
# update20240716
'''
直方图（Histogram）又称质量分布图，由一系列高度不等的纵向条纹或线段表示数据分布的情况。
一般用横轴表示数据类型，纵轴表示分布情况。
'''
# df.set_index('name', inplace=True)

df2 = pd.DataFrame({'a': np.random.randn(1000) + 1,
                    'b': np.random.randn(1000),
                    'c': np.random.randn(1000) - 1,
                    },
                   columns=['a', 'b', 'c']
                   )

# print(df2)
df2.plot.hist(alpha=0.1)  # alpha参数调控图形透明度，值越大透明度越小。

# 单直方图
df2.a.plot.hist(alpha=0.5)

# 堆叠，指定分箱数量
df2.plot.hist(stacked=True, bins=20)

# 绘制子图
df.hist(alpha=0.5)

# 可以单独绘制子图，指定分箱数量：
df2.a.hist(bins=20, alpha=0.5)
df2.hist('a', bins=20, alpha=0.5)  # 同上

'''by参数可用来进行分组，生成分组后的子图'''
# 分组
df.Q1.hist(by=df.team)

'''还可以传递matplotlib hist支持的其他关键字，详情请参考Matplotlib官方文档。'''
plt.show()

print()
print('------------------------------------------------------------')
print('\t16.2.5 箱形图plot.box')
# update20240717
'''
箱形图（Box Chart）又称盒形图、盒式图或箱线图，是一种用来显示一组数据分布情况的统计图。

从箱形图中我们可以观察到：
    一组数据的关键值，如中位数、最大值、最小值等；
    数据集中是否存在异常值，以及异常值的具体数值；
    数据是不是对称的；
    这组数据的分布是否密集、集中；
    数据是否扭曲，即是否有偏向性。

'''
# df.set_index('name', inplace=True)

df.plot.box()  # 所有列

# 其他的一些方法如下：
df.Q1.plot.box()  # 单列
df.boxplot()  # 自带背景
df.boxplot('Q1')

print('为图形中的一些元素指定颜色，')

color = {'boxes': 'DarkGreen',  # 箱体颜色
         'whiskers': 'DarkOrange',  # 连线颜色
         'medians': 'DarkBlue',  # 中位数颜色
         'caps': 'Gray'  # 极值颜色
         }

df.plot.box(color=color, sym='r+')

print('# 横向+位置调整')
df.plot.box(vert=False, positions=[1, 2, 5, 6])
df.plot.box(vert=False)

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.2.6 面积图plot.area')
# update20240717
'''
面积图（Area Chart）又叫区域图。将折线图中折线与自变量坐标
轴之间的区域使用颜色或纹理填充，这样的填充区域就叫作面积。

默认情况下，面积图是堆叠的。
要生成堆积面积图，每列必须全部为正值或全部为负值。
当输入数据包含NaN时，它将被自动填充0。
如果要删除或填充不同的值，请在调用图之前使用DataFrame.dropna()或DataFrame.fillna()。

'''
# df.set_index('name', inplace=True)

df4 = pd.DataFrame(np.random.randn(10, 4), columns=['a', 'b', 'c', 'd'])

df4 = df4.abs()  # 默认情况下，面积图是堆叠的。要生成堆积面积图，每列必须全部为正值或全部为负值。
df4.a.plot.area()  # 单列
print(df4)
df4.plot.area(alpha=0.5)

print('# 未堆积')
# df4.plot.area(stacked=False)

'''
ValueError: When stacked is True, each column must be either all positive or all negative. 
Column 'a' contains both positive and negative values.

这个错误是由于 plot.area() 方法默认情况下会堆叠（stack）各列的数据，
而堆叠时，要求每列数据要么全为正值，要么全为负值。如果某列数据同时包含正值和负值，就会引发这个错误。
'''

print('指定x轴和y轴：')
# df4.plot.area(y='a')  # 同单列
# df4.plot.area(y=['b', 'c'])
df4.plot.area(x='a')

plt.show()

print()
print('------------------------------------------------------------')
print('\t16.2.7 散点图plot.scatter')
# update20240717
'''
散点图（Scatter Graph）也叫x-y图，它将所有的数据以点的形式展现在直角坐标系上，
以显示变量之间的相互影响程度，点的位置由变量的数值决定。

散点图要求x轴和y轴为数字列，这些可以通过x和y关键字指定。

'''
# df.set_index('name', inplace=True)
print('Q1成绩与平均分的散点图')

(
    df.assign(avg=df.select_dtypes('number').mean(1))  # 增加一列平均分
    .plot
    # .scatter()  # TypeError: PlotAccessor.scatter() missing 2 required positional arguments: 'x' and 'y'
    .scatter(x='Q1', y='avg')  # success
)

'''通过散点图发现，学生Q1的成绩与平均成绩呈现一定的相关性。'''

print('# 指定颜色')

# df.plot.scatter(x='Q1', y='Q2', c='b', s=50)
'''
c可以取以下值：
    字符，RGB或RGBA码，如red、#a98d19；
    序列，颜色列表，对应每个点的颜色；
    列名称或位置，其值将用于根据颜色图为标记点着色。
'''

df.plot.scatter(x='Q1', y='Q2', c=['green', 'yellow'] * 3, s=50)
df.plot.scatter(x='Q1', y='Q2', c='DarkBlue')
df.plot.scatter(x='Q1', y='Q2', c='Q1', colormap='viridis')
df.plot.scatter(x='Q1', y='Q2', c='Q1', colormap='coolwarm')

'''
传入参数colorbar=True会在当前坐标区或图的右侧显示一个垂直颜色栏，
颜色栏显示当前颜色图并指示数据值到颜色图的映射。
'''
# 色阶栏
df.plot.scatter(x='Q1', y='Q2', c='DarkBlue', colorbar=True)
# df.plot.scatter(x='Q1', y='Q2', c='coolwarm', colorbar=True)  # error
# df.plot.scatter(x='Q1', y='Q2', c='DarkBlue', colormap='coolwarm')  # error

print('s用于指定点的大小：')
df.plot.scatter(x='Q1', y='Q2', s=df['Q1'] * 10)  # 值越大 点越大
df.plot.scatter(x='Q1', y='Q2', s=50)  # 每个点是同样大小


print()
print('------------------------------------------------------------')
print('\t16.2.8 六边形分箱图plot.hexbin')
# update20240717
'''
六边形分箱图（Hexagonal Binning）也称六边形箱体图，或简称六边形图，它是一种以六边形为主要元素的统计图表。
它比较特殊，既是散点图的延伸，又兼具直方图和热力图的特征。

使用DataFrame.plot.hexbin()创建六边形图。
如果你的数据过于密集而无法单独绘制每个点，则可以用六边形图替代散点图。
'''
# df.set_index('name', inplace=True)
# print('Q1成绩与平均分的散点图')
df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])
df['b'] = df['b'] + np.arange(1000)

print(df)

df.plot.hexbin(x='a', y='b')  # 默认散点尺寸，较小
df.plot.hexbin(x='a', y='b', gridsize=25)

plt.show()

'''
16.2.9 小结
本节介绍了数据分析中最为常见的可视化图形，基本涵盖了大多数数据可视化场景，需要熟练掌握本节内容。
这些功能与Matplotlib紧密相连，如果想了解更为高级的用法，可以到Matplotlib官网查看文档进行学习。
'''