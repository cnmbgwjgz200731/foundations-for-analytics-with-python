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
