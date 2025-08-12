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

