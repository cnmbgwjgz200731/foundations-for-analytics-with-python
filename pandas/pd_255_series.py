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
print(pd.json_normalize(df.col))  # 运行失败
print(df.col.apply(pd.Series))  # 运行失败
