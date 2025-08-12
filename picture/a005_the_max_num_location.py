import pandas as pd

# 17.2.5 全表最大值的位置
"""
在解决最强相关性的问题中，核心的工作就是找到DataFrame中最大值的标签，
除了以上方法，还有没有其他办法可以完成这个工作呢？
"""

df = pd.DataFrame({
    'A': [1, 2, 4, 5, -6],
    'B': [2, -1, 8, 2, 1],
    'C': [2, -1, 8, 2, 1]
},
    index=['x', 'y', 'z', 'h', 'j']
)

print(df)

"""先取到DataFrame中的最大值："""
print()
print(df.max().max())  # 得到全局最大值
'''8'''

print()
"""查出最大值，返回的DataFrame中非最大值的值都显示为NaN："""
print(df[df == df.max().max()])
'''
    A    B    C
x NaN  NaN  NaN
y NaN  NaN  NaN
z NaN  8.0  8.0
h NaN  NaN  NaN
j NaN  NaN  NaN
'''

"""将全为空的行和列删除："""
print()
# 找到最大值索引位
df_f = (
    df[df == df.max().max()]
    .dropna(how='all')  # 删除全为空的行
    .dropna(how='all', axis=1)  # 删除全为空的列
)
print(df_f)
'''
     B    C
z  8.0  8.0
'''

"""可见有两个最大值，在同一行的两列中，最后用axes得到轴信息："""
print()
# 找到最大值索引位
df_t = (
    df[df == df.max().max()]
    .dropna(how='all')  # 删除全为空的行
    .dropna(how='all', axis=1)  # 删除全为空的列
    .axes
)
print(df_t)
'''[Index(['z'], dtype='object'), Index(['B', 'C'], dtype='object')]'''

"""
这样，我们用另一种方法确定了最大值在DataFrame中的位置。
这可能不是最优解，但为我们提供了另一个思路，帮助我们熟悉了相关方法的用法。
"""