import pandas as pd

"""
需求如下，将A、B两列组合进行分组，同组内的数据显示在同一行，有多少条数据就放多少列：
"""
# 源数据
df = pd.DataFrame({'A': ['a', 'a', 'a', 'a', 'a', 'a', 'a'],
                   'B': ['b1', 'b1', 'b1', 'b2', 'b2', 'b2', 'b2'],
                   'C': ['c', 'c', 'c', 'c', 'c', 'c', 'c'],
                   'D': ['2001', '2003', '2005', '2001', '2002', '2003', '2004']})

# print(df.groupby(['A', 'B', 'C']).sum())

df_pivot = df.pivot(index=['A', 'B', 'C'], columns='D', values='D')

print(df_pivot)
print()


df_p2 = (
        df
        .pivot(index=['A', 'B', 'C'], columns='D', values='D')
        # .apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
        .apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
        .str.split(',', expand=True)
        )
print(df_p2)

'''
# 所需效果如下： 

           0     1     2     3
A B  C                        
a b1 c  2001  2003  2005  None
  b2 c  2001  2002  2003  2004
'''

"""
# 修改 列名称
(
    df
    .pivot(index=['A', 'B', 'C'], columns='D', values='D')
    # .apply(lambda x: ','.join(x.D.astype(str)))  # AttributeError: 'Series' object has no attribute 'D'
    .apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
    .str.split(',', expand=True)
    # .rename(columns={0:'年','1':'月','2':'日','3':'季度'})  # 数字类型 加引号 修改名称无效
    .rename(columns={0:'年',1:'月',2:'日',3:'季度'})

)
"""

