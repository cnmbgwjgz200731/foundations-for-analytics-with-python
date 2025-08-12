# 17.3.4 全国城市房价分析

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8.0, 5.0)
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_clipboard()

print(df)
print(df.dtypes)

df = (
    # 去掉千分位符并转为整数
    df.assign(平均单价=df['平均单价（元/㎡）'].str.replace(',', '').astype(int))  # 去掉千位符并转为整数
    .assign(同比=df.同比.str[:-1].astype(float))  # 去掉百分号并转为浮点数
    .assign(环比=df.环比.str.replace('%', '').astype(float))
    .loc[:, ['城市名称', '平均单价', '同比', '环比']]
)

print()
# print(df)
# print(df.dtypes)

# df = (
#     df.set_index('城市名称')
#     .平均单价
#     .plot
#     .bar()
#
# )


# df = (
#     df.set_index('城市名称')
#     .loc[:, ['同比', '环比']]
#     .plot
#     .bar(title='Top10各城市平均房价同比与环比')
# )

# 条形图
# (
#     df.style
#     .bar(subset=['平均单价'], color='yellow')
#     # .bar(subset=['平均单价'], color='yellow')
# )

# 数据样式综合

# (
#     df.style
#     .background_gradient(subset=['平均单价'], cmap='BuGn')
#     .format({'同比':"{:2}%", '环比':"{:2}%"})
#     .bar(
#         # subset=['同比','环比'],
#          subset=['同比'],
#          color=['#ffe4e4','#bbf9ce'],  # 上涨，下降的颜色
#          # vmin=0,vmax=15,  # 范围定为以0为基准的上下15
#          align='zero'
#         )
#     .bar(
#         subset=['环比'],
#         color=['red','green'],
#         # vmin=0, vmax=11,
#         align='zero'
#     )
# )

plt.show()