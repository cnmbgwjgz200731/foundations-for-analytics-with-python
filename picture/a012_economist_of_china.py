#  17.3.1 中国经济发展分析

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

temp_file = 'E:/bat/input_files/annual.xlsx'

df = pd.read_excel(temp_file, skiprows=2, skipfooter=1, index_col='指标')

df = \
    (
        df.T
        .reset_index()
        # .index
        # .columns
        .rename(columns={'index': '年份'})
        # .年份.apply(lambda s: s.replace('年','')).astype('int')
    )

df.年份 = df.年份.apply(lambda s: s.replace('年', '')).astype('int')

df.columns = df.columns.str.replace(r'(亿元)', '').str.replace('(元)', '')

print(df)

# print(df.set_index('年份').loc[:,'国内生产总值'])

df_gdp = (
    df.set_index('年份')
    .loc[:, '国内生产总值']
    .plot()
)

# plt.show()

# print(df_gdp)

# TODO 在三个产业增长趋势方面，第三产业和第二产业增长迅猛
df_three = (
    df.set_index('年份')
    .loc[:, '第一产业增加值':'第三产业增加值']
    .plot()

)

# plt.show()
# print(df_three)

# TODO 在第一产业占比趋势方面，第一产业的占比越来越低

df_rate = (df.assign(rate=df.第一产业增加值 / df.国内生产总值)
           .set_index('年份')
           .rate
           .plot()
           )

# print(df_rate)

# plt.show()

# TODO 在2000年前后新增GDP总量方面，可以看到绝大部分GDP是在2000。年以后产生的

df_four = (
    df.groupby(df.年份 >= 2000)
    .sum()
    .rename(index={True: "2000年以后", False: "2000年以前"})
    .国内生产总值
    .plot
    .pie()
)

# print(df_four)
# plt.show()

# TODO: 最后，我们计算出每五年的GDP之和：

df_five = (
    df.groupby(pd.cut(df.年份,
                      bins=[i for i in range(1952, 2024, 5)],
                      right=False
                      ),
               observed=False
               )
    .sum()
    .国内生产总值
    .sort_values(ascending=False)

)

print(df_five)
'''
年份
[2017, 2022)    4900636.2
[2012, 2017)    3210359.6
[2007, 2012)    1837914.1
[2002, 2007)     827737.0
[1997, 2002)     466618.1
[1992, 1997)     244658.7
[1987, 1992)      85413.2
[1982, 1987)      38147.9
[1977, 1982)      20552.6
[1972, 1977)      14164.4
[1967, 1972)      10237.1
[1962, 1967)       7503.1
[1957, 1962)       6533.6
[1952, 1957)       4305.6
Name: 国内生产总值, dtype: float64
'''