"""17.3.2 新冠肺炎疫情分析"""

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.3, 5.6)
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


covid_file = 'E:/bat/input_files/countries-aggregated.csv'

df = pd.read_csv(covid_file)

# print(df.tail())
# print(df.columns)
# print(len(df))

df.columns = [i.lower() for i in df.columns]
# print(df.columns)
# print(df)

"""
Date：日期。
Country：国家。
Confirmed, Recovered, Deaths当日累计确诊、康复、死亡人数。
"""

"""首先来看一下中国累计确诊人数趋势，可见爆发之初是快速上升的"""


# china_df = (
#     df.loc[df.country == 'China']  # 只选中国的
#     .set_index('date')  # 日期为索引
#     .confirmed  # 看确诊的
#     .plot()
# )

# print(china_df)
# plt.show()

print()
# -------------------------------
"""再看中国新增确诊趋势，在2020年2月初有一个确诊增加高峰."""

# china_newadd_df = (
#     df.loc[df.country == 'China']  # 只选中国的
#     .set_index('date')  # 日期为索引
#     .confirmed  # 看确诊的
#     # .assign(dif=df.confirmed.diff(1))  # AttributeError: 'Series' object has no attribute 'assign'. Did you mean: 'align'?
#     .diff()
#     .bfill()
#     .plot()
# )

# print(df.loc[df.country == 'China'])
# print(china_newadd_df)

print()
# -------------------------------
"""找出确诊病例在1万以上的国家中死亡率排名前十的国家"""

# death_rate = (
#     df.query('confirmed > 10_000')
#     .assign(rate=df.deaths/df.confirmed)
#     .groupby('country').max()
#     .sort_values(by='rate')
#     .tail(10)
#     .rate
#     .plot
#     .barh(title="图17-12 新冠疫情确诊上万国家死亡率前10")
# )

# print(death_rate)


print()
# -------------------------------
"""中美两国新冠肺炎确诊病例数量趋势"""

# ch_us_df = (
#     df.query("country == ['China', 'US']")
#     .loc[:, ['date', 'country', 'confirmed']]
#     .groupby(['date', 'country'])
#     .max()
#     # .sum('confirmed')
#     .unstack()
#     # .T
#     # .droplevel(0)
#     .plot()
# )

# print(ch_us_df)
# plt.show()

print()
# -------------------------------
"""中美两国新冠肺炎病例的死亡率对比"""

death_rate = (
    df.query("country in ['China', 'US']")
    .assign(death_rate=df.deaths/df.confirmed)
    .loc[:, ['date', 'country', 'death_rate']]
    .groupby(['date', 'country'])
    .max()
    .unstack()
    .plot()
)

print(death_rate)
plt.show()


# df.loc[(df.Country.isin(['China', 'US'])) & (df.Date == df.Date.max())