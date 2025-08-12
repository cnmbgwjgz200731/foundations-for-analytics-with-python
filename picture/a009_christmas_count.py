import pandas as pd
import matplotlib.pyplot as plt

# 17.2.9 圣诞节的星期分布

"""
我们想知道圣诞节在星期几多一些，针对这个问题，可以抽样近100年的圣诞节进行分析。
基本思路如下：
    用pd.date_range生成100年日期数据；
    筛选出12月25日的所有日期；
    将日期转换为所在星期几的数字；
    统计数字重复值的数量；
    绘图观察；
    得出结论。
接下来编写代码。第一步找到所有的圣诞节日期：
"""
# ts = pd.Timestamp('now').ceil('d')
ts = pd.Timestamp('2021-12-31').ceil('d')
# pd.Timestamp
# date_code = pd.date_range(end='2025-12-25', periods=100, freq='Y')
date_code = pd.date_range(end=ts, periods=100, freq='Y')
# print(date_code)

df = pd.DataFrame({'day_code': date_code})
# print(df.replace(day=25))
# print(df.assign(week_day=df.day_code.dt.weekday))
print(df)
print(df.day_code.dtype)
# print(df.replace(day=25))


df = df.assign(christmas_day= df.day_code - pd.Timedelta(days=6))
# df = df.assign(christmas_day= df.day_code.dt.replace(day=25))  # error
# df = df.assign(christmas_day=df.day_code.apply(lambda x: x.replace(day=25)))  # success

df = df.assign(week_day=df.christmas_day.dt.weekday + 1)

print(df)
print(df.groupby('week_day').count())

# df.groupby('week_day').count().plot()
# df.iloc[:, [1, 2]].groupby('week_day').count().plot.bar()  # SUCCESS
df.groupby('week_day').day_code.count().plot.bar()  # success

plt.show()

