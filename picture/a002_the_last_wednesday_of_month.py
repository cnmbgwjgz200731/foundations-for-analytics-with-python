# update20240725
# 17.2.2 当月最后一个星期三

"""本需求为给出一个日期，得到这个日期所在月份的最后一个星期三是哪天。"""

import pandas as pd

t = pd.Timestamp('2024-06-15')

t = t.replace(day=1)

"""用pd.date_range()构造出这个月的所有日期，结束时间取这个月的月底："""
index = pd.date_range(start=t,
                      end=(t + pd.offsets.MonthEnd()))
# print(t)
# print(index)

df = pd.DataFrame(index.weekday + 1, index=index.date, columns=['weekday'])
# df.set_axis(['date'], axis='columns', inplace=True)
# df.reset_index(inplace=True)

# df.index.rename('date', inplace=True)
df.index.set_names('date', inplace=True)

# 给定日期所在月的最后一个星期三
print(df.query('weekday==3')
      .tail(1)
      .index[0])


