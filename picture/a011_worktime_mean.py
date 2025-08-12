import pandas as pd
import numpy as np
from io import StringIO

# 17.2.11 计算平均打卡上班时间
"""
某员工一段时间上班打卡的时间记录如下，现在需要计算他在这期间的平均打卡时间。
"""

# 一周打卡时间记录
ts = \
    '''
2020-10-28 09:59:44
2020-10-29 10:01:32
2020-10-30 10:04:27
2020-11-02 09:55:43
2020-11-03 10:05:03
2020-11-04 09:44:34
2020-11-05 10:10:32
2020-11-06 10:02:37
'''
# print(ts)

"""
首先读取数据，并将数据类型转为时间类型，然后计算时间序列的平均值。

下列代码中的StringIO将字符串读入内存的缓冲区，
read_csv的parse_dates参数传入需要转换时间类型的列名：
"""

# 读取数据，类型设置为时间类型
df = pd.read_csv(StringIO(ts), names=['time'], parse_dates=['time'])

print(df)
print(df.dtypes)

# 对时间序列求平均
print(df.time.mean())  # 2020-11-02 04:00:31.500000

"""
我们发现，mean方法会对时间序列的时间戳求平均值，得出的值为11月2日凌晨4点，
这和我们的需求不符，因为我们不需要关心具体是哪天，只关注时间。

正确的做法是将日期归到同一天，再求平均时间。
时间的replace方法可以实现这个功能，结合函数的调用方法，
有以下三种办法可以实现同样的效果：
"""
# 将时间归为同一天，再求平均时间
# df.time.apply(lambda s: s.replace(year=2020, month=1, day=1)).mean()
# df.time.apply(pd.Timestamp.replace, year=2020, month=1, day=1).mean()
# df.time.agg(pd.Timestamp.replace, year=2020, month=1, day=1).mean()
# Timestamp('2020-01-01 10:00:31.500000')
print()
# df_mean = df.time.apply(lambda s: s.replace(year=2020, month=1, day=1))  # success
df_mean = df.time.apply(pd.Timestamp.replace, year=2020, month=1, day=1)  # success
# df_mean = df.time.agg(pd.Timestamp.replace, year=2020, month=1, day=1)  # success

print(df_mean)
print(df_mean.mean())  # 2020-01-01 10:00:31.500000
