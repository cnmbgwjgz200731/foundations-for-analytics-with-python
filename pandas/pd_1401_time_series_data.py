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
print('第14章 Pandas时序数据')
print('\t14.1 固定时间')
# update20240603
'''
本章将全面介绍Pandas在时序数据处理中的方法，
主要有时间的概念、时间的计算机表示方法、时间的属性操作和时间格式转换；
时间之间的数学计算、时长的意义、时长与时间的加减操作；
时间偏移的概念、用途、表示方法；时间跨度的意义、表示方法、单位转换等。
'''
# 小节注释
'''
本节介绍一些关于时间的基础概念，帮助大家建立对时间的表示方式和计算方式的一个简单认知。

▶ 以下是一些具体的使用方法举例：
'''
print()
print('------------------------------------------------------------')
print('\t14.1.1 时间的表示')

'''
固定时间是指一个时间点，如2020年11月11日00:00:00。
固定时间是时序数据的基础，一个固定时间带有丰富的信息，
如年份、周几、几月、哪个季度，需要我们进行属性的读取。
'''

from datetime import datetime

print('# 当前时间')
print(datetime.now())  # 当前行和上一行累加，上一行不存在 则结果为空值！
# 2024-06-03 10:55:17.711633
print('# 指定时间')
print(datetime(2020, 11, 1, 19))
# 2020-11-01 19:00:00
print(datetime(year=2020, month=11, day=11))
# 2020-11-11 00:00:00

# 个人测试
# print(datetime(2024,5,20,00,00,00))
# print(datetime(year=2024,month=5,day=20,hour=5,minute=20,second=00))

print()
print('------------------------------------------------------------')
print('\t14.1.2 创建时间点')

'''
pd.Timestamp()是Pandas定义时间的主要函数，代替Python中的datetime.datetime对象。下面介绍它可以传入的内容。
'''
import datetime

print('# 至少需要年、月、日')
print(pd.Timestamp(datetime.datetime(2020, 6, 8)))
# 2020-06-08 00:00:00
print('# 指定时、分、秒')
print(pd.Timestamp(datetime.datetime(2020, 6, 8, 16, 17, 18)))
# 2020-06-08 16:17:18

'''指定时间字符串：'''
print()
print(pd.Timestamp('2012-05-01'))
print(type(pd.Timestamp('2012-05-01')))
# 2012-05-01 00:00:00
# <class 'pandas._libs.tslibs.timestamps.Timestamp'>
print(pd.Timestamp('2017-01-01T12'))
# 2017-01-01 12:00:00
print(pd.Timestamp('2017-01-01T12:05:59'))
# 2017-01-01 12:05:59

'''指定时间位置数字，可依次定义year、month、day、hour、
minute、second、microsecond：'''

print(pd.Timestamp(2020, 5, 1))  # 2020-05-01 00:00:00
print(pd.Timestamp(2017, 1, 1, 12))  # 2017-01-01 12:00:00
print(pd.Timestamp(year=2017, month=1, day=1, hour=12))  # 2017-01-01 12:00:00

print('# 解析时间戳：')
print(pd.Timestamp(1513393355.5, unit='s'))  # 单位为秒 2017-12-16 03:02:35.500000
# print(pd.Timestamp(1513393355.5,unit='h')) # 测试 结果同上 why？

'''用tz指定时区，需要记住的是北京时间值为Asia/Shanghai：'''
print(pd.Timestamp(1513393355, unit='s', tz='US/Pacific'))  # 2017-12-15 19:02:35-08:00
print(pd.Timestamp(1513393355, unit='s'))  # 2017-12-16 03:02:35
print(pd.Timestamp(1513393355, unit='s', tz='Asia/Shanghai'))  # 2017-12-16 11:02:35+08:00
print()
print(pd.Timestamp(datetime.datetime.now()))  # 2024-06-03 13:27:04.434208
print(pd.Timestamp(datetime.datetime.now(), unit='s'))  # 同上
print(pd.Timestamp(datetime.datetime.now(), unit='s', tz='Asia/Shanghai'))  # 2024-06-03 13:28:03.099779+08:00

print('获取到当前时间，从而可通过属性取到今天的日期、年份等信息：')
print(pd.Timestamp('today'))  # 2024-06-03 13:29:14.516242
print(pd.Timestamp('now'))  # 同上
print(pd.Timestamp('today').date())  # 只取日期 2024-06-03

print('通过当前时间计算出昨天、明天等信息：')
# # 昨天
print(pd.Timestamp('now') - pd.Timedelta(days=1))  # 2024-06-02 14:04:42.100899
print(pd.Timedelta(days=1))  # 1 days 00:00:00
# 明天
print(pd.Timestamp('now') + pd.Timedelta(days=1))  # 2024-06-04 14:06:26.799704
# 当月初，一日
print(pd.Timestamp('now').replace(day=1))  # 2024-06-01 14:07:12.629509

'''pd.to_datetime()也可以实现上述功能，不过根据语义，它常用在时间转换上。'''
print()
print(pd.to_datetime('now'))  # 2024-06-03 14:14:45.508085
'''
由于Pandas以纳秒粒度表示时间戳，因此可以使用64位整数表示的时间跨度限制为大约584年，
意味着能表示的时间范围有最早和早晚的限制：
'''
print(pd.Timestamp.min)  # 1677-09-21 00:12:43.145224193
print(pd.Timestamp.max)  # 2262-04-11 23:47:16.854775807

'''
不过，Pandas也给出一个解决方案：使用PeriodIndex来解决。PeriodIndex后面会介绍。
'''

print()
print('------------------------------------------------------------')
print('\t14.1.3 时间的属性')

'''
一个固定的时间包含丰富的属性，包括时间所在的年份、月份、周几，是否月初，在哪个季度等。
利用这些属性，我们可以进行时序数据的探索。
我们先定义一个当前时间：
'''
time = pd.Timestamp('now')

'''以下是丰富的时间属性：'''
print(time.tz)
'''
time.asm8 # 返回NumPy datetime64格式（以纳秒为单位）
# numpy.datetime64('2020-06-09T16:30:54.813664000') || # 2024-06-03T14:23:26.961411000
time.dayofweek # 1（周几，周一为0）
time.dayofyear # 161（一年的第几天）  # 155
time.days_in_month # 30（当月有多少天）
time.daysinmonth # 30（同上）
time.freqstr # None（周期字符）
time.is_leap_year # True（是否闰年，公历的）
time.is_month_end # False（是否当月最后一天）
time.is_month_start # False（是否当月第一天）
time.is_quarter_end # False（是否当季最后一天）
time.is_quarter_start # False（是否当季第一天）
time.is_year_end # 是否当年最后一天
time.is_year_start # 是否当年第一天
time.quarter # 2（当前季度数）
# 如指定，会返回类似<DstTzInfo 'Asia/Shanghai' CST+8:00:00 STD>
time.tz # None（当前时区别名）
time.week # 24（当年第几周）
time.weekofyear # 24（同上）
time.day # 9（日）
time.fold # 0
time.freq # None（频度周期）
time.hour # 16
time.microsecond # 890462
time.minute # 46
time.month # 6
time.nanosecond # 0
time.second # 59
time.tzinfo # None
time.value # 1591721219890462000
time.year # 2020
'''

print()
print('------------------------------------------------------------')
print('\t14.1.4 时间的方法')

'''
可以对时间进行时区转换、年份和月份替换等一系列操作。我们取当前时间，并指定时区为北京时间：
'''
time = pd.Timestamp('now', tz='Asia/Shanghai')
print('# 转换为指定时区')
print(time.astimezone('UTC'))
# 2024-06-03 06:32:04.555630+00:00
print('# 转换单位，向上舍入')
print(time.ceil('s'))  # 2024-06-03 14:34:19+08:00
print(time.ceil('ns'))  # 转为以纳秒为单位 || 2024-06-03 14:34:40.371953+08:00
print(time.ceil('d'))  # 保留日 || 2024-06-04 00:00:00+08:00
print(time.ceil('h'))  # 保留时 || 2024-06-03 15:00:00+08:00

print('# 转换单位，向下舍入')
print(time.floor('h'))  # 保留时 | 2024-06-03 14:00:00+08:00
print(time.floor(freq='H'))  # 结果同上
print(time.floor('5T'))  # 2024-06-03 14:40:00+08:00
print('# 类似四舍五入')
print(time.round('h'))  # 保留时 |2024-06-03 15:00:00+08:00
print('# 返回星期名')
print(time.day_name())  # Monday
print('# 月份名称')
print(time.month_name())  # June
print('# 将时间戳规范化为午夜，保留tz信息')
print(time.normalize())  # 2024-06-03 00:00:00+08:00

print()
print('# 将时间元素替换datetime.replace，可处理纳秒')
print(time.replace(year=2019))  # 年份换为2019年
# 2019-06-03 14:50:47.318516+08:00
print(time.replace(month=8))  # 月份换为8月
# 2024-08-03 14:51:22.882376+08:00

print()
print('# 转换为周期类型，将丢失时区')
print(time.to_period(freq='h'))  # 2024-06-03 14:00
print('# 转换为指定时区')
print(time.tz_convert('UTC'))  # 转为UTC时间
# print(time)
# 2024-06-03 06:53:15.079697+00:00
print('# 本地化时区转换')
time = pd.Timestamp('now')  # 必须增加 否则下行代码运行失败
# print(time)
print(time.tz_localize('Asia/Shanghai'))

'''
对一个已经具有时区信息的时间戳再次进行本地化（tz_localize），这是不被允许的.
对于已经具有时区信息的时间戳，你应该使用tz_convert来转换时区
'''

print()
print('------------------------------------------------------------')
print('\t14.1.5 时间缺失值')
'''对于时间的缺失值，有专门的NaT来表示：'''
print(pd.Timestamp(pd.NaT))  # NaT
print(pd.Timedelta(pd.NaT))  # NaT
print(pd.Period(pd.NaT))  # NaT

print('# 类似np.nan')
print(pd.NaT == pd.NaT)  # False

'''
NaT可以代表固定时间、时长、时间周期为空的情况，类似于
np.nan可以参与到时间的各种计算中：
'''
print(pd.NaT + pd.Timestamp('20201001'))  # NaT
print(pd.NaT + pd.Timedelta('2 days'))  # NaT

'''
14.1.6 小结
时间序列是由很多个按照一定频率的固定时间组织起来的。
Pandas借助NumPy的广播机制，对时间序列进行高效操作。
因此熟练掌握时间的表示方法和一些常用的操作是至关重要的。
'''

print()
print('------------------------------------------------------------')
print('第14章 Pandas时序数据')
print('\t14.2 时长数据')
# update20240603
'''
本章将全面介绍Pandas在时序数据处理中的方法，
主要有时间的概念、时间的计算机表示方法、时间的属性操作和时间格式转换；
时间之间的数学计算、时长的意义、时长与时间的加减操作；
时间偏移的概念、用途、表示方法；时间跨度的意义、表示方法、单位转换等。
'''
# 小节注释
'''
前面介绍了固定时间，如果两个固定时间相减会得到什么呢？时间差或者时长。
时间差代表一个时间长度，它与固定时间已经没有了关系，没有指定的开始时间和结束时间，

▶ 以下是一些具体的使用方法举例：
'''
print()
print('------------------------------------------------------------')
print('\t14.2.1 创建时间差')

'''
pd.Timedelta()对象表示时间差，也就是时长，以差异单位表示，
例如天、小时、分钟、秒等。它们可以是正数，也可以是负数。
'''

from datetime import datetime

print('# 两个固定时间相减')
print(pd.Timestamp('2020-11-01 15') - pd.Timestamp('2020-11-01 14'))
# 0 days 01:00:00
print(pd.Timestamp('2020-11-01 08') - pd.Timestamp('2020-11-02 08'))
# -1 days +00:00:00

'''按以下格式传入字符串：'''
print('# 一天')
print(pd.Timedelta('1 days'))  # 1 days 00:00:00
print(pd.Timedelta('1 days 00:00:00'))  # 1 days 00:00:00
print(pd.Timedelta('1 days 2 hours'))  # 1 days 02:00:00
print(pd.Timedelta('-1 days 2 min 3us'))  # -2 days +23:57:59.999997

'''用关键字参数指定时间：'''
print(pd.Timedelta(days=5, seconds=10))  # 5 days 00:00:10
print(pd.Timedelta(minutes=3, seconds=2))  # 0 days 00:03:02

print('# 可以将指定分钟转换为天和小时')
print(pd.Timedelta(minutes=3242))  # 2 days 06:02:00

print('使用带周期量的偏移量别名：')
# 一天
print(pd.Timedelta('1D'))  # 1 days 00:00:00
# 两周
print(pd.Timedelta('2W'))  # 14 days 00:00:00
# 一天零2小时3分钟4秒
# print(pd.Timedelta('1D 2H 3M 4S')) # 报错！
print(pd.Timedelta('1D 2H 3m 4S'))  # 1 days 02:03:04
print(pd.Timedelta('1D2H3m4S'))  # 同上

print('带单位的整型数字：')
# 一天
print(pd.Timedelta(1, unit='d'))  # 1 days 00:00:00
# 100秒
print(pd.Timedelta(100, unit='s'))  # 0 days 00:01:40
# 4周
print(pd.Timedelta(4, unit='w'))  # 28 days 00:00:00

'''使用Python内置的datetime.timedelta或者NumPy的np.timedelta64：'''

import datetime
import numpy as np

print('# 一天零10分钟')
print(datetime.timedelta(days=1, minutes=10))  # 1 day, 0:10:00
print(pd.Timedelta(datetime.timedelta(days=1, minutes=10)))  # 1 days 00:10:00
# 100纳秒
print(pd.Timedelta(np.timedelta64(100, 'ns')))  # 0 days 00:00:00.000000100

print('负值：')
print(pd.Timedelta('-1min'))  # -1 days +23:59:00
print('# 空值，缺失值')
print(pd.Timedelta('nan'))  # NaT
print(pd.Timedelta('nat'))  # NaT

print('# ISO 8601 Duration strings')
# 标准字符串（ISO 8601 Duration strings）：
print(pd.Timedelta('P0DT0H1M0S'))  # 0 days 00:01:00
print(pd.Timedelta('P0DT0H0M0.000000123S'))  # 0 days 00:00:00.000000123
'''
使用时间偏移对象DateOffsets (Day, Hour, Minute, Second, Milli,Micro, Nano)直接创建：
'''
print('# 两分钟')
print(pd.Timedelta(pd.offsets.Minute(2)))  # 0 days 00:02:00
print(pd.Timedelta(pd.offsets.Day(3)))  # 3 days 00:00:00

'''
另外，还有一个pd.to_timedelta()可以完成以上操作，
不过根据语义，它会用在时长类型的数据转换上。
'''
print()
print(pd.to_timedelta(pd.offsets.Day(3)))  # 3 days 00:00:00
print(pd.to_timedelta('15.5min'))  # 0 days 00:15:30
print(pd.to_timedelta(124524564574835))  # 1 days 10:35:24.564574835

'''如时间戳数据一样，时长数据的存储也有上下限：'''
print(pd.Timedelta.min)  # -106752 days +00:12:43.145224193
print(pd.Timedelta.max)  # 106751 days 23:47:16.854775807

'''如果想处理更大的时长数据，可以将其转换为一定单位的数字类型。'''

print()
print('------------------------------------------------------------')
print('\t14.2.2 时长的加减')

'''时长可以相加，多个时长累积为一个更长的时长：'''
print('# 一天与5个小时相加')
print(pd.Timedelta(pd.offsets.Day(1)) + pd.Timedelta(pd.offsets.Hour(5)))
# 1 days 05:00:00
print('# 一天与5个小时相减')
print(pd.Timedelta(pd.offsets.Day(1)) - pd.Timedelta(pd.offsets.Hour(5)))
# 0 days 19:00:00

'''固定时间与时长相加或相减会得到一个新的固定时间：'''
print('# 11月11日减去一天')
print(pd.Timestamp('2024-11-11') - pd.Timedelta(pd.offsets.Day(1)))
# 2024-11-10 00:00:00
print('# # 11月11日加3周')
print(pd.Timestamp('2024-11-11') + pd.Timedelta('3W'))  # 2024-12-02 00:00:00

'''不过，此类计算我们使用时间偏移来操作，后面会介绍。'''

print()
print('------------------------------------------------------------')
print('\t14.2.3 时长的属性')

'''
时长数据中我们可以解析出指定时间计数单位的值，
比如小时、秒等，这对我们进行数据计算非常有用。
'''

tdt = pd.Timedelta('10 days 9 min 3 sec')
print(tdt)  # 10 days 00:09:03
print(tdt.days)  # 10
print(tdt.seconds)  # 543
print((-tdt).days)  # -11
print(tdt.value)  # 864543000000000

'''
14.2.4 时长索引
时长数据可以作为索引（TimedeltaIndex），它使用的场景比较少，
例如在一项体育运动中，分别有2分钟完成、4分钟完成、5分钟完成三类。
时长数据可能是完成人数、平均身高等。

14.2.5 小结
时长是两个具体时间的差值，是一个绝对的时间数值，没有开始和结束时间。
时长数据使用场景较少，但是它是我们在后面理解时间偏移和周期时间的基础。
'''

print()
print('------------------------------------------------------------')
print('\t14.3.1 时序索引')

'''
在时间序列数据中，索引经常是时间类型，我们在操作数据时经常
会与时间类型索引打交道，本节将介绍如何查询和操作时间类型索引。

DatetimeIndex是时间索引对象，一般由to_datetime()或date_range()来创建：

'''

import datetime

day_code = pd.to_datetime(['11/1/2020',  # 类时间字符串
                           np.datetime64('2020-11-02'),  # NumPy的时间类型
                           datetime.datetime(2020, 11, 3)  # Python自带时间类型
                           ])

print(day_code)
# DatetimeIndex(['2020-11-01', '2020-11-02', '2020-11-03'], dtype='datetime64[ns]', freq=None)

'''
date_range()可以给定开始或者结束时间，并给定周期数据、周期频率，
会自动生成在此范围内的时间索引数据：
'''
print('date_range()')
# 默认频率为天
print(pd.date_range('2020-01-01', periods=10))
# DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
#                '2020-01-05', '2020-01-06', '2020-01-07', '2020-01-08',
#                '2020-01-09', '2020-01-10'],
#               dtype='datetime64[ns]', freq='D')
print(pd.date_range('2020-01-01', '2020-01-10'))  # 结果同上
print(pd.date_range(end='2020-01-10', periods=10))  # 结果同上

'''pd.bdate_range()生成数据可以跳过周六日，实现工作日的时间索引序列：'''
print('# 频率为工作日')
print(pd.bdate_range('2024-06-01', periods=10))
# DatetimeIndex(['2024-06-03', '2024-06-04', '2024-06-05', '2024-06-06',
#                '2024-06-07', '2024-06-10', '2024-06-11', '2024-06-12',
#                '2024-06-13', '2024-06-14'],
#               dtype='datetime64[ns]', freq='B')
'''个人ps：无法跳过端午节 || 好像有参数可以操作！ '''

print()
print('------------------------------------------------------------')
print('\t14.3.2 创建时序数据')

'''
创建包含时序的Series和DataFrame与创建普通的Series和DataFrame一样，
将时序索引序列作为索引或者将时间列转换为时间类型。
'''

print('# 生成时序索引')
tidx = pd.date_range('2024-01-01', periods=5)
# 应用时序索引
s = pd.Series(range(len(tidx)), index=tidx)
print(s)
# 2024-01-01    0
# 2024-01-02    1
# 2024-01-03    2
# 2024-01-04    3
# 2024-01-05    4
# Freq: D, dtype: int64

'''如果将其作为Series的内容，我们会看到序列的数据类型为datetime64[ns]：'''
print(pd.Series(tidx))
# 0   2024-01-01
# 1   2024-01-02
# 2   2024-01-03
# 3   2024-01-04
# 4   2024-01-05
# dtype: datetime64[ns]

print('创建DataFrame：')
# 索引
tidx = pd.date_range('2024-1-1', periods=5)
# print(tidx)
df = pd.DataFrame({'A': range(len(tidx)), 'B': range(len(tidx))[::-1]}, index=tidx)
print(df)
#             A  B
# 2024-01-01  0  4
# 2024-01-02  1  3
# 2024-01-03  2  2
# 2024-01-04  3  1
# 2024-01-05  4  0

df1 = pd.DataFrame({'A': list(range(len(tidx))), 'B': list(range(len(tidx)))[::-1]}, index=tidx)
print(df1)  # 结果同上

print()
print('------------------------------------------------------------')
print('\t14.3.3 数据访问')

'''首先创建时序索引数据。以下数据包含2020年和2021年，以小时为频率：'''

idx = pd.date_range('1/1/2020', '12/1/2021', freq='H')
ts = pd.Series(np.random.randn(len(idx)), index=idx)
print(ts)
# 2020-01-01 00:00:00   -0.732463
# 2020-01-01 01:00:00   -1.235969
# 2020-01-01 02:00:00   -1.571011
# 2020-01-01 03:00:00    0.479507
# 2020-01-01 04:00:00    1.058393
#                          ...
# 2021-11-30 20:00:00    1.729897
# 2021-11-30 21:00:00   -0.211982
# 2021-11-30 22:00:00   -0.439532
# 2021-11-30 23:00:00   -0.130613
# 2021-12-01 00:00:00    0.541320
# Freq: H, Length: 16801, dtype: float64

'''查询访问数据时，和 []、loc等的用法一样，可以按切片的操作对数据进行访问，如：'''
print('# 指定区间的')
print(ts[5:10])
# 指定区间的
# 2020-01-01 05:00:00    0.340714
# 2020-01-01 06:00:00    1.132463
# 2020-01-01 07:00:00   -1.769089
# 2020-01-01 08:00:00   -1.752932
# 2020-01-01 09:00:00    1.112932
# Freq: H, dtype: float64
print('# 只筛选2020年的')
print(ts['2020'])
# 2020-01-01 00:00:00   -0.053891
# 2020-01-01 01:00:00    0.347673
#                          ...
# 2020-12-31 22:00:00   -0.548344
# 2020-12-31 23:00:00    1.526979
# Freq: H, Length: 8784, dtype: float64

'''还支持传入时间字符和各种时间对象：'''
print('# 指定天，结果相同')
print(ts['11/30/2020'])
# 2020-11-30 00:00:00   -0.507145
# 2020-11-30 01:00:00   -0.638819
# ...
# 2020-11-30 22:00:00   -1.498252
# 2020-11-30 23:00:00   -0.475406
# Freq: H, dtype: float64
print(ts['2020-11-30'])  # 结果同上
print(ts['20201130'])  # 结果同上

print('# 指定时间点')
print(ts[datetime.datetime(2020, 11, 30)])  # 0.22997129354557833
# print(datetime.datetime(2020,11,30)) # 2020-11-30 00:00:00
print(ts[pd.Timestamp(2020, 11, 30)])  # 结果同上
print(ts[pd.Timestamp('2020-11-30')])  # 结果同上
print(ts[np.datetime64('2020-11-30')])  # 结果同上

'''也可以使用部分字符查询一定范围内的数据：'''
print('部分字符串查询')
print(ts['2021'])  # 查询整个2021年的
print(ts['2021-06'])  # 查询2021年6月的
print(ts['2021-6'])  # 查询2021年6月的
print(ts['2021-6':'2021-10'])  # 查询2021年6月到10月的
print(ts['2021-1':'2021-2-28 00:00:00'])  # 精确时间
print(ts['2020-01-15':'2020-01-15 12:30:00'])
print(ts.loc['2020-01-15'])

'''如果想知道序列的粒度，即频率，可以使用ts.resolution查看（以上数据的粒度为小时）：'''
print()
print('# 时间粒度（频率）')
print(ts.index.resolution)  # hour

'''df.truncate()作为一个专门对索引的截取工具，可以很好地应用在时序索引上：'''
print('# 给定开始时间和结束时间来截取部分时间')
print(ts.truncate(before='2020-11-10 11:20', after='2020-12'))
# 2020-11-10 12:00:00   -0.968447
# 2020-11-10 13:00:00    0.249492
# ...
# 2020-11-30 23:00:00    1.833575
# 2020-12-01 00:00:00    0.051057
# Freq: H, Length: 493, dtype: float64

print()
print('------------------------------------------------------------')
print('\t14.3.4 类型转换')

'''
由于时间格式样式比较多，很多情况下Pandas并不能自动将时序数据识别为时间类型，
所以我们在处理前的数据清洗过程中，需要专门对数据进行时间类型转换。

astype是最简单的时间转换方法，它只能针对相对标准的时间格式，如以下数据的数据类型是object：
'''

s = pd.Series(['2020-11-01 01:10', '2020-11-11 11:10', '2020-11-30 20:10'])
print(s)
# 0    2020-11-01 01:10
# 1    2020-11-11 11:10
# 2    2020-11-30 20:10
# dtype: object

print('# 转为时间类型')
print(s.astype('datetime64[ns]'))
# 0   2020-11-01 01:10:00
# 1   2020-11-11 11:10:00
# 2   2020-11-30 20:10:00
# dtype: datetime64[ns]
print('修改频率：')
# 转为时间类型，指定频率为天
# print(s.astype('datetime64[D]')) # 报错！
# print(s.astype('datetime64[ns]').astype('datetime[D]')) # error
print(pd.to_datetime(s))  # 同print(s.astype('datetime64[ns]'))
# print(pd.to_datetime(s).astype('datetime[D]')) # error
# TypeError: data type 'datetime[D]' not understood
print(s.astype('datetime64[ns]').dt.floor('D'))  # 去除日期时间部分
# 0   2020-11-01
# 1   2020-11-11
# 2   2020-11-30
# dtype: datetime64[ns]
print('# 转为时间类型，指定时区为北京时间')
# print(pd.to_datetime(s))
# print(pd.to_datetime(s).astype('datetime64[ns,Asia/Shanghai]')) # 书中示例error
print(pd.to_datetime(s).dt)  # <pandas.core.indexes.accessors.DatetimeProperties object at 0x00000139C76A46A0>
print(pd.to_datetime(s).dt.tz_localize('Asia/Shanghai'))
# 0   2020-11-01 01:10:00+08:00
# 1   2020-11-11 11:10:00+08:00
# 2   2020-11-30 20:10:00+08:00
# dtype: datetime64[ns, Asia/Shanghai]

'''pd.to_datetime()也可以转换时间类型：'''
print('# 转为时间类型')
print(pd.to_datetime(s))
# 0   2020-11-01 01:10:00
# 1   2020-11-11 11:10:00
# 2   2020-11-30 20:10:00
# dtype: datetime64[ns]

'''pd.to_datetime()还可以将多列组合成一个时间进行转换：'''
df = pd.DataFrame({'year': [2020, 2020, 2020],
                   'month': [10, 11, 12],
                   'day': [10, 11, 12]
                   })

print(df)
print('# 转为时间类型')
print(pd.to_datetime(df))
# 0   2020-10-10
# 1   2020-11-11
# 2   2020-12-12
# dtype: datetime64[ns]

'''对于Series，pd.to_datetime()会智能识别其时间格式并进行转换：'''

s = pd.Series(['2020-11-01 01:10', '2020-11-11 11:10', None])
print(pd.to_datetime(s))
# 0   2020-11-01 01:10:00
# 1   2020-11-11 11:10:00
# 2                   NaT
# dtype: datetime64[ns]

'''对于列表，pd.to_datetime()也会智能识别其时间格式并转换为时间序列索引：'''
print()
# print(pd.to_datetime(['2020/11/11','2020.12.12'])) # 报错
print(pd.to_datetime(['2020/11/11', '2020.12.12'], dayfirst=True, errors='coerce'))
# DatetimeIndex(['2020-11-11', 'NaT'], dtype='datetime64[ns]', freq=None)
# print(pd.to_datetime(['2020/11/11','2020.12.12'],format='%Y-%m-%d',dayfirst=True)) # 报错

print(pd.to_datetime(['2020/11/11'], format='%Y/%m/%d', dayfirst=True))  # ,'2020.12.12'
# DatetimeIndex(['2020-11-11'], dtype='datetime64[ns]', freq=None)
print(pd.to_datetime(['2020.12.12'], format='%Y.%m.%d', dayfirst=True))  # ,'2020.12.12'
# DatetimeIndex(['2020-12-12'], dtype='datetime64[ns]', freq=None)
print(pd.to_datetime(['2020.12.12', '2020_11_11'], format='%Y.%m.%d', errors='coerce'))
# DatetimeIndex(['2020-12-12', 'NaT'], dtype='datetime64[ns]', freq=None)

'''用pd.DatetimeIndex直接转为时间序列索引：'''
print('# 转为时间序列索引，自动推断频率')
print(pd.DatetimeIndex(['20201101', '20201102', '20201103'], freq='infer'))
# DatetimeIndex(['2020-11-01', '2020-11-02', '2020-11-03'], dtype='datetime64[ns]', freq='D')

print('''针对单个时间，用pd.Timestamp()转换为时间格式：''')
print(pd.to_datetime('2020/11/12'))  # 2020-11-12 00:00:00
print(pd.Timestamp('2020/11/12'))  # 同上！

print()
print('------------------------------------------------------------')
print('\t14.3.5 按格式转换')

'''
如果原数据的格式为不规范的时间格式数据，可以通过格式映射来将其转为时间数据：
'''
print('# 不规则格式转换时间')
print(pd.to_datetime('2020_11_11', format='%Y_%m_%d'))  # 2020-11-11 00:00:00

print('# 可以让系统自己推断时间格式')
# print(pd.to_datetime('20200101', infer_datetime_format=True, errors='ignore'))
# 2020-01-01 00:00:00
'''多个时间格式，无法推断'''
'''提示该参数已弃用！ 默认 infer_datetime_format=True'''
# print(pd.to_datetime(['2020.12.12','2020_11_11','20200101'], infer_datetime_format=True, errors='coerce'))
# DatetimeIndex(['2020-12-12', 'NaT', 'NaT'], dtype='datetime64[ns]', freq=None)
print('''# 将errors参数设置为coerce，将不会忽略错误，返回空值''')
print(pd.to_datetime('20200101', format='%Y%m%d', errors='coerce'))
# 2020-01-01 00:00:00
print('fasdfas')
# print(pd.to_datetime(df.day.astype(str),format='%m/%d/%Y')) # 报错！

print('其它')
print(pd.to_datetime('01-10-2020 00:00', format='%d-%m-%Y %H:%M'))  # 2020-10-01 00:00:00
print('# 对时间戳进行转换，需要给出时间单位，一般为秒')
print(pd.to_datetime(1490195805, unit='s'))  # 2017-03-22 15:16:45
print(pd.to_datetime(1490195805433502912, unit='ns'))  # 2017-03-22 15:16:45.433502912

print('可以将数字列表转换为时间：')
print(pd.to_datetime([10, 11, 12, 15], unit='D', origin=pd.Timestamp('2020-11-01')))
# DatetimeIndex(['2020-11-11', '2020-11-12', '2020-11-13', '2020-11-16'], dtype='datetime64[ns]', freq=None)
'''个人测试'''
s = pd.Series([10, 11, 12, 15])
print(pd.to_datetime(s, unit='D', origin=pd.Timestamp('2024-01-01')))
# 0   2024-01-11
# 1   2024-01-12
# 2   2024-01-13
# 3   2024-01-16
# dtype: datetime64[ns]

print()
print('------------------------------------------------------------')
print('\t14.3.6 时间访问器.dt')
'''
之前介绍过了文本访问器（.str）和分类访问器（.cat），
对时间Pandas也提供了一个时间访问器.dt.<method>，
用它可以以time.dt.xxx的形式来访问时间序列数据的属性和调用它们的方法，返回对应值的序列。
'''

s = pd.Series(pd.date_range('2020-11-01', periods=5, freq='d'))
# print(s)
print('# 各天是星期几')
print(s.dt.day_name())
# 0       Sunday
# 1       Monday
# 2      Tuesday
# 3    Wednesday
# 4     Thursday
# dtype: object
print('# 时间访问器操作')
print(s.dt.date)
# 0    2020-11-01
# 1    2020-11-02
# 2    2020-11-03
# 3    2020-11-04
# 4    2020-11-05
# dtype: object
print(s.dt.time)
print(s.dt.timetz)  # 同上
# 0    00:00:00
# 1    00:00:00
# 2    00:00:00
# 3    00:00:00
# 4    00:00:00
# dtype: object

print('# 以下为时间各成分的值')
print(s.dt.year)  # 年
print(s.dt.month)  # 月份
print(s.dt.day)  # 天
print(s.dt.hour)  # 小时
print(s.dt.minute)  # 分钟
print(s.dt.second)  # 秒
print(s.dt.microsecond)
print(s.dt.nanosecond)

print('# 以下为与周、月、年相关的属性')

s = pd.Series(pd.date_range('2024-06-01', periods=3, freq='d'))
print(s)
# print(s.dt.week) # 警告已作废
print(s.dt.isocalendar().week)  # 年内第几周
print(s.dt.dayofweek)  # 星期一属于0
print(s.dt.weekday)  # 同上
print(s.dt.dayofyear)  # 一年中的第几天 元旦是第一天
print(s.dt.quarter)  # 季度
print(s.dt.is_month_start)  # 是否月初第一天
print(s.dt.is_month_end)  # 是否月初最后一天
print(s.dt.is_year_start)  # 是否年初第一天
print(s.dt.is_year_end)  # 是否年初最后一天
print(s.dt.is_leap_year)  # 是否闰年
print(s.dt.daysinmonth)  # 当月天数
print(s.dt.days_in_month)  # 同上
print(s.dt.tz)  # None
print(s.dt.freq)  # D
print()
print('# 以下为转换方法')
print(s.dt.to_period)
# <bound method PandasDelegate._add_delegate_accessors.<locals>._create_delegator_method.<locals>.f of <pandas.core.indexes.accessors.DatetimeProperties object at 0x00000207DC173A30>>
print(s.dt.to_pydatetime)
# <bound method DatetimeProperties.to_pydatetime of <pandas.core.indexes.accessors.DatetimeProperties object at 0x0000019B9C511A30>>
print(s.dt.to_pydatetime())
# [datetime.datetime(2024, 1, 1, 0, 0) datetime.datetime(2024, 1, 2, 0, 0)
#  datetime.datetime(2024, 1, 3, 0, 0) datetime.datetime(2024, 1, 4, 0, 0)
#  datetime.datetime(2024, 1, 5, 0, 0) datetime.datetime(2024, 1, 6, 0, 0)
#  datetime.datetime(2024, 1, 7, 0, 0) datetime.datetime(2024, 1, 8, 0, 0)]
print(s.dt.tz_convert)
print(s.dt.strftime)
print(s.dt.normalize)
print(s.dt.tz_localize)

print()
print(s.dt.round(freq='D'))  # 类似四舍五入 返回年月日
print(s.dt.floor(freq='D'))  # 向下舍入为天
print(s.dt.ceil(freq='D'))  # 向上舍入为天

print(s.dt.month_name())  # 月份名称
print(s.dt.day_name())  # 星期几的名称
print(s.dt.day)  # 当月第几天

print()
print('dfsdf')
s = pd.Series(pd.date_range('2024-01-30', periods=3, freq='d'))
# print(s.dt.start_time) # 报错警告
# print(s.dt.end_time()) # 报错警告
# print(s.dt.total_seconds) # 报错警告

print('# 个别用法举例')
# 将时间转为UTC时间，再转为美国东部时间
print(s.dt.tz_localize('UTC').dt.tz_convert('US/Eastern'))
# 0   2024-01-29 19:00:00-05:00
# 1   2024-01-30 19:00:00-05:00
# 2   2024-01-31 19:00:00-05:00
# dtype: datetime64[ns, US/Eastern]
# 输出时间显示格式
print(s.dt.strftime('%Y/%m/%d'))
# 0    2024/01/30
# 1    2024/01/31
# 2    2024/02/01
# dtype: object


print()
print('------------------------------------------------------------')
print('第14章 Pandas时序数据')
print('\t14.3 时间序列')
# update20240605
'''
本章将全面介绍Pandas在时序数据处理中的方法，
主要有时间的概念、时间的计算机表示方法、时间的属性操作和时间格式转换；
时间之间的数学计算、时长的意义、时长与时间的加减操作；
时间偏移的概念、用途、表示方法；时间跨度的意义、表示方法、单位转换等。
'''
# 小节注释
'''
上节介绍了固定时间，将众多的固定时间组织起来就形成了时间序列，即所谓的时序数据。

▶ 以下是一些具体的使用方法举例：
'''
print()
print('------------------------------------------------------------')
print('\t14.3.1 时序索引')

'''
在时间序列数据中，索引经常是时间类型，我们在操作数据时经常
会与时间类型索引打交道，本节将介绍如何查询和操作时间类型索引。

DatetimeIndex是时间索引对象，一般由to_datetime()或date_range()来创建：

'''

import datetime

day_code = pd.to_datetime(['11/1/2020',  # 类时间字符串
                           np.datetime64('2020-11-02'),  # NumPy的时间类型
                           datetime.datetime(2020, 11, 3)  # Python自带时间类型
                           ])

print(day_code)
# DatetimeIndex(['2020-11-01', '2020-11-02', '2020-11-03'], dtype='datetime64[ns]', freq=None)

'''
date_range()可以给定开始或者结束时间，并给定周期数据、周期频率，
会自动生成在此范围内的时间索引数据：
'''
print('date_range()')
# 默认频率为天
print(pd.date_range('2020-01-01', periods=10))
# DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
#                '2020-01-05', '2020-01-06', '2020-01-07', '2020-01-08',
#                '2020-01-09', '2020-01-10'],
#               dtype='datetime64[ns]', freq='D')
print(pd.date_range('2020-01-01', '2020-01-10'))  # 结果同上
print(pd.date_range(end='2020-01-10', periods=10))  # 结果同上

'''pd.bdate_range()生成数据可以跳过周六日，实现工作日的时间索引序列：'''
print('# 频率为工作日')
print(pd.bdate_range('2024-06-01', periods=10))
# DatetimeIndex(['2024-06-03', '2024-06-04', '2024-06-05', '2024-06-06',
#                '2024-06-07', '2024-06-10', '2024-06-11', '2024-06-12',
#                '2024-06-13', '2024-06-14'],
#               dtype='datetime64[ns]', freq='B')
'''个人ps：无法跳过端午节 || 好像有参数可以操作！ '''

print()
print('------------------------------------------------------------')
print('\t14.3.7 时长数据访问器')

'''
时长数据也支持访问器，可以解析出时长的相关属性，最终产出一个结果序列：
'''
# print(np.arange(5)) # [0 1 2 3 4]
ts = pd.Series(pd.to_timedelta(np.arange(5), unit='hour'))
print(ts)
# 0   0 days 00:00:00
# 1   0 days 01:00:00
# 2   0 days 02:00:00
# 3   0 days 03:00:00
# 4   0 days 04:00:00
# dtype: timedelta64[ns]

print('# 计算秒数')
print(ts.dt.seconds)
# 0        0
# 1     3600
# 2     7200
# 3    10800
# 4    14400
# dtype: int32
print('# 转为Python时间格式')
print(ts.dt.to_pytimedelta())
# [datetime.timedelta(0) datetime.timedelta(seconds=3600)
#  datetime.timedelta(seconds=7200) datetime.timedelta(seconds=10800)
#  datetime.timedelta(seconds=14400)]

print()
print('------------------------------------------------------------')
print('\t14.3.8 时序数据移动')

rng = pd.date_range('2024-06-01', '2024-06-05')
ts = pd.Series(range(len(rng)), index=rng)
print(ts)
# 2024-06-01    0
# 2024-06-02    1
# 2024-06-03    2
# 2024-06-04    3
# 2024-06-05    4
# Freq: D, dtype: int64
print('# 向上移动一位')
print(ts.shift(-1))
# 2024-06-01    1.0
# 2024-06-02    2.0
# 2024-06-03    3.0
# 2024-06-04    4.0
# 2024-06-05    NaN
# Freq: D, dtype: float64

'''shift()方法接受freq频率参数，该参数可以接受DateOffset类或其他类似timedelta的对象，也可以接受偏移别名：'''
print('# 向上移动一个工作日，06-01~06-02是周六、日')
print(ts.shift(-1, freq='B'))
# 2024-05-31    0
# 2024-05-31    1
# 2024-05-31    2
# 2024-06-03    3
# 2024-06-04    4
# dtype: int64

print()
print('------------------------------------------------------------')
print('\t14.3.9 频率转换')

'''
更换时间频率是将时间序列由一个频率单位更换为另一个频率单位，
实现时间粒度的变化。更改频率的主要功能是asfreq()方法。
以下是一个频率为自然日的时间序列：
'''

rng = pd.date_range('2020-11-01', '2020-12-01')
ts = pd.Series(range(len(rng)), index=rng)
print(ts)
'''
2020-11-01     0
2020-11-02     1
...
2020-11-30    29
2020-12-01    30
Freq: D, dtype: int64
'''
# 我们将它的频率变更为更加细的粒度，会产生缺失值：
print('# 频率转为12小时')
print(ts.asfreq(pd.offsets.Hour(12)))
# 2020-11-01 00:00:00     0.0
# 2020-11-01 12:00:00     NaN
# 2020-11-02 00:00:00     1.0
# 2020-11-02 12:00:00     NaN
#                        ...
# 2020-11-30 00:00:00    29.0
# 2020-11-30 12:00:00     NaN
# 2020-12-01 00:00:00    30.0
# Freq: 12H, Length: 61, dtype: float64

'''对于缺失值可以用指定值或者指定方法进行填充：'''
print('# 对缺失值进行填充')
print(ts.asfreq(freq='12h', fill_value=0))
# 2020-11-01 00:00:00     0
# 2020-11-01 12:00:00     0
# 2020-11-02 00:00:00     1
# 2020-11-02 12:00:00     0
# 2020-11-03 00:00:00     2
#                        ..
# 2020-11-29 00:00:00    28
# 2020-11-29 12:00:00     0
# 2020-11-30 00:00:00    29
# 2020-11-30 12:00:00     0
# 2020-12-01 00:00:00    30
# Freq: 12H, Length: 61, dtype: int64

print('# 对产生的缺失值使用指定方法填充')
print(ts.asfreq(pd.offsets.Hour(12), method='pad'))
# 2020-11-01 00:00:00     0
# 2020-11-01 12:00:00     0
# 2020-11-02 00:00:00     1
# 2020-11-02 12:00:00     1
# 2020-11-03 00:00:00     2
#                        ..
# 2020-11-29 00:00:00    28
# 2020-11-29 12:00:00    28
# 2020-11-30 00:00:00    29
# 2020-11-30 12:00:00    29
# 2020-12-01 00:00:00    30
# Freq: 12H, Length: 61, dtype: int64

'''
14.3.10 小结
时序数据由若干个固定时间组成，这些固定时间的分布大多呈现出一定的周期性。
时序数据经常作为索引，它也有可能在数据列中。
在数据分析业务实践中大量用到时序数据，因此本节内容是数序数据分析的关键内容，也是操作频率最高的内容。

'''

print()
print('------------------------------------------------------------')
print('第14章 Pandas时序数据')
print('\t14.4 时间偏移')
# update20240607
'''

'''
# 小节注释
'''
DateOffset类似于时长Timedelta，但它使用日历中时间日期的规则，
而不是直接进行时间性质的算术计算，让时间更符合实际生活。

▶ 以下是一些具体的使用方法举例：
'''
print()
print('------------------------------------------------------------')
print('\t14.4.1 DateOffset对象')

'''
我们通过夏令时来理解DateOffset对象。有些地区使用夏令时，每
日偏移时间有可能是23或24小时，甚至25个小时。

'''
print('# 生成一个指定的时间，芬兰赫尔辛基时间执行夏令时')
t = pd.Timestamp('2016-10-30 00:00:00', tz='Europe/Helsinki')
print(t)
# 2016-10-30 00:00:00+03:00

print(t + pd.Timedelta(days=1))  # 增加一个自然天
# 2016-10-30 23:00:00+02:00
print(t + pd.DateOffset(days=1))  # 增加一个时间偏移天
# 2016-10-31 00:00:00+02:00

print('工作日')
# 定义一个日期
d = pd.Timestamp('2024-06-07')
print(d.day_name())  # Friday
print('# 定义2个工作日时间偏移变量')
two_business_days = 2 * pd.offsets.BDay()
# print(two_business_days) # <2 * BusinessDays>
print('# 增加两个工作日')
# print(two_business_days.apply(d)) # 报错！
print(two_business_days + d)  # 2024-06-11 00:00:00

print('# 取增加两个工作日后的星期')
print((d + two_business_days).day_name())  # 'Tuesday'

print()
print(d + pd.Timedelta(days=3))  # 增加一个自然天
# 2024-06-10 00:00:00
print(d + pd.DateOffset(days=3))  # 增加一个时间偏移天
# 2024-06-10 00:00:00

'''
我们发现，与时长Timedelta不同，时间偏移DateOffset不是数学意义上的增加或减少，
而是根据实际生活的日历对现有时间进行偏移。
时长可以独立存在，作为业务的一个数据指标，而时间偏移DateOffset的意义是找到一个时间起点并对它进行时间移动。
所有的日期偏移对象都在pandas.tseries.offsets下，
其中pandas.tseries.offsets.DateOffset是标准的日期范围时间偏移类型，它默认是一个日历日。

from pandas.tseries.offsets import DateOffset
ts = pd.Timestamp('2020-01-01 09:10:11')
ts + DateOffset(months=3)
# Timestamp('2020-04-01 09:10:11')
ts + DateOffset(hours=2)
# Timestamp('2020-01-01 11:10:11')
ts + DateOffset()
# Timestamp('2020-01-02 09:10:11')
'''

print()
print('------------------------------------------------------------')
print('\t14.4.2 偏移别名')
'''
DateOffset基本都支持频率字符串或偏移别名，传入freq参数，时间
偏移的子类、子对象都支持时间偏移的相关操作。有效的日期偏移及频
率字符串见表14-1。

很多 请见该章节！
'''

add_week = 1 * pd.offsets.Week()
print(pd.Timestamp('now') + add_week)  # 增加一周
# 2024-06-14 17:07:11.829099

print()
print('------------------------------------------------------------')
print('\t14.4.3 移动偏移')

'''Offset通过计算支持向前或向后偏移：'''

ts = pd.Timestamp('2020-06-06 00:00:00')
print(ts.day_name())  # Saturday

# 定义一个工作小时偏移，默认是周一到周五9～17点，我们从10点开始
offset = pd.offsets.BusinessHour(start='10:00')
# print(offset) # <BusinessHour: BH=10:00-17:00>
print('# 向前偏移一个工作小时，是一个周一，跳过了周日')
print(offset.rollforward(ts))  # 2020-06-08 10:00:00

print()
print('# 向前偏移至最近的工作日，小时也会增加')
print(ts + offset)  # 2020-06-08 11:00:00
print('# 向后偏移，会在周五下班前的一个小时')
print(offset.rollback(ts))  # 2020-06-05 17:00:00

print(ts - pd.offsets.Day(1))  # 昨日
# 2020-06-05 00:00:00
print(ts - pd.offsets.Day(2))  # 前日
# 2020-06-04 00:00:00
print(ts - pd.offsets.Week(weekday=0))  # # 2020-06-01 00:00:00 本周一
print(ts - pd.offsets.Week(weekday=0) - pd.offsets.Day(14))  # 2020-05-18 00:00:00
print((ts - pd.offsets.Week(weekday=0) - pd.offsets.Day(14)).day_name())  # Monday 上上周一

print()
print(ts - pd.offsets.MonthEnd())  # 2020-05-31 00:00:00
print(ts - pd.offsets.MonthBegin())  # 2020-06-01 00:00:00
print(ts - pd.offsets.MonthEnd() - pd.offsets.MonthBegin())  # 2020-05-01 00:00:00  # 上月一日
'''
时间偏移操作会保留小时和分钟，有时候我们不在意具体的时间，
可以使用normalize()进行标准化到午夜0点：
'''
print(offset.rollback(ts).normalize())  # 2020-06-05 00:00:00
print(pd.Timestamp('now').normalize())  # 2024-06-07 00:00:00

print()
print('------------------------------------------------------------')
print('\t14.4.4 应用偏移')
'''update20240611 已作废！ apply可以使偏移对象应用到一个时间上'''

ts = pd.Timestamp('2020-06-01 09:00')
# print(ts)
day = pd.offsets.Day()  # 定义偏移对象
# print(day.apply(ts)) # 将偏移对象应用到时间上
print(ts + day)  # 2020-06-02 09:00:00
print((ts + day).normalize())  # 2020-06-02 00:00:00

ts = pd.Timestamp('2020-06-01 22:00')
hour = pd.offsets.Hour()
print(ts + hour)  # 2020-06-01 23:00:00
print((ts + hour).normalize())  # 2020-06-01 00:00:00

print()
print('------------------------------------------------------------')
print('\t14.4.5 偏移参数')

'''
之前我们只偏移了偏移对象的一个单位，可以传入参数来偏移多个单位和对象中的其他单位：
'''

import datetime

d = datetime.datetime(2020, 6, 1, 9, 0)
# d = datetime.datetime(2024,6,6,9,0)
print(d)  # 2020-06-01 09:00:00

print(d + pd.offsets.Week())  # # 偏移一周
# 2020-06-08 09:00:00
print(d + pd.offsets.Week(weekday=4))  # # 偏移4周中的日期
# 2020-06-05 09:00:00 ??? 看不懂 逻辑如下注释中 看懂了

'''
pd.offsets.Week(weekday=4)：偏移对象的 weekday 参数指定目标日期为周五。(周一是0，所以周五是4)
偏移逻辑：从初始日期 d 开始，找到最近的周五并进行偏移。
结果：从2020年6月1日（星期一）偏移到2020年6月5日（星期五）。
'''

print('# 取一周第几天')
print((d + pd.offsets.Week(weekday=4)).weekday())  # 4
print(d - pd.offsets.Week())  # 向后一周

print('参数也支持标准化normalize：')
print(d + pd.offsets.Week(normalize=True))  # 2020-06-08 00:00:00
print(d - pd.offsets.Week(normalize=True))  # 2020-05-25 00:00:00

print('YearEnd支持用参数month指定月份：')
print(d + pd.offsets.YearEnd())  # 2020-12-31 09:00:00
print(d + pd.offsets.YearEnd(month=6))  # 2020-06-30 09:00:00

'''
不同的偏移对象支持不同的参数，可以通过代码编辑器的代码提示进行查询。
'''

print()
d = pd.Timestamp('2024-06-07')
print(d + pd.offsets.DateOffset(years=5))  # 2029-06-07 00:00:00 增加年份  +s 年月日时分秒类似
print(d + pd.offsets.DateOffset(year=5))  # 0005-06-07 00:00:00 # 替换年份 无s 年月日时分秒类似

print()
print('------------------------------------------------------------')
print('\t14.4.6 相关查询')

'''当使用日期作为索引的DataFrame时，此函数可以基于日期偏移量选择最后几行：'''

i = pd.date_range('2018-04-09', periods=4, freq='2D')
ts = pd.DataFrame({'A': [1, 2, 3, 4]}, index=i)
print(ts)
#             A
# 2018-04-09  1
# 2018-04-11  2
# 2018-04-13  3
# 2018-04-15  4
# 取最后3天，请注意，返回的是最近3天的数据
# 而不是数据集中最近3天的数据，因此未返回2018-04-11的数据
print(ts.last('3D'))
#             A
# 2018-04-13  3
# 2018-04-15  4
print('# 前3天')
print(ts.first('3D'))
#             A
# 2018-04-09  1
# 2018-04-11  2
'''可以用at_time()来指定时间：'''
print('# 指定时间')

i = pd.date_range('2018-04-09', periods=4, freq='12H')
ts = pd.DataFrame({'A': [1, 2, 3, 4]}, index=i)

print(ts.at_time('12:00'))
#                      A
# 2018-04-09 12:00:00  2
# 2018-04-10 12:00:00  4
'''用between_time()来指定时间区间：'''
i = pd.date_range('2018-04-09', periods=4, freq='15T')
ts = pd.DataFrame({'A': [1, 2, 3, 4]}, index=i)
print(ts)
print(ts.between_time('0:15', '0:45'))
#                      A
# 2018-04-09 00:15:00  2
# 2018-04-09 00:30:00  3
# 2018-04-09 00:45:00  4
print()
print('------------------------------------------------------------')
print('\t14.4.7 与时序的计算')
'''
可以对Series或DatetimeIndex时间索引序列应用时间偏移，
与其他时间序列数据一样，时间偏移后的数据一般会作为索引。

举例类似上述内容 不写了
'''

print()
print('------------------------------------------------------------')
print('\t14.4.8 锚定偏移')

i = pd.date_range('2024-06-01', periods=14, freq='D')
# ts = pd.DataFrame({'A':pd.range(14)[::-1]},index=i)
ts = pd.DataFrame({'A': range(14)}, index=i)
# print(ts)
# 别名     说明
# W-SUN   周(星期日)，同"W"
print()
# 生成一个以星期日为结束日的周时间序列
date_range = pd.date_range('2024-06-01', periods=14, freq='W-SUN')  # W-MON ~ W-SAT
print(date_range)
# DatetimeIndex(['2024-06-02', '2024-06-09', '2024-06-16', '2024-06-23',
#                '2024-06-30', '2024-07-07', '2024-07-14', '2024-07-21',
#                '2024-07-28', '2024-08-04', '2024-08-11', '2024-08-18',
#                '2024-08-25', '2024-09-01'],
#               dtype='datetime64[ns]', freq='W-SUN')

# 创建一个数据框，以日期序列为索引
data = pd.DataFrame({'A': range(14)}, index=date_range)
print(data)

print()
'''
Q-JAN 季，结束于1月
...
Q-NOV 季，结束于11月
'''
date_range = pd.date_range('2024-06-01', periods=14, freq='Q')  # 季，结束于12月，同“Q
print(date_range)
# DatetimeIndex(['2024-06-30', '2024-09-30', '2024-12-31', '2025-03-31',
#                '2025-06-30', '2025-09-30', '2025-12-31', '2026-03-31',
#                '2026-06-30', '2026-09-30', '2026-12-31', '2027-03-31',
#                '2027-06-30', '2027-09-30'],
#               dtype='datetime64[ns]', freq='Q-DEC')
date_range = pd.date_range('2024-06-01', periods=14, freq='Q-NOV')  # 季，结束于11月
print(date_range)

date_range = pd.date_range('2024-06-01', periods=14, freq='A')  # 年，结束于12月
print(date_range)
# DatetimeIndex(['2024-12-31', '2025-12-31', '2026-12-31', '2027-12-31',
#                '2028-12-31', '2029-12-31', '2030-12-31', '2031-12-31',
#                '2032-12-31', '2033-12-31', '2034-12-31', '2035-12-31',
#                '2036-12-31', '2037-12-31'],
#               dtype='datetime64[ns]', freq='A-DEC')

print(pd.Timestamp('2020-01-02') - pd.offsets.MonthBegin(n=4))  # 2019-10-01 00:00:00

print()
print('------------------------------------------------------------')
print('\t14.4.9 自定义工作时间')
'''
由于不同地区不同文化，工作时间和休息时间不尽相同。
在数据分析时需要考虑工作日、周末等文化差异带来的影响，
比如，埃及的周末是星期五和星期六。

可以向Cday或CustomBusinessDay类传入节假日参数来自定义一个工作日偏移对象：
'''
import datetime

weekmask_egypt = 'Sun Mon Tue Wed Thu'

# 定义出五一劳动节的日期，因为放假
holidays = ['2018-05-01',
            datetime.datetime(2019, 5, 1),
            np.datetime64('2020-05-01')
            ]

# 自定义工作日中传入休假日期，一个正常星期工作日的顺序
bday_egypt = pd.offsets.CustomBusinessDay(holidays=holidays,
                                          weekmask=weekmask_egypt)

# 指定一个日期
dt = datetime.datetime(2020, 4, 30)
# 偏移两个工作日，跳过了休假日
print(dt + 2 * bday_egypt)  # 2020-05-04 00:00:00
print(dt + bday_egypt)  # 2020-05-03 00:00:00 || 2号、3号 是周六日

print()
print('# 输出时序及星期几')
idx = pd.date_range(dt, periods=5, freq=bday_egypt)
print(pd.Series(idx.weekday + 1, index=idx))
# 2020-04-30    4
# 2020-05-03    7
# 2020-05-04    1
# 2020-05-05    2
# 2020-05-06    3
# Freq: C, dtype: int32

'''
BusinessHour表是开始和结束工作的小时时间，默认的工作时间是9:00—17:00，
与时间相加超过一个小时会移到下一个小时，超过一天会移动到下一个工作日。

'''
print('BusinessHour')
bh = pd.offsets.BusinessHour()
print(bh)  # <BusinessHour: BH=09:00-17:00>
print('# 2020-08-01是周六')
print(pd.Timestamp('2020-08-01 10:00').weekday())  # 5
print('# 增加一个工作小时')
# 跳过周末
print(pd.Timestamp('2020-08-01 10:00') + bh)  # 2020-08-03 10:00:00
print(pd.Timestamp('2020-07-31 10:00') + bh)  # 2020-07-31 11:00:00

# 一旦计算就等于上班了，等同于pd.Timestamp('2020-08-01 09:00') + bh
print(pd.Timestamp('2020-08-01 08:00') + bh)  # 2020-08-03 10:00:00

print('# 计算后已经下班了，就移到下一个工作小时（跳过周末）')
print(pd.Timestamp('2020-07-31 16:00') + bh)  # 2020-08-03 09:00:00
print(pd.Timestamp('2020-08-01 16:30') + bh)  # 2020-08-03 10:00:00

print('# 偏移两个工作小时')
print(pd.Timestamp('2024-06-19') + pd.offsets.BusinessHour(2))  # 2024-06-19 11:00:00

print('# 减去3个工作小时')
print(pd.Timestamp('2024-06-19 10:00') + pd.offsets.BusinessHour(-3))  # 2024-06-18 15:00:00
print(pd.Timestamp('now') + pd.offsets.BusinessHour(-3))  # 2024-06-19 12:58:44.417465

print()
'''
可以自定义开始和结束工作的时间，格式必须是hour:minute字符串，不支持秒、微秒、纳秒。
'''
print('# 11点开始上班')
# print(datetime.time(20,0,0)) # 20:00:00
bh = pd.offsets.BusinessHour(start='11:00', end=datetime.time(20, 0))
# print(bh) # <BusinessHour: BH=11:00-20:00>

print(pd.Timestamp('2024-06-19 13:00') + bh)  # 2024-06-19 14:00:00
print(pd.Timestamp('2024-06-19 09:00') + bh)  # 2024-06-19 12:00:00
print(pd.Timestamp('2024-06-19 18:00') + bh)  # 2024-06-19 19:00:00

print()
'''start时间晚于end时间表示夜班工作时间。此时，工作时间将从午夜延至第二天。'''
bh = pd.offsets.BusinessHour(start='17:00', end='09:00')
print(bh)  # <BusinessHour: BH=17:00-09:00>

print(pd.Timestamp('2024-06-19 17:00') + bh)  # 2024-06-19 18:00:00
print(pd.Timestamp('2024-06-19 23:00') + bh)  # 2024-06-20 00:00:00

'''
# 尽管2024年6月15日是周六，但因为工作时间从周五开始，因此也有效
'''
print(pd.Timestamp('2024-06-15 07:00') + bh)  # 2024-06-15 08:00:00
print(pd.Timestamp('2024-06-15 08:00') + bh)  # 2024-06-17 17:00:00

'''# 尽管2024年6月17日是周一，但因为工作时间从周日开始，超出了工作时间'''
print(pd.Timestamp('2024-06-17 04:00') + bh)  # 2024-06-17 18:00:00

'''
14.4.10 小结
时间偏移与时长的根本不同是它是真实的日历上的时间移动，在数据分析中时间偏移的意义是大于时长的。
另外，通过继承pandas.tseries.holiday.AbstractHolidayCalendar创建子类，
可以自定义假期日历，完成更为复杂的时间偏移操作，可浏览Pandas官方文档了解。
'''

print()
print('------------------------------------------------------------')
print('第14章 Pandas时序数据')
print('\t14.5 时间段')
# update20240621
'''

'''
# 小节注释
'''
Pandas中的Period()对象表示一个时间段，比如一年、一个月或一个季度。
与时间长度不同，它表示一个具体的时间区间，有时间起点和周期频率。

▶ 以下是一些具体的使用方法举例：
'''
print()
print('------------------------------------------------------------')
print('\t14.5.1 Period对象')

'''
我们来利用pd.Period()创建时间段对象：
'''

print('# 创建一个时间段（年）')
print(pd.Period('2020'))  # 2020
print(type(pd.Period('2020')))  # <class 'pandas._libs.tslibs.period.Period'>

print('# 创建一个时间段（季度）')
print(pd.Period('2020Q4'))  # 2020Q4
print(type(pd.Period('2020Q4')))  # <class 'pandas._libs.tslibs.period.Period'>

print('# 2020-01-01全天的时间段')
print(pd.Period(year=2020, freq='D'))  # 2020-01-01

print('# 一周')
print(pd.Period('20240621', freq='W'))  # 2024-06-17/2024-06-23 || (周日为23)

print('# 默认周期，对应到最细粒度——分钟')
print(pd.Period('2020-11-11 23:00'))  # 2020-11-11 23:00
print(type(pd.Period('2020-11-11 23:00')))  # <class 'pandas._libs.tslibs.period.Period'>

print('# 指定周期')
print(pd.Period('2020-11-11 23:00', 'D'))  # 2020-11-11
print(pd.Period('2020-11-11 23:00', freq='D'))  # 同上

print()
print('------------------------------------------------------------')
print('\t14.5.2 属性方法')

'''
一个时间段有开始和结束时间，可以用如下方法获取：
'''

print('# 定义时间段')
p = pd.Period('2020Q4')
print('# 开始与结束时间')
print(p.start_time)  # 2020-10-01 00:00:00
print(p.end_time)  # 2020-12-31 23:59:59.999999999

'''如果当前时间段不符合业务实际，可以转换频率：'''
print('# 将频率转换为天')
print(p.asfreq('D'))  # 2020-12-31
print(p.asfreq('D', how='start'))  # 2020-10-01

'''其他的属性方法如下：'''
print(p.freq)  # <QuarterEnd: startingMonth=12> | （时间偏移对象）
print(p.freqstr)  # Q-DEC | （时间偏移别名）
print(p.is_leap_year)  # True（是否闰年）
print(p.to_timestamp())  # 2020-10-01 00:00:00

'''# 以下日期取时间段内最后一天'''
print()
print(p.day)  # 31
print(p.dayofweek)  # 3(周四)
print(p.dayofyear)  # 366
print(p.hour)  # 0
print(p.week)  # 53
print(p.minute)  # 0
print(p.second)  # 0
print(p.month)  # 12
print(p.quarter)  # 4
print(p.qyear)  # 2020 | (财年)
print(p.year)  # 2020
print(p.days_in_month)  # 31 （当月第几天）
print(p.daysinmonth)  # 31（当月共多少天）
'''
strptime()：用于将日期时间字符串解析为datetime对象，适用于处理用户输入、读取文件等场景。
strftime()：用于将datetime对象格式化为字符串，适用于格式化输出、生成文件名等场景。
'''
print(p.strftime('%Y年%m月'))  # 2020年12月
from datetime import datetime

user_input = "2024-05-28"
date_object = datetime.strptime(user_input, "%Y-%m-%d")
print(date_object)  # 输出：2024-05-28 00:00:00
user_two = '2024/06/21'
# print(datetime.strptime(user_two,'%Y-%m-%d')) # ValueError: time data '2024/06/21' does not match format '%Y-%m-%d'
print(datetime.strptime(user_two, '%Y/%m/%d'))  # 2024-06-21 00:00:00

print()
print('------------------------------------------------------------')
print('\t14.5.3 时间段的计算')
'''
时间段可以做加减法，表示将此时间段前移或后移相应的单位：
'''
print('# 在2020Q4上增加一个周期')
print(pd.Period('2020Q4') + 1)  # 2021Q1
print('# 在2020Q4上减少一个周期')
print(pd.Period('2020Q4') - 1)  # 2020Q3

'''当然，时间段对象也可以和时间偏移对象做加减：'''
print('# 增加一小时')
print(pd.Period('20200105 15'))  # 2020-01-05 15:00
# print(pd.Period('2020010515')) # ValueError: year 2020010515 is out of range
print(pd.Period('20200105 15') + pd.offsets.Hour())  # 2020-01-05 16:00
print(pd.Period('20200105 15') + pd.offsets.Hour(1))  # 同上

print('# 增加10天')
print(pd.Period('20200101') + pd.offsets.Day(10))  # 2020-01-11

'''
如果偏移量频率与时间段不同，则其单位要大于时间段的频率，否则会报错：
'''
print()
print(pd.Period('20200101 14') + pd.offsets.Day(10))  # 2020-01-11 14:00
# print(pd.Period('20200101 14') + pd.offsets.Minute(10)) # ValueError: Cannot losslessly convert units
print(pd.Period('20200101 1432') + pd.offsets.Minute(10))  # 2020-01-01 14:42
print(pd.Period('2020 10'))  # 2020-10
# print(pd.Period('202010')) # pandas._libs.tslibs.parsing.DateParseError: month must be in 1..12: 202010
print(pd.Period('2020 10') + pd.offsets.MonthEnd(3))  # 2021-01
# print(pd.Period('2020 10') + pd.offsets.MonthBegin()) error
print(pd.Period('2020 10') - pd.offsets.MonthEnd(3))  # 2020-07

print('时间段也可以和时间差相加减：')
print(pd.Period('20200101 14') + pd.Timedelta('1 days'))  # 2020-01-02 14:00
# print(pd.Period('20200101 14') + pd.Timedelta('1 seconds')) # error
# pandas._libs.tslibs.period.IncompatibleFrequency: Input cannot be converted to Period(freq=H)

'''相同频率的时间段实例之差将返回它们之间的频率单位数'''
print()
print(pd.Period('20200101 14') - pd.Period('20200101 10'))  # <4 * Hours>
print(pd.Period('2020Q4') - pd.Period('2020Q1'))  # <3 * QuarterEnds: startingMonth=12>

print()
print('------------------------------------------------------------')
print('\t14.5.4 时间段索引')

'''
类似于时间范围pd.date_range()生成时序索引数据，
pd.period_range()可以生成时间段索引数据：
'''
print('# 生成时间段索引对象')
print(pd.period_range('2020-11-01 10:00', periods=10, freq='H'))
# PeriodIndex(['2020-11-01 10:00', '2020-11-01 11:00', '2020-11-01 12:00',
#              '2020-11-01 13:00', '2020-11-01 14:00', '2020-11-01 15:00',
#              '2020-11-01 16:00', '2020-11-01 17:00', '2020-11-01 18:00',
#              '2020-11-01 19:00'],
#             dtype='period[H]')

print('# 指定开始和结束时间')
print(pd.period_range('2020Q1', '2021Q4', freq='Q-NOV'))
'''
上例定义了一个从2020年第一季度到2021第四季度共8个季度的时间段，一年以11月为最后时间。

PeriodIndex(['2020Q1', '2020Q2', '2020Q3', '2020Q4', '2021Q1', '2021Q2',
             '2021Q3', '2021Q4'],
            dtype='period[Q-NOV]')
'''

print('# 通过传入时间段对象来定义')
print(pd.period_range(start=pd.Period('2020Q1', freq='Q'),
                      end=pd.Period('2021Q2', freq='Q'), freq='M'
                      ))
'''
PeriodIndex(['2020-03', '2020-04', '2020-05', '2020-06', '2020-07', '2020-08',
             '2020-09', '2020-10', '2020-11', '2020-12', '2021-01', '2021-02',
             '2021-03', '2021-04', '2021-05', '2021-06'],
            dtype='period[M]')
'''

print('时间段索引可以应用于数据中：')
print(pd.Series(pd.period_range('2020Q1', '2021Q4', freq='Q-NOV')))
# 0    2020Q1
# 1    2020Q2
# 2    2020Q3
# 3    2020Q4
# 4    2021Q1
# 5    2021Q2
# 6    2021Q3
# 7    2021Q4
# dtype: period[Q-NOV]

print(pd.Series(range(8), index=pd.period_range('2020Q1', '2021Q4', freq='Q-NOV')))
# 2020Q1    0
# 2020Q2    1
# 2020Q3    2
# 2020Q4    3
# 2021Q1    4
# 2021Q2    5
# 2021Q3    6
# 2021Q4    7
# Freq: Q-NOV, dtype: int64

print()
print('------------------------------------------------------------')
print('\t14.5.5 数据查询')

'''在数据查询时，支持切片操作：'''

s = pd.Series(1, index=pd.period_range('2020-10-01 10:00', '2021-10-01 10:00', freq='H'))
print(s)
'''
2020-10-01 10:00    1
2020-10-01 11:00    1
2020-10-01 12:00    1
2020-10-01 13:00    1
2020-10-01 14:00    1
                   ..
2021-10-01 06:00    1
2021-10-01 07:00    1
2021-10-01 08:00    1
2021-10-01 09:00    1
2021-10-01 10:00    1
Freq: H, Length: 8761, dtype: int64
'''

print(s['2020'])
'''
2020-10-01 10:00    1
2020-10-01 11:00    1
2020-10-01 12:00    1
2020-10-01 13:00    1
2020-10-01 14:00    1
                   ..
2020-12-31 19:00    1
2020-12-31 20:00    1
2020-12-31 21:00    1
2020-12-31 22:00    1
2020-12-31 23:00    1
Freq: H, Length: 2198, dtype: int64
'''

print('# 进行切片操作')
print(s['2020-10':'2020-11'])
'''
2020-10-01 10:00    1
2020-10-01 11:00    1
2020-10-01 12:00    1
2020-10-01 13:00    1
2020-10-01 14:00    1
                   ..
2020-11-30 19:00    1
2020-11-30 20:00    1
2020-11-30 21:00    1
2020-11-30 22:00    1
2020-11-30 23:00    1
Freq: H, Length: 1454, dtype: int64
'''

'''数据的查询方法与之前介绍过的时序查询一致。'''

print()
print('------------------------------------------------------------')
print('\t14.5.6 相关类型转换')

'''
astype()可以在几种数据之间自由转换，如DatetimeIndex转PeriodIndex：
'''

ts = pd.date_range('20201101', periods=100)
print(ts)
'''
DatetimeIndex(['2020-11-01', '2020-11-02', '2020-11-03', '2020-11-04',
                ...
               '2021-02-05', '2021-02-06', '2021-02-07', '2021-02-08'],
              dtype='datetime64[ns]', freq='D')
'''

print('# 转为PeriodIndex，频率为月')
print(ts.astype('period[M]'))
# PeriodIndex(['2020-11', '2020-11', '2020-11', '2020-11', '2020-11', '2020-11',
#              ...
#              '2021-01', '2021-01', '2021-02', '2021-02', '2021-02', '2021-02',
#              '2021-02', '2021-02', '2021-02', '2021-02'],
#             dtype='period[M]')

print('PeriodIndex转DatetimeIndex：')

ts = pd.period_range('2020-11', periods=100, freq='M')
print(ts)
'''
PeriodIndex(['2020-11', '2020-12', '2021-01', '2021-02', '2021-03', '2021-04',
             ...
             '2028-11', '2028-12', '2029-01', '2029-02'],
            dtype='period[M]')
'''

print('# 转为DatetimeIndex')
print(ts.astype('datetime64[ns]'))
'''
DatetimeIndex(['2020-11-01', '2020-12-01', '2021-01-01', '2021-02-01',
               '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01',
                ...
               '2028-07-01', '2028-08-01', '2028-09-01', '2028-10-01',
               '2028-11-01', '2028-12-01', '2029-01-01', '2029-02-01'],
              dtype='datetime64[ns]', freq='MS')
'''

print('# 频率从月转为季度')
print(ts.astype('period[Q]'))
'''
PeriodIndex(['2020Q4', '2020Q4', '2021Q1', '2021Q1', '2021Q1', '2021Q2',
             '2021Q2', '2021Q2', '2021Q3', '2021Q3', '2021Q3', '2021Q4',
             ........
             '2028Q2', '2028Q2', '2028Q3', '2028Q3', '2028Q3', '2028Q4',
             '2028Q4', '2028Q4', '2029Q1', '2029Q1'],
            dtype='period[Q-DEC]')
'''

'''
14.5.7 小结
时间段与时长和时间偏移不同的是，时间段有开始时间（当然也能推出结束时间）和长度，
在分析周期性发生的业务数据时，它会让你如鱼得水。
'''

print()
print('------------------------------------------------------------')
print('第14章 Pandas时序数据')
print('\t14.6 时间操作')
# update20240621
'''

'''
# 小节注释
'''
在前面几大时间类型的介绍中，我们需要进行转换、时间解析和输出格式化等操作，
本节就来介绍一些与之类似的通用时间操作和高级功能。

▶ 以下是一些具体的使用方法举例：
'''
print()
print('------------------------------------------------------------')
print('\t14.6.1 时区转换')

'''
Pandas使用pytz和dateutil库或标准库中的
datetime.timezone对象为使用不同时区的时间戳提供了丰富的支持。
可以通过以下方法查看所有时区及时区的字符名称：

'''

import pytz

# print(pytz.common_timezones) # out:长列表

# for i in pytz.common_timezones:  # 434种时区
#     print(i)

'''如果没有指定，时间一般是不带时区的：'''
ts = pd.date_range('11/11/2020 00:00', periods=10, freq='D')
print(ts.tz)  # None

'''进行简单的时区指定，中国通用的北京时区使用'Asia/Shanghai'定义：'''
print(pd.date_range('2020-01-01', periods=10, freq='D', tz='Asia/Shanghai'))
'''
DatetimeIndex(['2020-01-01 00:00:00+08:00', '2020-01-02 00:00:00+08:00',
               '2020-01-03 00:00:00+08:00', '2020-01-04 00:00:00+08:00',
               '2020-01-05 00:00:00+08:00', '2020-01-06 00:00:00+08:00',
               '2020-01-07 00:00:00+08:00', '2020-01-08 00:00:00+08:00',
               '2020-01-09 00:00:00+08:00', '2020-01-10 00:00:00+08:00'],
              dtype='datetime64[ns, Asia/Shanghai]', freq='D')
'''

print('简单指定时区的方法：')
print(pd.Timestamp('2020-01-01', tz='Asia/Shanghai'))  # 2020-01-01 00:00:00+08:00

print('以下是指定时区的更多方法：')
# 使用pytz支持
rng_pytz = pd.date_range('11/11/2020 00:00', periods=3, freq='D', tz='Europe/London')
print(rng_pytz.tz)  # Europe/London

# 使用dateutil支持
rng_dateutil = pd.date_range('11/11/2020 00:00', periods=3, freq='D')
# 转为伦敦所在的时区
rng_dateutil = rng_dateutil.tz_localize('dateutil/Europe/London')
print(rng_dateutil.tz)  # tzfile('Europe/Belfast')

print()
print('# 使用dateutil指定为UTC时间')
rng_utc = pd.date_range('11/11/2020 00:00', periods=3, freq='D', tz=dateutil.tz.tzutc())
print(rng_utc.tz)  # tzutc()

print()
print('# datetime.timezone')
rng_utc = pd.date_range('11/11/2020 00:00', periods=3, freq='D', tz=datetime.timezone.utc)
print(rng_utc.tz)  # UTC

'''从一个时区转换为另一个时区，使用tz_convert方法：'''
print('tz_convert')
print(rng_pytz.tz_convert('US/Eastern'))
'''
DatetimeIndex(['2020-11-10 19:00:00-05:00', '2020-11-11 19:00:00-05:00',
               '2020-11-12 19:00:00-05:00'],
              dtype='datetime64[ns, US/Eastern]', freq='D')
'''

print('其他方法：')
# 示例数据：UTC时间戳的交易记录
s_naive = pd.Series(pd.date_range('2024-05-27 00:00', periods=5, freq='H'))
print(s_naive)
# 0   2024-05-27 00:00:00
# 1   2024-05-27 01:00:00
# 2   2024-05-27 02:00:00
# 3   2024-05-27 03:00:00
# 4   2024-05-27 04:00:00
# dtype: datetime64[ns]

print('# 将时间戳本地化为UTC，然后转换为美国东部时间')
s_eastern = s_naive.dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
print(s_eastern)
# 0   2024-05-26 20:00:00-04:00
# 1   2024-05-26 21:00:00-04:00
# 2   2024-05-26 22:00:00-04:00
# 3   2024-05-26 23:00:00-04:00
# 4   2024-05-27 00:00:00-04:00
# dtype: datetime64[ns, US/Eastern]

# 直接转换为带有美国东部时间的时间戳
# s_naive = pd.Series(pd.date_range('2024-05-27 00:00', periods=5, freq='H'))
# s_eastern_2 = s_naive.astype('datetime64[ns, US/Eastern]')
# print(s_eastern_2)
'''该示例实际会 报错！'''

print('# 转换为不带时区信息的numpy数组')
# 示例数据：带有时区信息的时间戳
s_aware = pd.Series(pd.date_range('2024-05-27 00:00', periods=5, freq='H', tz='US/Eastern'))
s_no_tz = s_aware.to_numpy(dtype='datetime64[ns]')

print(s_no_tz)
# ['2024-05-27T04:00:00.000000000' '2024-05-27T05:00:00.000000000'
#  '2024-05-27T06:00:00.000000000' '2024-05-27T07:00:00.000000000'
#  '2024-05-27T08:00:00.000000000']

print()
print('------------------------------------------------------------')
print('\t14.6.2 时间的格式化')
# update20240625
'''
在数据格式解析、输出格式和格式转换过程中，需要用标识符来匹配日期元素的位置，
Pandas使用了Python的格式化符号系统，如：
'''

print('# 解析时间格式')
print(pd.to_datetime('2020*11*12', format='%Y*%m*%d'))  # 2020-11-12 00:00:00

print('# 输出的时间格式')
print(pd.Timestamp('now').strftime('%Y年%m月%d日'))  # 2024年06月25日

print(pd.Timestamp('now').strftime('%a'))  # Tue
print(pd.Timestamp('now').strftime('%A'))  # Tuesday
print(pd.Timestamp('now').strftime('%b'))  # Jun
print(pd.Timestamp('now').strftime('%B'))  # June
print(pd.Timestamp('now').strftime('%c'))  # Tue Jun 25 10:13:02 2024 |本地相应的日期表示和时间表示
print(pd.Timestamp('now').strftime('%j'))  # 177 | 年内的一天(001~366)

print(pd.Timestamp('now').strftime('%w'))  # 2  | 星期(0~6)，星期天为星期的开始
print(pd.Timestamp('now').strftime('%W'))  # 26 | 一年中的星期数(00~53)，星期一为星期的开始
print(pd.Timestamp('now').strftime('%U'))  # 25 | 一年中的星期数(00~53)，星期天为星期的开始

print(pd.Timestamp('now').strftime('%x'))  # 06/25/24 | 本地相应的日期表示
print(pd.Timestamp('now').strftime('%X'))  # 10:16:35 | 本地相应的时间表示
print(pd.Timestamp('now').strftime('%Z'))  # 空白 | 当前时区的名称
print(pd.Timestamp('now').strftime('%%'))  # % | %号本身

print(pd.Timestamp('now').strftime('%p'))  # AM | 本地 A.M.或 PM.的等价符

print()
print('------------------------------------------------------------')
print('\t14.6.3 时间重采样')

'''
Pandas可以对时序数据按不同的频率进行重采样操作，例如，原时序数据频率为分钟，
使用resample()可以按5分钟、15分钟、半小时等频率进行分组，然后完成聚合计算。
时间重采样在资金流水、金融交易等业务下非常常用。
'''

idx = pd.date_range('2020-01-01', periods=500, freq='Min')
ts = pd.Series(range(len(idx)), index=idx)
print(ts)
'''
2020-01-01 00:00:00      0
2020-01-01 00:01:00      1
                      ... 
2020-01-01 08:18:00    498
2020-01-01 08:19:00    499
Freq: T, Length: 500, dtype: int64
'''

print('# 每5分钟进行一次聚合')
print(ts.resample('5Min').sum())
'''
2020-01-01 00:00:00      10
2020-01-01 00:05:00      35
                       ... 
2020-01-01 08:10:00    2460
2020-01-01 08:15:00    2485
Freq: 5T, Length: 100, dtype: int64
'''

'''
重采样功能非常灵活，你可以指定许多不同的参数来控制频率转换和重采样操作。
通过类似于groupby聚合后的各种统计函数实现数据的分组聚合，
包括sum、mean、std、sem、max、min、mid、median、first、last、ohlc。
'''
print()
print(ts.resample('5Min').mean())  # 平均
'''
2020-01-01 00:00:00      2.0
2020-01-01 00:05:00      7.0
                       ...  
2020-01-01 08:10:00    492.0
2020-01-01 08:15:00    497.0
Freq: 5T, Length: 100, dtype: float64
'''
print(ts.resample('5Min').max())  # 最大值
'''
2020-01-01 00:00:00      4
2020-01-01 00:05:00      9
2020-01-01 00:10:00     14
2020-01-01 00:15:00     19
2020-01-01 00:20:00     24
                      ... 
2020-01-01 07:55:00    479
2020-01-01 08:00:00    484
2020-01-01 08:05:00    489
2020-01-01 08:10:00    494
2020-01-01 08:15:00    499
Freq: 5T, Length: 100, dtype: int64
'''

'''
其中ohlc是又叫美国线（Open-High-Low-Close chart，OHLCchart），
可以呈现类似股票的开盘价、最高价、最低价和收盘价：
'''
print('# 两小时频率的美国线')
print(ts.resample('2h').ohlc())
'''
                     open  high  low  close
2020-01-01 00:00:00     0   119    0    119
2020-01-01 02:00:00   120   239  120    239
2020-01-01 04:00:00   240   359  240    359
2020-01-01 06:00:00   360   479  360    479
2020-01-01 08:00:00   480   499  480    499
'''

print()
print('closed参数设置')
'''可以将closed参数设置为“left”或“right”，以指定开闭区间的哪一端：'''
print(ts.resample('2h').mean())
# 2020-01-01 00:00:00     59.5
# 2020-01-01 02:00:00    179.5
# 2020-01-01 04:00:00    299.5
# 2020-01-01 06:00:00    419.5
# 2020-01-01 08:00:00    489.5
# Freq: 2H, dtype: float64
print(ts.resample('2h', closed='left').mean())  # 结果同上
print(ts.resample('2h', closed='right').mean())
# 2019-12-31 22:00:00      0.0
# 2020-01-01 00:00:00     60.5
# 2020-01-01 02:00:00    180.5
# 2020-01-01 04:00:00    300.5
# 2020-01-01 06:00:00    420.5
# 2020-01-01 08:00:00    490.0
# Freq: 2H, dtype: float64

print()
print('# label参数')
'''使用label可以控制输出结果显示左还是右，但不像closed那样影响计算结果：'''
print(ts.resample('5Min').mean())  # 默认 label='left'
print(ts.resample('5Min', label='right').mean())  # 计算结果一样 可能是格式显示不一样吧 肉眼不大看出来

print()
print('------------------------------------------------------------')
print('\t14.6.4 上采样')

'''
上采样（upsampling）一般应用在图形图像学中，目的是放大图像。
由于原数据有限，放大图像后需要对缺失值进行内插值填充。

在时序数据中同样存在着类似的问题，上例中的数据频率是分钟，我们要对其按30秒重采样：
'''
print(ts.head(3).resample('30S').asfreq())
'''
2020-01-01 00:00:00    0.0
2020-01-01 00:00:30    NaN
2020-01-01 00:01:00    1.0
2020-01-01 00:01:30    NaN
2020-01-01 00:02:00    2.0
Freq: 30S, dtype: float64
'''

'''我们发现由于原数据粒度不够，出现了缺失值，这就需要用.ffill()和.bfill()来计算填充值：'''
print(ts.head(3).resample('30S').ffill())  # 向前填充
'''
2020-01-01 00:00:00    0
2020-01-01 00:00:30    0
2020-01-01 00:01:00    1
2020-01-01 00:01:30    1
2020-01-01 00:02:00    2
Freq: 30S, dtype: int64
'''
print(ts.head(3).resample('30S').bfill())  # 向后填充
'''
2020-01-01 00:00:00    0
2020-01-01 00:00:30    1
2020-01-01 00:01:00    1
2020-01-01 00:01:30    2
2020-01-01 00:02:00    2
Freq: 30S, dtype: int64
'''

print()
print('------------------------------------------------------------')
print('\t14.6.5 重采样聚合')

'''类似于agg API、groupby API和窗口方法API，重采样也适用于相关的统计聚合方法：'''

df = pd.DataFrame(np.random.randn(1000, 3),
                  index=pd.date_range('1/1/2020', freq='S', periods=1000),
                  columns=['A', 'B', 'C']
                  )
print(df)
'''
                            A         B         C
2020-01-01 00:00:00 -1.339569  0.392938 -0.329884
2020-01-01 00:00:01 -0.451720 -0.841686  1.594416
2020-01-01 00:00:02 -0.925979 -0.040797 -0.101541
2020-01-01 00:00:03  0.202807  1.097003 -0.153193
2020-01-01 00:00:04  0.804477 -0.703568 -0.159791
...                       ...       ...       ...
2020-01-01 00:16:35 -2.575767 -0.316534  0.295881
2020-01-01 00:16:36 -2.310698  0.571659 -0.301216
2020-01-01 00:16:37 -0.464210  0.208668 -1.025299
2020-01-01 00:16:38  0.275189 -1.978887  0.257300
2020-01-01 00:16:39  0.238310  0.276412  0.492659

[1000 rows x 3 columns]
'''

print('# 生成Resampler重采样对象')
r = df.resample('3T')
# print(r) DatetimeIndexResampler [freq=<3 * Minutes>, axis=0, closed=left, label=left, convention=start,
# origin=start_day]
print(r.mean())
'''
                            A         B         C
2020-01-01 00:00:00  0.029331  0.071450 -0.052328
2020-01-01 00:03:00  0.174249  0.033578  0.099816
2020-01-01 00:06:00  0.021705  0.113303  0.014374
2020-01-01 00:09:00 -0.026221  0.053145  0.082342
2020-01-01 00:12:00 -0.075547  0.100557 -0.097778
2020-01-01 00:15:00 -0.082372 -0.119609 -0.076147
'''

print()
print('有多个聚合方式：')
print(r['A'].agg([np.sum, np.mean, np.std]))
'''
                           sum      mean       std
2020-01-01 00:00:00  10.195274  0.056640  1.141293
2020-01-01 00:03:00 -11.834165 -0.065745  1.016519
2020-01-01 00:06:00  12.333678  0.068520  0.956461
2020-01-01 00:09:00  -3.122459 -0.017347  0.942420
2020-01-01 00:12:00  14.771384  0.082063  0.977905
2020-01-01 00:15:00  -3.185838 -0.031858  0.910143
'''

print('# 每个列')
print(r.agg([np.sum, np.mean]))
'''
                           sum      mean  ...        sum      mean
2020-01-01 00:00:00   4.317965  0.023989  ...   2.472293  0.013735
2020-01-01 00:03:00   8.653921  0.048077  ...   4.473018  0.024850
2020-01-01 00:06:00   7.267614  0.040376  ...  -0.635775 -0.003532
2020-01-01 00:09:00   5.423832  0.030132  ...   4.825570  0.026809
2020-01-01 00:12:00 -10.215428 -0.056752  ... -13.677265 -0.075985
2020-01-01 00:15:00  -4.046951 -0.040470  ...  -8.146950 -0.081469

[6 rows x 6 columns]
'''

print()
print(r.agg({'A': np.sum,
             'B': lambda x: np.std(x, ddof=1)}))
'''
                             A         B
2020-01-01 00:00:00  -3.465165  0.954057
2020-01-01 00:03:00   7.450780  1.125230
2020-01-01 00:06:00   5.596317  0.980065
2020-01-01 00:09:00   8.468608  1.050400
2020-01-01 00:12:00 -14.944876  0.983348
2020-01-01 00:15:00  -0.679696  0.944394
'''

print()
print('# 用字符指定')
print(r.agg({'A': 'sum', 'B': 'std'}))
'''
                             A         B
2020-01-01 00:00:00  12.566308  1.052866
2020-01-01 00:03:00  -9.757681  0.972359
2020-01-01 00:06:00  18.627415  1.000087
2020-01-01 00:09:00   2.906557  0.917702
2020-01-01 00:12:00  -2.211160  0.994550
2020-01-01 00:15:00  -5.333259  0.934275
'''

print(r.agg({'A': ['sum', 'std'], 'B': ['mean', 'std']}))
'''
                             A                   B          
                           sum       std      mean       std
2020-01-01 00:00:00 -12.735055  1.089689 -0.040202  1.023641
2020-01-01 00:03:00  15.675315  0.927174 -0.005197  1.023730
2020-01-01 00:06:00  22.353234  0.930815  0.017241  0.938834
2020-01-01 00:09:00   5.565829  0.907279  0.183779  0.980010
2020-01-01 00:12:00   2.016427  1.065811 -0.019264  1.027851
2020-01-01 00:15:00 -16.844252  0.832082 -0.031579  0.999082
'''

'''如果索引不是时间，可以指定采样的时间列：'''
print('# date是一个普通列')
# print(df.resample('M',on='date').sum()) # KeyError: 'The grouper name date is not found'
# print(df.resample('M', level='d').sum()) # 多层索引 | # ValueError: The level d is not valid

# 创建示例 DataFrame
data = {
    'date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
    'value': range(100)
}
df = pd.DataFrame(data)

# 确保 'date' 列存在
print('# date是一个普通列')
# print(df.head())  # 查看前几行数据，确保 'date' 列存在

# 使用 'date' 列进行重采样
resampled_df = df.resample('M', on='date').sum()

print(resampled_df)
#             value
# date
# 2024-01-31    465
# 2024-02-29   1305
# 2024-03-31   2325
# 2024-04-30    855


# 创建示例 DataFrame
data = {
    'date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
    'category': ['A'] * 50 + ['B'] * 50,
    'value': range(100)
}
df = pd.DataFrame(data)
print(df)
'''
         date category  value
0  2024-01-01        A      0
1  2024-01-02        A      1
2  2024-01-03        A      2
3  2024-01-04        A      3
4  2024-01-05        A      4
..        ...      ...    ...
95 2024-04-05        B     95
96 2024-04-06        B     96
97 2024-04-07        B     97
98 2024-04-08        B     98
99 2024-04-09        B     99

[100 rows x 3 columns]
'''

# 设置多层索引
df = df.set_index(['date', 'category'])
print(df)
'''
                     value
date       category       
2024-01-01 A             0
2024-01-02 A             1
2024-01-03 A             2
2024-01-04 A             3
2024-01-05 A             4
...                    ...
2024-04-05 B            95
2024-04-06 B            96
2024-04-07 B            97
2024-04-08 B            98
2024-04-09 B            99

[100 rows x 1 columns]
'''

# 使用 'date' 级别进行重采样
resampled_df = df.resample('M', level='date').sum()

# 查看重采样后的 DataFrame
print("\\nResampled DataFrame:")
print(resampled_df)
#             value
# date
# 2024-01-31    465
# 2024-02-29   1305
# 2024-03-31   2325
# 2024-04-30    855

print()
print('迭代采样对象：')
# r 是重采样对象
for name, group in r:
    print("Group:", name)
    print("-" * 20)
    print(group, end="\n\n")

'''输出结果为6组 前5组[180 rows x 3 columns] 最后一组[100 rows x 3 columns]'''

print()
print('------------------------------------------------------------')
print('\t14.6.6 时间类型间转换')

'''介绍一下不同时间概念之间的相互转换。to_period()将DatetimeIndex转换为PeriodIndex：'''

print('# 转换为时间周期')
p = pd.date_range('1/1/2020', periods=5)
print(p)
'''
DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
               '2020-01-05'],
              dtype='datetime64[ns]', freq='D')
'''
print(p.to_period())
'''
PeriodIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
             '2020-01-05'],
            dtype='period[D]')
'''

print()
print('# to_timestamp()')
'''to_timestamp()将默认周期的开始时间转换为DatetimeIndex：'''
pt = pd.period_range('1/1/2020', periods=5)
print(pt.to_timestamp())
'''
DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',
               '2020-01-05'],
              dtype='datetime64[ns]', freq='D')
'''

print()
print('------------------------------------------------------------')
print('\t14.6.7 超出时间戳范围时间')
'''
在介绍时间表示方法时我们说到Pandas原生支持的时间范围大约在
1677年至2262年之间，那么如果分析数据不在这个区间怎么办呢？
可以使用PeriodIndex来进行计算，我们来测试一下：
'''

print('# 定义一个超限时间周期')
print(pd.period_range('1111-01-01', '8888-01-01', freq='D'))
'''
PeriodIndex(['1111-01-01', '1111-01-02', '1111-01-03', '1111-01-04',
             '1111-01-05', '1111-01-06', '1111-01-07', '1111-01-08',
             '1111-01-09', '1111-01-10',
             ...
             '8887-12-23', '8887-12-24', '8887-12-25', '8887-12-26',
             '8887-12-27', '8887-12-28', '8887-12-29', '8887-12-30',
             '8887-12-31', '8888-01-01'],
            dtype='period[D]', length=2840493)
'''

print()
'''可以正常计算和使用。还可以将时间以数字形式保存，在计算的时候再转换为周期数据：'''
print(pd.Series([123_1111, 2008_10_01, 8888_12_12]))
# 0     1231111
# 1    20081001
# 2    88881212
# dtype: int64

# 将整型转为时间周期类型
print(pd.Series([123_1111, 2008_10_01, 8888_12_12])
      .apply(lambda x: pd.Period(year=x // 10000,
                                 month=x // 100 % 100,
                                 day=x % 100,
                                 freq='D')))
'''
0    0123-11-11
1    2008-10-01
2    8888-12-12
dtype: period[D]
'''

# print(123_1111 // 100) # 整除
# print(123_1111 / 100) # 除数
# print(123_1111 % 100) # 取模

print()
print('------------------------------------------------------------')
print('\t14.6.8 区间间隔')
# update20240626

'''
pandas.Interval可以解决数字区间和时间区间的相关问题，
它实现一个名为Interval的不可变对象，该对象是一个有界的切片状间隔。
构建Interval对象的方法如下：
'''
print('# Interval对象构建')
print(pd.Interval(left=0, right=5, closed='right'))  # (0, 5]

print('# 4 是否在1～10之间')
print(4 in pd.Interval(1, 10))  # True
print(pd.Interval(1, 10))  # (1, 10]
print(pd.Interval(1, 10, closed='left'))  # [1, 10)

print('# 10 是否在1～9之间')
print(10 in pd.Interval(1, 10, closed='left'))  # False

# for i in pd.Interval(1,10):  # TypeError: 'pandas._libs.interval.Interval' object is not iterable
#     print(i)

print()
print('将区间转换为整数列表')
# 尝试将区间转换为整数列表 报错！
# print(list(pd.Interval(1,10))) # TypeError: 'pandas._libs.interval.Interval' object is not iterable

# openai建议：
# 创建一个区间
interval = pd.Interval(1, 10, closed='both')
print(interval)  # [1, 10]
# 手动创建区间内的整数列表
int_list = list(range(interval.left, interval.right + 1))

print(int_list)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# for i in int_list:
#     print(i)

'''
参数的定义如下。
left：定值，间隔的左边界。
right：定值，间隔的右边界。
closed：字符，可选right、left、both、neither，分别代表区间是
        在右侧、左侧、同时闭合、都不闭合。默认为right。
'''

print()
'''Interval可以对数字、固定时间、时长起作用，以下是构建数字类型间隔的方法和案例：'''
iv = pd.Interval(left=0, right=5)
print(iv)  # (0, 5]

print('# 可以检查元素是否属于它')
print(3.5 in iv)  # True
print(5.5 in iv)  # False

print('# 可以测试边界值')
# closed ='right'，所以0 < x <= 5
print(0 in iv)  # False
print(5 in iv)  # True
print(0.00001 in iv)  # True

print()
print('创建时间区间间隔：')
print('# 定义一个2020年的区间')
year_2020 = pd.Interval(pd.Timestamp('2020-01-01 00:00:00'),
                        pd.Timestamp('2021-01-01 00:00:00'),
                        closed='left')
print(year_2020)  # [2020-01-01, 2021-01-01)

# 检查指定时间是否在2020年区间里
print(pd.Timestamp('2020-01-01 00:00:00') in year_2020)  # True

# 2020年时间区间的长度
print(year_2020.length)  # 366 days 00:00:00

print()
print('创建时长区间间隔：')
# 定义一个时长区间，3秒到1天

time_deltas = pd.Interval(pd.Timedelta('3 seconds'),
                          pd.Timedelta('1 days'),
                          closed='both'
                          )
print(time_deltas)  # [0 days 00:00:03, 1 days 00:00:00]

# 5分钟是否在时间区间里
print(pd.Timedelta('5 minutes') in time_deltas)  # True

# 时长区间长度
print(time_deltas.length)  # 0 days 23:59:57

print()
print('pd.Interval支持以下属性：')
# 区间闭合之处
print(iv.closed)  # right
# 检查间隔是否在左侧关闭
print(iv.closed_left)  # False
# 检查间隔是否在右侧关闭
print(iv.closed_right)  # True
# 间隔是否为空，表示该间隔不包含任何点
print(iv.is_empty)  # False
# 间隔的左边界
print(iv.left)  # 0
# 间隔的右边界
print(iv.right)  # 5
# 间隔的长度
print(iv.length)  # 5
# 间隔的中点
print(iv.mid)  # 2.5
# 间隔是否在左侧为开区间
print(iv.open_left)  # True
# 间隔是否在右侧为开区间
print(iv.open_right)  # False

print()
print(pd.Interval(0, 1, closed='right').is_empty)  # False
print('# 不包含任何点的间隔为空')
print(pd.Interval(0, 0, closed='right').is_empty)  # True
print(pd.Interval(0, 0, closed='left').is_empty)  # True
print(pd.Interval(0, 0, closed='neither').is_empty)  # True

print('# 包含单个点的间隔不为空')
print(pd.Interval(0, 0, closed='both').is_empty)  # False

'''
# 一个IntervalArray或IntervalIndex返回一个布尔ndarray
# 它在位置上指示Interval是否为空
'''
ivs = [pd.Interval(0, 0, closed='neither'),
       pd.Interval(1, 2, closed='neither')]
print(ivs)  # [Interval(0, 0, closed='neither'), Interval(1, 2, closed='neither')]
print(pd.arrays.IntervalArray(ivs))
'''
<IntervalArray>
[(0, 0), (1, 2)]
Length: 2, dtype: interval[int64, neither]
'''
print(pd.arrays.IntervalArray(ivs).is_empty)  # [ True False]

print('# 缺失值不为空')
ivs = [pd.Interval(0, 0, closed='neither'), np.nan]
print(ivs)  # [Interval(0, 0, closed='neither'), nan]
print(pd.IntervalIndex(ivs))  # IntervalIndex([(0.0, 0.0), nan], dtype='interval[float64, neither]')
print(pd.IntervalIndex(ivs).is_empty)  # [ True False]

print()
'''
pd.Interval.overlaps检查两个Interval对象是否重叠。
如果两个间隔至少共享一个公共点（包括封闭的端点），则它们重叠。
'''
i1 = pd.Interval(0, 2)
i2 = pd.Interval(1, 3)
print(i1.overlaps(i2))  # True

i3 = pd.Interval(4, 5)
print(i1.overlaps(i3))  # False

print('共享封闭端点的间隔重叠：')
i4 = pd.Interval(0, 1, closed='both')
i5 = pd.Interval(1, 2, closed='both')
print(i4.overlaps(i5))  # True

'''只有共同的开放端点的间隔不会重叠：'''
i6 = pd.Interval(1, 2, closed='neither')
print(i4.overlaps(i6))  # False

print()
'''
间隔对象能使用+和*与一个固定值进行计算，此操作将同时应用于对象的两个边界，
结果取决于绑定边界值数据的类型。
以下是边界值为数字的示例：
'''
print(iv)  # (0, 5]
shift_iv = iv + 3
print(shift_iv)  # (3, 8]
extended_iv = iv * 10.0
print(extended_iv)  # (0.0, 50.0]

'''
另外，Pandas还不支持两个区间的合并、取交集等操作，可以使用Python的第三方库portion来实现。

14.6.9 小结
本节主要介绍了在时间操作中的一些综合功能。
由于大多数据库开发规范要求存储时间的时区为UTC，因此我们拿到数据就需要将其转换为北京时间。

时区转换是数据清洗整理的一个必不可少的环节。
对数据的交付使用需要人性化的显示格式，时间的格式化让我们能够更好地阅读时间。
时间重采样让我们可以如同Pandas的groupby那样方便地聚合分组时间。

14.7 本章小结
时序数据是数据类型中一个非常庞大的类型，在我们生活中无处不在，学习数据分析是无法绕开时序数据的。
Pandas构建了多种时间数据类型，提供了多元的时间处理方法，为我们打造了一个适应各种时间场景的时序数据分析平台。

Pandas有关时间处理的更多强大功能有待我们进一步挖掘，也值得我们细细研究。
'''
