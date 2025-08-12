import pandas as pd
import faker

# 17.2.6 编写年会抽奖程序

"""
某公司年会设有抽奖环节，奖品设有三个等级：
    一等奖一名，
    二等奖两名，
    三等奖三名。
    要求一个人只能中一次奖。
"""

f = faker.Faker('zh-cn')
df = pd.DataFrame([f.name() for i in range(50)], columns=['name'])
# print(df)
# 增加一列用于存储结果
df['等级'] = ''
print()
print(df.tail())
"""上面构造了50名员工的名单，抽奖时执行以下代码："""

print()
# 配置信息，第一位为抽奖人数，第二位为奖项等级
win_info = (3, '三等奖')

# df.sample(win_info[0])

# 创建一个筛选器变量
# filter = df.index.isin(df.sample(win_info[0]).index)
# filter = df.index.isin(df.sample(win_info[0]).index) & (df.等级.isna())
filter = df.index.isin(df.sample(win_info[0]).index) & ~(df.等级.isna())
# 执行抽奖，将等级写入
df.loc[filter, '等级'] = win_info[1]
# filter
# df.等级.isna()  # True
# ~(df.等级.isna()) # False
# print(filter)
print(df.loc[filter, '等级'])
# 显示本次抽奖结果
# print(df.loc[df.等级 == win_info[1]])
print(df.loc[df.等级 == win_info[1]])


print()
# 配置信息，第一位为抽奖人数，第二位为奖项等级
win_info_t = (2, '二等奖')

filter = df.index.isin(df.sample(win_info_t[0]).index) & ~(df.等级.isna())

# 执行抽奖，将等级写入
df.loc[filter, '等级'] = win_info_t[1]

print()
# 配置信息，第一位为抽奖人数，第二位为奖项等级
win_info_r = (1, '一等奖')

filter = df.index.isin(df.sample(win_info_r[0]).index) & ~(df.等级.isna())

# 执行抽奖，将等级写入
df.loc[filter, '等级'] = win_info_r[1]


# 显示所有结果
print(df[~(df.等级=='')].sort_values(by='等级'))
print(df[~(df.等级=='')].groupby(['等级','name']).max())