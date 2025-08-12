import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 构造数据
import faker

f = faker.Faker('zh-cn')
num = 30_000

df = pd.DataFrame({
    '用户': [f.name() for i in range(num)],
    '购买日期': [f.date_between(start_date='-1y', end_date='today') for i in range(num)],
    '金额': [f.random_int(10, 100) for i in range(num)]
})

# 数据类型转换
df = df.astype({'购买日期': 'datetime64[ns]'})

# 计算R值
r = (
    df.groupby('用户')
    .apply(lambda x: (pd.Timestamp('today') - x['购买日期'].max()))
    .dt
    .days
)

# 计算F值
f = (
    df.groupby('用户')
    .apply(lambda x: x['购买日期'].nunique())
)

# 计算每个用户的总金额
total_amount = df.groupby('用户')['金额'].sum()

# 合并RFM数据并评分
df_score = (
    pd.DataFrame({'r': r, 'f': f})
    .assign(m=lambda x: total_amount / x.f)
    .assign(r_s=lambda x: pd.qcut(x.r, q=3, labels=[3, 2, 1]).astype(int))
    .assign(f_s=lambda x: pd.cut(x.f, bins=[0, 2, 5, float('inf')], labels=[1, 2, 3], right=False).astype(int))
    .assign(m_s=lambda x: pd.cut(x.m, bins=[0, 30, 60, float('inf')], labels=[1, 2, 3], right=False).astype(int))
)

# 打印结果
print(df_score)
print(df_score.dtypes)

# 分值归一化
df_score_rfm = (
    df_score
    .assign(r_e=lambda x: (x.r_s > x.r_s.mean()).astype(int))
    .assign(f_e=lambda x: (x.f_s > x.f_s.mean()).astype(int))
    .assign(m_e=lambda x: (x.m_s > x.m_s.mean()).astype(int))
    .assign(label=lambda x: x.r_e * 100 + x.f_e * 10 + x.m_e)
)

# 打印结果
print(df_score_rfm)

# 标签映射
label_names = {
    111: '重要价值用户',
    110: '一般价值用户',
    101: '重要发展用户',
    100: '一般发展用户',
    11: '重要保持用户',
    10: '一般保持用户',
    1: '重要挽留用户',
    0: '一般挽留用户'
}

df_score_rfm = df_score_rfm.assign(label_names=lambda x: x.label.map(label_names))

# 打印结果
print(df_score_rfm)

# 统计占比数据
df_filter = (
    df_score
    .reset_index()
    .groupby('label')
    .用户
    .count()
    .reset_index()
    .assign(rate=lambda d: d.用户 / d.用户.sum())
    # .sort_values()
    # .plot
    # .bar()
)

df_filter.columns = ['label', 'total', 'rate']
df_filter


# 图形可视化
from matplotlib.ticker import FuncFormatter
# 使用FuncFormatter类创建一个格式化函数formatter，将Y轴的刻度值乘以100并格式化为两位小数的百分数。

# 按照 total 列从大到小排序
df_filter = df_filter.sort_values(by='total', ascending=False).reset_index(drop=True)

# 创建图形和子图
fig, ax1 = plt.subplots()

# 创建条形图（total）
ax1.bar(df_filter['label'], df_filter['total'], color='b', alpha=0.6)
ax1.set_xlabel('label')
ax1.set_ylabel('Total', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# 创建共享X轴的第二个Y轴
ax2 = ax1.twinx()
ax2.plot(df_filter['label'], df_filter['rate'], color='r', marker='o')
ax2.set_ylabel('Rate', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# 设置右Y轴刻度格式为百分数并保留两位小数
formatter = FuncFormatter(lambda y, _: f'{y * 100:.2f}%')
ax2.yaxis.set_major_formatter(formatter)

# 设置图形标题
plt.title('Total and Rate by Label')

# 显示图形
plt.show()