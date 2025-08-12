import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
csv_file = 'E:/bat/input_files/20240307_110738.csv'
csv_file2 = 'E:/bat/input_files/20240307_110739.csv'

df = pd.read_csv(csv_file2,encoding='GBK')

# 过滤出 '交易日' 大于等于 '2024-03-01' 的数据
# df['交易日'] = pd.to_datetime(df['交易日'])  # 确保 '交易日' 是日期类型
# df1 = df[df['交易日'] >= '2024-02-01']

# 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 将 'trade_day' 列转换为 datetime 类型
# 注意：这里假设你的日期列名为 'trade_day'，请根据你的实际情况更改
df['trade_day'] = pd.to_datetime(df['trade_day'], format='%Y/%m/%d')

# 确定数据集中最近的日期
# 如果CSV文件已经是按日期排序的，那么最近的日期应该是最后一行的日期
# 如果不是排序的，你需要先排序或者使用 max() 函数找到最近的日期
latest_date = df['trade_day'].max()

# 筛选出最近30天的数据
# df_last_30_days = df[(df['trade_day'] > (latest_date - pd.Timedelta(days=30))) & (df['trade_day'] <= latest_date)]
df1 = df[(df['trade_day'] > (latest_date - pd.Timedelta(days=30))) & (df['trade_day'] <= latest_date)]
# 打印结果
# print(df_last_30_days)


# print(df.head())
# print(df.dtypes)


# 获取不同的 '刷卡行' 数量
unique_banks = df1['bank_name'].unique()
n_banks = len(unique_banks)
# print(unique_banks)

# 创建一个足够大的图来容纳所有子图
fig, axes = plt.subplots(n_banks, 1, figsize=(10, 5 * n_banks), sharex=True)

# 如果只有一个刷卡行，将 axes 包装为列表
if n_banks == 1:
    axes = [axes]

# 绘制每个刷卡行的子图
for ax, bank in zip(axes, unique_banks):
    # 筛选出当前刷卡行的数据
    df_bank = df1[df1['bank_name'] == bank]

    # 将数据转换为 numpy 数组
    dates = df_bank['trade_day'].values  # 转换为 numpy 数组
    transactions = df_bank['minus_lag_num'].values  # 转换为 numpy 数组

    # 绘制折线图（确保使用 numpy 数组）
    ax.plot(dates, transactions, label=bank)

    # 设置子图标题
    ax.set_title(f'bank_name：{bank}')

    # 显示图例
    ax.legend()

# 调整子图布局
plt.tight_layout()

# 显示图表
plt.show()