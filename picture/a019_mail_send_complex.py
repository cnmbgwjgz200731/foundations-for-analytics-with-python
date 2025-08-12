# update20240827
# 17.3.7 自动邮件报表
# update20240925 带html效果的邮件
# http://localhost:8889/notebooks/pandas_test001.ipynb
# jupyter notebook有测试 很多邮件中的参数 比如为附件添加最新值 多个附件等 可以参考

import os
import pandas as pd
from drymail import SMTPMailer, Message

# 获取配置
host = os.getenv('SMTP_HOST', 'mail.shengpay.com')
port = 465  # 端口
user = os.getenv('SMTP_USER', 'liuxin05@shengpay.com')
password = os.getenv('SMTP_PASSWORD', 'lx@BS23546')  # 只要变量中密码正确，默认密码错误也能发送邮件成功！

# 配置发件服务
client = SMTPMailer(
    host=host,
    port=port,
    user=user,
    password=password,
    ssl=True
)

# 构造邮件
message = Message(
    subject='数据日报',  # 邮件主题
    sender=('刘鑫', 'liuxin05@shengpay.com'),  # 发件人
    receivers=['liuxin05@shengpay.com', 'lsj_883721@qq.com'],  # 收件人
    cc=['liuxin05@shengpay.com'],  # 抄送
    bcc=['lsj_883721@qq.com']  # 密送
)

# 构造附件
try:
    with open('E:/bat/output_files/pandas_out_20240814031_005.xlsx', 'rb') as pdf_file:
        message.attach(
            filename='pandas_out_20240814031_005.xlsx',
            data=pdf_file.read(),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
except FileNotFoundError:
    print("附件文件未找到，请检查文件路径。")

# 创建 DataFrame
df1 = pd.DataFrame({
    'day': pd.date_range('2024-08-01', '2024-08-07'),
    'gmv': [1, 2, 3, 4, 5, 6, 7]
})

df2 = pd.DataFrame({
    'day': pd.date_range('2024-08-01', periods=7),
    'gmv': [7, 6, 5, 4, 3, 2, 1]
})

dfs = (
    pd.merge(df1, df2, on='day', suffixes=('1', '2'))
    .assign(diff=lambda x: (x.gmv1 - x.gmv2) / x.gmv2)
    .assign(diff_v=lambda x: (x.gmv2 - x.gmv1))
    .sort_values('diff', ascending=False)
    .query('diff != 0')
    .reset_index()
    .loc[:, ['day', 'diff', 'diff_v']]
    .style
    .format({'diff': "{:.6%}", 'diff_v': "{:,.2f}"})
    .bar(subset=['diff'], vmin=0, vmax=0.001, color='yellow')
)

df_html = dfs.to_html(caption='数据日报')

# 定义样式
css = '''
<style>
    table {
        border: 1px solid #aac1de;
        border-collapse: collapse;
        border-spacing: 0;
        color: black;
        text-align: center;
        font-size: 11px;
        min-width: 100%;
    }

    thead {
        border-bottom: 1px solid #aac1de;
        vertical-align: bottom;
        background-color: #eff5fb;
    }

    tr {
        border: 1px dotted #aac1de;
    }

    td {
        vertical-align: middle;
        padding: 0.5em;
        line-height: normal;
        white-space: normal;
        max-width: 150px;
    }

    th {
        font-weight: bold;
        vertical-align: middle;
        padding: 0.5em;
        line-height: normal;
        white-space: normal;
        max-width: 150px;
        text-align: center;
    }
</style>    
'''

# 组合 HTML 内容
html_content = css + df_html

# 设置邮件正文
message.html = html_content

# 发出邮件
try:
    client.send(message)
    print("邮件发送成功！")
except Exception as e:
    print(f"邮件发送失败：{e}")