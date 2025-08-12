# update20240827
# 17.3.7 自动邮件报表

import os
import sys
from drymail import SMTPMailer, Message
import matplotlib.pyplot as plt



# 获取配置
host = os.getenv('SMTP_HOST', 'mail.shengpay.com')
port = 465  # 端口
user = os.getenv('SMTP_USER', 'liuxin05@shengpay.com')
password = os.getenv('SMTP_PASSWORD', 'lx@BS12345')

# 配置发件服务
client = SMTPMailer(
    host=host,  # 发件服务器
    port=port,  # 端口号
    user=user,  # 账号
    password=password,  # 密码
    # tls=True  # 如果使用SSL则改为ssl=True
    ssl=True
)

# 构造邮件
message = Message(
    subject='Congrats on the new job!',  # 邮件主题
    # sender=('John Doe', 'john@email.com'),  # 发件人
    # receivers=[('Jane Doe', 'jane@message.com'), 'jane.doe@mail.io'],  # 收件人
    # cc=[('Jane Doe', 'jane@message.com')],  # 抄送
    sender=('刘鑫', 'liuxin05@shengpay.com'),  # 发件人
    receivers=['liuxin05@shengpay.com', 'lsj_883721@qq.com'],  # 收件人
    cc=['liuxin05@shengpay.com'],  # 抄送
    bcc=['lsj_883721@qq.com'],  # 密送
    text='When is the party? ;)',  # 纯文本
    html='<h1>Hello World!</h1>'  # HTML优先
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

# 发出邮件
try:
    client.send(message)
    print("邮件发送成功！")
except Exception as e:
    print(f"邮件发送失败：{e}")
