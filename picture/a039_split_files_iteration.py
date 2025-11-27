"""
update20251114
优化按照BD维度拆分多工作簿、sheets拆分数据；
将拆分好后的文件，按照BD维度建立文件夹，并创建对应的文件压缩包。
只处理指定的BD（张三、李四、王二），并为每个BD发送邮件通知。

update20251117
取消 target_bds 替换为 bd_emails 中的bd 避免修改多个变量

update20251125
1、避免文件打开无读取权限
2、如果一个工作簿中所有sheet没有对应的bd，则不输出bd对应的文件  避免有bd知道其它的业务
"""

from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from email.header import Header
from datetime import datetime
from glob import glob

import pandas as pd
import zipfile
import smtplib
import shutil
import time
import os

# 文件路径设置
out_folder = 'E:/bat/output_files/split_dir/split_bd/202504'
# out_folder = 'E:/bat/output_files/split_dir/split_bd/202511'
# 创建输出目录
os.makedirs(out_folder, exist_ok=True)

# source_dir = 'E:\桌面08\测试\临时拆分\拆分数据-胡国强\拆分分润数据工具\胡国强拆分文件_每月\\2025-06\PLXSQ20210421001_SIM卡已扣费明细_月报03_202511_流程.xlsx'
source_dir = 'E:/桌面08/测试/临时拆分/拆分数据-胡国强/拆分分润数据工具/胡国强拆分文件_每月/2025-05/*.xlsx'
# source_files = glob(source_dir)
source_files = [f for f in glob(source_dir) if not os.path.basename(f).startswith('~$')]

# 记录开始时间
start_time = datetime.now()
# month_code = datetime.now().strftime('%Y-%m')
month_code = (pd.Timestamp('now') - pd.DateOffset(months=1)).strftime('%Y-%m')
# 只处理这些BD
# target_bds = ['张三', '李四', '王二', '宗和']  # 测试环境 取消

# 邮件配置
mail_host = os.getenv('SMTP_HOST')  # 根据您的邮件服务商更改
mail_port = os.getenv('SMTP_PORT')  # 通常587用于TLS  # 465端口应该使用SMTP_SSL
mail_user = os.getenv('SMTP_USER')
mail_pass = os.getenv('SMTP_PASS')  # 建议使用应用专用密码而非登录密码
company_name = 'SFT'

bd_emails_file = r'E:\bat\output_files\split_dir\split_bd\bd_emails.xlsx'
df_bd = pd.read_excel(bd_emails_file)  # 使用pandas读取Excel文件

bd_emails = {}

for i, row in df_bd.iterrows():
    if row.get('status') == 1:
        to_list = [to_email.strip() for to_email in str(row['email']).split(';') if to_email.strip()]
        cc_list = [cc_email.strip() for cc_email in str(row['cc_emails']).split(';') if cc_email.strip()]
        bd_emails[row['name']] = {
            'to': to_list,
            'cc': cc_list
        }

# 存储每个BD的文件路径，用于后续压缩和邮件发送
bd_files_dict = {}

for source_file in source_files:
    # 读取Excel文件的所有sheet
    all_sheets = pd.read_excel(source_file, sheet_name=None)

    file_name = os.path.splitext(os.path.basename(source_file))[0]

    # 为每个目标BD创建独立工作簿
    valid_bd_count = 0
    for bd in list(bd_emails.keys()):
        # 先检查该BD在所有工作表中是否有数据
        has_data = False
        sheets_to_write = {}  # 存储有数据的工作表

        # 遍历所有工作表检查是否有数据
        for sheet_name, df in all_sheets.items():
            # 确保BD列存在
            if 'BD' not in df.columns:
                continue

            # 统一数据类型
            df['BD'] = df['BD'].astype(str)

            # 筛选当前BD的数据
            bd_data = df[df['BD'] == bd]

            if not bd_data.empty:
                has_data = True
                sheets_to_write[sheet_name] = bd_data

        # 如果没有数据，跳过文件创建
        if not has_data:
            print(f"BD [{bd}] 在本文件中无数据，不创建文件")
            continue

        # 创建文件名（替换非法字符）
        safe_bd = str(bd).replace('/', '_').replace('\\', '_').replace(':', '_')
        bd_file_name = f'{safe_bd}_{month_code}_{file_name}.xlsx'

        # 创建BD维度目录
        output_bd_path = os.path.join(out_folder, safe_bd)
        os.makedirs(output_bd_path, exist_ok=True)

        # 记录BD文件夹路径，用于后续压缩
        if safe_bd not in bd_files_dict:
            bd_files_dict[safe_bd] = {
                'path': output_bd_path,
                'original_name': bd
            }

        output_path = os.path.join(output_bd_path, bd_file_name)

        # 创建Excel写入器并写入有数据的工作表
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, bd_data in sheets_to_write.items():
                bd_data.to_excel(
                    writer,
                    sheet_name=sheet_name[:30],  # 确保sheet名称不超过31字符
                    index=False,
                    freeze_panes=(1, 1)
                )

        valid_bd_count += 1
        print(f"已创建BD [{bd}] 的工作簿: {bd_file_name}")

# 为每个BD创建压缩包并发送邮件
print("开始创建BD压缩包并发送邮件...")

# 计算并输出耗时
end_time = datetime.now()
total_seconds = (end_time - start_time).total_seconds()
print(f'程序执行完成，耗时: {total_seconds:.4f} 秒')
print(f'共生成 {len(bd_emails)} 个目标BD工作簿')
