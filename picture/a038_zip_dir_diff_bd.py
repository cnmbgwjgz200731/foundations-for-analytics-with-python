"""
1、 指定文件夹中有一批xlsx文件，文件名称格式： 《姓名_年月日_数据内容名称.xlsx》 ；
2、 想将这批xlsx文件，按照姓名分别移动到 该姓名的文件夹中，且创建对应的zip文件，zip文件名称格式为 《姓名_年月日.zip》
"""

from glob import glob

import pandas as pd
import zipfile
import smtplib
import shutil
import time
import os

# 附件文件夹路径
attach_folder = r'E:\bat\output_files\split_dir\split_bd\202504'

# target_bds = ['周元', '徐雅雯', '杨春艳', '曾聪', '肖伟', '张晓红', '龚建强', '李彩霞', '岳琦琦']  # 生产环境
# target_bds = ['汪玉敏', '蒋爽', '姜银花', '王晶', '丁亮', '徐建亮', '刘鑫']  # 测试环境
month_code = (pd.Timestamp('now') - pd.DateOffset(months=1)).strftime('%Y%m')
# 存储每个BD的文件路径，用于后续压缩和邮件发送
bd_files_dict = {}


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

for bd in list(bd_emails.keys()):
    # 创建文件名（替换非法字符）
    safe_bd = str(bd).replace('/', '_').replace('\\', '_').replace(':', '_')
    # bd_file_name = f'{safe_bd}_{month_code}_{file_name}.xlsx'

    # 创建BD维度目录
    output_bd_path = os.path.join(attach_folder, safe_bd)
    os.makedirs(output_bd_path, exist_ok=True)

    # 记录BD文件夹路径，用于后续压缩
    if safe_bd not in bd_files_dict:
        bd_files_dict[safe_bd] = {
            'path': output_bd_path,
            'original_name': bd
        }

# print(bd_files_dict)
# print(pd.DataFrame(bd_files_dict))

# 为每个目标BD创建压缩包
for bd_name, bd_info in bd_files_dict.items():
    original_bd_name = bd_info['original_name']
    bd_folder_path = bd_info['path']
    # print(bd_folder_path) # E:\bat\output_files\split_dir\split_bd\202505\岳琦琦

    # 创建zip文件路径
    zip_path = os.path.join(attach_folder, f"{bd_name}_{month_code}.zip")

    # print(zip_path)

    # 创建zip文件
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_f:
        # 遍历BD文件夹中的所有文件
        for root, dirs, files in os.walk(bd_folder_path):
            # print(files)
            for file in files:
                # print(file)  # E:\bat\output_files\split_dir\split_bd\202505\岳琦琦_202511.zip
                file_path = os.path.join(root, file)
                # print(file_path)  # E:\bat\output_files\split_dir\split_bd\202505\岳琦琦_202511.zip
#                 # 在zip文件中创建相对路径
                attachment = os.path.relpath(file_path, bd_folder_path)
                # print(attachment)  # E:\bat\output_files\split_dir\split_bd\202505\岳琦琦_202511.zip
                zip_f.write(file_path, attachment)








#
    print(f"已创建BD [{original_bd_name}] 的压缩包: {zip_path}")
