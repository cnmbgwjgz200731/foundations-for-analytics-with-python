"""
update20251112

1、该项目 测试 移动指定文件夹中 的 xlsx文件 文件格式《姓名_文件名》
2、将文件按照姓名移动到指定 相同姓名的文件夹中

3、将不同姓名文件夹中的xlsx文件 移出到指定父文件夹中。

"""

from datetime import datetime
from glob import glob

import pandas as pd
import zipfile
import smtplib
import shutil
import time
import os

# 文件路径设置
out_folder = 'E:/bat/output_files/split_dir/split_bd/202505'
month_code = pd.Timestamp('now').strftime('%Y%m')
# 创建输出目录
# os.makedirs(out_folder, exist_ok=True)
# print(os.listdir(out_folder))

# bd = []

# for file in os.listdir(out_folder):
#     # print(file)
#     if file.endswith('.xlsx'):
#         # print(file.split('_')[0])
#         bd_name = file.split('_')[0]
#         bd.append(bd_name)
#         # zip_folder = os.path.join(f'{out_folder}', bd_name)
#         # os.makedirs(zip_folder, exist_ok=True)
#         # print(zip_folder)
# print(bd)
# print(set(bd))
#
# for name in set(bd):
#     # print(name)
#     zip_folder = os.path.join(out_folder, name)
#     # print(zip_folder)
#     os.makedirs(zip_folder, exist_ok=True)


# 存放xlsx文件的根目录（根据实际情况修改）
# root_dir = "./excel_files"

# 遍历根目录下的所有xlsx文件
for filename in os.listdir(out_folder):
    # 只处理xlsx文件
    if filename.endswith(".xlsx"):
        try:
            # 提取姓名（假设文件名格式严格为“姓名_xxx.xlsx”）
            name = filename.split("_")[0]
            # 构造源文件路径
            src_path = os.path.join(out_folder, filename)
            # 构造目标文件夹路径
            target_dir = os.path.join(out_folder, name)
            # 构造目标文件路径（移动后的路径）
            target_path = os.path.join(target_dir, filename)

            # 若目标文件夹不存在，则创建
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)  # exist_ok=True避免文件夹已存在时报错

            # 移动文件
            shutil.move(src_path, target_path)
            print(f"已移动：{filename} -> {target_dir}")

        except IndexError:
            # 处理文件名格式错误（如没有“_”分隔符）
            print(f"文件名格式错误，跳过：{filename}")
        except Exception as e:
            # 捕获其他异常（如权限问题）
            print(f"移动失败 {filename}：{str(e)}")

# for folder in os.listdir(out_folder):
#     # if os.path.isdir(folder):
#     pattern = os.path.join(out_folder, folder)
#     # print(pattern)
#     if os.path.isdir(pattern):
#         # for filename in glob(os.path.join(pattern, f'*.xlsx')):
#         for filename in os.listdir(pattern):
#             # print(filename)
#             source_dir = os.path.join(pattern, filename)
#             # print(source_dir)
#             target_dir = os.path.join(out_folder, filename)
#
#             shutil.move(source_dir, target_dir)