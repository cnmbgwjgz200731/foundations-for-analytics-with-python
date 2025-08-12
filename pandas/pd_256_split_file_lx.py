import pandas as pd
import numpy as np
import sys
import time
import os

start_time = time.perf_counter()

df = pd.read_excel('E:/bat/input_files/dfszdfgasrfg.xlsx')
# df = pd.read_csv('E:/bat/input_files/mpos_split_file.csv', encoding='gbk')

# outfile = 'E:/bat/output_files/pandas_out_20240103033.xlsx'
output_dir = 'E:/bat/output_files/split_dir'

# 按照组名列分组
grouped = df.groupby('grandpa_promoter_bd')  # 请替换为实际的组名列名称

# 为每个组创建独立的xlsx文件
for group_name, group_df in grouped:
    # 创建文件名（确保文件名是有效的，没有特殊字符）
    output_file = f'DATA20240107022_各业务多维度数据_{group_name}.xlsx'
    # 完整的输出路径
    output_path = os.path.join(output_dir, output_file)
    # print(output_path)
    print(output_path)
    # 将每个组的数据保存成新的xlsx文件
    group_df.to_excel(output_path, index=False)

print("拆分完成!")

end_time = time.perf_counter()
print(end_time - start_time)
