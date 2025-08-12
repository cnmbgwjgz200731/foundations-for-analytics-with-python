import pandas as pd
import os
import chardet
import time
import datetime

start_time = time.perf_counter()
# 定义 CSV 文件路径和目标文件夹
input_file = 'E:/bat/input_files/temp20240528023_split_file.csv'
output_path = 'E:/bat/output_files/split_dir'

'''
update20240530
读取指定文件目录中，指定类型（如xlsx、csv、pdf等）的文件；
然后将文件名称，文件创建时间、文件更新时间、文件大小、文件类型5个字段保存为dataframe

然后将数据保存到指定文件目录的文件中。
'''

# 指定文件目录
directory = 'E:/bat/input_files'

# 获取目录下所有文件
files = os.listdir(directory)

# print(files)

data = []

# 遍历文件
for file in files:
    file_path = os.path.join(directory, file)

    if not os.path.isfile(file_path) or not file.lower().endswith('.xlsx'):
        continue  # 跳过非文件项和非xlsx文件

    file_stat = os.stat(file_path)
    # # print(file_path)
    # print(file_stat)

    # 提取文件信息
    file_name = file
    create_time = datetime.datetime.fromtimestamp(file_stat.st_ctime)
    update_time = datetime.datetime.fromtimestamp(file_stat.st_mtime)
    # print(create_time,update_time)
    file_size = file_stat.st_size
    file_type = file.split('.')[-1]
    # print(file_size,file_type)
    data.append([file_name, create_time, update_time, file_size, file_type])

# 创建DataFrame
df = pd.DataFrame(data, columns=['文件名称', '创建时间', '更新时间', '文件大小', '文件类型'])

# 保存为xlsx文件
output_file = 'E:/bat/output_files/file_info_20240530041.xlsx'
df.to_excel(output_file, index=False)

print("文件信息已保存至:", output_file)
