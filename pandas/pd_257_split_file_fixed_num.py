import pandas as pd
import os
import chardet

# 定义 CSV 文件路径和目标文件夹
# E:/bat/input_files/temp20240426055_split_csv_double_column_js.csv
# 'E:/bat/input_files/temp20240426055_split_data_js.csv'
# E:/bat/input_files/import_data_source.csv
csv_file_path = 'E:/bat/input_files/temp20240426055_split_data_js.csv'
# csv_file_path = 'E:/bat/input_files/import_data_source.xlsx'
outfile_path = 'E:/bat/output_files/split_dir'  # 确保此文件夹已存在或由脚本创建

# 确保目标文件夹存在
if not os.path.exists(outfile_path):
    os.makedirs(outfile_path)

# 使用 chardet 来检测前 10000 个字节的编码
# with open(csv_file_path, 'rb') as f:
#     # 只读取前 10000 个字节进行编码检测
#     raw_data = f.read(10000)
#     result = chardet.detect(raw_data)
#     file_encoding = result['encoding']

# 读取 CSV 文件
# 按照激活数量 降序 授权商户号 顺序 排列
# 仅取2列
df = pd.read_csv(csv_file_path, delimiter=',', encoding='GBK')
df = df.sort_values(['quanity', 'auth_merchant_no'], ascending=[False, True])
df = df[['auth_merchant_no', 'artif_nm']]
print(len(df))
# print(df)

# 使用 chardet 检测文件编码
# with open(csv_file_path, 'rb') as f:
#     result = chardet.detect(f.read())
#     file_encoding = result['encoding']


# 打印检测到的编码
# print("Detected encoding:", file_encoding)

# 每个 Excel 文件的行数
rows_per_file = 90000
# 计算需要分割成多少个文件
num_files = len(df) // rows_per_file + int(len(df) % rows_per_file != 0)

# // 取整运算符 例： 10 // 3 = 3 ，11 // 3 = 3， 12 // 3 = 4
# % 取模运算符 例： 10 % 3 = 1， 11 % 3 = 2， 12 % 3 = 0


# num_files = sum(1 for chunk in df_chunks)
# print(num_files)


# 新增列
# df['num'] = (df.index + 2)/2 # round((df.index + 1)/2,0) # (df.index + 1)/2  len(df)/2
# df['num'] = df['num'].astype(int)
# code_str = '000'
# df['num'] = code_str + df['num']


# print(df)

# 分割并写入 Excel 文件
for i in range(num_files):
    # 计算每个文件的起始和结束索引
    start_index = i * rows_per_file
    end_index = (i + 1) * rows_per_file if (i + 1) * rows_per_file < len(df) else len(df)

    # 拆分 DataFrame
    df_subset = df.iloc[start_index:end_index]

    # 定义 Excel 文件路径
    excel_file_path = os.path.join(outfile_path, f'splitted_file_{i + 1}.xlsx')

    # 将 DataFrame 写入 Excel 文件
    df_subset.to_excel(excel_file_path, index=False)

print(f'Successfully split the CSV into {num_files} Excel files.')
