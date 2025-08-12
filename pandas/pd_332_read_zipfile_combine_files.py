import pandas as pd
import zipfile
import os

'''
读取输入文件：E:/bat/input_files/Desktop.zip

该zip压缩包里有3个.xlsx文件,且结构一致，列标题一致。
要求如下：
使用pandas读取该指定压缩包中的3个.xlsx文件；
读取指定3个文件中的Sheet1表中第2列的数据，
并将所有数据合并后,去除重复值
输出到指定文件E:/bat/output_files/pandas_out_20240124031.csv'中

step by one step please!
'''

# 定义输入压缩文件路径和输出文件路径
input_zip = 'E:/bat/input_files/Desktop.zip'
output_csv = 'E:/bat/output_files/pandas_out_20240124031.csv'

# 使用 zipfile 库打开压缩文件
with zipfile.ZipFile(input_zip, 'r') as z:
    # 获取压缩包内所有文件的列表
    filenames = z.namelist()

    # 对文件名进行解码尝试，这里假设文件名使用的是GBK编码
    # 如果您知道正确的编码，可以将 'gbk' 替换为正确的编码
    decoded_filenames = [name.encode('cp437').decode('gbk') for name in filenames]
    # decoded_filenames = [name.encode('CP437').decode('GB2312') for name in filenames]
    #
    # print(decoded_filenames)
    # 尝试解决中文乱码问题
    # 这里列出一些常见的中文编码，可以根据需要尝试不同的编码
    # encodings = ['utf-8', 'gbk', 'gb2312', 'big5']
    # for encoding in encodings:
    #     try:
    #         # 对文件名进行解码尝试
    #         decoded_filenames = [name.encode('cp437').decode(encoding) for name in filenames]
    #         print("Decoded with encoding:", encoding)
    #         print(decoded_filenames)
    #         break  # 成功解码，跳出循环
    #     except UnicodeDecodeError:
    #         # 解码失败，尝试下一种编码
    #         continue

    # 过滤出所有的 .xlsx 文件
    # xlsx_files = [f for f in filenames if f.endswith('.xlsx')]
    xlsx_files = [f for f in decoded_filenames if f.endswith('.xlsx')]
    # print(xlsx_files)
    #
    # 初始化一个空的 DataFrame 用于存储所有数据
    all_data = pd.DataFrame()
    #
    # 遍历所有的 .xlsx 文件
    for xlsx in xlsx_files:
        # print(xlsx)
        #         # 从压缩包中读取 .xlsx 文件内容
        with z.open(xlsx) as f:
            # 使用 pandas 读取工作表中的第二列数据
            # 由于列是从 0 开始计数的，所以第二列的索引是 1
            df = pd.read_excel(f, sheet_name='Sheet1', usecols=[1])

            # 将数据添加到汇总的 DataFrame 中
            all_data = pd.concat([all_data, df], ignore_index=True)
#
# 去除重复数据
all_data = all_data.drop_duplicates()

# 将合并后的数据输出到 CSV 文件
all_data.to_csv(output_csv, index=False)
