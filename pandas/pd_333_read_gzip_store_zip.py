import numpy as np
import pandas as pd

from io import StringIO
from io import BytesIO

'''
update20240124
读取指定tar.gz文件，分块后，读取2w条数据到指定文件，并将该文件压缩到指定zip文件中
'''

print()
print('# tar.gz文件超大文件分块处理')
print()
'''
compression（压缩格式）用于对磁盘数据进行即时解压缩。如果为“infer”,
且filepath_or_buffer是以.gz、.bz2、.zip或.xz结尾的字符串，
则使用gzip、bz2、zip或xz，否则不进行解压缩。
如果使用zip，则ZIP文件必须仅包含一个要读取的数据文件。设置为None将不进行解压缩。

# 可选值有'infer'、'gzip'、'bz2'、'zip'、'xz'和None，默认为'infer'
'''
in_file = 'E:/bat/input_files/jiangshuang/mchnt.tar.gz'
in_file_thr = 'E:/bat/input_files/jiangshuang/48200000_20231225_091446_jiangs_imchnt'

out_zip = 'E:/bat/output_files/pandas_out_20240124034.zip'

chunk_size = 5000  # 你可以根据你的内存大小调整这个值
# encoding='gbk报错 || encoding='utf-8'也报错
df = pd.read_csv(in_file, compression='gzip', encoding='ISO-8859-1',
                 chunksize=chunk_size, usecols=[0], skiprows=1, names=['id'], header=None, low_memory=False,
                 skip_blank_lines=True, engine='c')  # ,dtype={0:str} 添加后没用

chunks = []  # # 用于存储数据块的列表

# compression_opts = dict(method='zip',archive_name='out.csv')
# df.to_csv('E:/bat/output_files/pandas_out_20240124033.zip',mode='w',header=True,index=False,compression=compression_opts)

# 运行成功  耗时大概2min 有800万+
for index, chunk in enumerate(df):
    # print(index,chunk)
    if index == 0:
        # chunk.to_csv('E:/bat/output_files/pandas_out_20240118041.csv',mode='w',header=True,index=False)
        # chunk.to_csv('E:/bat/output_files/pandas_out_20240124033.zip', mode='w', header=True, index=False,\
        #           compression=compression_opts)
        chunks.append(chunk)
    elif 1 <= index <= 3:
        # chunk.to_csv('E:/bat/output_files/pandas_out_20240118041.csv', mode='a', header=False, index=False)
        # chunk.to_csv('E:/bat/output_files/pandas_out_20240124033.zip', mode='a', header=False, index=False,\
        #           compression=compression_opts)
        chunks.append(chunk)
    else:
        break
        # chunk.to_csv('E:/bat/output_files/pandas_out_20240118041.csv', mode='a',header=False,index=False)
    # break  # 如果你只想看第一个 chunk 的话，可以使用 break 语句

# 合并所有数据块
df_combined = pd.concat(chunks, ignore_index=True)

# 创建一个包含out.csv的压缩文件out.zip
# archive_name='pandas_out_20240124035.csv' 如果加绝对路径，会有循环目录生成 |直接输入文件名即可！
compression_opts = dict(method='zip', archive_name='pandas_out_20240124035.csv')

# 将合并后的DataFrame写入到压缩文件
df_combined.to_csv(out_zip, index=False, compression=compression_opts)
