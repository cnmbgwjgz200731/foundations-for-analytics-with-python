import numpy as np
import pandas as pd
import warnings
import xlrd

from io import StringIO
from io import BytesIO

input_file = 'E:/bat/input_files/sales_2013.xlsx'
lx_test = 'E:/bat/input_files/split_mpos_less.xlsx'
xls_file = 'E:/bat/input_files/sales_2013.xls'
m_file = 'E:/bat/input_files/winequality-red.csv'
big_file = 'E:/bat/input_files/dq_split_file.xlsx'

path = 'E:/bat/output_files/pandas_read_csv_20240118.csv'

# 忽略特定的警告 | 当遇到 openpyxl 中的 UserWarning 类型的警告时，它们将不会被打印出来。
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

print()
print('------------------------------------------------------------')
print('\t3.3 读取Excel')
print('\t3.3.1 语法')
print()
'''
update20240122

data = 'a,b,c,1\n0,1,2,3\n1,2,3,5\n2,5,3,4'

pd.read_excel(io, sheet_name=0, header=0,
              names=None, index_col=None,
              usecols=None, squeeze=False,
              dtype=None, engine=None,
              converters=None, true_values=None,
              false_values=None, skiprows=None,
              nrows=None, na_values=None,
              keep_default_na=True, verbose=False,
              parse_dates=False, date_parser=None,
              thousands=None, comment=None, skipfooter=0,
              convert_float=True, mangle_dupe_cols=True, **kwds)
'''

print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.3.2 读取文件')
print()

# 如果文件与代码文件在同目录下
pd.read_excel('data.xls')
# 本地绝对路径：
pd.read_excel('E:/bat/input_files/sales_2013.xlsx')
# 使用网址 url
# pd.read_excel('https://www.gairuo.com/file/data/dataset/team.xlsx')


print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.3.3 读取指定sheet表')
print()

# str, int, list, or None, default 0

print('# 默认sheet')
df = pd.read_excel(input_file)
print(df)
print()

print('# 第二个 sheet')
df = pd.read_excel(input_file, sheet_name=1)
print(df)
print()

print('# 按 sheet 的名字')
df = pd.read_excel(input_file, sheet_name='march_2013')
print(df)
print()

print('# 修改标题列，取前五行')
df.columns = [column.lower() for column in df.columns.str.replace(' ', '_')]
print(df.head())
print()

print('# 取第一个、第二个、名为 march_2013 的，返回一个 df 组成的字典')
dfs = pd.read_excel(input_file, sheet_name=[0, 1, "march_2013"])
print(dfs)
print()

print('# 所有的 sheet')
dfs = pd.read_excel(input_file, sheet_name=None)  # 所有的 sheet
print(dfs)
print()
print('# 读取时按 sheet 名')
print(dfs['february_2013'])
print()

print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.3.4 表头 header')
print()

# int, list of int, default 0

print('# 不设表头')  # || 原表头成为第一行数据，表头成为阿拉伯数字索引
df = pd.read_excel(input_file, header=None)
print(df)
print()

print('# 第3行为表头')  # 输出显示跳过前2行
df = pd.read_excel(input_file, header=2)
print(df)
print()

print('# 两层表头，多层索引')
df = pd.read_excel(input_file, header=[0, 1])
print(df)
print()
# df = pd.read_excel('tmp.xlsx', header=None)  # 不设表头
# pd.read_excel('tmp.xlsx', header=2)  # 第3行为表头
# pd.read_excel('tmp.xlsx', header=[0, 1])  # 两层表头，多层索引

print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.3.5 列名 names')
print()
'''
用names指定列名，也就是表头的名称，如不指定，默认为表头的名称。

如指定部分列名，则输出列名与对应列不一致
'''
# array-like, default None
c_list = ['Customer ID', 'Customer Name', 'Invoice Number', 'Sale Amount', 'Purchase Date']

print('# 指定列名')
df = pd.read_excel(input_file, names=['Customer ID', 'Customer Name'])
print(df)
print()
'''
out:
                                  Customer ID Customer Name
1234 John Smith         100-0002         1200    2013-01-01
2345 Mary Harrison      100-0003         1425    2013-01-06
3456 Lucy Gomez         100-0004         1390    2013-01-11
4567 Rupert Jones       100-0005         1257    2013-01-18
5678 Jenny Walters      100-0006         1725    2013-01-24
6789 Samantha Donaldson 100-0007         1995    2013-01-31
'''

df = pd.read_excel(input_file, names=['Customer ID', 'Customer Name', 'Invoice Number', 'Sale Amount', 'Purchase Date'])
print(df)
print(df.dtypes)
print()

print('# 传入列表变量')
df = pd.read_excel(input_file, names=c_list)
print(df)
print()

print('# 没有表头，需要设置为 None')
df = pd.read_excel(input_file, header=None, names=None)
print(df)

print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.3.6 索引列 index_col')
print()

# int, list of int, default None
print('# 指定第一列为索引')
df = pd.read_excel(input_file, index_col=0)
print(df)
print()

print('# 前两列，多层索引')
df = pd.read_excel(input_file, index_col=[0, 1])
print(df)
print()

print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.3.7 使用列 usecols')
print()

print('# 取 A 和 B 两列')
df = pd.read_excel(input_file, usecols='A,B')
print(df)
print()

print('# 取 A 和 B 两列')
# 大写
df = pd.read_excel(input_file, usecols='A:D')
print(df)
print()
# 小写 | 运行成功
df = pd.read_excel(input_file, usecols='a:e')
print(df)
print()

print('# 取 a列和b列，再加c到e列')
df = pd.read_excel(input_file, usecols='a,b,c:e')
print(df)
print()

print('# 取前两列')
df = pd.read_excel(input_file, usecols=[0, 1])
print(df)
print()

print('# 取指定列名的列')
df = pd.read_excel(input_file, usecols=['Customer ID', 'Sale Amount'])
print(df)
print()

print('# 表头包含 m 的')  # 大小写敏感
df = pd.read_excel(input_file, usecols=lambda x: 'm' in x)
print(df)
print()

# print('# 返回序列 squeezebool') # squeezebool=True 报错！
# df = pd.read_excel(input_file,usecols='a')
# print(df.dtypes)
# print()

print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.3.8 数据类型 dtype')
print()
'''数据类型，如果不传则自动推断。如果被 converters 处理则不生效。'''
df = pd.read_excel(input_file)
print(df.dtypes)
print()

print('# 所有数据均为此数据类型')  # | 但是字符串列、日期列转换为浮点类型 会报错
df = pd.read_excel(input_file, usecols=['Customer ID', 'Sale Amount'], dtype=np.float64)
print(df.dtypes)  # ValueError: Unable to convert column Customer Name to type float64 (sheet: 0)
print()

print('# 指定字段的类型')
df = pd.read_excel(input_file, dtype={'Customer ID': float, 'Sale Amount': str})
print(df.dtypes)
print()

print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.3.9 处理引擎 engine')
print()

'''
可接受的参数值是 “xlrd”, “openpyxl” 或者 “odf”，如果文件不是缓冲或路径，就需要指定，用于处理 excel 使用的引擎，三方库。
如果未指定engine参数，pandas会尝试根据文件扩展名自动选择合适的引擎。
xlrd需要2.0.1及以上版本 ，  openpyxl及xlrd需要单独安装！
'''

print('# openpyxl: 用于读取.xlsx文件')  # 这是Excel 2007及以后版本的文件格式。
df = pd.read_excel(input_file, engine='openpyxl')
print(df)
print()

# xlrd: 早期被广泛使用来读取.xls和.xlsx文件，但从xlrd 2.0.0版本开始，不再支持.xlsx文件。
print('# xlrd: 用于读取.xls文件')
df = pd.read_excel(xls_file, engine='xlrd')
print(df)
print()

# pyxlsb: 用于读取Excel的二进制文件格式.xlsb。
df = pd.read_excel('tmp.xlsb', engine='pyxlsb')
print(df)
print()

print('# 未指定引擎，pandas自动选择合适的引擎')
df = pd.read_excel(xls_file)
print(df)
print()

# odf: 用于读取OpenDocument格式的.ods文件，这是LibreOffice和OpenOffice使用的格式。
df = pd.read_excel('tmp.ods', engine='odf')
print(df)
print()

print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.3.10 列数据处理 converters')
print()

'''
对列的数据进行转换，列名与函数组成的字典。key 可以是列名或者列的序号。
'''


# dict, default None
def foo(p):
    return p + 's'


print('# Customer Name 应用函数, Sale Amount 使用 lambda')
df = pd.read_excel(input_file, converters={'Customer Name': foo, 'Sale Amount': lambda x: x * 2})
print(df)
'''
out:
   Customer ID        Customer Name Invoice Number  Sale Amount Purchase Date
0         1234          John Smiths       100-0002         2400    2013-01-01
1         2345       Mary Harrisons       100-0003         2850    2013-01-06
2         3456          Lucy Gomezs       100-0004         2780    2013-01-11
3         4567        Rupert Joness       100-0005         2514    2013-01-18
4         5678       Jenny Walterss       100-0006         3450    2013-01-24
5         6789  Samantha Donaldsons       100-0007         3990    2013-01-31
'''

print('# 使用列索引')
df = pd.read_excel(input_file, converters={1: foo, 3: lambda x: x * 0.5})
print(df)
print()
'''
out:
   Customer ID        Customer Name Invoice Number  Sale Amount Purchase Date
0         1234          John Smiths       100-0002        600.0    2013-01-01
1         2345       Mary Harrisons       100-0003        712.5    2013-01-06
2         3456          Lucy Gomezs       100-0004        695.0    2013-01-11
3         4567        Rupert Joness       100-0005        628.5    2013-01-18
4         5678       Jenny Walterss       100-0006        862.5    2013-01-24
5         6789  Samantha Donaldsons       100-0007        997.5    2013-01-31
'''

print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.4.2 数据输出 Excel')
print()

'''

'''

out_file = 'E:/bat/output_files/pandas_read_excel_20240124036.xlsx'

df = pd.read_excel(input_file)
df.columns = [column.lower() for column in df.columns.str.replace(' ', '_')]
print(df)
print()

# 导入文件到指定路径
# df.to_excel('E:/bat/output_files/pandas_read_excel_20240124032.xlsx',index=False)

# 指定sheet名，不要索引
df.to_excel(out_file, sheet_name='out_data', index=False)

# 指定索引名，不合并单元格
# index_label='label' # 该参数 指定第0列为索引列 名为label
df.to_excel(out_file, index_label='label', merge_cells=False)  # 不合并单元格
df.to_excel(out_file, index_label='label', merge_cells=True)  # 合并单元格 ||  和上面一个没啥区别啊？
print()

# 创建一个具有多层次索引的DataFrame
# df = pd.DataFrame({
#     'Value': [1, 2, 3, 4]
# }, index=[['A', 'A', 'B', 'B'], [1, 2, 1, 2]])

# 测试 [1, 1, 2, 2] 该列 没有合并单元格
df = pd.DataFrame({
    'Value': [1, 2, 3, 4]
}, index=[['A', 'A', 'B', 'B'], [1, 1, 2, 2]])

print(df)
# 将DataFrame写入Excel，合并相同的索引值
df.to_excel(out_file, index_label='label', merge_cells=True)

# 将DataFrame写入Excel，不合并相同的索引值
df.to_excel('E:/bat/output_files/pandas_read_excel_20240124037.xlsx', index_label='label', merge_cells=False)

print('---------------------------------------------------------------------')
print('# 多个数据导出到同一工作簿 不同sheet中')
out_file = 'E:/bat/output_files/pandas_read_excel_20240125046.xlsx'

df = pd.read_excel(input_file)
df.columns = [column.lower() for column in df.columns.str.replace(' ', '_')]
print(df)
print()

df1 = pd.read_excel(input_file, usecols='a:c')
print(df1)
print()

df2 = pd.read_excel(input_file)
# print(df2)
# print()
out_third = df2[df2['Customer ID'] >= 4567]
print(out_third)
print()
# print(df2[df2['Customer Name'].str.startswith('J')])
# print()

# 多个数据导出到同一工作簿 不同sheet中
with pd.ExcelWriter(out_file) as writer:
    df.to_excel(writer, sheet_name='first', index=False)
    df1.to_excel(writer, sheet_name='second', index=False)
    out_third.to_excel(writer, sheet_name='third', index=False)

print()
print('------------------------------------------------------------')
# print('\t3.3 读取Excel')
print('\t3.4.3 数据输出 导出引擎')
print()

'''
在pandas中，当使用to_excel()方法将DataFrame写入到Excel文件时，
可以选择不同的引擎来处理写入操作。常用的引擎包括openpyxl、xlsxwriter、xlwt等。

openpyxl:
用于读写.xlsx文件，支持Excel 2010及以上版本的文件格式。
支持创建图表和修改Excel文件的高级功能，如设置单元格样式、过滤器、条件格式等。
df.to_excel('path_to_file.xlsx', sheet_name='Sheet1', engine='openpyxl')

xlsxwriter:
仅用于写入.xlsx文件，提供了丰富的格式化选项，如单元格格式、图表、图像插入等。
通常用于需要高度定制化的Excel报告生成。
writer = pd.ExcelWriter('path_to_file.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()

xlwt:
用于写入.xls文件，支持老版本的Excel文件格式（Excel 97-2003）。
不支持.xlsx文件格式，功能上比openpyxl和xlsxwriter受限。
writer = pd.ExcelWriter('path_to_file.xls', engine='xlwt')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()
'''

# 指定操作引擎
df.to_excel('path_to_file.xlsx', sheet_name='Sheet1', engine='xlsxwriter')  # 在'engine'参数中设置ExcelWriter使用的引擎
writer = pd.ExcelWriter('path_to_file.xlsx', engine='xlsxwriter')
df.to_excel(writer)
writer.save()

# 设置系统引擎
'''
当你在to_excel()方法中指定engine参数时，你是在为单次操作选择一个特定的引擎。
这种方式适用于你只想在特定情况下使用某个引擎，而不改变全局默认设置。
'''
from pandas import options  # noqa: E402

options.io.excel.xlsx.writer = 'xlsxwriter'
df.to_excel('path_to_file.xlsx', sheet_name='Sheet1')
