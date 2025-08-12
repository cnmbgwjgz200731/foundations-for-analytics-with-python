# update20241016
# 合并工作簿中多个sheet数据

import pandas as pd
import warnings

# 忽略 UserWarning 类型的警告
warnings.simplefilter(action='ignore', category=UserWarning)

input_file = 'E:/bat/input_files/sales_2013.xlsx'

df = pd.read_excel(input_file, sheet_name=None, engine='openpyxl')

# print(df['january_2013'])

merge_df = []

# print(df.items())
for sheet_name, data in df.items():
    # print(sheet_name)
    # print(data)
    # print(data.)
    # print(type(data))

    # print(merge_df)

    data = data.assign(source=sheet_name)
    data.columns = [i.lower().replace(' ', '_') for i in data.columns]
    # print(data)
    # merge_df = pd.DataFrame(data)

    merge_df.append(data)
    # print(merge_df)

merge_file = pd.concat(merge_df, axis=0, ignore_index=True)
print(merge_file)

# 如果不需要知道数据来源于哪张sheet，使用下列代码更简洁！
# print(pd.concat(df.values(), axis=0, ignore_index=True))
