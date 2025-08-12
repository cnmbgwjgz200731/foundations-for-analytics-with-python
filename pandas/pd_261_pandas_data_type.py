import numpy as np
import pandas as pd

print()
print('2.6 pandas的数据类型')
print('\t2.6.1 数据类型查看')
print()
# update20240102

input_file = 'E:/bat/input_files/supplier_data.csv'

df = pd.read_csv(input_file, index_col=False)

df.columns = [heading.lower() for heading in df.columns.str.replace(' ', '_')]

print(df.columns)
# 查看指定列的数据类型
print(df.supplier_name.dtype)

print()
print('2.6 pandas的数据类型')
print('\t2.6.3 数据检测')
print()
s = pd.Series([14.34, 21.43, 5.08], name='gdp', index=[4, 5, 6])
print(s)
print()
'''
可以使用类型判断方法检测数据的类型是否与该方法中指定的类型一致，
如果一致，则返回True，注意传入的是一个Series：
'''
print(pd.api.types.is_bool_dtype(s))  # False
print(pd.api.types.is_categorical_dtype(s))  # False
print(pd.api.types.is_datetime64_dtype(s))  # False
print(pd.api.types.is_datetime64_any_dtype(s))  # False
print(pd.api.types.is_datetime64_ns_dtype(s))  # False
print(pd.api.types.is_float_dtype(s))  # True
print(pd.api.types.is_int64_dtype(s))
print(pd.api.types.is_numeric_dtype(s))  # True
print(pd.api.types.is_object_dtype(s))
print(pd.api.types.is_string_dtype(s))
print(pd.api.types.is_timedelta64_dtype(s))  # False
