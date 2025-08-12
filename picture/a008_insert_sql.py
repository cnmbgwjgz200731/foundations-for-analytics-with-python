import pandas as pd

# 17.2.8 生成SQL

"""
现有以下2020年节假日的数据，需要将其插入数据库的holiday表里，
holiday除了以下三列，还有一个年份字段year。
"""

temp_file = 'E:/bat/input_files/holidays.xlsx'

df = pd.read_excel(temp_file)

print(df)

sql = ''
for i, r in df.iterrows():
    # print(r)
    r_sql = f"INSERT INTO holiday(holiday, year, start_date, end_date)" \
            f" VALUES('{r['节日']}', '{r['结束日期'][:4]}', '{r['开始日期']}', '{r['结束日期']}');"
    sql = sql + r_sql + '\n'
    # print(r_sql)

# print()
# print(r_sql)
print()
print(sql)



