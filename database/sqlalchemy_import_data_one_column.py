import pandas as pd
import time
import os

from sqlalchemy import create_engine, text
from urllib.parse import quote

os.environ['SQLALCHEMY_WARN_20'] = '0'

start_time = time.perf_counter()

# 初始化数据库连接
password = "shyybcx@"
encoded_password = quote(password)  # 对密码进行URL编码
database_url = f'starrocks://shyyb:{encoded_password}@StarRocksJQ.sdpintra.com:9031/ODS'
engine = create_engine(database_url, echo=False)


file = 'D:/pdi6/input_files/import_data_source.xlsx'

df = pd.read_excel(file, engine='C')

# print(df)

# 将 DataFrame 写入数据库表
# df.to_sql('table_name', engine, if_exists='append', index=False, chunksize=5000)
# df.to_sql('LIUXIN_TMP.tmp_lx_temporary_merchant_base_20240105',engine,if_exists='append',index=False,chunksize=5000)

# 读取 SQL 文件
sql_file_path = 'D:/pdi6/input_files/create_table_001.sql'  # 更新为你的 SQL 文件实际路径
with open(sql_file_path, 'r', encoding='utf-8') as file:
    sql_command = file.read()
    # print(sql_command)


# 执行 SQL 命令
with engine.connect() as connection:
    connection.execute(text(sql_command))

# 重命名 DataFrame 的列以匹配数据库表的列名
df.rename(columns={'id': 'merchant_id'}, inplace=True)  # 重要 注释 会报错！

# 确保 'merchant_id' 列的数据类型与数据库表中的列兼容
# 例如，如果数据库中 'merchant_id' 列是字符串，确保 DataFrame 中也是字符串
df['merchant_id'] = df['merchant_id'].astype(str) # 好像没有这行也能执行

# 将 DataFrame 写入数据库表
df.to_sql('tmp_lx_temporary_merchant_base_20240105', engine, schema='LIUXIN_TMP',
          if_exists='append', index=False, chunksize=5000)

query = "select count(*) from LIUXIN_TMP.tmp_lx_temporary_merchant_base_20240105 ;"  # 排序有效
df1 = pd.read_sql(query, engine)
print(df1)

end_time = time.perf_counter()

print(f'耗时{end_time - start_time} 秒！')

'''
导入数据 数量 耗时：
   count(*)
0      6398
耗时1.7244192771613598 秒！

   count(*)
0    991940
耗时78.07627197634429 秒！

'''