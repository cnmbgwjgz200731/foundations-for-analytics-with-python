from sqlalchemy import create_engine, text
from urllib.parse import quote
import pandas as pd


# username = "shyybb"
# password = "shybbxyz@"
# host = "www.baidu.com"
# port = 2345
# database = "ODS"


def create_db_engine(db_type, username, password, host, port, database):
    encode_password = quote(password)
    if db_type == "starrocks":
        url = f'starrocks://{username}:{encode_password}@{host}:{port}/{database}'
    elif db_type == "mysql":
        url = f'mysql+pymysql://{username}:{encode_password}@{host}:{port}/{database}'
    else:
        raise ValueError("Unsupported the database!")

    try:
        engine = create_engine(url, echo=False)
        return engine
    except Exception as e:
        print(f'unsupported this database:{e}')
        return None


sr_engine = create_db_engine(
    db_type='starrocks',
    username='shyxyz',
    password='shyxyz@',
    host='www.baidu.com',
    port=2345,
    database='ODS'
)


# 查询数据示例:
def query_data(engine, sql):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn)
            return df
    except Exception as e:
        print(f'Unsupported this database!')
        return None


# 查询示例:
query_sql = """
select * from table_name ;
"""
result_df = query_data(sr_engine, query_sql)


# 4.更新数据示例
def update_data(engine, sql):
    try:
        with engine.connect() as conn:
            conn.execute(text(sql))
            conn.commit()
            print("数据更新成功！")
    except Exception as e:
        print(f'更新出错：{e}')


# 更新示例：
update_sql = """
update orders
set amount = 150.0
where order_id = 1001
;
"""
update_data(sr_engine, update_sql)

# delete数据类似于update


# ----------------------------------------------
# update20241120 实际 将外部数据 读取后 插入指定表中！
import pandas as pd
import time
import os

from sqlalchemy import create_engine, text
from urllib.parse import quote

os.environ['SQLALCHEMY_WARN_20'] = '0'

start_time = time.perf_counter()

# 初始化数据库连接
username="shyyb"
password = "shyybcx@"
encoded_password = quote(password)  # 对密码进行URL编码
host="StarRocksJQ.sdpintra.com"
port="9031"
database="ODS"
database_url = f'starrocks://{username}:{encoded_password}@{host}:{port}/{database}'
engine = create_engine(database_url, echo=False)


file = 'D:/pdi6/input_files/pandas_20240912044_001.csv'

df = pd.read_csv(file)

# print(df)

# 将 DataFrame 写入数据库表
# df.to_sql('table_name', engine, if_exists='append', index=False, chunksize=5000)
# df.to_sql('LIUXIN_TMP.tmp_lx_temporary_merchant_base_20240105',engine,if_exists='append',index=False,chunksize=5000)

# 读取 SQL 文件
sql_file_path = 'D:/pdi6/input_files/create_table_002.sql'  # 更新为你的 SQL 文件实际路径
with open(sql_file_path, 'r', encoding='utf-8') as file:
    sql_command = file.read()
    # print(sql_command)


# 执行 SQL 命令
with engine.connect() as connection:
    # connection.execute(sql_command)
    connection.execute(text(sql_command))  # 使用text()函数将SQL命令转换为SQLAlchemy的TextClause对象，以确保兼容SQLAlchemy 2.0。

# 重命名 DataFrame 的列以匹配数据库表的列名
# df.rename(columns={'id': 'sn'}, inplace=True)  # 重要 注释 会报错！

# 确保 'merchant_id' 列的数据类型与数据库表中的列兼容
# 例如，如果数据库中 'merchant_id' 列是字符串，确保 DataFrame 中也是字符串
# df['merchant_id'] = df['merchant_id'].astype(str) # 好像没有这行也能执行

# 将 DataFrame 写入数据库表
df.to_sql('tmp_lx_temporary_merchant_base_20240701012', engine, schema='LIUXIN_TMP',
          if_exists='append', index=False, chunksize=5000)

query = "select count(*) from LIUXIN_TMP.tmp_lx_temporary_merchant_base_20240701012 ;"  # 排序有效
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