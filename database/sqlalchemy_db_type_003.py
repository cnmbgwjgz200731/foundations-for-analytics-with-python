import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote

username = "shybby"
password = "shybbcxy@"
encode_password = quote(password)
host = "StarRocksJQ.sdpintra.com"
port = "9031"
database = "ODS"
service_name = "your_service_name"  # 如果使用Oracle数据库需要指定

database_url = f'starrocks://{username}:{encode_password}@{host}:{port}/{database}'

sr_engine = create_engine(database_url, echo=False)


def create_db_engine(db_type, username, password, host, port, database, service_name=None):
    encode_password = quote(password)
    if db_type == 'mysql':
        url = f'mysql+pymysql://{username}:{encode_password}@{host}:{port}/{database}'
    elif db_type == 'oracle':
        url = f'oracle+cx_oracle://{username}:{encode_password}@{host}:{port}/?service_name={service_name}'
    elif db_type == 'starrocks':
        url = f'starrocks://{username}:{encode_password}@{host}:{port}/{database}'
    elif db_type == 'hive':
        url = f'hive://{username}:{encode_password}@{host}:{port}/{database}'
    elif db_type == 'spark':
        url = f'spark://{username}:{encode_password}@{host}:{port}/{database}'
    else:
        raise ValueError("unsupported database type!")

    # engine = create_engine(url, echo=False)
    # return engine

    try:
        engine = create_engine(url, echo=False)
        return engine
    except Exception as e:
        print(f"Error creating engine: {e}")
        return None


# 创建数据库引擎
engine = create_db_engine("starrocks", username, password, host, port, database)

if engine:
    query = "SELECT COUNT(1) FROM table_name;"
    try:
        df = pd.read_sql(query, engine)
        print(df)
    except Exception as e:
        print(f"Error executing query: {e}")
else:
    print("Failed to create database engine.")




delete_query = """
DELETE FROM users
WHERE user_id = :user_id
"""

# 数据
delete_data = {
    'user_id': 1
}

with engine.connect() as conn:
    conn.execute(delete_query, **delete_data)


update_query = """
UPDATE users
SET email = :email
WHERE user_id = :user_id
"""

# 数据
update_data = {
    'email': 'new_email@example.com',
    'user_id': 1
}

with engine.connect() as conn:
    conn.execute(update_query, **update_data)


# 读取 SQL 文件
sql_file_path = 'D:/pdi6/input_files/create_table_001.sql'  # 更新为你的 SQL 文件实际路径
with open(sql_file_path, 'r') as file:
    sql_command = file.read()
    # print(sql_command)


# 执行 SQL 命令
with engine.connect() as connection:
    # connection.execute(sql_command)
    connection.execute(text(sql_command))  # 使用text()函数将SQL命令转换为SQLAlchemy的TextClause对象，以确保兼容SQLAlchemy 2.0。


# 插入数据示例
insert_query = """
INSERT INTO users (user_id, user_name, email)
VALUES (:user_id, :user_name, :email)
"""

# 数据
data = {
    'user_id': 1,
    'user_name': 'John Doe',
    'email': 'john.doe@example.com'
}

with engine.connect() as conn:
    conn.execute(insert_query, **data)
