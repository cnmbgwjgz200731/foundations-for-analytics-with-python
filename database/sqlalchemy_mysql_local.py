"""
create20250311 创建连接本地mysql 测试连接成功，读数成功！
"""

from sqlalchemy import create_engine, text
from urllib.parse import quote

import logging
import sqlparse
import pandas as pd
import os


def create_db_database(db_type, username, password, host, port, database,
                       service_name=None):
    """创建数据库连接"""
    encode_password = quote(password)
    if db_type == "starrocks":
        url = f'starrocks://{username}:{encode_password}@{host}:{port}/{database}'
    elif db_type == "mysql":
        url = f'mysql+pymysql://{username}:{encode_password}@{host}:{port}/{database}'
    else:
        raise ValueError(f'Unsupported this database!')

    engine_kwargs = {
        'pool_size': 5,
        'max_overflow': 10,
        'pool_timeout': 60,
        'pool_recycle': 1800,
        'echo': True,
        'future': True,
        'pool_pre_ping': True
    }

    try:
        engine = create_engine(url, **engine_kwargs)
        return engine
    except Exception as e:
        logging.error(f'Fail to connect this database: {e} !')
        return None


if __name__ == "__main__":

    ms_engine = create_db_database(
        db_type='mysql',
        username=os.getenv('DB_USERNAME'),
        password=os.getenv('DB_PASSWORD'),
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_DATABASE')
    )

    # try:
    #     with ms_engine.connect() as conn:
    #         # result = conn.execute(text(f'select * from suppliers;'))
    #         df = pd.read_sql(text(f'select * from suppliers;'), conn)
    #         print(df)
    # except Exception as e:
    #     print(f'can not read data: {e} !')
    #     raise

    # update20250401 执行读取本地sql文件，且循环 参数执行！

    sql = 'E:/bat/input_files/test325.sql'

    # 修改执行部分代码
    with open(sql, 'r', encoding='utf-8') as file:
        sql_commands = [cmd.strip() for cmd in file.read().split(';') if cmd.strip()]
        # sql_commands = file.read()

    try:
        with ms_engine.connect() as conn:
            with conn.begin():
                for cmd in sql_commands:
                    if cmd:  # 跳过空语句
                        conn.execute(text(cmd)) # 如果不主动commit() 结果会自动回滚，实际数据库没有插入数据；

            # conn.execute(text(sql_commands))
            df = pd.read_sql(text(f'select * from my_suppliers.number001;'), conn)
            print(df)
        # conn.commit()  # 显式提交事务
    except Exception as e:
        print(f'执行失败: {e}')
        conn.rollback()
        raise
