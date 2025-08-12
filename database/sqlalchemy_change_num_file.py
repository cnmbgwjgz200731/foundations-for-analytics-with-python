"""
create20250311 创建连接本地mysql 测试连接成功，读数成功！

update20250402 读取指定sql文件，且循环执行sql，且将变量参数 循环替换； success！
"""

from sqlalchemy import create_engine, text
from urllib.parse import quote

import logging
import sqlparse
import pandas as pd
import time
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
        'echo': False,
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

    sql = 'E:/bat/input_files/test325.sql'

    # 修改执行部分代码
    for i in range(1, 3):
        try:
            with open(sql, 'r', encoding='utf-8') as file:
                sql_cont = file.read()
                sql_content = sql_cont.format(num=i)
                # sql_commands = [cmd.strip() for cmd in sql_content.split(';') if cmd.strip()] # 也可以运行
                # 使用sqlparse.split分割，然后过滤空语句 更安全
                sql_commands = [cmd.strip() for cmd in sqlparse.split(sql_content) if cmd.strip()]


        except Exception as ep:
            print(f'can not read this file: {ep} !')
            continue  # 继续下一次循环

        try:
            with ms_engine.connect() as conn:
                with conn.begin():
                    for cmd in sql_commands:
                        try:
                            # 执行单条命令
                            conn.execute(text(cmd))
                            print(f"执行成功: {cmd[:50]}...")  # 打印前50字符便于调试
                            # time.sleep(1)
                        except Exception as cmd_error:
                            print(f"命令执行失败: {cmd[:50]}... \n错误详情: {cmd_error}")
                            raise  # 抛出异常触发回滚

                    # for cmd in sql_commands:
                    #     if cmd:  # 跳过空语句
                    #         conn.execute(text(cmd))  # 如果不主动commit() 结果会自动回滚，实际数据库没有插入数据；

                # conn.execute(text(sql_commands))
                # df = pd.read_sql(text(f'select * from my_suppliers.number001;'), conn)
                # print(df)
            # conn.commit()  # 显式提交事务
        except Exception as e:
            print(f'主流程异常,执行失败: {e}')
            conn.rollback()
            raise

    df = pd.read_sql(text(f'select * from my_suppliers.number001;'), ms_engine)
    print(df)
