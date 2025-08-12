"""这是什么"""

from logging.handlers import RotatingFileHandler
from sqlalchemy import create_engine, text
from urllib.parse import quote
import logging
import pandas as pd
import time
import os

from test006 import temp_sql


def create_db_engine(db_type, username, password, host, port,
                     database, service_name=None):
    """创建数据库引擎"""
    encode_password = quote(password)
    if db_type == "starrocks":
        url = f'starrocks://{username}:{encode_password}@{host}:{port}/{database}'
    elif db_type == "mysql":
        url = f'mysql+pymysql://{username}:{encode_password}@{host}:{port}/{database}'
    else:
        raise ValueError(f'不支持这种数据库类型')

    engine_kwargs = {
        'pool_size': 5,  # 连接池中保持的数据库连接数。
        'max_overflow': 10,  # 连接池中允许超出的最大连接数。|| 在连接池中的连接数达到 pool_size 后，允许额外创建的连接数。超出部分不会被连接池维护，使用后会被销毁。
        'pool_timeout': 30,  # 从连接池获取连接的超时时间（秒）。
        'pool_recycle': 600,  # 连接在连接池中存活的时间（秒），超过此时间的连接会被断开并重新创建。
        'echo': False,  # 是否打印生成的 SQL 语句 || 用于调试，开启后会在控制台打印所有生成的 SQL 语句
        'future': True,  # 使用未来版本的 SQLAlchemy 行为。 || 启用后将使用未来版本（2.0）的行为特性。
        'pool_pre_ping': True  # 在连接被检出前测试连接是否可用。| 预防数据库连接因超时或其他原因断开
    }

    try:
        engine = create_engine(url, **engine_kwargs)
        logging.info(f'连接数据库成功！')
        return engine
    except Exception as er:
        logging.error(f'连接数据库失败: {er}')
        return None


def setup_log(program_name):
    current_date = pd.Timestamp('now').strftime('%Y%m%d')
    log_folder = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_folder, exist_ok=True)

    log_file = os.path.join(log_folder, f'{program_name}_{current_date}.log')

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(),
                                  RotatingFileHandler(log_file,
                                                      maxBytes=10 * 1024 * 1024,
                                                      backupCount=5
                                                      )
                                  ]
                        )

    return logging.getLogger(program_name)


def get_next_id(conn):
    result = conn.execute(text(f'select max(id) from table_name;')).fetchone()
    return result[0] + 1 if result[0] is not None else 10_000


def insert_log(engine, id_name, start_time, stop_time, status):
    next_id = get_next_id(engine)
    insert_query = text("""
        INSERT INTO table_name(id, program_name, start_time, stop_time, status)
        VALUES(:id, :program_name, :start_time, :stop_time, :status)
        """
                        )

    try:
        with engine.connect() as conn:
            conn.execute(insert_query, {
                'id': next_id,
                'id_name': id_name,
                'start_time': start_time,
                'stop_time': stop_time,
                'status': status
            })
            conn.commit()
            logger.info(f'insert data success !')
    except Exception as e:
        logger.error(f'insert data failed: {e} !')
        raise

    return None


def execute_sql(engine):
    try:
        with engine.begin() as conn:
            for i in range(1, 3):
                conn.execute(temp_sql(i))
                logger.info(f'execute sql success !')
                time.sleep(1)
    except Exception as e:
        logger.error(f'execute sql failed {e} !')
        return None


if __name__ == "__main__":
    program = os.path.splitext(os.path.basename(__file__))[0]
    begin_time = pd.Timestamp('now')

    logger = setup_log(program)

    sr_engine = create_db_engine(
        db_type='starrocks',
        username=os.getenv('SR_USERNAME'),
        password=os.getenv('SR_PASSWORD'),
        host=os.getenv('SR_HOST'),
        port=os.getenv('SR_PORT'),
        database=os.getenv('SR_DATABASE')
    )

    if sr_engine is None:
        logger.error(f'数据库连接失败！')
        exit(1)

    try:
        execute_sql(sr_engine)
        remark = 'success'
        logger.info(f'程序执行成功！')
    except Exception as e:
        remark = 'error'
        logger.info(f'程序执行失败：{e} !')
    finally:
        end_time = pd.Timestamp('now')
        insert_log(sr_engine, program, begin_time, end_time, remark)

        total_seconds = (end_time - begin_time).total_seconds()

        logger.info(f'this program spent time : {total_seconds: .4f} !')
        logger.info("\n" + "=" * 50 + "program end" + "=" * 50 + "\n")
