"""
update20250313
读取excel数据 多列
导入到指定数据库中（指定表也行 if_exists='append'） to_sql() 可以自动新建表

df.to_sql(
    name='table_name',          # 目标表名
    con=engine,                # 数据库连接对象
    schema=None,               # 数据库Schema（模式）
    if_exists='fail',          # 表存在时的处理策略
    index=False,               # 是否写入DataFrame索引
    index_label=None,          # 索引列的数据库列名 || 类似于主键
    chunksize=None,            # 批量写入的每批数据量 chunksize=50000,  # 每次写入5万条
    dtype=None,                # 列数据类型映射
    method=None               # 插入方法 || 和参数chunksize 一起使用 method='multi'    # 开启批量模式
)

4. if_exists
作用：表存在时的处理策略（fail/replace/append）

策略	适用场景	业务案例
fail	防误覆盖生产数据	重要客户信息表写入前存在性检查
replace	初始化或重建表结构	每日凌晨清空并刷新商品库存表
append	增量数据追加	实时追加用户行为日志到历史表

8. method
作用：控制SQL插入方式，优化写入性能

方法	适用场景	性能对比（10万条/s）
None（默认）	小数据量（<1万条）	基准值1x
'multi'	通用批量插入	3x~5x
'fast_executemany'	SQL Server/Oracle专用	10x+

"""
from sqlalchemy import create_engine, text
from urllib.parse import quote

import logging
import pandas as pd
import os

# 本地文件导入数据
# files = 'E:/bat/input_files/sim_jyh.xlsx'
# df = pd.read_excel(files)

# 复制导入
df = pd.read_clipboard()

# print(df.head(5))
# ['msisdn', 'iccid', 'imsi', 'open_date', 'active_date', 'net_type', 'province', 'status', 'desc']

# df.columns = ['msisdn', 'iccid', 'imsi', 'open_date', 'active_date', 'net_type', 'province', 'status', 'remark']
df.columns = ['date', 'num']


# print(df.columns)
# print(df.head(5))
# print(df.describe())

df = df.replace(r'\t+$', '', regex=True)  # 仅替换行尾的 \t  // 大数据集 会更快


# 或使用正则表达式清理（更彻底）
# df = df.map(lambda x: x.rstrip('\t') if isinstance(x, str) else x)

# print(df.columns)
# print(df.head(5))
# print(df.describe())
# print(df.dtypes)


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
    except Exception as ep:
        logging.error(f'Fail to connect this database: {ep} !')
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

    try:
        with ms_engine.connect() as conn:
            # result = conn.execute(text(f'select * from suppliers;'))
            df.to_sql("sim_data_two", conn, schema='my_suppliers',
                      if_exists='replace', index=False, chunksize=5000)
    except Exception as e:
        print(f'can not read data: {e} !')
