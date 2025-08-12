from logging.handlers import RotatingFileHandler
from sqlalchemy import create_engine, text
from urllib.parse import quote
from datetime import datetime
import pandas as pd
import time
import logging
import os

# update20241206 测试日志表成功 无论成功还是 失败（除无法连接数据库外） 都会记录一条结果！


# 自定义的数据库日志处理器
class DatabaseLogHandler(logging.Handler):
    def __init__(self, engine, table_name):
        logging.Handler.__init__(self)
        self.engine = engine
        self.table_name = table_name
        self.start_time = datetime.now()
        self.program_name = os.path.splitext(os.path.basename(__file__))[0]
        self.status = "Running"  # 初始状态为运行中

    def emit(self, record):
        # 仅在程序结束时插入日志记录
        if record.msg in ["Program finished", "Program failed"]:
            end_time = datetime.now()
            try:
                with self.engine.connect() as conn:
                    insert_query = text(f"""
                    INSERT INTO {self.table_name} (program_name, start_time, end_time, status)
                    VALUES (:program_name, :start_time, :end_time, :status)
                    """)
                    conn.execute(insert_query, {
                        'program_name': self.program_name,
                        'start_time': self.start_time,
                        'end_time': end_time,
                        'status': self.status
                    })
                    conn.commit()
            except Exception as e:
                print(f"Error while inserting log to StarRocks: {e}")


# 创建StarRocks数据库引擎
def create_db_engine(db_type, username, password, host, port, database):
    encode_password = quote(password)
    if db_type == "starrocks":
        url = f'starrocks://{username}:{encode_password}@{host}:{port}/{database}'
    elif db_type == "mysql":
        url = f'mysql+pymysql://{username}:{encode_password}@{host}:{port}/{database}'
    else:
        raise ValueError("Unsupported the database!")

    engine_kwargs = {
        'pool_size': 5,
        'max_overflow': 10,
        'pool_timeout': 30,
        'pool_recycle': 30,
        'echo': False,
        'future': True
    }

    try:
        engine = create_engine(url, **engine_kwargs)
        return engine
    except Exception as e:
        print(f'Unsupported this database: {e}')
        return None


# 配置StarRocks数据库连接
sr_engine = create_db_engine(
    db_type='starrocks',
    username='liuxin',
    password='jk8j34hsy3mfdjdw773j2huq',
    host='StarRocksJQ.sdpintra.com',
    port=9031,
    database='ODS'
)

# 获取当前时间并格式化为字符串，替换非法字符
current_time = datetime.now().strftime("%Y%m%d")
program_name = os.path.splitext(os.path.basename(__file__))[0]  # 日志名称包含 程序名称

# 创建日志文件夹路径
log_folder = 'D:/pdi6/log'
os.makedirs(log_folder, exist_ok=True)

# 日志文件路径
log_file = os.path.join(log_folder, f'{program_name}_{current_time}.log')

# 配置日志记录，保留本地日志文件功能
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)

# 添加自定义的数据库日志处理器
db_handler = DatabaseLogHandler(sr_engine, "LIUXIN_TMP.tmp_lx_python_log_20241206")
db_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
db_handler.setFormatter(formatter)
logger.addHandler(db_handler)

start_time = time.perf_counter()


# 查询数据示例
def query_data(engine, sql):
    try:
        with engine.connect() as conn:
            conn.execute(text(sql))
            conn.commit()
            logger.info('Insert successful')
    except Exception as e:
        logger.error(f'Insert failure: {e}')
        db_handler.status = "Failure"  # 设置状态为失败
        return None


# 查询示例
def temp_sql(day_offsets):
    query_sql = f"""
    insert into LIUXIN_TMP.tmp_lx_rfm_loss_rate_20240820 
    with days_base as
        (
        select date_sub(current_date(),interval {day_offsets} day) day_code -- union
        )
        ,
    trade_base as
        (    
        select 
        cr.tx_time,
        cr.order_id,
        cr.merchant_no,
        cr.pay_channel,
        cr.amount,
        cr.mc_type,
        cr.policy_id,
        cr.mcc,
        b.policy_category,
        cr.promoter_no_first promoter_no
        from LIUXIN_TMP.permanent_lx_policy_category b
        left join ODS.ods_pcs_clearing_record cr on cr.policy_id = b.policy_id 
        inner join days_base a on 1=1
        where cr.type = 1 and cr.fundin_status >= 1
        and cr.clearing_status <> '8'
        and b.policy_category not in ('手机POS','碰一碰')
        and b.mc_type = '0001' 
        and cr.mc_type = '0001' 
        and cr.tx_time > date_sub(a.day_code, interval 1 year)
        and cr.tx_time <= a.day_code
        and cr.amount > 1
        and exists (select 1 from LIUXIN_TMP.permanent_lx_policy_category a where cr.policy_id = a.policy_id and a.mc_type = '0001' 
                          and a.policy_category not in ('手机POS','碰一碰')
                          )
        )
        ,
    loss_rate_base as
        ( 
        select 
        merchant_no,
        policy_category,
        max(tx_time) max_time
        from trade_base a
        group by merchant_no,policy_category
        )
        ,
    filter_base as
        (    
        select
        a.merchant_no,
        a.policy_category,
        datediff(b.day_code,a.max_time) minus_days
        from loss_rate_base a
        inner join days_base b on 1=1
        )

    select 
    a.day_code,
    count(distinct case when minus_days > 30 then merchant_no end) loss_cnt, 
    count(distinct merchant_no) total_cnt, 
    count(distinct case when minus_days > 30 then merchant_no end)/count(distinct merchant_no) thirty_rate_loss,
    count(distinct case when minus_days > 60 then merchant_no end) sixty_loss_cnt,
    count(distinct case when minus_days > 60 then merchant_no end)/count(distinct merchant_no) sixty_rate_loss,
    count(distinct case when minus_days > 90 then merchant_no end) ninty_loss_cnt,
    count(distinct case when minus_days > 90 then merchant_no end)/count(distinct merchant_no) ninty_rate_loss,
    count(distinct case when minus_days > 180 then merchant_no end) halfyear_loss_cnt,
    count(distinct case when minus_days > 180 then merchant_no end)/count(distinct merchant_no) halfyear_rate_loss
    from filter_base b
    left join days_base a on 1=1
    group by day_code
    ;
    """
    return query_sql


# 执行SQL并记录日志
def execute_sql():
    for i in range(0, 3, 1):
        i = i + 1
        logger.info(f'第{i}次执行：')
        with sr_engine.begin() as conn:
            conn.execute(text(temp_sql(i)))
            logger.info(f'Insert success!')
            time.sleep(1)


try:
    execute_sql()
    logger.info(f'全部执行成功!')
    db_handler.status = "Success"  # 设置状态为成功
except Exception as e:
    logger.error(f'执行失败: {e}')
    db_handler.status = "Failure"  # 设置状态为失败
finally:
    end_time = time.perf_counter()
    time_consuming = end_time - start_time

    logger.info(f'耗时：{time_consuming:.4f} s!')
    logger.info("\n" + "=" * 50 + " Program End " + "=" * 50 + "\n")

    # 手动调用 emit 方法插入最终日志记录
    db_handler.emit(
        logging.LogRecord(
            name=logger.name,
            level=logging.INFO if db_handler.status == "Success" else logging.ERROR,
            pathname=__file__,
            lineno=0,
            msg="Program finished" if db_handler.status == "Success" else "Program failed",
            args=(),
            exc_info=None
        )
    )
