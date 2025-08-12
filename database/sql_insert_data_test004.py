from sqlalchemy import create_engine, text
from urllib.parse import quote
from logging.handlers import RotatingFileHandler
from datetime import datetime
import pandas as pd
import time
import logging
import os

# 获取当前时间并格式化为字符串，替换非法字符
# current_time = datetime.now().strftime("%Y_%m_%d %H:%M:%S")  # error :为非法字符
# current_time = datetime.now().strftime("%Y_%m_%d %H_%M_%S")  # success
current_time = datetime.now().strftime("%Y_%m_%d")
program_name = os.path.splitext(os.path.basename(__file__))[0]  # 日志名称包含 程序名称

# 配置日志
# logging.basicConfig(level=logging.INFO)
# 配置日志记录
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     handlers=[logging.StreamHandler()])

# filemode='w' 覆盖之前内容  filemode='a' 增量内容
# logging.basicConfig(filename='example.log',filemode='w',level=logging.DEBUG)
# TODO 注意：上面level设置的是显示的最低严重级别，小于level设置的最低严重级别将不会打印出来


# 配置日志记录 制定文件保存

# 创建日志文件夹路径
log_folder = 'D:/pdi6/log'
os.makedirs(log_folder, exist_ok=True)

# 日志文件路径
# log_file = os.path.join(log_folder, f'database_operations_{current_time}.log')
log_file = os.path.join(log_folder, f'{program_name}_{current_time}.log')

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(lineno)d - %(name)s -%(levelname)s - %(message)s',
                    handlers=[
                        RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)

start_time = time.perf_counter()


# username = "liuxin"
# password = "jk8j34hsy3mfdjdw773j2huq"
# host = "StarRocksJQ.sdpintra.com"
# port = 9031
# database = "ODS"


def create_db_engine(db_type, username, password, host, port, database):
    encode_password = quote(password)
    if db_type == "starrocks":
        url = f'starrocks://{username}:{encode_password}@{host}:{port}/{database}'
    elif db_type == "mysql":
        url = f'mysql+pymysql://{username}:{encode_password}@{host}:{port}/{database}'
    else:
        raise ValueError("Unsupported the database!")

        # 添加连接池配置
    engine_kwargs = {
        'pool_size': 5,  # 连接池大小
        'max_overflow': 10,  # 最大溢出连接数
        'pool_timeout': 30,  # 连接池超时时间
        'pool_recycle': 30,  # 连接回收时间
        'echo': False,
        'future': True
    }

    try:
        # 添加future=True参数以支持SQLAlchemy 2.0
        # engine = create_engine(url, echo=False, future=True)  # 也可以运行
        engine = create_engine(url, **engine_kwargs)
        logger.info(f"Database engine created for {db_type} at {host}:{port}/{database}")  # 添加日志
        # engine = create_engine(url, echo=False)
        return engine
    except Exception as e:
        logger.error(f"Unsupported this database: {e}")  # 添加日志
        return None


sr_engine = create_db_engine(
    db_type='starrocks',
    username='liuxin',
    password='jk8j34hsy3mfdjdw773j2huq',
    host='StarRocksJQ.sdpintra.com',
    port=9031,
    database='ODS'
)


# 查询数据示例:
def query_data(engine, sql):
    try:
        with engine.connect() as conn:
            conn.execute(text(sql))
            conn.commit()
            logger.info('Insert successful')
    except Exception as e:
        logger.info(f'insert failure')
        return None


# 查询示例:
def temp_sql(day_offsets):
    query_sql = f"""
    insert into LIUXIN_TMP.superset_lx_dqj_loss_statistic_20241120 
    with days_base as
        (
        -- date_format(date_sub(current_date(),interval 13 day),'%Y-%m-%d')
        select date_sub(current_date(),interval {day_offsets} day) day_code -- union 
    --    select date_sub(current_date(),interval 1 day) union
    --    select date'2024-08-18'
        )
        ,
    trade_base as
        (    
        select 
        -- a.day_code,
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
        from LIUXIN_TMP.permanent_lx_policy_category b -- where cr.policy_id = a.policy_id and a.mc_type = '0001' 
        left join ODS.ods_pcs_clearing_record cr on cr.policy_id = b.policy_id 
        inner join days_base a on 1=1
        where cr.type = 1 and cr.fundin_status >= 1
        and cr.clearing_status <> '8'
        and b.policy_category not in ('手机POS','碰一碰')
        and b.mc_type = '0001' 
        and cr.mc_type = '0001' 
        -- and cr.pay_channel in ('36','57','85','SP','z1','c1','c3','c7','c8','n1','n2','n3','n4') ---POS交易
        and cr.tx_time > date_sub(a.day_code, interval 1 year) -- date_sub(current_date(), interval 1 year) --date'2024-01-01'
        and cr.tx_time <= a.day_code -- date_add(current_date(),0) 
        and cr.amount > 1
        and exists (select 1 from LIUXIN_TMP.permanent_lx_policy_category a where cr.policy_id = a.policy_id and a.mc_type = '0001' 
                          -- and a.policy_category = '盛钱呗'
                          and a.policy_category not in ('手机POS','碰一碰')
                          )
    --    limit 1000
        -- and ppi.srv_entry_mode not in ('033','034')
        )
        ,
    loss_rate_base as
        ( 
        select 
        merchant_no,
        policy_category,
        max(tx_time) max_time
        -- count(distinct merchant_no) cnt  -- 1425710 , 876600,  873899
        from trade_base a
        group by merchant_no,policy_category
        -- limit 100
        -- where
        ) 
        ,
    filter_base as
        (    
        select
        a.merchant_no,
        a.policy_category,
        -- datediff(current_date(),max_time) minus_days
        datediff(b.day_code,a.max_time) minus_days
        -- days_diff(max_time,current_date()) cnt
        from loss_rate_base a
        inner join days_base b on 1=1
        -- limit 1000
        )

        select 
        date_format(a.day_code,'%Y-%m-%d') day_code,
        b.policy_category,
        count(distinct case when minus_days > 30 then merchant_no end) loss_cnt, -- 
        count(distinct merchant_no) total_cnt, -- 过去一年内有交易商户总量
        count(distinct case when minus_days > 30 then merchant_no end)/count(distinct merchant_no) thirty_rate_loss,
        count(distinct case when minus_days > 60 then merchant_no end) sixty_loss_cnt,
        count(distinct case when minus_days > 60 then merchant_no end)/count(distinct merchant_no) sixty_rate_loss,
        count(distinct case when minus_days > 90 then merchant_no end) ninty_loss_cnt,
        count(distinct case when minus_days > 90 then merchant_no end)/count(distinct merchant_no) ninty_rate_loss,
        count(distinct case when minus_days > 180 then merchant_no end) halfyear_loss_cnt,
        count(distinct case when minus_days > 180 then merchant_no end)/count(distinct merchant_no) halfyear_rate_loss
        from filter_base b
        left join days_base a on 1=1
        group by day_code,policy_category

        union all 

        select 
        date_format(a.day_code,'%Y-%m-%d') day_code,
        '电签_汇总' policy_category,
        count(distinct case when minus_days > 30 then merchant_no end) loss_cnt, -- 
        count(distinct merchant_no) total_cnt, -- 过去一年内有交易商户总量
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


# for i in range(3):
#     print(i)
#     query_data(sr_engine, temp_sql(i))
#     time.sleep(0.1)

sql = """select * from LIUXIN_TMP.tmp_lx_rfm_loss_rate_20240820 ;"""

for i in range(0, 3, 1):
    i = i + 1
    logger.info(f'第{i}次执行：')
    with sr_engine.begin() as conn:
        # 这样会自动处理事务和提交
        # with sr_engine.connect() as conn:
        conn.execute(text(temp_sql(i)))
        logger.info(f'insert success!')
        # conn.commit()
        time.sleep(1)

logger.info(f'全部执行成功!')

end_time = time.perf_counter()

time_consuming = end_time - start_time

# print(f'耗时：{time_consuming} s!')
logger.info(f'耗时：{time_consuming:.4f} s!')

# 添加运行结束的分隔线
logger.info("\n" + "="*50 + " Program End " + "="*50 + "\n")