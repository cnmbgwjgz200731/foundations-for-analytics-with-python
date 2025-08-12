import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)


from sqlalchemy import create_engine, text  # 导入text函数


# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy import Column, Integer, String, DateTime
#
# from sqlalchemy.orm import sessionmaker

# con = MySQLdb.connect(host='localhost',port=3306,db='my_suppliers',user='root',passwd='12345678')
# c = con.cursor()

# 设置sqlalchemy.engine的日志级别为WARNING，这样只会显示警告和错误信息，而不会显示INFO级别的消息
# logging.CRITICAL（只有最严重的消息才会被记录）或使用logging.disable(logging.CRITICAL)完全禁用日志。

# logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING) # 运行没用
# logging.getLogger('sqlalchemy.engine').setLevel(logging.CRITICAL)


# 初始化数据库连接
database_url = 'mysql+mysqldb://root:12345678@localhost:3306/my_suppliers'
engine = create_engine(database_url, echo=False) # echo=True 开启日志输出 || 运行成功！


logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
# 获取连接并执行查询
with engine.connect() as connection:
    result = connection.execute(text("SELECT * FROM suppliers"))  # merchants
    for row in result:
        print(row)

'''这里，text("SELECT * FROM suppliers")将明确地告诉SQLAlchemy这是一个需要执行的SQL语句字符串。'''
'''
2024-04-19 10:19:44,231 INFO sqlalchemy.engine.Engine SELECT DATABASE()
2024-04-19 10:19:44,231 INFO sqlalchemy.engine.Engine [raw sql] ()
2024-04-19 10:19:44,231 INFO sqlalchemy.engine.Engine SELECT @@sql_mode
2024-04-19 10:19:44,231 INFO sqlalchemy.engine.Engine [raw sql] ()
2024-04-19 10:19:44,231 INFO sqlalchemy.engine.Engine SELECT @@lower_case_table_names
2024-04-19 10:19:44,231 INFO sqlalchemy.engine.Engine [raw sql] ()
2024-04-19 10:19:44,231 INFO sqlalchemy.engine.Engine BEGIN (implicit)
2024-04-19 10:19:44,231 INFO sqlalchemy.engine.Engine SELECT * FROM suppliers
2024-04-19 10:19:44,231 INFO sqlalchemy.engine.Engine [generated in 0.00011s] ()
(1, 'Supplier X', '001-1001', '2341', '500.00', datetime.datetime(2014, 1, 20, 0, 0))
(2, 'Supplier X', '001-1001', '2341', '500.00', datetime.datetime(2014, 1, 20, 0, 0))
(3, 'Supplier X', '001-1001', '5467', '750.00', datetime.datetime(2014, 1, 20, 0, 0))
(4, 'Supplier X', '001-1001', '5467', '750.00', datetime.datetime(2014, 1, 20, 0, 0))
(5, 'Supplier Y', '50-9501', '7009', '250.00', datetime.datetime(2014, 1, 30, 0, 0))
(6, 'Supplier Y', '50-9501', '7009', '250.00', datetime.datetime(2014, 1, 30, 0, 0))
(7, 'Supplier Y', '50-9505', '6650', '125.00', datetime.datetime(2014, 2, 3, 0, 0))
(8, 'Supplier Y', '50-9505', '6650', '125.00', datetime.datetime(2014, 2, 3, 0, 0))
(9, 'Supplier Z', '920-4803', '3321', '615.00', datetime.datetime(2014, 2, 3, 0, 0))
(10, 'Supplier Z', '920-4804', '3321', '615.00', datetime.datetime(2014, 2, 10, 0, 0))
(11, 'Supplier Z', '920-4805', '3321', '6015.00', datetime.datetime(2014, 2, 17, 0, 0))
(12, 'Supplier Z', '920-4806', '3321', '1006015.00', datetime.datetime(2014, 2, 24, 0, 0))
2024-04-19 10:19:44,231 INFO sqlalchemy.engine.Engine ROLLBACK
'''


# 测试删除语句 执行成功
# with engine.connect() as connection:
#     result = connection.execute(text("drop table if exists suppliers"))  # merchants
    # for row in result:
    #     print(row)



print()
# 使用pandas的read_sql函数执行SQL查询，并直接返回DataFrame
# 运行成功！
query = "SELECT * FROM suppliers order by purchase_date desc"  # 排序有效
df = pd.read_sql(query, engine)

# 打印DataFrame
print(df)
print()
print(df['supplier_name'])

df1 = df['supplier_name'].astype(str)
# print(df1.lower())
for i in df1:
    print(i.lower())