import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote

username = "root"
password = "12345678"
host = "localhost"
port = "3306"
database = "my_suppliers"


def create_db_engine(db_type, username, password, host, port, database):
    encode_password = quote(password)
    if db_type == "mysql":
        url = f'mysql+pymysql://{username}:{encode_password}@{host}:{port}/{database}'
    else:
        raise ValueError("unsupported this database!")

    try:
        engine = create_engine(url, echo=False)
        return engine
    except Exception as e:
        print(f'Error creating engine: {e}')
        return None


mysql_engine = create_db_engine("mysql", username, password, host, port, database)
# print(mysql_engine)  # Engine(mysql+pymysql://root:***@localhost:3306/my_suppliers)

query = "select * from suppliers;"
df = pd.read_sql(query, mysql_engine)

print(df)

create_table_sql = """
-- drop table if exists products; 不能同时执行2个命令 否则报错！

create table if not exists products(
product_id int not null auto_increment comment '产品编号',
product_name varchar(255) null comment '产品名称',
create_time datetime not null default current_timestamp comment '创建时间',
primary key(product_id)
)
;
"""

# 测试执行建表成功 ；
with mysql_engine.connect() as con:
    con.execute(text(create_table_sql))

"""
# 分开执行
drop_table_sql = "drop table if exists products;"

# 测试执行建表成功 ；
with mysql_engine.connect() as con:
    con.execute(text(drop_table_sql))
    con.execute(text(create_table_sql))
"""

# TODO 执行插入语句 测试成功！
df_product = pd.DataFrame({
    'product_id': ['00001', '00002', '0003'],
    'product_name': ['辣条', '小雨点', '口香糖']
})

# df_product.to_sql("products", if_exists='append', mysql_engine, index=False) # 顺序错误会 报错！
df_product.to_sql("products", mysql_engine, if_exists='append', index=False)

product_query = "select * from products;"
print(pd.read_sql(product_query, mysql_engine))
