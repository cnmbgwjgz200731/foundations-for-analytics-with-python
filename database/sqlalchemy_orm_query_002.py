from sqlalchemy import Column, Integer, String, DateTime,create_engine
from sqlalchemy.orm import sessionmaker,declarative_base

import pandas as pd


# 初始化数据库连接 及 定义模式
database_url = 'mysql+mysqldb://root:12345678@localhost:3306/my_suppliers'
engine = create_engine(database_url, echo=False)

Base = declarative_base()

# 定义模型
'''
你需要定义一个模型Supplier，该模型映射了数据库中的suppliers表。
这一步是使用SQLAlchemy ORM的关键，因为它让我们可以用面向对象的方式来处理数据库操作。
'''

# 可以删除某列（如列 purchase_date = Column(DateTime)） 即只定义自己想要查询的某些列即可
class Supplier(Base):
    __tablename__ = 'suppliers'
    supplier_id = Column(Integer, primary_key=True)  # 假设这是表的主键
    supplier_name = Column(String(255))
    invoice_number = Column(String(255))
    part_number = Column(String(255))
    cost = Column(String(255))
    purchase_date = Column(DateTime)


# 创建Session类
'''
Session是进行数据库操作的起点。通过与数据库绑定的engine创建Session实例，
我们可以在数据库会话中执行查询、添加或修改记录。
'''
Session = sessionmaker(bind=engine)
# 创建Session实例
session = Session()

# 使用ORM方式查询数据库
query = session.query(Supplier)
# print(query)
df = pd.read_sql(query.statement, query.session.bind)

# 打印DataFrame
print(df)