#! /usr/bin/env python3
# -*- coding:utf-8 -*-

# --------------------------------------------------------------------------
# ############################################################################
# page120
# update20230907 1db_count_rows.py
print("第4章 数据库")
print('\t4.1 Python内置的sqlite3模块')
# print('\t\t\t基础python')
print()
"""
创建连接对象con来代表数据库
在该示例中，使用了专用名称':memory:',在内存中创建了一个数据库。
如果你想使这个数据库持久化，就需要提供另外的字符串。
例如：如果我使用字符串'My_database.db'或'C:\\Users\Clinton\Desktop\my_database.db' 而不是':memory:',
那么数据库对象就会永久保存在当前目录或你的桌面上。

"""

import sqlite3

# 创建SQLlite3内存数据库
# 创建带有4个属性的sales表

con = sqlite3.connect(':memory:')
query = """CREATE TABLE IF NOT EXISTS sales
                (customer VARCHAR(20),
                product VARCHAR(40),
                amount FLOAT,
                data DATE          
                );"""
con.execute(query)
con.commit() # 提交数据库并保存 ||
# 当你对数据库做出修改时，必须使用commit()方法来保存你的修改。否则这种修改不会保存到数据库中。

# 在表中插入几行数据
# 元组列表
data = [('Richard Lucas','Notepad',2.50,'2014-01-02'),
        ('Jenny Kim','Binder',4.15,'2014-01-15'),
        ('Svetlana Crow','Printer',155.75,'2014-02-03'),
        ('Stephen Randolph','Computer',679.40,'2014-02-20'),
        ]

"""
'?'在此表示占位符，表示你想在SQL命令中使用的值。然后在连接对象的execute或executemany方法中,
你需要提供一个包含4个值的元组，元组中的值会按位置替换到SQL命令中。
相对于使用字符串操作组装SQL命令的方法，这种参数替换的方法可以使你的代码不易受到SQL的注入攻击。

"""
statement = "INSERT INTO sales VALUES(?,?,?,?)"
con.executemany(statement,data) # 因为data有4个元组  所以executemany执行了4次
con.commit()

# 查询sales表
cursor = con.execute("SELECT * FROM sales")
rows = cursor.fetchall()

# 个人测试
# print(rows)
# print()

# 计算查询结果中行的数量
row_counter = 0
for row in rows:
        print(row)
        row_counter += 1
print('Number of rows: %d' %(row_counter))