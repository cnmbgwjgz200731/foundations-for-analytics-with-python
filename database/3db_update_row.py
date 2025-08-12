#! /usr/bin/env python3
# -- coding:utf-8

# --------------------------------------------------------------------------
# ############################################################################
# page128
# update20230912 3db_update_row.py
print("第4章 数据库")
print('\t4.1 Python内置的sqlite3模块')
print('\t\t4.1.2 更新表中记录')
# print('\t\t 1.基础python')
# print('\t\t\t基础python')
print()


import sqlite3
import sys
import csv

"""
该示例展示 批量更新数据表中已有记录的方法，这种方法使用csv输入文件来提供
更新特定记录的数据。

在本例中，csv输入文件中数据的顺序也要同查询中属性的顺序一样。
即查询中的属性从左到右分别是amount,date和customer；
所以csv输入文件中的列从左到右也应该是金额，日期和客户名称。

如果输入csv文件中提供的列多于创建表的列数量 报错：
Traceback (most recent call last):
File "D:/Pycharm/pythonproject/venv/foundations_for_analytics/database/3db_update_row.py", line 56, in <module>
con.execute("UPDATE sales SET amount=?,date=? WHERE customer=?;",data)
sqlite3.ProgrammingError: Incorrect number of bindings supplied. The current statement uses 3, and there are 5 supplied.
"""

# csv输入文件的路径和文件名
# input_file = sys.argv[1]
# 个人替换
import time

input_file = 'E:/bat/input_files/data_for_updating.csv'

# 创建sqlite3的内存数据库
# 创建带有4个属性的sales表
con = sqlite3.connect(":memory:")

# 删除表
drop_query = """DROP TABLE IF EXISTS sales;"""
con.execute(drop_query)
con.commit()

# 创建表
query = """
        CREATE TABLE IF NOT EXISTS sales
        (customer VARCHAR(20),
        product VARCHAR(40),
        amount FLOAT,
        date DATE
        );"""
con.execute(query)
con.commit()

# 向表中插入几行数据
data = [('Richard Lucas', 'Notepad', 2.50, '2014-01-02'),
		('Jenny Kim', 'Binder', 4.15, '2014-01-15'),
		('Svetlana Crow', 'Printer', 155.75, '2014-02-03'),
		('Stephen Randolph', 'Computer', 679.40, '2014-02-20')]
# for tuple in data:
# 	print(tuple)
statement = "INSERT INTO sales VALUES(?,?,?,?)"
con.executemany(statement,data)
con.commit()

# 读取csv文件并更新特定的行
file_reader = csv.reader(open(input_file,'r'),delimiter = ',')
header = next(file_reader,None)
# print(header)
for row in file_reader:
	data = []
	for column_index in range(len(header)):
		data.append(row[column_index])
	# print(data)
	# time.sleep(0.1)
	con.execute("UPDATE sales SET amount=?,date=? WHERE customer=?;",data)
	# con.execute("UPDATE sales SET amount=?,date=? WHERE customer=? AND product=?;", data)
con.commit()

# 查询sales表
cursor = con.execute("SELECT * FROM sales")
rows = cursor.fetchall()
# print(rows) # 输出： [('Richard Lucas', 'Notepad', 4.25, '5/11/2014'), ('Jenny Kim', 'Binder', 6.75, '5/12/2014'), ('Svetlana Crow', 'Printer', 155.75, '2014-02-03'), ('Stephen Randolph', 'Computer', 679.4, '2014-02-20')]
for row in rows:
	output = []
	for column_index in range(len(row)):
		output.append(str(row[column_index]))
	print(output)


"""
output:
['Richard Lucas', 'Notepad', '4.25', '5/11/2014']
['Jenny Kim', 'Binder', '6.75', '5/12/2014']
['Svetlana Crow', 'Printer', '155.75', '2014-02-03']
['Stephen Randolph', 'Computer', '679.4', '2014-02-20']
"""