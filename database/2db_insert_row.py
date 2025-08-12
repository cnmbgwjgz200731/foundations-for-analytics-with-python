#! /usr/bin/env python3
# -*- coding:utf-8 -*-

# --------------------------------------------------------------------------
# ############################################################################
# page120
# update20230908 2db_insert_row.py
print("第4章 数据库")
print('\t4.1 Python内置的sqlite3模块')
print('\t\t4.1.1 向表中插入新记录')
# print('\t\t 1.基础python')
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
import csv
import sys
import time
# csv输入文件的路径和文件名

# input_file = sys.argv[1]
# cd D:\pycharm\pythonproject\\venv\\foundations_for_analytics\database
# python 2db_insert_row.py 'E:\\bat\input_files\supplier_data.csv'

# 个人替换
input_file = 'E:/bat/input_files/supplier_data.csv'

start_time = time.time()

# 创建SQLite3内存数据库  || 指定永久数据库 电脑重启后 数据不会消失
con = sqlite3.connect('E:/bat/output_files/Supplizers.db')

# 从这里到第1个commit(个人测试部分不包括) || 创建一个光标及一个多行SQL语句  用来创建suppliers表
c = con.cursor()

# 个人测试删除表 || 成功！
drop_table = """DROP TABLE IF EXISTS Suppliers;"""
c.execute(drop_table)
con.commit()

# 创建带有5个属性的suppliers表
create_table = """CREATE TABLE IF NOT EXISTS Suppliers
                (Supplier_Name VARCHAR(20),
                Invoice_Number VARCHAR(20),
                Part_Number VARCHAR(20),
                Cost FLOAT,
                Purchase_Date DATE
                );
                """
c.execute(create_table)
con.commit()

# 读取csv文件
# 向Suppliers表中插入数据
file_reader = csv.reader(open(input_file,'r'),delimiter=',')
header = next(file_reader,None)
for row in file_reader:
    data = []
    for column_index in range(len(header)):
        data.append(row[column_index])
    # print(data)
    c.execute("INSERT INTO Suppliers VALUES (?,?,?,?,?);",data)
con.commit()
print('')

# 查询Suppliers表
output = c.execute("SELECT * FROM Suppliers")
rows = output.fetchall()
for row in rows:
    output = []
    for column_index in range(len(row)):
        output.append(str(row[column_index]))
    print(output)


print()
# time.sleep(1)
end_time = time.time()
print(end_time-start_time)