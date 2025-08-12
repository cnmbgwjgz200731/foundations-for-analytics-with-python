#! /usr/bin/env python3
# -*- coding:utf-8 -*-

# --------------------------------------------------------------------------
# ############################################################################
# page142
# update20230918 6db_mysql_update_from_csv.py
print("第4章 数据库")
print('\t4.2 MYSQL数据库')
print('\t\t4.2.3 更新表中记录')
# print('\t\t 1.基础python')
# print('\t\t\t基础python')
print()


import csv
import sys
import MySQLdb

# csv输入文件的路径和文件名
# input_file = sys.argv[1]
input_file = 'E:/bat/input_files/data_for_updating_mysql.csv'

# 连接mysql数据库
con = MySQLdb.connect(host='localhost',port=3306,db='my_suppliers',user='root',passwd='12345678')
c = con.cursor()

# 读取csv文件并更新特定的行
file_reader = csv.reader(open(input_file,'r',newline=''),delimiter=',')
header = next(file_reader)
for row in file_reader:
    data=[]
    for column_index in range(len(header)):
        data.append(str(row[column_index]).strip())
    print(data)
    c.execute("UPDATE Suppliers SET Cost=%s,Purchase_Date=%s WHERE Supplier_Name=%s;",data)
con.commit()
print('')

# 查询suppliers表
c.execute("SELECT * FROM Suppliers;")
rows = c.fetchall()
for row in rows:
    output = []
    for column_index in range(len(row)):
        output.append(str(row[column_index]))
    print(output)