#! /usr/bin/env python3
# -*- coding:utf-8 -*-

# --------------------------------------------------------------------------
# ############################################################################
# page140
# update20230915 5db_mysql_write_to_file.py
print("第4章 数据库")
print('\t4.2 MYSQL数据库')
print('\t\t4.2.2 查询一个表并将输出写入CSV文件')
# print('\t\t 1.基础python')
# print('\t\t\t基础python')
print()

import MySQLdb
import sys
import csv
import time

from datetime import datetime,date




start_time = time.time()
# CSV输出文件的路径和文件名
# output_file = sys.argv[1]
output_file = 'E:/bat/output_files/5db_mysql_write_to_file_20231108032.csv' # 原始
# output_file = 'E:/bat/output_files/5db_mysql_write_to_file_20230913033.xls' # 文件内容合并为一列
# output_file = 'E:/bat/output_files/5db_mysql_write_to_file_20230913034.xlsx' # 文件无法打开
# 连接mysql数据库
con = MySQLdb.connect(host='localhost',port=3306,db='my_suppliers',user='root',passwd='12345678')
c=con.cursor()
# 创建写文件的对象，并写入标题行
filewriter = csv.writer(open(output_file,'w',newline=''),delimiter=',')
header = ['Supplier Name','Invoice Number','Part Number','Cost','Purchase Date']
filewriter.writerow(header)

# 查询suppliers表，并将结果写入CSV输出文件
c.execute("SELECT * FROM Suppliers WHERE Cost > 700.0;")
rows = c.fetchall()

for row in rows:
    filewriter.writerow(row)

end_time = time.time()
print(end_time-start_time)