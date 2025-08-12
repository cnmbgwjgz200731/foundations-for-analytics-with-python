#! /usr/bin/env python3
# -- coding:utf-8

# --------------------------------------------------------------------------
# ############################################################################
# page135
# update20230913 4db_mysql_load_from_csv.py
print("第4章 数据库")
print('\t4.2 MYSQL数据库')
print('\t\t4.2.1 向表中插入新记录')
# print('\t\t 1.基础python')
# print('\t\t\t基础python')
print()


import MySQLdb
import sys
import csv
from datetime import datetime,date

# csv输入文件的路径和文件名
# input_file = sys.argv[1]
input_file = 'E:/bat/input_files/supplier_data.csv'
# 连接MYSQL数据库
con = MySQLdb.connect(host='localhost',port=3306,db='my_suppliers',user='root',passwd='12345678')
c = con.cursor()


# 删除表
drop_table = """
             DROP TABLE IF EXISTS suppliers;
             """
c.execute(drop_table)
con.commit()

# 创建表 || 含主键 自动增长 时间列默认为当前时间
create_table = """
                CREATE TABLE IF NOT EXISTS suppliers(
                supplier_id int not null auto_increment,
                supplier_name varchar(30) not null,
                invoice_number varchar(10) not null,
                part_number varchar(10) not null,
                cost varchar(10) not null,
                purchase_date timestamp not null default current_timestamp,
                primary key(supplier_id)
                );                
                """

c.execute(create_table)
con.commit()


# 清空表 || 测试成功！ 如果不执行该段 则每次运行下面的语句 suppliers表数据会越来越多;
truncate_table = "TRUNCATE suppliers;"
c.execute(truncate_table)
con.commit()


# 向Suppliers表中插入数据
# file_reader = csv.reader(open(input_file,'r',newline='')) # 原始
file_reader = csv.reader(open(input_file,'r',newline=''),delimiter=',') # 个人修改 也行！
header = next(file_reader)

for row in file_reader:
    data = []
    for column_index in range(len(header)):
        if column_index < 4:
            data.append(str(row[column_index]).lstrip('$').replace(',','').strip())
        else:
            a_date = datetime.date(datetime.strptime(str(row[column_index]),'%m/%d/%y'))
            # %Y: year is 2015; %y: year is 15
            a_date = a_date.strftime('%Y-%m-%d')
            data.append(a_date)
    # print(data)

    # c.execute("INSERT INTO Suppliers VALUES(%s,%s,%s,%s,%s);", data) # 原代码 update20240419 使用下段代替

    c.execute('''
              INSERT INTO Suppliers(
              supplier_name, invoice_number, part_number, cost, purchase_date)
              VALUES(%s,%s,%s,%s,%s);
              ''',data) # 指定具体列名 主键列自动增加 不需要指定
# con.autocommit(TRUE)
con.commit() # 成功 命令行查询 已插入数据
print("")
# 查询suppliers表
c.execute("SELECT * FROM Suppliers;")
rows = c.fetchall()
# print(rows)
for row in rows:
    row_list_output = []
    for column_index in range(len(row)):
        row_list_output.append(str(row[column_index]))
    print(row_list_output)