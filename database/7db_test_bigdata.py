# /usr/bin/env python3
# -*- coding:utf-8 -*-

import csv
import time
import MySQLdb

# input_file = 'E:/bat/input_files/supplier_data.csv'
# input_file = 'E:/bat/input_files/xjl.csv'
input_file = 'E:/bat/output_files/supplier_data_out20230922054.csv'
# output_file = 'E:/bat/output_files/supplier_data_out20230922055.csv'

con = MySQLdb.connect(host='localhost',port=3306,db='my_suppliers',user='root',passwd='12345678')
c = con.cursor()

# 删除表
drop_table = """
             DROP TABLE IF EXISTS merchants;
             """
c.execute(drop_table)
con.commit()

# 创建表 || 含主键 自动增长 时间列默认为当前时间
create_table = """
                CREATE TABLE IF NOT EXISTS merchants(
                id int not null auto_increment,
                insert_time timestamp not null default current_timestamp,
                merchant_no varchar(10) not null,
                promoter_no varchar(10) not null,
                promoter_id varchar(10) not null,
                merchant_type varchar(6),
                primary key(id) 
                );                
                """

c.execute(create_table)
con.commit()

# 导入csv文件数据

data = []
with open(input_file,'r',newline='') as csv_in_file:
    filereader = csv.reader(csv_in_file,delimiter=',')
    header = next(filereader)
    for row in filereader:
        row = [str(value).strip() for value in row]
        for column in range(len(header)):
            data.append(row[column])


# --------------------------------------------------------------
# 写入100万数据 耗时 229.72386914199998 s  约5000行/秒
# update20231024022 新电脑100万数据仅耗时56.37秒
import csv
import time
import MySQLdb

# input_file = 'E:/bat/input_files/supplier_data.csv'
# input_file = 'E:/bat/input_files/xjl.csv'
input_file = 'E:/bat/output_files/supplier_data_out20230922058.csv'

start_time = time.perf_counter()
# output_file = 'E:/bat/output_files/supplier_data_out20230922055.csv'

con = MySQLdb.connect(host='localhost',port=3306,db='my_suppliers',user='root',passwd='12345678')
c = con.cursor()

# 清空表
trunc_table = 'TRUNCATE TABLE merchants;'
c.execute(trunc_table)
con.commit()

# 将csv文件数据导入merchants表
with open(input_file,'r',newline='') as csv_in_file:
    filereader = csv.reader(csv_in_file,delimiter=',')
    header = next(filereader)
    for row in filereader:
        data = []
        row = [str(value).strip() for value in row]
        # print(row)
        for column in range(len(header)):
            data.append(row[column])

        # print(data)

        c.execute("""
                  INSERT INTO merchants(merchant_no,promoter_no,promoter_id,merchant_type)
                  VALUES(%s,%s,%s,%s);
                  """,data)
con.commit()


end_time = time.perf_counter()

print(end_time-start_time)