"""
update20251113
将指定文件夹中的不同xlsx文件 按照所属bd拆分后 归类到指定文件夹中。
"""

# 导入所需库
from email.mime.application import MIMEApplication  # 邮件附件处理
from email.mime.multipart import MIMEMultipart  # 构建多部分邮件
from email.mime.text import MIMEText  # 处理邮件正文
from email.header import Header  # 邮件头编码处理
from datetime import datetime  # 时间处理
from glob import glob  # 文件路径模式匹配

import pandas as pd  # Excel数据处理
import zipfile  # 创建ZIP压缩包
import smtplib  # SMTP协议邮件发送
import shutil  # 高级文件操作
import time  # 时间相关操作
import os  # 操作系统接口

# ========== 文件路径配置 ==========
# 输出目录路径
out_folder = 'E:/bat/output_files/split_dir/split_bd/202505'
# 创建输出目录（exist_ok=True表示如果目录已存在不报错）
os.makedirs(out_folder, exist_ok=True)

# 源文件路径模式匹配（获取所有xlsx文件）
source_dir = 'E:/桌面08/测试/临时拆分/拆分数据-胡国强/拆分分润数据工具/胡国强拆分文件_每月/2025-06/*.xlsx'
source_files = glob(source_dir)  # 获取匹配的文件列表

# ========== 运行时信息记录 ==========
start_time = datetime.now()  # 记录程序开始时间
month_code = datetime.now().strftime('%Y%m')  # 生成年月格式字符串（如202505）

# ========== BD配置 ==========
# target_bds = ['刘鑫', '徐建亮', '王二']  # 测试环境BD列表
# target_bds = ['汪玉敏', '蒋爽', '姜银花', '王晶', '丁亮', '徐建亮', '刘鑫']  # 测试环境
# target_bds = ['周元', '徐雅雯', ...]  # 生产环境BD列表（已注释）

# ========== 邮件服务器配置 ==========
mail_host = os.getenv('SMTP_HOST')  # 从环境变量获取SMTP主机
mail_port = os.getenv('SMTP_PORT')  # SMTP端口（通常587用于TLS，465用于SSL）
mail_user = os.getenv('SMTP_USER')  # 发件邮箱账号
mail_pass = os.getenv('SMTP_PASS')  # 邮箱密码/授权码
company_name = 'SFT'  # 公司名称（用于邮件签名）

# ========== 读取BD邮箱配置 ==========
bd_emails_file = r'E:\bat\output_files\split_dir\split_bd\bd_emails.xlsx'
df_bd = pd.read_excel(bd_emails_file)  # 使用Pandas读取Excel文件

# 构建BD邮箱字典
bd_emails = {}
for i, row in df_bd.iterrows():
    if row.get('status') == 1:  # 只处理状态为1的BD
        # 拆分收件人列表（分号分隔）
        to_list = [to_mail.strip() for to_mail in str(row['email']).split(';') if to_mail.strip()]
        # 拆分抄送列表（分号分隔）
        cc_list = [cc_mail.strip() for cc_mail in str(row['cc_emails']).split(';') if cc_mail.strip()]
        # 存储到字典
        bd_emails[row['name']] = {
            'to': to_list,
            'cc': cc_list
        }


target_bds = list(bd_emails.keys())
# ========== 主处理逻辑 ==========
# 存储每个目录结构的字典（用于后续压缩）
bd_files_dict = {}

# 遍历所有源文件
for source_file in source_files:
    # 读取Excel文件的所有工作表（sheet_name=None表示读取所有sheet）
    all_sheets = pd.read_excel(source_file, sheet_name=None)

    # 获取源文件名（不含扩展名）
    file_name = os.path.splitext(os.path.basename(source_file))[0]

    # 遍历每个目标BD
    # 为每个目标BD创建独立工作簿
    valid_bd_count = 0
    for bd in target_bds:
        # 处理特殊字符（创建安全目录名）
        safe_bd = str(bd).replace('/', '_').replace('\\', '_').replace(':', '_')
        # 构建输出文件名
        bd_file_name = f'{safe_bd}_{month_code}_{file_name}.xlsx'

        # 创建BD专属目录
        output_bd_path = os.path.join(out_folder, safe_bd)
        os.makedirs(output_bd_path, exist_ok=True)  # 递归创建目录

        # 记录目录结构（用于后续压缩）
        if safe_bd not in bd_files_dict:
            bd_files_dict[safe_bd] = {
                'path': output_bd_path,  # 物理存储路径
                'original_name': bd  # 原始BD名称
            }

        # 构建完整输出路径
        output_path = os.path.join(output_bd_path, bd_file_name)

        # 使用ExcelWriter创建xlsx文件
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            has_data = False  # 标记是否有有效数据

            # 遍历所有工作表
            for sheet_name, df in all_sheets.items():
                # 检查必需字段
                if 'BD' not in df.columns:
                    continue  # 跳过没有BD列的工作表

                # print(sheet_name)  # 打印当前处理的工作表名
                # time.sleep(1)  # 防止过快操作（可能用于调试）

                # 统一数据类型（避免类型不匹配）
                df['BD'] = df['BD'].astype(str)

                # 筛选当前BD的数据
                bd_data = df[df['BD'] == bd]

                if not bd_data.empty:  # 如果有数据
                    has_data = True
                    # 写入Excel（限制sheet名称长度）
                    bd_data.to_excel(
                        writer,
                        sheet_name=sheet_name[:30],  # 确保sheet名<=31字符
                        index=False,  # 不写入索引
                        freeze_panes=(1, 1)  # 冻结首行首列
                    )

            # 无数据处理
            if not has_data:
                # 创建提示工作表
                pd.DataFrame({'提示': [f'BD [{bd}] 在本月所有工作表中无数据']}).to_excel(
                    writer,
                    sheet_name='无数据',
                    index=False
                )
            else:
                valid_bd_count += 1  # 计数有效BD文件

# ========== 后续处理 ==========
print("开始创建BD压缩包并发送邮件...")
print(valid_bd_count)
# （此处应接续压缩和发送邮件的代码）