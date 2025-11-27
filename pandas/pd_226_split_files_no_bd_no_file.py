"""
update20251114
优化按照BD维度拆分多工作簿、sheets拆分数据；
将拆分好后的文件，按照BD维度建立文件夹，并创建对应的文件压缩包。
只处理指定的BD（张三、李四、王二），并为每个BD发送邮件通知。

update20251117
取消 target_bds 替换为 bd_emails 中的bd 避免修改多个变量

update20251125
1、避免文件打开无读取权限
2、如果一个工作簿中所有sheet没有对应的bd，则不输出bd对应的文件  避免有bd知道其它的业务
"""

from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from email.header import Header
from datetime import datetime
from glob import glob

import pandas as pd
import zipfile
import smtplib
import shutil
import time
import os

# 文件路径设置
out_folder = 'E:/bat/output_files/split_dir/split_bd/202505'
# out_folder = 'E:/bat/output_files/split_dir/split_bd/202511'
# 创建输出目录
os.makedirs(out_folder, exist_ok=True)

source_dir = 'E:/桌面08/测试/临时拆分/拆分数据-胡国强/拆分分润数据工具/胡国强拆分文件_每月/2025-06/*.xlsx'
# source_dir = 'E:/桌面08/测试/临时拆分/拆分数据-胡国强/拆分分润数据工具/胡国强拆分文件_每月/2025-11/*.xlsx'
# source_files = glob(source_dir)
source_files = [f for f in glob(source_dir) if not os.path.basename(f).startswith('~$')]

# 记录开始时间
start_time = datetime.now()
# month_code = datetime.now().strftime('%Y-%m')
month_code = (pd.Timestamp('now') - pd.DateOffset(months=1)).strftime('%Y-%m')
# 只处理这些BD
# target_bds = ['张三', '李四', '王二', '宗和']  # 测试环境 取消

# 邮件配置
mail_host = os.getenv('SMTP_HOST')  # 根据您的邮件服务商更改
mail_port = os.getenv('SMTP_PORT')  # 通常587用于TLS  # 465端口应该使用SMTP_SSL
mail_user = os.getenv('SMTP_USER')
mail_pass = os.getenv('SMTP_PASS')  # 建议使用应用专用密码而非登录密码
company_name = 'SFT'

bd_emails_file = r'E:\bat\output_files\split_dir\split_bd\bd_emails.xlsx'
df_bd = pd.read_excel(bd_emails_file)  # 使用pandas读取Excel文件

bd_emails = {}

for i, row in df_bd.iterrows():
    if row.get('status') == 0:
        to_list = [to_email.strip() for to_email in str(row['email']).split(';') if to_email.strip()]
        cc_list = [cc_email.strip() for cc_email in str(row['cc_emails']).split(';') if cc_email.strip()]
        bd_emails[row['name']] = {
            'to': to_list,
            'cc': cc_list
        }

# 存储每个BD的文件路径，用于后续压缩和邮件发送
bd_files_dict = {}

for source_file in source_files:
    # 读取Excel文件的所有sheet
    all_sheets = pd.read_excel(source_file, sheet_name=None)

    file_name = os.path.splitext(os.path.basename(source_file))[0]

    # 为每个目标BD创建独立工作簿
    valid_bd_count = 0
    for bd in list(bd_emails.keys()):
        # 先检查该BD在所有工作表中是否有数据
        has_data = False
        sheets_to_write = {}  # 存储有数据的工作表

        # 遍历所有工作表检查是否有数据
        for sheet_name, df in all_sheets.items():
            # 确保BD列存在
            if 'BD' not in df.columns:
                continue

            # 统一数据类型
            df['BD'] = df['BD'].astype(str)

            # 筛选当前BD的数据
            bd_data = df[df['BD'] == bd]

            if not bd_data.empty:
                has_data = True
                sheets_to_write[sheet_name] = bd_data

        # 如果没有数据，跳过文件创建
        if not has_data:
            print(f"BD [{bd}] 在本文件中无数据，不创建文件")
            continue

        # 创建文件名（替换非法字符）
        safe_bd = str(bd).replace('/', '_').replace('\\', '_').replace(':', '_')
        bd_file_name = f'{safe_bd}_{month_code}_{file_name}.xlsx'

        # 创建BD维度目录
        output_bd_path = os.path.join(out_folder, safe_bd)
        os.makedirs(output_bd_path, exist_ok=True)

        # 记录BD文件夹路径，用于后续压缩
        if safe_bd not in bd_files_dict:
            bd_files_dict[safe_bd] = {
                'path': output_bd_path,
                'original_name': bd
            }

        output_path = os.path.join(output_bd_path, bd_file_name)

        # 创建Excel写入器并写入有数据的工作表
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, bd_data in sheets_to_write.items():
                bd_data.to_excel(
                    writer,
                    sheet_name=sheet_name[:30],  # 确保sheet名称不超过31字符
                    index=False,
                    freeze_panes=(1, 1)
                )

        valid_bd_count += 1
        print(f"已创建BD [{bd}] 的工作簿: {bd_file_name}")

# 为每个BD创建压缩包并发送邮件
print("开始创建BD压缩包并发送邮件...")


def send_email_with_attachment(to_list, cc_list, bd_name, zip_path):
    """发送带附件的邮件函数 - 修复版本"""
    try:
        print(f"正在准备发送邮件给 {bd_name} ({to_list})...")

        with smtplib.SMTP_SSL(mail_host, int(mail_port), timeout=30) as sender:
            # 创建邮件对象
            sender.login(mail_user, mail_pass)

            msg = MIMEMultipart('mixed')
            msg['From'] = formataddr((str(Header('线下商户团队', 'utf-8')), mail_user))

            to_format = [formataddr((str(Header(f'{bd_name}', 'utf-8')), to_mail.strip()))
                         for to_mail in to_list if to_mail.strip()]
            if to_format:
                msg['To'] = ','.join(to_format)

            cc_format = [formataddr((str(Header(cc_mail.split('@')[0], 'utf-8')), cc_mail.strip()))
                         for cc_mail in cc_list if cc_mail.strip()]
            if cc_format:
                msg['Cc'] = ','.join(cc_format)
            # msg['Subject'] = f'{company_name} - {bd_name}的{month_code}数据报告'
            msg['Subject'] = Header(f'{month_code}_分润及返现数据_{bd_name}', 'utf-8').encode()

            # 邮件正文
            html_body = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            font-size: 13px;  /* 原通常为14px，现调小一号 */
            line-height: 1.5;  /* 相应调整行高 */
            color: #333333;
            margin: 0;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        .email-content {{
            max-width: 600px;
            margin: 0 auto;
            background: white;
            padding: 25px;  /* 相应调整内边距 */
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            font-family: 'Microsoft YaHei', Arial, sans-serif;  /* 确保内容区也使用微软雅黑 */
        }}
        .greeting {{
            font-size: 14px;  /* 原16px调小为14px */
            margin-bottom: 18px;
            font-family: 'Microsoft YaHei', Arial, sans-serif;
        }}
        .content {{
            margin: 12px 0;
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            font-size: 13px;  /* 明确设置内容字号 */
        }}
        .signature {{
            margin-top: 25px;
            padding-top: 18px;
            border-top: 1px solid #eeeeee;
            color: #666666;
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            font-size: 13px;
        }}
        .highlight {{
            color: #1890ff;
            font-weight: bold;
            font-family: 'Microsoft YaHei', Arial, sans-serif;
        }}
        p, strong {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;  /* 确保所有文本元素使用微软雅黑 */
            font-size: 13px;
            margin: 8px 0;  /* 调整段落间距 */
        }}
    </style>
</head>
<body>
    <div class="email-content">
        <div class="greeting">
            <p>尊敬的 <span class="highlight">{bd_name}</span>：</p>
        </div>

        <div class="content">
            <p>您好，</p>
            <p>附件是您<span class="highlight"><strong> {month_code}月 <strong></span>的<strong>分润及返现数据明细</strong>，烦请查阅。</p>
        </div>

        <div class="content">
            <p>如有任何问题，请随时联系我们，谢谢！</p>
        </div>

        <div class="signature">
            <p>谨祝商祺！</p>
            <p><strong>{company_name}商户运营团队</strong></p>
        </div>
    </div>
</body>
</html>
'''
            # 创建HTML部分
            html_part = MIMEText(html_body, 'html', 'utf-8')
            msg.attach(html_part)
            # msg.attach(MIMEText(body, 'plain', 'utf-8'))

            # 添加附件
            print(f"  添加附件: {os.path.basename(zip_path)}")
            with open(zip_path, "rb") as f:
                attach = MIMEApplication(f.read(), _subtype="zip")
                # attach.add_header('Content-Disposition', 'attachment',
                #                   filename=os.path.basename(zip_path))
                # 对附件文件名进行编码，确保中文正确显示 || 如果没有这行 附件名称会显示乱码（outlook邮箱）
                filename_header = Header(os.path.basename(zip_path), 'utf-8').encode()
                attach.add_header('Content-Disposition', 'attachment', filename=filename_header)
                msg.attach(attach)

            sender.send_message(msg)
            time.sleep(3)

            print(f"√ 成功发送邮件给 {bd_name} ({to_list})")
            return True

    except Exception as e:
        print(f"× 发送邮件给 {bd_name} 失败: {str(e)}")
        # 如果是认证错误，给出更详细的提示
        if "authentication" in str(e).lower():
            print("  请检查邮箱用户名和密码是否正确")
        elif "connection" in str(e).lower():
            print("  网络连接问题，请检查网络设置或SMTP服务器配置")
        return False


# 为每个目标BD创建压缩包并发送邮件
for bd_name, bd_info in bd_files_dict.items():
    original_bd_name = bd_info['original_name']
    bd_folder_path = bd_info['path']

    # 创建zip文件路径
    zip_path = os.path.join(out_folder, f"{bd_name}_{month_code}.zip")

    # 创建zip文件
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历BD文件夹中的所有文件
        for root, dirs, files in os.walk(bd_folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # 在zip文件中创建相对路径
                arcname = os.path.relpath(file_path, bd_folder_path)
                zipf.write(file_path, arcname)

    print(f"已创建BD [{original_bd_name}] 的压缩包: {zip_path}")

    # 发送邮件给对应BD
    if original_bd_name in bd_emails:
        recipient_email = bd_emails[original_bd_name]
        send_email_with_attachment(
            recipient_email['to'],
            recipient_email['cc'],
            original_bd_name,
            zip_path
        )
    else:
        print(f"未找到BD [{original_bd_name}] 的邮箱地址，跳过邮件发送")

# 计算并输出耗时
end_time = datetime.now()
total_seconds = (end_time - start_time).total_seconds()
print(f'程序执行完成，耗时: {total_seconds:.4f} 秒')
print(f'共生成 {len(bd_emails)} 个目标BD工作簿')
