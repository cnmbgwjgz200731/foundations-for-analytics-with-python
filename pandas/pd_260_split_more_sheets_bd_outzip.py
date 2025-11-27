"""
update20250903
优化按照BD维度拆分多工作簿、sheets拆分数据；
将拆分好后的文件，按照BD维度建立文件夹，并创建对应的文件压缩包。

"""

import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime
from glob import glob
import zipfile

# 文件路径设置
out_folder = 'E:/bat/output_files/split_dir/split_bd/202509'
# 创建输出目录
os.makedirs(out_folder, exist_ok=True)

source_dir = 'E:/桌面08/测试/临时拆分/拆分数据-胡国强/拆分分润数据工具/胡国强拆分文件_每月/2025-09/*.xlsx'
source_files = glob(source_dir)

# 记录开始时间
start_time = datetime.now()
month_code = datetime.now().strftime('%Y%m%d')

# 存储每个BD的文件路径，用于后续压缩
bd_files_dict = {}

# 只处理这些BD
target_bds = ['张三', '李四', '王二']

for source_file in source_files:
    # 读取Excel文件的所有sheet
    all_sheets = pd.read_excel(source_file, sheet_name=None)

    # 获取所有BD列表（排除空值）
    all_bds = set()
    valid_sheets = {}  # 存储有效工作表

    for sheet_name, df in all_sheets.items():
        # 确保有BD列且非空
        if 'BD' in df.columns and not df['BD'].empty:
            # 处理可能的混合类型
            df['BD'] = df['BD'].astype(str)

            # 过滤空字符串和NaN
            non_empty_bds = df['BD'][df['BD'].notna() & (df['BD'] != '') & (df['BD'] != 'nan')].unique()

            # 添加到集合
            all_bds.update(non_empty_bds)
            valid_sheets[sheet_name] = df

    print(f"发现 {len(all_bds)} 个有效的BD")
    print(f"有效BD列表: {list(all_bds)[:10]}")  # 打印前10个示例

    file_name = os.path.splitext(os.path.basename(source_file))[0]

    # 为每个有效BD创建独立工作簿
    valid_bd_count = 0
    for bd in all_bds:
        # 跳过无效BD（二次验证）
        if pd.isna(bd) or bd == '' or bd == 'nan':
            print(f"跳过无效BD: {bd}")
            continue

        # 创建文件名（替换非法字符）
        safe_bd = str(bd).replace('/', '_').replace('\\', '_').replace(':', '_')
        bd_file_name = f'{safe_bd}_{month_code}_{file_name}.xlsx'

        # 测试使用新的BD维度目录
        output_bd_path = os.path.join(out_folder, safe_bd)
        # 创建输出目录
        os.makedirs(output_bd_path, exist_ok=True)

        # 记录BD文件夹路径，用于后续压缩
        if safe_bd not in bd_files_dict:
            bd_files_dict[safe_bd] = output_bd_path

        output_path = os.path.join(output_bd_path, bd_file_name)

        # 创建Excel写入器
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            has_data = False

            # 遍历所有有效工作表
            for sheet_name, df in valid_sheets.items():
                # 确保BD列存在
                if 'BD' not in df.columns:
                    continue

                # 统一数据类型
                df['BD'] = df['BD'].astype(str)

                # 筛选当前BD的数据
                bd_data = df[df['BD'] == bd]

                if not bd_data.empty:
                    has_data = True
                    bd_data.to_excel(
                        writer,
                        sheet_name=sheet_name[:30],  # 确保sheet名称不超过31字符
                        index=False,
                        freeze_panes=(1, 1)
                    )

            # 检查是否有数据，如果没有则创建提示工作表
            if not has_data:
                pd.DataFrame({'提示': [f'BD [{bd}] 在本月所有工作表中无数据']}).to_excel(
                    writer,
                    sheet_name='无数据',
                    index=False
                )
            else:
                valid_bd_count += 1
                print(f"已创建BD [{bd}] 的工作簿: {bd_file_name}")

# 为每个BD创建压缩包
print("开始创建BD压缩包...")
for bd_name, bd_folder_path in bd_files_dict.items():
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

    print(f"已创建BD [{bd_name}] 的压缩包: {zip_path}")

    # 可选：删除原始BD文件夹（如果只需要压缩包）
    # shutil.rmtree(bd_folder_path)

# 计算并输出耗时
end_time = datetime.now()
total_seconds = (end_time - start_time).total_seconds()
print(f'程序执行完成，耗时: {total_seconds:.4f} 秒')
print(f'共生成 {valid_bd_count} 个有效BD工作簿')
print(f'共创建 {len(bd_files_dict)} 个BD压缩包')