import pandas as pd
import numpy as np
import os
from datetime import datetime
from glob import glob

# 文件路径设置
# source_file = 'E:/bat/input_files/POS推广商服务费-3月.xlsx'
# source_file = 'E:/桌面08/测试/临时拆分/拆分数据-胡国强/拆分分润数据工具/胡国强拆分文件_每月/2025-07/6月-21版赚钱吧D0奖励交易量明细（给BD）.xlsx'

out_folder = 'E:/bat/output_files/split_dir/split_bd'

# 创建输出目录
os.makedirs(out_folder, exist_ok=True)

source_dir = 'E:/桌面08/测试/临时拆分/拆分数据-胡国强/拆分分润数据工具/胡国强拆分文件_每月/2025-07/6月调价明细-SFT（给BD）*.xlsx'
source_files = glob(source_dir)

# 记录开始时间
start_time = datetime.now()
month_code = datetime.now().strftime('%Y%m%d')

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
        output_path = os.path.join(out_folder, bd_file_name)

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

                    # 必须有至少一个可见工作表
                    if not has_data:
                        # 创建提示工作表
                        pd.DataFrame({'提示': [f'BD [{bd}] 在本月所有工作表中无数据']}).to_excel(
                            writer,
                            sheet_name='无数据',
                            index=False
                        )

                    valid_bd_count += 1
                    print(f"已创建BD [{bd}] 的工作簿: {bd_file_name}")

# 计算并输出耗时
end_time = datetime.now()
total_seconds = (end_time - start_time).total_seconds()
print(f'程序执行完成，耗时: {total_seconds:.4f} 秒')
print(f'共生成 {valid_bd_count} 个有效BD工作簿')
