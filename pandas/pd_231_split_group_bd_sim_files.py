import os.path

import pandas as pd

# register_file = r'E:\桌面08\POS活动统计_更新至2025年\POS统计--2025年\POS统计--202510\sim_deregistration_month_202510.xlsx'

begin_time = pd.Timestamp('now')

register_file = r'E:\桌面08\POS活动统计_更新至2025年\POS统计--2025年\POS统计--202510\warning_sim_split_bd_202510.xlsx'

df = pd.read_excel(register_file)

# df_group = df.groupby('顶级代理商BD')  # 关闭sim文件

df_group = df.groupby('归属顶级代理商BD')  # 预警文件

current_date = pd.Timestamp('now').strftime('%Y%m')

out_dir = r'E:\bat\output_files\split_dir'

for df_name, df_data in df_group:
    # print(df_name)
    num = len(df_data)
    # out_file = os.path.join(out_dir, f'{current_date}_内置sim卡拟关停商户明细_{num}条_{df_name}.xlsx')
    out_file = os.path.join(out_dir, f'{current_date}_内置sim卡续费_预警商户明细_{num}条_{df_name}.xlsx')
    df_data.to_excel(out_file, index=False, freeze_panes=(1, 1))

end_time = pd.Timestamp('now')

total_seconds = (end_time - begin_time).total_seconds()
print(f'this program spent time: {total_seconds: .4f} seconds.')
