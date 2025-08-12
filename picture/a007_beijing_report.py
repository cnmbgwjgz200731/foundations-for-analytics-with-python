import pandas as pd

# 17.2.7 北京各区无新增新冠肺炎确诊病例天数
"""
2020年新冠肺炎疫情期间，“北京发布”微信公众号每天会发布北京
市上一日疫情数据，其中会介绍全市16区无报告病例天数情况。
原始数据如下：
"""

bj_file = 'E:/bat/input_files/bei_jing_file.xlsx'

# print(pd.read_clipboard())  # 复制
df = pd.read_excel(bj_file)

# print(df)

t = pd.Timestamp('today')
# print(t)
# print(t.strftime('%Y-%m-%d'))
# print(t.strftime('%Y-%m-%d').dt.strptime('%Y-%m-%d'))  # error

df_f = \
    (
        df.replace('Nan', pd.NaT)  # 将缺失值转为空时间
        #  将确诊日期转为时间格式
        .assign(最后一例确诊日期=lambda x: x['最后一例确诊日期'].astype('datetime64[ns]'))
        # .dtypes  #
        # 增加无报告病例天数列，当日与确诊日期相减
        # .assign(无报告病例天数=lambda x: pd.Timestamp('2020-11-16')-x['最后一例确诊日期'])
        .assign(无报告病例天数=lambda x: t - x['最后一例确诊日期'])  # test
        # 计算出天数
        .assign(无报告病例天数=lambda x: x['无报告病例天数'].dt.days)  # test
        # 排序，空值在前，重排索引
        # .sort_values('无报告病例天数', ascending=False, na_position='first', ignore_index=False)
        .sort_values('无报告病例天数', ascending=False, na_position='last', ignore_index=True)
        # .style.background_gradient(subset=['无报告病例天数'], cmap='spring')  # 显示有效果
        # .background_gradient(subset=df.select_dtypes('number').columns, cmap='BuGn')  # 没效果
    )

print(df_f)

'''
      地区   最后一例确诊日期  无报告病例天数
0    延庆区 2020-01-23   1652.0
1    怀柔区 2020-02-06   1638.0
2    顺义区 2020-02-08   1636.0
3    密云区 2020-02-11   1633.0
4   石景山区 2020-06-14   1509.0
5   门头沟区 2020-06-15   1508.0
6    房山区 2020-06-15   1508.0
7    东城区 2020-06-16   1507.0
8    通州区 2020-06-20   1503.0
9    朝阳区 2020-06-21   1502.0
10   西城区 2020-06-22   1501.0
11   海淀区 2020-06-25   1498.0
12   大兴区 2020-06-30   1493.0
13   丰台区 2020-07-05   1488.0
14   昌平区 2020-08-06   1456.0
15   平谷区        NaT      NaN

'''
