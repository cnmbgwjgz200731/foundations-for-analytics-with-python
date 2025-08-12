import datetime

import dateutil.tz
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import glob
import os
# import jinja2

import matplotlib

from io import StringIO
from io import BytesIO

input_file = 'E:/bat/input_files/sales_2013.xlsx'
lx_test = 'E:/bat/input_files/split_mpos_less.xlsx'
xls_file = 'E:/bat/input_files/sales_2013.xls'
m_file = 'E:/bat/input_files/winequality-red.csv'
big_file = 'E:/bat/input_files/dq_split_file.xlsx'
team_file = 'E:/bat/input_files/team.xlsx'
csv_file = 'E:/bat/input_files/sales_january_2014.csv'
csv_feb_file = 'E:/bat/input_files/sales_february_2014.csv'
multindex_file = 'E:/bat/input_files/pandas_out_20240509042.xlsx'

path = 'E:/bat/output_files/pandas_read_csv_20240118.csv'
# df1.to_excel('E:/bat/output_files/pandas_out_20240510053.xlsx')

# TODO 忽略特定的警告 | 当遇到 openpyxl 中的 UserWarning 类型的警告时，它们将不会被打印出来。
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

'''
                第六部分
                 可视化
        可视化是数据分析的终点也是起点。
得益于生动呈现的可视化数据效果，我们能够跨越对于数据的认知鸿沟。
本部分主要介绍Pandas的样式功能如何让数据表格更有表现力，
Pandas的绘图功能如何让数据自己说话，如何定义不同类型的数据图形，
以及如何对图形中的线条、颜色、字体、背景等进行细节处理。
'''

print()
print('------------------------------------------------------------')
print('第15章 Pandas样式')
# update20240626
'''
Pandas的样式是一个被大多数人忽视的可视化方法，它不仅仅是美化数据、提高数据的可视化效果这么简单。
回忆一下，你在Excel中是不是经常对特定的数据加粗、标红、背景标黄？
这些操作就是为了让数据更加醒目清晰，突显数据的逻辑和特征。

在本章中，我们将介绍Pandas的一些内置样式，如何使用这些样式功能快速实现可视化效果，
如何自定义一些个性化的样式，还将介绍内容的格式化显示方法，
最后介绍样式的函数调用、复用、清除、带样式文件导出等操作。
'''
print('\t15.1 内置样式')
# 小节注释
'''
Pandas的样式在Jupyter Notebook和JupyterLab等代码编辑工具上获得了非常好的数据展示效果，让数据呈现更加专业，更加友好。
本节介绍Pandas样式并告诉大家一些它内置的、非常有用的样式功能。

▶ 以下是一些具体的使用方法举例：
'''
print()
print('------------------------------------------------------------')
print('\t15.1.1 样式功能')

'''
如同给Excel中的数据设置各种颜色、字体一样，Pandas提供的样式功能可实现：
    数值格式化，如千分号、小数位数、货币符号、日期格式、百分比等；
    凸显某些数据，对行、列或特定的值（如最大值、最小值）使用样式，如字体大小、黄色、背景；
    显示数据关系，如用颜色深浅代表数据大小；
    迷你条形图，如在一个百分比的格子里，用颜色比例表达占比；
    表达趋势，类似Excel中每行代表趋势变化的迷你走势图（sparkline）。

  我们发现，样式和可视化图形的区别是，数据图形化不关注具体数据内容，
  而样式则在保留具体内容的基础上进行修饰，让可读性更强。
  有时候两者有交叉共用的情况。


'''

print()
print('------------------------------------------------------------')
print('\t15.1.2 Styler对象')

'''
DataFrame有一个df.style Styler对象，用来生成数据的样式，样式是使用CSS来完成的。
如果你懂点CSS的知识会得心应手，不过也不用担心，CSS非常简单，基本就是一个字典，单词也是我们最常见的。

这里有个使用技巧，仅使用df.style就可以在Jupyter Notebook未给样式的情况下显示所有数据：
'''

# 读取数据
df = pd.read_excel(team_file)
print(df)
print(df.style)  # <pandas.io.formats.style.Styler object at 0x000001DF0829A250>
# 查看类型
print(type(df.style))  # <class 'pandas.io.formats.style.Styler'>

'''
Pandas提供了几个非常实用的内置样式，它们也是我们经常要使用的，
接下来将对这些提高计数效率的样式功能进行介绍。
'''

print()
print('------------------------------------------------------------')
print('\t15.1.3 空值高亮')

'''style.highlight_null()对为空的值高亮标示，增加背景颜色，使其更为醒目：'''
# 将一个值改为空
df.iloc[1, 1] = np.NaN
print(df)
print('# 将空值高亮，默认为红色背景')
print(df.head().style.highlight_null())

# 将空值高亮，默认为红色背景
styled_df = df.style.highlight_null()
# <pandas.io.formats.style.Styler object at 0x0000023C5DD9B250>

# 输出.xlsx文件 测试正常！
# styled_df.to_excel('E:/bat/output_files/pandas_out_20240626032.xlsx',index=False)
# df.head().style.highlight_null().to_excel('E:/bat/output_files/pandas_out_20240626034.xlsx',index=False)

print('可以指定颜色：')
# 使用颜色名
# blue_style_df = df.style.highlight_null(color='blue') # 指定蓝色
# blue_style_df.to_excel('E:/bat/output_files/pandas_out_20240626035.xlsx',index=False)

# 使用颜色值 | 使用十六进制颜色代码
# num_style_df = df.style.highlight_null(color='#ccc') # 指定灰色
# num_style_df.to_excel('E:/bat/output_files/pandas_out_20240626036.xlsx',index=False)

# 使用 RGB 颜色值 || 报错 pandas不支持rgb格式
# green_style_df = df.style.highlight_null(color='rgb(0, 255, 0)')
# green_style_df.to_excel('E:/bat/output_files/pandas_out_20240626037.xlsx',index=False)

'''
颜色名和颜色值与CSS中的颜色表示方法相同，可以用CSS颜色名
和CSS合法颜色值表示，相关内容将会在第16章详细介绍。
'''

print()
print('------------------------------------------------------------')
print('\t15.1.4 极值高亮')

'''
分别将最大值高亮显示，最小值高亮显示，二者同时高亮显示并指定颜色，
示例代码如下，效果分别如图15-2～图15-4所示。
'''
print('# 将最大值高亮，默认为黄色背景')

# print(df.select_dtypes(include='number'))
df_select = df.select_dtypes(include='number')
# max_df = df_select.style.highlight_max(color='yellow') # 仅输出数值列 非数值列没有显示！
# max_df.to_excel('E:/bat/output_files/pandas_out_20240626038_01.xlsx',index=False)

max_df = df.style.highlight_max(color='yellow', subset=df_select.columns)
max_df.to_excel('E:/bat/output_files/pandas_out_20240627041_001.xlsx', index=False)

print()
print('# 将最小值高亮')
# min_df = df.style.highlight_min(color='green')
# min_df.to_excel('E:/bat/output_files/pandas_out_20240626039.xlsx',index=False)
'''
# 测试成功！
min_df = df.style.highlight_min(color='green',subset=['Q1']) # 仅应用Q1列
min_df = df.style.highlight_min(color='green',subset=df.select_dtypes(include='number').columns) # 应用每个数值列 成功！
min_df = df.style.highlight_min(color='green',subset=df_select.columns,axis=1) # 应用每行数值最小值 成功！
min_df.to_excel('E:/bat/output_files/pandas_out_20240626039_004.xlsx',index=False)
'''

'''
# update20240627 openai提供的函数应用方法 测试成功！参考！

# 每个数值列中的最大值

# 定义一个函数来应用高亮
# highlight_max 函数会对每列的数据进行检查，如果数据是最大值，则返回相应的高亮样式。
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

# 选择数值列
numeric_cols = df.select_dtypes(include='number').columns

# 应用高亮到数值列
# 通过 apply 方法对数值列应用高亮函数。
styled_df = df.style.apply(highlight_max, subset=numeric_cols)
# styled_df = df.style.apply(highlight_max, subset=numeric_cols,axis=1) # axis=1 表示每行最大值应用高亮

# 保存到 Excel 文件中
output_path = 'E:/bat/output_files/pandas_out_20240626039.xlsx'
styled_df.to_excel(output_path, engine='openpyxl', index=False)

print(f'文件已成功保存到 {output_path}')

'''

"""
# 所有数值列中的最大值 显示高亮

# 选择数值列
numeric_df = df.select_dtypes(include='number')

# 找到所有数值列中的全局最大值
global_max = numeric_df.max().max()

# 定义一个函数来应用高亮
def highlight_global_max(val):
    color = 'background-color: yellow' if val == global_max else ''
    return color

# 应用高亮到数值列
styled_df = df.style.applymap(highlight_global_max, subset=numeric_df.columns)

# 保存到 Excel 文件中
output_path = 'E:/bat/output_files/pandas_out_20240626039_01.xlsx'
styled_df.to_excel(output_path, engine='openpyxl', index=False)

print(f'文件已成功保存到 {output_path}')

"""

print()
print('# 以上同时使用并指定颜色')

mi_mx_df = (df.style
            .highlight_max(color='lime', subset=df_select.columns)
            .highlight_min(subset=df_select.columns)  # 默认颜色为黄色
            )
mi_mx_df.to_excel('E:/bat/output_files/pandas_out_20240627041_002.xlsx', index=False)

print('指定行级')

mi_mx_df_row = (df.style
                .highlight_max(color='lime', subset=df_select.columns, axis=1)
                .highlight_min(subset=df_select.columns, axis=1)  # 默认颜色为黄色
                )
mi_mx_df_row.to_excel('E:/bat/output_files/pandas_out_20240627041_003.xlsx', index=False)

print()
print('也可以作用于指定行：')
# 只对Q1起作用
one_colmn_df = df.style.highlight_min(subset=['Q1'])
one_colmn_df.to_excel('E:/bat/output_files/pandas_out_20240627041_004.xlsx', index=False)

# 对Q1、Q2两列起作用
two_columns_df = df.style.highlight_max(subset=['Q1', 'Q2'])
two_columns_df.to_excel('E:/bat/output_files/pandas_out_20240627041_005.xlsx', index=False)

# 使用pd.IndexSlice索引器（和loc[]类似）
# 注意，数据是所有数据，算最小值的范围而不是全部
cross_df = df.style.highlight_max(subset=pd.IndexSlice[:3, ['Q1', 'Q3']])
cross_df.to_excel('E:/bat/output_files/pandas_out_20240627041_006.xlsx', index=False)

# 按行，只在这两列进行
intersec_df = df.style.highlight_min(axis=1, subset=['Q1', 'Q2'])
intersec_df.to_excel('E:/bat/output_files/pandas_out_20240627041_007.xlsx', index=False)

print()
print('------------------------------------------------------------')
print('\t15.1.5 背景渐变')

'''
根据数值的大小，背景颜色呈现梯度渐变，越深表示越大，越浅表示越小，
类似于Excel中的色阶样式。颜色指定为Matplotlib库的色系表（Matplotlib colormap）中的色系名，
可通过这个网址查看色系名和颜色示例：https://matplotlib.org/devdocs/gallery/color/colormap_reference.html。
background_gradient()会对数字按大小用背景色深浅表示，

示例代码如下，效果如图15-6所示。
'''

print('# 数字类型按列背景渐变')

df_select = df.select_dtypes('number')
# print(df_select)

# background_df = df.style.background_gradient(subset=df_select.columns,)
# 指定数值列
background_df = df.style.background_gradient(subset=df_select.columns, cmap='BuGn')
background_df.to_excel('E:/bat/output_files/pandas_out_20240627042_003.xlsx', index=False)

df.style.background_gradient()
print(df.style.background_gradient())  # <pandas.io.formats.style.Styler object at 0x0000013FF2032AF0>

print(df.style.background_gradient(subset=['Q1'], cmap='BuGn'))
# <pandas.io.formats.style.Styler object at 0x00000134A9592AF0>
print('指定具体一列')
back_df = df.style.background_gradient(subset=['Q1'], cmap='BuGn')
back_df.to_excel('E:/bat/output_files/pandas_out_20240627042_001.xlsx', index=False)

print()
print(pd.__version__)  # 2.1.4
print(matplotlib.__version__)  # 3.8.4

print()
print('# 低百分比和高百分比范围, 更换颜色时避免使用所有色域')
# high_df = df.style.background_gradient(subset=df_select.columns, low=0.6, high=0)
# high_df = df.style.background_gradient(subset=df_select.columns,cmap='viridis', low=0.6, high=0) # cmap参数
# high_df = df.style.background_gradient(subset=df_select.columns, low=0, high=1)  #
# high_df = df.style.background_gradient(subset=df_select.columns, low=1, high=0)  # 这个也是从低到高颜色加深 如上
# high_df = df.style.background_gradient(subset=df_select.columns, low=0.6, high=0)
# high_df = df.style.background_gradient(subset=df_select.columns, low=0, high=0.6)
high_df = df.style.background_gradient(subset=df_select.columns, low=0.4, high=0.6)
high_df = df.style.background_gradient(subset=df_select.columns, low=0.1, high=0.3)
print(high_df)
high_df.to_excel('E:/bat/output_files/pandas_out_20240701012_008.xlsx', index=False)
'''
默认 low 和 high（即 low=0 和 high=1）：颜色渐变从最小值到最大值均匀分布。
low=0.6 和 high=0：颜色渐变集中在数据的高值部分，低值部分几乎没有颜色。
low=0 和 high=0.4：颜色渐变集中在数据的低值部分，高值部分几乎没有颜色。
通过调整 low 和 high 参数，你可以控制颜色渐变的范围和集中区域，在数据可视化时突出显示特定部分的数据。

个人测试上述情况 感觉上述解释 不靠谱
除了颜色有些变化，感觉参数调整高低值后，还是按照数值从小到大 颜色渐深  
'''

print()
print('# 内容的颜色，取0～1（深色到浅色），方便凸显文本')
# text_df = df.style.background_gradient(text_color_threshold=0.5)
# text_df = df.style.background_gradient(text_color_threshold=0.9)  # 低值看不清值
text_df = df.style.background_gradient(text_color_threshold=0.1)  # 中高值看不清值 最高值可以看清
# text_df = df.style.background_gradient(text_color_threshold=0.6)
text_df.to_excel('E:/bat/output_files/pandas_out_20240701012_012.xlsx', index=False)

print()
print('# 颜色应用的取值范围，不在这个范围的不应用')
# vin_df = df.style.background_gradient(vmin=60, vmax=100)

# vin_df = df.style.background_gradient(vmin=30, vmax=50)  # 看起来颜色更深了 极低值好像没颜色。
# vin_df = df.style.background_gradient(vmin=0, vmax=50)  # 比上个看起来颜色更深。
vin_df = df.style.background_gradient(vmin=50, vmax=70)  # 颜色分明
vin_df.to_excel('E:/bat/output_files/pandas_out_20240701012_016.xlsx', index=False)

print()
print('# 链式方法使用样式')
vin_df = (df.style
          .background_gradient(subset=['Q1'], cmap='spring')  # 指定色系
          .background_gradient(subset=['Q2'], vmin=60, vmax=100)  # 指定应用值区间
          .background_gradient(subset=['Q3'], low=0.6, high=0)  # 高低百分比范围
          .background_gradient(subset=['Q4'], text_color_threshold=0.9)  # 文本色深
          )

vin_df.to_excel('E:/bat/output_files/pandas_out_20240701012_017.xlsx', index=False)

print()
print('------------------------------------------------------------')
print('\t15.1.6 条形图')

'''
条形图在表格里一般以横向柱状图的形式代表这个值的大小。
'''

print('# 显示Q4列的条形图')

df_select = df.select_dtypes('number')
# print(df_select)

# bar_df = df.style.bar(subset=['Q4'], vmin=50, vmax=100) # 输出没有效果！
bar_df = df.style.bar(subset=['Q4'])

bar_df.to_excel('E:/bat/output_files/pandas_out_20240701012_019.xlsx', index=False)

'''
Pandas的Styler对象使用background_gradient方法生成的渐变色样式可以在导出到Excel文件时保留，
因为该方法直接在Excel单元格中应用背景颜色。
而bar方法生成的条形图样式是通过CSS样式实现的，这些样式在导出到Excel文件时不会被保留，因此你无法在Excel文件中看到条形图。
'''

print('一些常用的参数及方法：')
'''
# 基本用法，默认对数字应用
df.style.bar()
# 指定应用范围
df.style.bar(subset=['Q1'])
# 定义颜色
df.style.bar(color='green')
df.style.bar(color='#ff11bb')
# 以行方向进行计算和展示
df.style.bar(axis=1)
# 样式在格中的占位百分比，0～100，100占满
df.style.bar(width=80)
# 对齐方式：
# 'left'：最小值开始
# 'zero'：0值在中间
# 'mid'：(max-min)/2 值在中间，负（正）值0在右（左）
df.style.bar(align='mid')
# 大小基准值
df.style.bar(vmin=60, vmax=100)


以下是一个综合示例，
# 以下是一个综合示例，代码效果如图15-9所示。
(df.head(10)
    .assign(avg=df.mean(axis=1, numeric_only=True)) # 增加平均值
    .assign(diff=lambda x: x.avg.diff()) # 和前一位同学的差值
    .style
    .bar(color='yellow', subset=['Q1'])
    .bar(subset=['avg'],width=90,align='mid',vmin=60, vmax=100,color='#5CADAD')
    .bar(subset=['diff'],color=['#ffe4e4','#bbf9ce'], # 上涨和下降的颜色
         # vmin=0, vmax=30, # 范围定为以0为基准的上下30 || 负数没有颜色
         vmin=-20, vmax=30, # 范围定为以0为基准的上下30 || 负数显示颜色
         align='zero') # 0 值居中
    )


15.1.7 小结
Pandas的内置样式也是我们在Excel操作中经常用到的功能，这些功能非常实用又方便操作，
希望大家在数据处理的最后环节不要忘记给数据增加样式。

'''

print('\t15.2 显示格式')
# update20240702
# 小节注释
'''
我们在最终输出数据以进行查看时，需要对数据进行相应的格式化，常见的如加货币符号、加百分号、增加千分位等，
目的是让计数更加场景化，明确列表一定的业务意义。
Styler.format是专门用来处理格式的方法。

▶ 以下是一些具体的使用方法举例：
'''
print()
print('------------------------------------------------------------')
print('\t15.2.1 语法结构')

'''
Styler.format的语法格式为：
# 语法格式
Styler.format(self, formatter,
                subset=None,
                na_rep: Union[str, NoneType]=None)

'''
# 读取数据
df = pd.read_excel(team_file)

'''
以上语法中的formatter可以是（str,callable, dict, None）中的任意一个，
一般是一个字典（由列名和格式组成），也可以是一个函数。
关于字符的格式化可参考Python的格式化字符串方法。
'''

print('# 给所有数据加一个方括号')
# 在jupyter 环境中 显示正常，输出文件没有显示
format_df = df.style.format("[{}]")
format_df.to_excel('E:/bat/output_files/pandas_out_20240702021_005.xlsx', index=False)

# 百分号
percent_df = df.style.format("{:.2%}", subset=df.select_dtypes('number').columns)
# percent_df.to_excel('E:/bat/output_files/pandas_out_20240702021_007.xlsx', index=False)
print(percent_df)

print()
print('------------------------------------------------------------')
print('\t15.2.2 常用方法')
'''
由于支持Python的字符串格式，Styler.format可以实现丰富多样的数据格式显示，以下为常用的格式方法：
'''

'''
# 百分号
df.style.format("{:.2%}")
# 指定列全变为大写
df.style.format({'name': str.upper})
# B，保留四位；D，两位小数并显示正负号
df.style.format({'B': "{:0<4.0f}", 'D': '{:+.2f}'})
# 应用lambda
df.style.format({"B": lambda x: "±{:.2f}".format(abs(x))})
# 缺失值的显示格式
df.style.format("{:.2%}", na_rep="-")
# 处理内置样式函数的缺失值
df.style.highlight_max().format(None, na_rep="-")
# 常用的格式
{'a': '¥{0:,.0f}', # 货币符号
'b': '{:%Y-%m}', # 年月
'c': '{:.2%}', # 百分号
'd': '{:,f}', # 千分位
'e': str.upper} # 大写
'''

print()
print('------------------------------------------------------------')
print('\t15.2.3 综合运用')
'''
显示格式可以多次设定，也可以与颜色相关样式一起使用。以下是一个综合的应用案例：
'''

# 链式方法使用格式
intergrative_df = (df
                   .assign(avg=df.select_dtypes('number').mean(axis=1, numeric_only=True) / 100)  # 增加平均值百分比
                   .assign(diff=lambda x: x.avg.diff())  # 与前一位同学的差值
                   .style
                   .format({'name': str.lower})
                   .format({'avg': "{:.2%}"})
                   .format({'diff': "¥{:.2f}"}, na_rep='-')
                   )
print(intergrative_df)

'''
15.2.4 小结
为数据增加常用的格式（如大小写、千分位符、百分号、正负号等），
既可以让数据表达更加直观清晰，也可以让数据的显示更加专业。
'''

print()
print('------------------------------------------------------------')
print('\t15.3.1 样式配置操作')

'''
.set_caption('xxx')给显示的表格数据增加一个标题，
以下代码的效果如图15-11所示。
'''
# 读取数据
df = pd.read_excel(team_file)

'''
以上语法中的formatter可以是（str,callable, dict, None）中的任意一个，
一般是一个字典（由列名和格式组成），也可以是一个函数。
关于字符的格式化可参考Python的格式化字符串方法。
'''
# 添加标题
title_df = df.head().style.set_caption('学生成绩表')
# print(title_df)
# 测试文件输出 还是没效果
# title_df.to_excel('E:/bat/output_files/pandas_out_20240703031_001.xlsx', index=False)

hide_df = df.style.hide(subset=[1, 2], axis=0)  # 隐藏指定行
# 输出文件没有隐藏效果
hide_df.to_excel('E:/bat/output_files/pandas_out_20240703031_002.xlsx', index=False)

print('# 隐藏指定行列')
# df.style.hide_columns(['Q1','Q2']) # AttributeError: 'Styler' object has no attribute 'hide_columns'

df.style.hide(axis=1)  # 不显示列名（列索引）
df.style.hide(axis='columns')  # 同上

# df.style.hide(axis=1, names = ['Q1','Q2']) # 输出没效果 == df  依然显示全部行列索引
# df.style.hide(axis=1, subset = ['Q1','Q2']) # 隐藏指定列
# df.style.hide(subset = ['Q1','Q2'], axis=1) # 隐藏指定列
# df.style.hide(subset = ['Q1','Q2'], axis=1, names=True) # 隐藏指定列
# df.style.hide(subset = ['Q1','Q2'], axis=1, names=False) # 隐藏指定列 || false设置后 依然显示隐藏列名

df.style.hide(subset=[1, 2], axis=0)  # 隐藏指定行

print()
print('------------------------------------------------------------')
print('\t15.3.2 表格CSS样式')

'''
我们知道，Pandas的样式是通过生成和修改输出的HTML让浏览器渲染而得到一定的显示效果的，
如果内置样式无法实现，可以通过直接指定HTML树节点上的CSS样式来实现复杂的功能。

因此，在理解以下功能时需要有一定的HTML和CSS等前端编程基础。
.set_properties()给单元格配置CSS样式，以下代码的效果如图15-14所示。
'''

print('# 将Q1列文字设为红色')

# 测试输出文件 单元格文字显示 红色 （标题没有颜色）
red_df = df.style.set_properties(subset=['Q1'], **{'color': 'red'})
red_df.to_excel('E:/bat/output_files/pandas_out_20240703031_003.xlsx', index=False)

print('# 一些其他示例')

# 测试输出文件 单元格文字显示 白色 （标题没有颜色）
# mmp 输出的文字 太白了 差点以为只有标题 没有内容
white_df = df.style.set_properties(color='white', align='right')
white_df.to_excel('E:/bat/output_files/pandas_out_20240703031_004.xlsx', index=False)

# 单元格背景色 黄色
white_df = df.style.set_properties(**{'background-color': 'yellow'})
white_df.to_excel('E:/bat/output_files/pandas_out_20240703031_005.xlsx', index=False)

# 字体增大
white_df = df.style.set_properties(**{'width': '100px', 'font-size': '18px'})
white_df.to_excel('E:/bat/output_files/pandas_out_20240703031_006.xlsx', index=False)

# border-color 边框颜色  | lawngreen 草坪绿
white_df = df.style.set_properties(**{'background-color': 'black',
                                      'color': 'lawngreen',
                                      'border-color': 'white'})
white_df.to_excel('E:/bat/output_files/pandas_out_20240703031_007.xlsx', index=False)

'''
.set_table_attributes()用于给<table>标签增加属性，可以随意给定属性名和属性值：
'''
white_df = df.style.set_table_attributes('class="pure-table"')
# 没看出来输出文件有什么变化
white_df.to_excel('E:/bat/output_files/pandas_out_20240703031_009.xlsx', index=False)

white_df = df.style.set_table_attributes('id="gairuo-table"')
print(white_df)
# 同样没看出来输出文件有什么变化
white_df.to_excel('E:/bat/output_files/pandas_out_20240703031_010.xlsx', index=False)

'''
set_table_styles()用于设置表格样式属性，用来实现极为复杂的显示功能。
可以带有选择器和props键的字典。
选择器selector的值是样式将应用此CSS样式的内容（自动以表的UUID为前缀），
props是由CSS样式属性和值组成的元组列表。
如下例：
'''

# 给所有的行（tr标签）的hover方法设置黄色背景
# 效果是当鼠标移动上去时整行背景变黄  jupyter中有效果！
new_df = df.style.set_table_styles([{'selector': 'tr:hover',
                                     'props': [('background-color', 'yellow')]
                                     }])
# 输出文件无效果！
new_df.to_excel('E:/bat/output_files/pandas_out_20240703031_012.xlsx', index=False)

print()
# 给所有的行（tr标签）的hover方法设置黄色背景
# 效果是当鼠标移动上去时整行背景变黄

print('# 为每个表格增加一个相同的符缀')
# 没看出来有啥不一样
df.style.set_uuid(9999)

# 加"gairuo"
# 同样没看出来有啥不一样
df.style.set_uuid('gairuo')

print()
print('------------------------------------------------------------')
print('\t15.3.3 应用函数')
# update20240704
'''
像Series和DataFrame一样，Styler也可以使用apply()和applymap()定义复杂的样式。
如用函数实现将最大值显示为红色：
'''


# 将最大值显示红色
def highlight_max(x):
    return ['color : red' if v == x.max() else '' for v in x]


print('# 应用函数')
func_style = df.style.apply(highlight_max)
print(func_style)
# 文件输出有颜色效果
func_style.to_excel('E:/bat/output_files/pandas_out_20240704041_001.xlsx', index=False)

print('# 按条件为整行加背景色（样式）')
# row.name 行数 ||
'''
len(row) 
len(row)返回当前行的长度，即该行中列的数量。
['background-color: red'] * len(row)生成一个长度与当前行相同的列表，每个元素都是字符串'background-color: red'。
例如，如果当前行有4个单元格，那么['background-color: red'] * 4就会生成['background-color: red', 'background-color: red', 
'background-color: red', 'background-color: red']。

这样做的目的是确保为每个单元格都应用相同的样式。
'''


def background_color_red(row):
    if row.name >= 5:
        return ['background-color: red'] * len(row)
    elif row.name >= 3:
        return ['background-color: yellow'] * len(row)
    return [''] * len(row)


# 应用函数
# df.style.apply(background_color, axis=1)

print('# 按照行数 奇偶性 填充颜色')


# 按照行数 奇偶性 填充颜色
# 浅灰：rgb(211, 211, 211)  中灰：rgb(169, 169, 169)  深灰：rgb(105, 105, 105)
# 十六进制值: 使用#AAAAAA表示中灰色。你可以根据需要调整颜色的深浅，例如#D3D3D3表示浅灰色，#696969表示深灰色。
def background_color(row):
    if row.name % 2 == 0:
        return ['background-color: #D3D3D3'] * len(row)
        # return ['background-color: rgb(211, 211, 211)'] * len(row)
    else:
        return ['background-color: #AAAAAA'] * len(row)


# 应用函数
df.style.apply(background_color, axis=1)

# 应用函数
print(df.style.apply(background_color, axis=1))
# 测试 支持单词 或者 16进制 输出  不支持rgb格式
df.style.apply(background_color, axis=1).to_excel('E:/bat/output_files/pandas_out_20240704041_004.xlsx', index=False)

print('# 简单的整行背景设置 || 修改后：')
# 测试正常！
df.style.apply(lambda x: ['background-color: yellow'] * len(x) if x.Q1 > 68 else [''] * len(x), axis=1)

# 定义函数，只对数字起作用，将大于90的值的背景设置为黄色
bg = lambda x: 'background-color: yellow' if type(x) == int and x > 90 else ''  # 应用函数
# df.style.applymap(bg)
'''
FutureWarning: Styler.applymap has been deprecated. Use Styler.map instead.
  df.style.applymap(bg)
'''
df.style.map(bg)  # 测试正常

print('subset可以限制应用的范围：')
# 指定列表（值大于0）加背景色
df.style.map(lambda x: 'background-color: grey' if x > 0 else '',
             subset=pd.IndexSlice[:, ['Q1', 'Q2']])

print('# 定义样式函数')


# 了将字体放大和将name列全小写
# 'font-size': '-200%' 字体缩小  'font-size': '200%' ：字体放大
def my_style(styler):
    return (styler.set_properties(**{'font-size': '-200%'})
            .format({'name': str.lower}))


# 应用管道方法
df.style.pipe(my_style)

print()
print('------------------------------------------------------------')
print('\t15.3.4 样式复用')
# update20240704
'''
可以将数据和样式应用到新表格中：
'''

print('# 将df的样式赋值给变量')

# 示例数据
data1 = {
    'A': [1, -2, 3, -4, 5],
    'B': [-1, 2, -3, 4, -5]
}
df = pd.DataFrame(data1)

data2 = {
    'C': [10, -20, 30, -40, 50],
    'D': [-10, 20, -30, 40, -50]
}
df2 = pd.DataFrame(data2)


# 定义函数，为负值添加红色背景
# def color_negative_red(val):
#     color = 'grey' if val < 0 else ''
#     return f'background-color: {color}'

# 定义函数，为负值添加灰色背景
def color_negative_red(val):
    if val < 0:
        return 'background-color: grey'
    else:
        return 'background-color: transparent'


# 将df的样式赋值给变量
style1 = df.style.map(color_negative_red)

# df2的样式为style2
style2 = df2.style

# style2使用style1的样式
style2.use(style1.export())

# 在Jupyter Notebook中显示结果
style2

# 输出到Excel文件 | 有效
# style2.to_excel('E:/bat/output_files/pandas_out_20240704041_006.xlsx', index=False)


print()
print('------------------------------------------------------------')
print('\t15.3.5 样式清除')
# update20240704
'''
df.style.clear()会返回None。如下清除所有样式：
'''

print('# 定义为一个变量')



print()
print('------------------------------------------------------------')
print('\t15.3.6 导出Excel')
# update20240705
'''
可以将样式生成HTML和导出Excel。
生成HTML可以用它来发邮件，做网页界面，生成Excel可以做二次处理或者传播。
样式导出Excel后会保留原来定义的大多数样式，方法如下：
'''

print('# 导出Excel')

# 定义为一个变量
print(df)

# 测试空值
# df.iloc[2, 2] = np.nan
# 测试浮点数
df['Q1'] = df['Q1'].astype(float)  # 转换类型 否则小数值插入整数列 会有报警
df.iloc[2, 2] = 3.1415926


def highlight_max(x):
    return ['color : red' if v == x.max() else '' for v in x]


dfs = df.loc[:, 'Q1':'Q4'].style.apply(highlight_max)

# df.style.to_excel('E:/bat/output_files/pandas_out_20240705051_001.xlsx', index=False)
# df.style.to_excel('E:/bat/output_files/pandas_out_20240705051_003.xlsx', index=False, engine='openpyxl')
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_002.xlsx', index=False)
print('# 指定标签页名称，sheet_name')
# df.style.to_excel('E:/bat/output_files/pandas_out_20240705051_004.xlsx', index=False, sheet_name='test')
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_005.xlsx', index=False, sheet_name='test')
print('# 指定缺失值的处理方式')
# df.style.to_excel('E:/bat/output_files/pandas_out_20240705051_006.xlsx', index=False, na_rep='-')
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_007.xlsx', index=False, na_rep='-')
print('# 浮点数字格式，下例将0.1234转为0.12')
# 测试 输出没有显示浮点数格式
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_008.xlsx', float_format="%.2f")
# 浮点数 转化格式 测试成功！
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_009.xlsx', float_format="%.2f")
print('# 只要这两列')
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_010.xlsx', columns=['Q1', 'Q3'])
print('# 不带表头、索引')
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_011.xlsx', header=False, index=False)
print('# 指定索引，多个值代表多层索引')
'''dfs没有team、name列'''
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_012.xlsx', index_label=['team', 'name'])
'''没看出来有啥 除了索引列名称为team 原name、team列均正常'''
# df.to_excel('E:/bat/output_files/pandas_out_20240705051_013.xlsx', index_label=['team', 'name'])
'''关闭默认索引，试试看|| 输出文件没变化， 猜测可能team、name列是索引'''
# df.to_excel('E:/bat/output_files/pandas_out_20240705051_014.xlsx', index_label=['team', 'name'], index=False)
print('# 从哪行取，从哪列取')
'''数据全部输出，数据向右平移2列，向下平移3行'''
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_015.xlsx', startrow=3, startcol=2)

'''
# 不合并单元格
dfs.to_excel('gairuo.xlsx', merge_cells=False)
# 指定编码格式
dfs.to_excel('gairuo.xlsx', encoding='utf-8')
# 无穷大表示法（Excel中没有无穷大的本机表示法）
dfs.to_excel('gairuo.xlsx', inf_rep='inf')
# 在错误日志中显示更多信息
dfs.to_excel('gairuo.xlsx', verbose=True)
'''

print('# 指定要冻结的最底行和最右列')
'''A、B列被冻结，但看不出最底行冻结效果'''
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_016.xlsx', freeze_panes=(0, 2))
'''Q1列被冻结'''
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_017.xlsx', freeze_panes=(0, 1), index=False)
'''
(1, 0) 表示冻结首行。
1 表示在第 1 行之前冻结，所以实际冻结的是第 1 行。
0 表示不冻结任何列。
'''
# 测试成功，首行首列被冻结！
# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_018.xlsx', freeze_panes=(1, 1), index=False)

# dfs.to_excel('E:/bat/output_files/pandas_out_20240705051_019.xlsx', freeze_panes=(-1, 1), index=False)
'''UserWarning: Row number -1 must be >= 0
  warn("Row number %d must be >= 0" % row)'''

print()
print('------------------------------------------------------------')
print('\t15.3.7 生成HTML')
# update20240708

'''
Styler.render()可以输出样式的HTML代码，它可以传入以下参数：
head
cellstyle
body
uuid
precision
table_styles
caption
table_attributes
生成的HTML代码可用于网页显示、邮件正文内容等场景，方法如下：
'''


# 定义样式函数，为负值添加灰色背景
def color_negative_red(val):
    color = 'grey' if val > 90 else ''
    return f'background-color: {color}'


# 将样式应用到 DataFrame
styled_df = df.select_dtypes('number').style.map(color_negative_red)
# styled_df
print(styled_df)
# 生成 HTML 字符串
html = styled_df.to_html()
# html # 仅展示编码
# 使用 IPython.display.HTML 来展示
# HTML(html)
print(html)  # 生成HTML编码


'''
15.3.8 小结
本节介绍了Pandas样式的一些高级用法，这些是除了Pandas提供的内置方法外，
为有HTML和CSS基础的用户提供的超级功能，可以用它们来实现任何复杂的展示效果。
'''

