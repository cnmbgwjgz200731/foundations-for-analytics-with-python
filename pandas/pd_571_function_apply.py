print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.7 函数应用')
print('\t5.7.1 pipe()')
print()
# update20240412
'''
我们知道，函数可以让复杂的常用操作模块化，既能在需要使用时直接调用，达到复用的目的，也能简化代码。
Pandas提供了几个常用的调用函数的方法。
pipe()：应用在整个DataFrame或Series上。
apply()：应用在DataFrame的行或列中，默认为列。
applymap()：应用在DataFrame的每个元素中。
map()：应用在Series或DataFrame的一列的每个元素中。
'''
# 小节注释
'''
Pandas提供的pipe()叫作管道方法，它可以让我们写的分析过程标准化、流水线化，达到复用目标，
它也是最近非常流行的链式方法的重要代表。
DataFrame和Series都支持pipe()方法。pipe()的语法结构为df.pipe(<函数名>, <传给函数的参数列表或字典>)。
它将DataFrame或Series作为函数的第一个参数（见图5-1），
可以根据需求返回自己定义的任意类型数据。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# pipe()可以将复杂的调用简化:')

df1 = df.select_dtypes(include='number')
# print(df1)

def h(df1):
    # 假设 h 函数将所有值增加 1
    return df1 + 1

# print(h(df1)) # 测试正常

def g(df1,arg1):
    # g 函数 将 所有值乘以 arg1
    return df1 * arg1

# print(g(df1,2)) # 测试正常

def f(df1,arg2,arg3):
    # f 函数 将所有值 加上 arg2 和 arg3 的和
    return df1 + arg2 + arg3

# print(f(df1,-80,1)) # 测试正常

df2 = pd.DataFrame({'A':[1,2,3],'B':[4,5,6]})
# print(df2)

print()
print('# 不使用 pipe,直接嵌套调用函数')
result1 = f(g(h(df2),arg1=2),arg2=3,arg3=4)
print(result1)
#     A   B
# 0  11  17
# 1  13  19
# 2  15  21

print('# 使用 pipe 连接函数')
result2 = (df2.pipe(h)
           .pipe(g,arg1 = 2)
           .pipe(f,arg2 = 3,arg3 = 4)
           )

print(result2)
#     A   B
# 0  11  17
# 1  13  19
# 2  15  21


print()
print('# 实际案例 2：')
# 定义一个函数，给所有季度的成绩加n，然后增加平均数
# 其中n中要加的值为必传参数
def add_mean(rdf,n):
    # pass
    df3 = rdf.copy()
    df3 = df3.loc[:,'Q1':'Q4'].applymap(lambda x: x + n)
    df3['avg'] = df3.loc[:,'Q1':'Q4'].mean(1)
    return df3

print(df.pipe(add_mean,100))
#     Q1   Q2   Q3   Q4     avg
# 0  189  121  124  164  149.50
# 1  136  137  137  157  141.75
# 2  157  160  118  184  154.75
# 3  193  196  171  178  184.50
# 4  165  149  161  186  165.25
# 5  200  199  197  200  199.00

print()
print('# 使用lambda')
# 筛选出Q1大于等于80且Q2大于等于90的数据
# df.pipe(lambda df_, x, y: df_[(df_.Q1 >= x) & (df_.Q2 >= y)], 80, 90)

result3 = df.pipe(lambda df_,x,y:df_[(df_.Q1 >= x) & (df_.Q2 >= y)],80,90)
print(result3)
#     name team   Q1  Q2  Q3   Q4
# 3  Eorge    C   93  96  71   78
# 5   Rick    B  100  99  97  100

print()
print(df.loc[(df.loc[:,'Q2'] >= 90) & (df.loc[:,'Q1'] >= 80)]) # 结果同上


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.7 函数应用')
print('\t5.7.2 apply()')
print()
# update20240412
# 小节注释
'''
apply()可以对DataFrame按行和列（默认）进行函数处理，也支持Series。
如果是Series，逐个传入具体值，DataFrame逐行或逐列传入，

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 将name全部变为小写:')
# print(df.name)
print(df.name.apply(lambda x: x.lower()))
# 0    liver
# 1     arry
# 2      ack
# 3    eorge
# 4      oah
# 5     rick
# Name: name, dtype: object
print()
print('# 案例二：DataFrame例子')
print('# 去掉一个最高分和一个最低分再算出平均分')

# s = df.Q1

def my_mean(s):
    max_min_ser = pd.Series([-s.max(),-s.min()])
    return s._append(max_min_ser).sum()/(s.count()-2)

# 对数字列应用函数
print(df.select_dtypes(include='number').apply(my_mean))
# print(my_mean(s))
# Q1    76.00
# Q2    60.50
# Q3    48.25
# Q4    78.00
# dtype: float64

print()
print('# 同样的算法以学生为维度计算')
print(df.set_index('name')
      .select_dtypes(include='number')
      .apply(my_mean,axis=1)
      ) # 横向计算
# name
# Liver    44.0
# Arry     37.0
# Ack      58.5
# Eorge    85.5
# Oah      63.0
# Rick     99.5
# dtype: float64

print()
print('# 判断一个值是否在另一个类似列表的列中')

df11 = pd.DataFrame({'s':[1,2,3,6],
                     's_list':[[1,2],[2,3],[3,4],[4,5]]})

# print(df11)

bool_series = df11.apply(lambda d: d.s in d.s_list,axis=1)
print(bool_series)
# 0    True
# 1    True
# 2    True
# 3    False
# dtype: bool

print()
print('# 将布尔序列转换为 0 和 1 序列')
int_series = bool_series.astype(int)
print(int_series)
# 0    1
# 1    1
# 2    1
# 3    0
# dtype: int32

# 它常被用来与NumPy库中的np.where()方法配合使用，如下例：
print()
print('# 函数，将大于90分的数字标记为good')
fun = lambda x: np.where(x.team == 'A' and x.Q1 > 30,'good','other') #
print(df.apply(fun,axis=1))
# 0    other
# 1    other
# 2     good
# 3    other
# 4    other
# 5    other
# dtype: object

print('# 结果同上')
print(df.apply(lambda x: x.team=='A' and x.Q1 > 30,axis=1)
      .map({True:'good',False:'other'}))

# df.apply(lambda x: 'good' if x.team=='A' and x.Q1>90 else '', axis=1)
# print(df.apply(lambda x: 'good' if x.team == 'A' and x.Q1>30 else '',axis=1)) # 逻辑同上

# result = df.apply(lambda x: 'good' if x.team == 'A' and x.Q1>30 else '',axis=1)
# result = df.apply(lambda x: True if x.team == 'A' and x.Q1>30 else False,axis=1)
# print(df.where(result).dropna()) # 不符合填充空值 删除空值！

print()
print('小节')
# 总结一下，apply()可以应用的函数类型如下：
# df.apply(fun) # 自定义
# df.apply(max) # Python内置函数
# df.apply(lambda x: x*2) # lambda
# df.apply(np.mean) # NumPy等其他库的函数
# df.apply(pd.Series.first_valid_index) # Pandas自己的函数

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.7 函数应用')
print('\t5.7.3 applymap()')
print()
# update20240416

# 小节注释
'''
df.applymap()可实现元素级函数应用，即对DataFrame中所有的元素（不包含索引）应用函数处理

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 计算数据的长度:')
# 使用lambda时，变量是指每一个具体的值。
def mylen(x):
    return len(str(x))

print(df.applymap(lambda x:mylen(x))) # 应用函数
#    name  team  Q1  Q2  Q3  Q4
# 0     5     1   2   2   2   2
# 1     4     1   2   2   2   2
# 2     3     1   2   2   2   2
# 3     5     1   2   2   2   2
# 4     3     1   2   2   2   2
# 5     4     1   3   2   2   3

print()
print(df.applymap(mylen)) # 结果同上

print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.7 函数应用')
print('\t5.7.4 map()')
print()
# update20240416
# 小节注释
'''
map()根据输入对应关系映射值返回最终数据，用于Series对象或DataFrame对象的一列。
传入的值可以是一个字典，键为原数据值，值为替换后的值。
可以传入一个函数（参数为Series的每个值），
还可以传入一个字符格式化表达式来格式化数据内容。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 枚举替换:')
print()
print(df.team.map({'A':'一班','B':'二班','C':'三班','D':'四班'})) # 枚举替换
# 0    NaN
# 1     三班
# 2     一班
# 3     三班
# 4     四班
# 5     二班
# Name: team, dtype: object

print()
print(df.team.map('I am a {}'.format))
# 0    I am a E
# 1    I am a C
# 2    I am a A
# 3    I am a C
# 4    I am a D
# 5    I am a B
# Name: team, dtype: object

# na_action='ignore' 参数指定如果遇到 NaN 值，则忽略它，不对其应用任何操作。
print(df.team.map('I am a {}'.format,na_action='ignore')) # 结果同上

# t = pd.Series({'six':6.,'seven':7.})
# s.map(t)
print(t)
# 应用函数
def f(x):
    return len(str(x))

print(df['name'].map(f))
# 0    5
# 1    4
# 2    3
# 3    5
# 4    3
# 5    4
# Name: name, dtype: int64


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.7 函数应用')
print('\t5.7.5 agg()')
print()
# update20240416
# 小节注释
'''
agg()一般用于使用指定轴上的一项或多项操作进行汇总，
可以传入一个函数或函数的字符，还可以用列表的形式传入多个函数。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 每列的最大值')
print(df.agg('max'))
# name    Rick
# team       E
# Q1       100
# Q2        99
# Q3        97
# Q4       100
# dtype: object

print()
print('# 将所有列聚合产生sum和min两行')
print(df.agg(['sum','min']))
#                          name    team   Q1   Q2   Q3   Q4
# sum  LiverArryAckEorgeOahRick  ECACDB  440  362  308  469
# min                       Ack       A   36   21   18   57

print()
print('# 序列多个聚合')
print(df.agg({'Q1':['sum','min'],'Q2':['min','max']}))
#         Q1    Q2
# sum  440.0   NaN
# min   36.0  21.0
# max    NaN  99.0

print()
print('# 分组后聚合')
print(df.groupby('team').agg('max'))
#        name   Q1  Q2  Q3   Q4
# team
# A       Ack   57  60  18   84
# B      Rick  100  99  97  100
# C     Eorge   93  96  71   78
# D       Oah   65  49  61   86
# E     Liver   89  21  24   64
print()
print(df.Q1.agg(['sum','mean']))
# sum     440.000000
# mean     73.333333
# Name: Q1, dtype: float64

'''
另外，agg()还支持传入函数的位置参数和关键字参数，支持每个列
分别用不同的方法聚合，支持指定轴的方向。
'''

print()
print('# 每列使用不同的方法进行聚合')
print(df.agg(a = ('Q1',max),
             b = ('Q2','min'),
             c = ('Q3',np.mean),
             d = ('Q4',lambda s:s.sum() + 1)
             ))
#       Q1    Q2         Q3     Q4
# a  100.0   NaN        NaN    NaN
# b    NaN  21.0        NaN    NaN
# c    NaN   NaN  51.333333    NaN
# d    NaN   NaN        NaN  470.0
print()
print(df.groupby('name').agg(a = ('Q1',max),
             b = ('Q2','min'),
             c = ('Q3',np.mean),
             d = ('Q4',lambda s:s.sum() + 1)
             ))

#          a   b     c    d
# name
# Ack     57  60  18.0   85
# Arry    36  37  37.0   58
# Eorge   93  96  71.0   79
# Liver   89  21  24.0   65
# Oah     65  49  61.0   87
# Rick   100  99  97.0  101

print()
print('# 按行聚合')
print(df.loc[:,'Q1':].agg('mean',axis='columns'))
# 0    49.50
# 1    41.75
# 2    54.75
# 3    84.50
# 4    65.25
# 5    99.00
# dtype: float64
print(df.loc[:,'Q1':].agg('mean'))
# Q1    73.333333
# Q2    60.333333
# Q3    51.333333
# Q4    78.166667
# dtype: float64

print()
print('# 利用pd.Series.add方法对所有数据加分，other是add方法的参数')

print(df.loc[:,'Q1':].agg(pd.Series.add,other=10))
#     Q1   Q2   Q3   Q4
# 0   99   31   34   74
# 1   46   47   47   67
# 2   67   70   28   94
# 3  103  106   81   88
# 4   75   59   71   96
# 5  110  109  107  110

'''agg()的用法整体上与apply()极为相似。'''


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.7 函数应用')
print('\t5.7.6 transform()')
print()
# update20240417

# 小节注释
'''
DataFrame或Series自身调用函数并返回一个与自身长度相同的数据。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 应用匿名函数')
print(df.transform(lambda x: x*2))
#          name team   Q1   Q2   Q3   Q4
# 0  LiverLiver   EE  178   42   48  128
# 1    ArryArry   CC   72   74   74  114
# 2      AckAck   AA  114  120   36  168
# 3  EorgeEorge   CC  186  192  142  156
# 4      OahOah   DD  130   98  122  172
# 5    RickRick   BB  200  198  194  200

print()
print('# 调用多个函数')
# print(df.transform([np.sqrt,np.exp])) # 报错！
print(df.select_dtypes('number').transform([np.sqrt,np.exp])) # 运行正常
#           Q1                      Q2  ...            Q3         Q4
#         sqrt           exp      sqrt  ...           exp       sqrt           exp
# 0   9.433981  4.489613e+38  4.582576  ...  2.648912e+10   8.000000  6.235149e+27
# 1   6.000000  4.311232e+15  6.082763  ...  1.171914e+16   7.549834  5.685720e+24
# 2   7.549834  5.685720e+24  7.745967  ...  6.565997e+07   9.165151  3.025077e+36
# 3   9.643651  2.451246e+40  9.797959  ...  6.837671e+30   8.831761  7.498417e+33
# 4   8.062258  1.694889e+28  7.000000  ...  3.104298e+26   9.273618  2.235247e+37
# 5  10.000000  2.688117e+43  9.949874  ...  1.338335e+42  10.000000  2.688117e+43

print()
print('# 调用多个函数 例2')
print(df.select_dtypes('number').transform([np.abs,lambda x: x + 1]))
#         Q1                Q2                Q3                Q4
#   absolute <lambda> absolute <lambda> absolute <lambda> absolute <lambda>
# 0       89       90       21       22       24       25       64       65
# 1       36       37       37       38       37       38       57       58
# 2       57       58       60       61       18       19       84       85
# 3       93       94       96       97       71       72       78       79
# 4       65       66       49       50       61       62       86       87
# 5      100      101       99      100       97       98      100      101

print()
print('# 调用函数 例3')
print(df.select_dtypes('number').transform({'abs'}))
#    abs abs abs  abs
# 0   89  21  24   64
# 1   36  37  37   57
# 2   57  60  18   84
# 3   93  96  71   78
# 4   65  49  61   86
# 5  100  99  97  100

print()
print('# 调用函数 例4 lambda x:x.abs()')
print(df.select_dtypes('number').transform(lambda x:x.abs())) # 结果同上！

print()
print('# 对比2个操作')
# transform sum 每行都有
print(df.groupby('team').sum())
#            name   Q1   Q2   Q3   Q4
# team
# A           Ack   57   60   18   84
# B          Rick  100   99   97  100
# C     ArryEorge  129  133  108  135
# D           Oah   65   49   61   86
# E         Liver   89   21   24   64
print(df.groupby('team').transform(sum))
#         name   Q1   Q2   Q3   Q4
# 0      Liver   89   21   24   64
# 1  ArryEorge  129  133  108  135
# 2        Ack   57   60   18   84
# 3  ArryEorge  129  133  108  135
# 4        Oah   65   49   61   86
# 5       Rick  100   99   97  100
'''
分组后，直接使用计算函数并按分组显示合计数据。
使用transform()调用计算函数，返回的是原数据的结构，
但在指定位置上显示聚合计算后的结果，这样方便我们了解数据所在组的情况。
'''


print()
print('------------------------------------------------------------')
print('第5章 pandas高级操作')
print('\t5.7 函数应用')
print('\t5.7.7 copy()')
print()
# update20240417
# 小节注释
'''
类似于Python中copy()函数，df.copy()方法可以返回一个新对象，
这个新对象与原对象没有关系。

▶ 以下是一些具体的使用方法举例：
'''
df = pd.read_excel(team_file) # Q1 name ,index_col='name'
# df = pd.read_excel(team_file, index_col='name') # Q1 name
print(df)
# print(df.dtypes)
print()

print('# 应用匿名函数')

s = pd.Series([1,2],index=['a','b'])
s_1 = s
s_copy = s.copy()
print(s_1 is s)  # True
print(s_copy is s) # False

# print(s)
# print(s_1)
# print(s_copy)
