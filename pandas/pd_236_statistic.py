import numpy as np

print()
print('2.3 Numpy')
print('\t2.3.6 统计计算')
print()
# update20231228

# 两个数组间的操作采用行列式的运算规则
a = np.array([10, 20, 30, 40])
b = np.array([1, 2, 3, 4])
c = np.array(((2, 5, 4), (12, 13, 46)))

print(a[:3])  # 支持类似列表的切片 |out：[10 20 30]

print(a + b)  # 矩阵相加 | out:[11 22 33 44]
print(a - 1)  # out:[ 9 19 29 39]
print(np.sin(a))
print(4 * np.sin(a))

# 以下是一些数学函数的例子，还支持非常多的数学函数
print(a.max())  # out:40
print(a.min())  # out:10
print(a.sum())  # out:100
print(a.std())  # out:11.180339887498949
print(a.all())  # out:True
print(a.cumsum())  # out:[ 10  30  60 100]

'''
# print(b.sum(axis=1)) # 运行报错
出现这个错误的原因是因为在尝试对一个一维数组b使用axis=1进行求和操作。
一维数组没有第二个轴（axis 1），因为它只有一个维度（axis 0）。
numpy.AxisError表明你尝试访问的轴超出了数组的维度范围。
在这个例子中，数组b是一个一维数组，所以你只能沿着axis=0（即数组的唯一维度）进行操作。
解决这个问题的方法是，如果你想对数组b的所有元素求和，你应该使用axis=0或者不指定axis参数，因为默认就是沿着所有的轴求和。
'''
print(b.sum())  # out:10
print(b.sum(axis=0))  # out:10
print(c)
print(c.sum(axis=1))  # 多维可以指定方向 || out:[11 71]
print(c.sum(axis=0))  # out:[14 18 50]
print(c.sum())  # out:82
