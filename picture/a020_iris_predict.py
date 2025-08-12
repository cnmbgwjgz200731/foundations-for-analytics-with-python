# update20240929
# 17.3.8 鸢尾花品种预测

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

input_file = 'E:/bat/input_files/iris.xlsx'

# 读取Excel文件
df = pd.read_excel(input_file, names=['萼片长度', '萼片宽度', '花瓣长度', '花瓣宽度', '种类名称'])

# 添加种类编号列
df_add_column = df.assign(种类=df['种类名称'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2}))

# 提取特征值和目标值
'''
提取特征值和目标值。特征值就是测量数据，
目标值是最终是哪个种类的结论。需要将数据结构转为array：
'''
x = df_add_column[['萼片长度', '萼片宽度', '花瓣长度', '花瓣宽度']].to_numpy()
y = df_add_column['种类'].to_numpy()

# 切分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 建立模型并设置迭代次数
lgr = LogisticRegression(max_iter=3000)

# 训练模型
lgr.fit(x_train, y_train)

# 测试结果
predictions = lgr.predict(x_test)
accuracy = (predictions == y_test).mean()
print(f"模型准确率: {accuracy:.2f}")  # 模型准确率: 1.00

# 单个样本预测
sample = [[5.1, 3.5, 5.4, 2.1]]  # 注意要使用双重方括号，表示二维数组
sample_scaled = scaler.transform(sample)
prediction = lgr.predict(sample_scaled)
print(f"预测结果: {prediction[0]}")  # 预测结果: 2

# 训练集上的准确度评分
train_set = lgr.score(x_train, y_train)
print(f"训练集准确率: {train_set:.2f}")  # 训练集准确率: 0.97

# 测试集上的准确度评分
test_set = lgr.score(x_test, y_test)
print(f"测试集准确率: {test_set:.2f}")  # 测试集准确率: 1.00

# 全局验证模型准确性
global_df = (
    df_add_column.assign(预测种类=lgr.predict(df_add_column.loc[:, '萼片长度':'花瓣宽度'].to_numpy()))
    .assign(是否正确=lambda x: x['种类'] == x.预测种类)
    .是否正确
    .value_counts(normalize=True)
)

# print(global_df)
'''
是否正确
False    0.666667
True     0.333333
Name: proportion, dtype: float64
'''

# openai修改后 提升准确性
"""
数据标准化：在训练和测试时，你对数据进行了标准化处理，但在全局验证时没有对数据进行相同的标准化处理。
模型的过拟合：模型在训练集和测试集上的表现很好，但在整个数据集上表现较差，可能是因为模型过拟合了训练数据。
"""
x_global = scaler.transform(df_add_column[['萼片长度', '萼片宽度', '花瓣长度', '花瓣宽度']].to_numpy())
df_add_column['预测种类'] = lgr.predict(x_global)
df_add_column['是否正确'] = df_add_column['种类'] == df_add_column['预测种类']

global_df = df_add_column['是否正确'].value_counts(normalize=True)
print(global_df)
'''
是否正确
True     0.973333
False    0.026667
Name: proportion, dtype: float64
'''


# 多个样本预测
sample = [[5.8, 2.7, 5.1, 1.9],
          [5, 3.3, 1.4, 0.2],
          [7, 3.2, 4.7, 1.4]]  # 注意要使用双重方括号，表示二维数组
sample_scaled = scaler.transform(sample)
prediction = lgr.predict(sample_scaled)
print(f"预测结果: {prediction}")  # 预测结果: [2 0 1]


# TODO 新csv文件 预测目标值！示例！
'''
在实际业务中，你可能需要将训练好的模型应用于新的数据集。
假设你已经有一个新的CSV文件，其中包含新的特征值。
你可以读取该CSV文件的特征值，使用训练好的模型进行预测，并将预测结果保存到一个新的文件中。
下面是一个详细的步骤和示例代码，展示如何实现这一过程。
'''
# 新CSV文件路径
new_input_file = 'E:/bat/input_files/new_iris_data.csv'

# 读取新的CSV文件
new_data = pd.read_csv(new_input_file, names=['萼片长度', '萼片宽度', '花瓣长度', '花瓣宽度'])

# 对新数据进行标准化
new_data_scaled = scaler.transform(new_data)

# 使用训练好的模型进行预测
new_predictions = lgr.predict(new_data_scaled)

# 将预测结果添加到新数据集中
new_data['预测种类'] = new_predictions

# 保存预测结果到新的CSV文件
output_file = 'E:/bat/output_files/predicted_iris_data.csv'
new_data.to_csv(output_file, index=False)

print(f"预测结果已保存到 {output_file}")