import pandas as pd
import faker  # 安装：pip install faker

f = faker.Faker('zh-cn')
# f = faker.Faker('zh-tw')
# f = faker.Faker()
df = pd.DataFrame({
    '客户姓名': [f.name() for i in range(10)],
    '年龄': [f.random_int(5, 90) for i in range(10)],
    '最后去电时间': [f.date_between(start_date='-3y', end_date='today')
               .strftime('%Y年%m月%d日') for i in range(10)],
    '意向': [f.random_element(('有', '无')) for i in range(10)],
    '地址': [f.street_address() for i in range(10)],
    '联系电话': [f.phone_number() for i in range(10)],
    '护照': [f.passport_number() for i in range(10)],
    '号码': [f.name_male() for i in range(10)]
})

print(df)