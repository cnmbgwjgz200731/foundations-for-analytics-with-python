import pandas as pd
import requests
import time
import random

# 创建一个Session
s = requests.Session()

# 添加请求头信息
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'Referer': 'https://bj.lianjia.com/',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive'
}

# 设置代理
proxies = {
    'http': 'http://your_proxy_ip:your_proxy_port',
    'https': 'http://your_proxy_ip:your_proxy_port'
}

# 增加随机延迟
time.sleep(random.uniform(1, 3))

# 访问首页，获取Cookies
s.get('https://bj.lianjia.com/', headers=headers)

# 增加随机延迟
time.sleep(random.uniform(1, 3))


# 爬虫类
# class PaChong(object):
#     def __init__(self, x):
#         self.s = requests.session()
#         self.xq = self.s.get(f'https://bj.lianjia.com/xiaoqu/{x}/')
#         self.name = self.xq.text.split('detailTitle">')[1].split('</h1>')[0]
#         self.price = self.xq.text.split('xiaoquUnitPrice">')[1].split('</span>')[0]

# 爬虫类
class PaChong(object):
    def __init__(self, x):
        self.s = requests.session()
        self.headers = headers
        # self.proxies = proxies
        self.url = f'https://bj.lianjia.com/xiaoqu/{x}/'
        self.name = self.get_name()
        self.price = self.get_price()
        self.desc = self.get_desc()

    def get_name(self):
        try:
            # xq = self.s.get(self.url, headers=self.headers, proxies=self.proxies)
            xq = self.s.get(self.url, headers=self.headers)
            if xq.status_code == 200:
                name = xq.text.split('detailTitle">')[1].split('</h1>')[0]
                return name
            else:
                print(f"Failed to fetch data for ID {self.url}. Status code: {xq.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching data for ID {self.url}: {e}")
            return None

    def get_price(self):
        try:
            # xq = self.s.get(self.url, headers=self.headers, proxies=self.proxies)
            xq = self.s.get(self.url, headers=self.headers)
            if xq.status_code == 200:
                price = xq.text.split('xiaoquUnitPrice">')[1].split('</span>')[0]
                return price
            else:
                print(f"Failed to fetch data for ID {self.url}. Status code: {xq.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching data for ID {self.url}: {e}")
            return None

    def get_desc(self):
        try:
            # xq = self.s.get(self.url, headers=self.headers, proxies=self.proxies)
            xq = self.s.get(self.url, headers=self.headers)
            if xq.status_code == 200:
                price = xq.text.split('xiaoquUnitPriceDesc">')[1].split('</span>')[0]
                return price
            else:
                print(f"Failed to fetch data for ID {self.url}. Status code: {xq.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching data for ID {self.url}: {e}")
            return None


# 小区列表
xqs = [1111027377595, 1111027382589,
       1111027378611, 1111027374569,
       1111027378069, 1111027374228,
       116964627385853]

# 构造数据
df = pd.DataFrame(xqs, columns=['小区'])
df['小区'] = df.小区.astype(str)

df_pc = (
    df
    .assign(小区名=df.小区.apply(lambda x: PaChong(x).name))
    .assign(房价=df.小区.apply(lambda x: PaChong(x).price))
    .assign(描述=df.小区.apply(lambda x: PaChong(x).desc))
)

print(df_pc)
df_pc.to_excel('E:/bat/output_files/pandas_out_20240814031_005.xlsx', index=False)

# TODO----------------------------------------------------------------

import pandas as pd
import requests
import time
import random

"""
详细解释：
    添加请求头：通过设置与浏览器类似的请求头，减少被WAF识别为爬虫的风险。
    使用代理：通过代理服务器发送请求，可以隐藏真实的IP地址，分散请求源，减少被拦截的可能性。
    增加延迟：在请求之间增加随机延迟，模拟人类的浏览行为，减少被WAF识别为爬虫的风险。


详细解释
    检查响应状态码：在每次请求后检查响应的状态码，以确保请求成功 (status_code == 200)。
    添加错误处理：在 try...except 块中处理异常，以确保即使发生错误，程序也不会崩溃。
    增加日志记录：在发生错误或请求失败时打印详细信息，以便调试。    
"""

# 创建一个Session
s = requests.Session()

# 添加请求头信息
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'Referer': 'https://bj.lianjia.com/',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive'
}

# 设置代理
proxies = {
    'http': 'http://your_proxy_ip:your_proxy_port',
    'https': 'http://your_proxy_ip:your_proxy_port'
}

# 增加随机延迟
time.sleep(random.uniform(1, 3))

# 访问首页，获取Cookies
s.get('https://bj.lianjia.com/', headers=headers)

# 增加随机延迟
time.sleep(random.uniform(1, 3))


# 获取小区名称的函数
def pa_name(x):
    try:
        xq = s.get(f'https://bj.lianjia.com/xiaoqu/{x}/', headers=headers)
        if xq.status_code == 200:
            name = xq.text.split('detailTitle">')[1].split('</h1>')[0]
            return name
        else:
            print(f"Failed to fetch data for ID {x}. Status code: {xq.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching data for ID {x}: {e}")
        return None


# 获取平均房价的函数
def pa_price(x):
    try:
        xq = s.get(f'https://bj.lianjia.com/xiaoqu/{x}/', headers=headers)
        if xq.status_code == 200:
            price = xq.text.split('xiaoquUnitPrice">')[1].split('</span>')[0]
            return price
        else:
            print(f"Failed to fetch data for ID {x}. Status code: {xq.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching data for ID {x}: {e}")
        return None


# 获取描述列的函数
def pa_desc(x):
    try:
        xq = s.get(f'https://bj.lianjia.com/xiaoqu/{x}/', headers=headers)
        if xq.status_code == 200:
            desc = xq.text.split('xiaoquUnitPriceDesc">')[1].split('</span>')[0]
            return desc
        else:
            print(f"Failed to fetch data for ID {x}. Status code: {xq.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching data for ID {x}: {e}")
        return None


# 小区列表
xqs = [1111027377595, 1111027382589,
       1111027378611, 1111027374569,
       1111027378069, 1111027374228,
       116964627385853]

# 构造数据
df = pd.DataFrame(xqs, columns=['小区'])

# 爬取小区名
df['小区名'] = df.小区.apply(lambda x: pa_name(x))
# 爬取房价
df['房价'] = df.小区.apply(lambda x: pa_price(x))
# 爬取描述
df['类型'] = df.小区.apply(lambda x: pa_desc(x))

print()
"""可以先用Python的类改造函数，再用链式方法调用："""

df.to_excel('E:/bat/output_files/pandas_out_20240814031_001.xlsx', index=False)

print(df)

