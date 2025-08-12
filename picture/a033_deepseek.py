from openai import OpenAI

client = OpenAI(api_key="sk-09926c61f77a4cc3afc9e81c129cfbab", base_url="https://api.deepseek.com/v1")

# Round 1
messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages
)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content



"""
Traceback (most recent call last):
  File "C:\Users\liuxin05\PycharmProjects\pythonProject\venv\foundations_for_analytics\picture\a033_deepseek.py", line 7, in <module>
    response = client.chat.completions.create(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\liuxin05\anaconda3\Lib\site-packages\openai\_utils\_utils.py", line 279, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\liuxin05\anaconda3\Lib\site-packages\openai\resources\chat\completions.py", line 850, in create
    return self._post(
           ^^^^^^^^^^^
  File "C:\Users\liuxin05\anaconda3\Lib\site-packages\openai\_base_client.py", line 1283, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\liuxin05\anaconda3\Lib\site-packages\openai\_base_client.py", line 960, in request
    return self._request(
           ^^^^^^^^^^^^^^
  File "C:\Users\liuxin05\anaconda3\Lib\site-packages\openai\_base_client.py", line 1064, in _request
    raise self._make_status_error_from_response(err.response) from None
openai.APIStatusError: Error code: 402 - {'error': {'message': 'Insufficient Balance', 'type': 'unknown_error', 'param': None, 'code': 'invalid_request_error'}}

进程已结束,退出代码1

402 - 余额不足	原因：账号余额不足
解决方法：请确认账户余额，并前往 充值 页面进行充值

"""