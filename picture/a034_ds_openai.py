import requests

# Define API Key and Base URL
API_KEY = "sk-09926c61f77a4cc3afc9e81c129cfbab"  # 替换为你的真实 API Key
BASE_URL = "https://api.deepseek.com/v1"  # 确保这是 DeepSeek 提供的正确 URL

# Headers including authentication
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Define the endpoint and payload
endpoint = "/chat/completions"  # 根据 DeepSeek 文档调整路径
url = f"{BASE_URL}{endpoint}"

# Define the request payload
payload = {
    "model": "deepseek-chat",  # 确保这是 DeepSeek 支持的模型名称
    "messages": [
        {"role": "user", "content": "9.11 and 9.8, which is greater?"}
    ]
}

# Send the POST request to the API
try:
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raise an error for HTTP codes 4xx/5xx

    # Parse and print the response
    result = response.json()
    print("Response content:", result)

    # Access specific fields in the response
    if "choices" in result:
        reasoning_content = result["choices"][0]["message"].get("reasoning_content", "No reasoning content provided.")
        content = result["choices"][0]["message"].get("content", "No content provided.")
        print("Reasoning Content:", reasoning_content)
        print("Content:", content)
    else:
        print("Unexpected response format:", result)

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")