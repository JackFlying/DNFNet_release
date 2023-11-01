import requests

url = "http://0.0.0.0:5001/wtext"  # 替换为你要发送GET请求的目标URL

response = requests.get(url)

if response.status_code == 200:
    print(response.text)
else:
    print("请求失败，状态码:", response.status_code)