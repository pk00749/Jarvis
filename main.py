import requests
import json

header = {
    "Content-Type": "application/json"
}

data = {
    "messages": [
        {
            "role": "user",
            "content": "Hello"
        }
    ]
}
response = requests.post(url='http://localhost:1234/v1/chat/completions', headers=header, json=data)
response_json = json.loads(response.content)
message_content = response_json['choices'][0]['message']['content']
print(message_content)
