import requests

url = "http://localhost:5000/chatbot"

data = {
    "mensaje": "no se mm"
}

response = requests.post(url, json=data)

print(response.json())