import requests
import certifi

response = requests.get("https://www.google.com")
print("status code:", response.status_code)
print("First 200 characters:", response.text[:200])