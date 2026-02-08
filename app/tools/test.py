import os

import requests


if __name__ == "__main__":
    url = f"https://restapi.amap.com/v3/weather/weatherInfo?city=110101&key={os.getenv('AMAP_TOKEN')}"
    response = requests.get(url)
    print("Response from amap_weather tool:", response.json())