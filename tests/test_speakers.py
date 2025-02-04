import requests
import json

class TestSpeakers:
    def test_speakers(self):
        response = requests.get(url="http://localhost:9880/speakers")
        print(json.loads(response.content))
        assert response.status_code == 200
