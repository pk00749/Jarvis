import requests
import json

class TestTTSToAudio:
    def test_tts_to_audio(self):
        header = {
            "Content-Type": "application/json"
        }
        data = {
            "text": "呢几个字都表达唔到，我想讲嘅意思。恭喜发财，生意兴隆，身体健康，大吉大利",
            "speaker": "粤语女"
        }
        response = requests.post(url="http://localhost:9880/tts_to_audio", headers=header, json=data)
        print(json.loads(response.content))
        assert response.status_code == 200
