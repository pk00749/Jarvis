import requests
import json

from listen.listen import Listen
from influence.influence import Influence
from speak.speak import Speak


class Jarvis:
    def __init__(self):
        pass

    @staticmethod
    def listener():
        Listen.record()

    def influencer(self):
        pass

    def speaker(self):
        pass


if __name__ == "__main__":
    j = Jarvis()
    j.listener()
    # completion_text = chat_completion(text="你好，你是谁")
    # text_to_audio(completion_text)
