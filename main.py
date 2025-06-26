from listen.listen import Listen
from influence.influence import Influence
from speak.speak import Speak
from frontend.home import ui_launch


class Jarvis:
    def __init__(self):
        pass

    @staticmethod
    def listener():
        Listen.record()

    @staticmethod
    def influencer():
        Influence.audio_to_text("./tests/yue.mp3")

    @staticmethod
    def speaker():
        pass

    @staticmethod
    def face():
        ui_launch(fn=Influence.audio_to_text("./tests/yue.mp3"))


if __name__ == "__main__":
    j = Jarvis()
    # j.listener()
    # j.influencer()
    j.face()
