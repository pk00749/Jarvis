from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

class Influence:
    def __init__(self):
        pass

    @staticmethod
    def audio_to_text(file_path):
        model_dir = "iic/SenseVoiceSmall"

        model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            remote_code="./model.py",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
        )

        print(model.model_path)

        res = model.generate(
            input=file_path,
            cache={},
            language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  #
            merge_length_s=15,
        )
        text = rich_transcription_postprocess(res[0]["text"])
        print(text)
