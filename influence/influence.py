from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from mlx_lm import load, generate

class Influence:
    def __init__(self):
        pass

    @staticmethod
    def voice_to_text(file_path):
        model_dir = "iic/SenseVoiceSmall"

        model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            remote_code="./model.py",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
            disable_update=True
        )

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
        print(f'Me: {text}')
        return text

    @staticmethod
    def llm(prompt):
        # 加载模型和分词器
        model, tokenizer = load(
            "/Users/yorkhxli/.lmstudio/models/mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx")

        # 将提示文本转换为模型所需的格式
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # 生成文本
        text = generate(model, tokenizer, prompt=prompt, verbose=True)
        return text
