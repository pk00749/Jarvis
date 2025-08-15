import os
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from mlx_lm import load, generate
from dashscope import Generation
from http import HTTPStatus

class Influence:
    def __init__(self):
        pass

    @staticmethod
    def voice_to_text(file_path):
        # MacBook Air M3 优化: 使用本地缓存模型，避免重复下载
        # 本地缓存路径配置
        local_sense_voice_path = "/Users/yorkhxli/.cache/modelscope/hub/models/iic/SenseVoiceSmall"
        local_vad_path = "/Users/yorkhxli/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"

        # 检查本地模型是否存在，优先使用本地路径
        if os.path.exists(local_sense_voice_path):
            model_dir = local_sense_voice_path
            print(f"Using local SenseVoice model: {model_dir}")
        else:
            model_dir = "iic/SenseVoiceSmall"
            print(f"Downloading SenseVoice model: {model_dir}")

        if os.path.exists(local_vad_path):
            vad_model_path = local_vad_path
            print(f"Using local VAD model: {vad_model_path}")
        else:
            vad_model_path = "fsmn-vad"
            print(f"Downloading VAD model: {vad_model_path}")

        model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            remote_code=f"{os.path.dirname(os.path.abspath(__file__))}/model.py",
            vad_model=vad_model_path,
            vad_kwargs={"max_single_segment_time": 30000},
            device="mps",  # MacBook Air M3 GPU加速: 使用MPS设备
            disable_update=True,
            disable_log=True  # MacBook Air M3 优化: 减少日志输出
        )

        res = model.generate(
            input=file_path,
            cache={},
            language="yue",  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  #
            merge_length_s=15,
        )
        text = rich_transcription_postprocess(res[0]["text"])
        print(f'User: {text}')
        return text

    @staticmethod
    def _create_cantonese_prompt(user_input):
        """
        创建专门的粤语提示词模板
        确保LLM生成纯正的粤语回复，不包含指令文本
        """
        cantonese_prompt = f"""你係一個粤语语音助手，專門用粤语同老人家傾偈。請你：
    
        1. 用純正嘅粤语回答所有問題
        2. 用詞要簡單易明，適合老人家理解
        3. 语氣要親切友善，好似同屋企人傾偈咁
        4. 回答要簡潔明了，唔好太冗長
        5. 用粤语嘅常用詞彙，避免書面語
        6. 【重要】絕對唔好重複或者包含任何指令文本，只返回純粹嘅粤语回答內容
        7. 【重要】唔好包含"用粤语"、"说这句话"、"<|endofprompt|>"等指令相關內容
    
        用户話：{user_input}
    
        請用粤语回答："""
        return cantonese_prompt

    def llm(self, user_text):
        # MacBook Air M3 优化: 本地LLM模型路径
        local_llm_path = "/Users/yorkhxli/.lmstudio/models/mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"

        # 加载模型和分词器 - GPU加速
        print(f"Loading LLM model from: {local_llm_path}")
        model, tokenizer = load(local_llm_path)

        # 将提示文本转换为模型所需的格式
        messages = [{"role": "user", "content": self._create_cantonese_prompt(user_text)}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # 生成文本 - MacBook Air M3 GPU加速
        text = generate(model, tokenizer, prompt=prompt, verbose=True)
        return text

    def llm_by_api(self, user_text):
        messages = [{"role": "user", "content": self._create_cantonese_prompt(user_text)}]
        responses = Generation.call(
            model="qwen-turbo",
            messages=messages,
            result_format="message",  # set result format as 'message'
            stream=True,  # enable stream output
            incremental_output=True,  # enable incremental output
        )
        for response in responses:
            if response.status_code == HTTPStatus.OK:
                return response.output.choices[0]["message"]["content"]
