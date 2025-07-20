import os
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from mlx_lm import load, generate
import re

class Influence:
    def __init__(self):
        pass

    @staticmethod
    def voice_to_text(file_path):
        model_dir = "iic/SenseVoiceSmall"

        model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            remote_code=f"{os.path.dirname(os.path.abspath(__file__))}/model.py",
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
    def _create_cantonese_prompt(user_input):
        """
        创建专门的粤语提示词模板
        确保LLM生成纯正的粤语回复
        """
        cantonese_prompt = f"""你係一個粤语语音助手，專門用粤语同老人家傾偈。請你：

1. 用純正嘅粤语回答所有問題
2. 用詞要簡單易明，適合老人家理解
3. 语氣要親切友善，好似同屋企人傾偈咁
4. 回答要簡潔明了，唔好太冗長
5. 用粤语嘅常用詞彙，避免書面語

用户話：{user_input}

請用粤语回答："""
        return cantonese_prompt

    @staticmethod
    def _is_cantonese(text):
        """
        簡單檢測文本是否為粤语
        通過常見粤语詞彙和語法特征判斷
        """
        # 粤语常見詞彙和語法特征
        cantonese_indicators = [
            # 粤语疑問詞
            '咩', '點', '邊', '乜', '幾時', '點解', '咁', '嘅', '係', '唔係',
            # 粤语語氣詞
            '呀', '啦', '嘛', '咋', '喎', '㗎', '囉', '呢', '嚟', '嘞',
            # 粤语動詞
            '搞', '嚟', '去', '返', '講', '話', '睇', '食', '飲', '著',
            # 粤语形容詞
            '好', '唔好', '靚', '醜', '新', '舊', '大', '細', '多', '少',
            # 粤语介詞和連詞
            '同', '俾', '畀', '幫', '为', '因为', '所以', '不过', '但係',
            # 粤语特殊用法
            '有冇', '冇', '咪', '即係', '就係', '真係', '實在', '當然'
        ]

        # 計算粤语特征詞出現次數
        cantonese_count = sum(1 for word in cantonese_indicators if word in text)

        # 如果文本中包含3個或以上粤语特征詞，判定為粤语
        return cantonese_count >= 3

    @staticmethod
    def _validate_cantonese_response(response):
        """
        驗證LLM回復是否為粤语
        如果不是粤语，返回提示信息
        """
        if not Influence._is_cantonese(response):
            return False, "回復唔係粤语，需要重新生成"
        return True, "回復係粤语，驗證通過"

    @staticmethod
    def llm(prompt):
        """
        優化後的LLM方法，確保生成粤语回復
        """
        # 加載模型和分詞器
        model, tokenizer = load(
            "/Users/yorkhxli/.lmstudio/models/mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx")

        # 使用粤语提示詞模板
        cantonese_prompt = Influence._create_cantonese_prompt(prompt)

        # 將提示文本轉換為模型所需的格式
        messages = [{"role": "user", "content": cantonese_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # 生成文本
        response = generate(model, tokenizer, prompt=formatted_prompt, verbose=True)

        # 驗證回復是否為粤语
        is_cantonese, validation_msg = Influence._validate_cantonese_response(response)
        print(f"粤语驗證結果: {validation_msg}")

        # 如果不是粤语，嘗試重新生成一次
        if not is_cantonese:
            print("重新生成粤语回復...")
            # 使用更強的粤语提示詞
            stronger_prompt = f"""你必須用純正粤语回答，絕對唔可以用普通話或者英文！

用户問題：{prompt}

請一定要用粤语回答，例如：
- 用"係"而唔係"是"
- 用"咩"而唔係"什麼"
- 用"點解"而唔係"為什麼"
- 用"乜"而唔係"什麼"
- 用"嘅"而唔係"的"

回答："""

            messages = [{"role": "user", "content": stronger_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            response = generate(model, tokenizer, prompt=formatted_prompt, verbose=True)

        print(f'Jarvis (粤语): {response}')
        return response
