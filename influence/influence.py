import os.path

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from mlx_lm import load, generate

class Influence:
    def __init__(self):
        pass

    @staticmethod
    def voice_to_text(file_path):
        print(f"开始处理音频文件: {file_path}")

        model_dir = "iic/SenseVoiceSmall"

        model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            remote_code=f"{os.path.dirname(os.path.abspath(__file__))}/model.py",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cpu",  # 在Mac M3上使用CPU
            disable_update=True
        )

        # 优化粤语识别：尝试不同的参数配置
        recognition_configs = [
            # 配置1：粤语识别，较短合并时间
            {
                "language": "yue",
                "name": "粤语",
                "merge_length_s": 5,
                "batch_size_s": 20
            },
            # 配置2：自动识别
            {
                "language": "auto",
                "name": "自动识别",
                "merge_length_s": 15,
                "batch_size_s": 60
            },
            # 配置3：中文识别
            {
                "language": "zh",
                "name": "中文",
                "merge_length_s": 15,
                "batch_size_s": 60
            }
        ]

        for config in recognition_configs:
            try:
                print(f"尝试使用{config['name']}识别...")

                res = model.generate(
                    input=file_path,
                    cache={},
                    language=config["language"],
                    use_itn=True,
                    batch_size_s=config["batch_size_s"],
                    merge_vad=True,
                    merge_length_s=config["merge_length_s"],
                    disable_punc=False,  # 保留标点符号
                    disable_update=True,
                    ban_emo_unk=True,  # 禁用情感识别和未知标记
                )

                print(f"原始识别结果: {res}")

                # 确保res是正确的格式
                if not res or len(res) == 0:
                    print(f'{config["name"]}识别无结果，尝试下一个...')
                    continue

                # 提取文本
                text = res[0].get("text", "") if isinstance(res[0], dict) else str(res[0])
                print(f"提取的文本: {text}")

                # 使用后处理
                text = rich_transcription_postprocess(text)
                print(f"后处理文本: {text}")

                # 主动清理emo标记和情感符号
                text = Influence._clean_emo_markers(text)
                print(f"清理后文本: {text}")

                # 检查识别结果是否合理
                if text and len(text.strip()) > 0:
                    # 检查是否是无意义的结果
                    invalid_results = ["we.", ".", "，", " ", "we", "the", "a", "an"]
                    if text.strip().lower() not in invalid_results:
                        print(f'Me ({config["name"]}): {text}')
                        return text.strip()
                    else:
                        print(f'{config["name"]}识别结果无效: "{text}"，尝试下一个...')
                else:
                    print(f'{config["name"]}识别结果为空，尝试下一个...')

            except Exception as e:
                print(f'{config["name"]}识别失败: {e}')
                continue

        # 如果所有配置都失败，返回默认消息
        print('���有识别配置都失败')
        return "语音识别失败"

    @staticmethod
    def llm(prompt):
        # 加载模型和分词器 - 使用MLX框架，专为Apple Silicon优化
        model, tokenizer = load(
            "/Users/yorkhxli/.lmstudio/models/mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx")

        # 创建强制粤语回答的系统提示词
        system_prompt = """你是一个智能语音助手，名叫 Jarvis。无论用户用什么语言提问，你都必须用粤语回答。

重要规则：
1. 必须用粤语回答，不能用普通话或其他语言
2. 回答要自然流畅，符合粤语的表达习惯
3. 可以使用粤语特有的词汇和语气助词
4. 保持友好和乐于助人的态度
5. 如果不确定某个词的粤语表达，可以用相近的粤语词汇代替

例如：
- 用��问："你好吗？" → 你答："你好啊！我几好啊，多谢你问！"
- 用户问："今天天气怎么样？" → 你答："今日天气几好啊，好舒服！"
- 用户问："你会做什么？" → 你答："我可以帮你解答问题，陪你倾偈，有咩需要帮手就话我知啦！"

记住：无论如何都要用粤语回答！"""

        # 将用户输入和系统提示组合
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        # 应用聊天模板
        formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # 生成文本 - MLX的generate函数参数修正
        try:
            text = generate(
                model,
                tokenizer,
                prompt=formatted_prompt,
                verbose=False,  # 减少输出噪音
                max_tokens=512
            )

            # 确保返回字符串
            if isinstance(text, list):
                text = text[0] if text else "抱歉，我现在无法处理您的请求。"

            print(f'Jarvis (粤语): {text}')
            return str(text)

        except Exception as e:
            print(f'LLM生成错误: {e}')
            # 返回粤语默认回答
            fallback_response = "唔好意思，我而家处理唔到你嘅问题，你可以试一次吗？"
            print(f'Jarvis (粤语-默认): {fallback_response}')
            return fallback_response

    @staticmethod
    def _clean_emo_markers(text):
        """清理语音识别结果中的情感标记和emo符号"""
        import re

        if not text:
            return text

        # 清理SenseVoiceSmall模型��能输出的情感标记
        # 常见的情感标记格式：<|HAPPY|>, <|SAD|>, <|ANGRY|>, <|NEUTRAL|> 等
        text = re.sub(r'<\|[A-Z_]+\|>', '', text)

        # 清理其他可能的情感标记格式
        text = re.sub(r'\\[EMO\\]', '', text)
        text = re.sub(r'\\[EMOTION\\]', '', text)
        text = re.sub(r'<EMO>', '', text)
        text = re.sub(r'<EMOTION>', '', text)

        # 清理emoji表情符号
        # 清理常见的emoji范围
        text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)  # 表情符号
        text = re.sub(r'[\U0001F300-\U0001F5FF]', '', text)  # 符号和象形文字
        text = re.sub(r'[\U0001F680-\U0001F6FF]', '', text)  # 交通和地图符号
        text = re.sub(r'[\U0001F1E0-\U0001F1FF]', '', text)  # 国旗
        text = re.sub(r'[\U00002600-\U000027BF]', '', text)  # 各种符号

        # 清理可能的音效标记
        text = re.sub(r'<\|[^>]+\|>', '', text)  # 任何 <|xxx|> 格式的标记
        text = re.sub(r'\\[[^]]+\\]', '', text)    # 任何 [xxx] 格式的标记

        # 清理多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text
