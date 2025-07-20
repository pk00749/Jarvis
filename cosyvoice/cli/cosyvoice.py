# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
import torch
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoice2Model
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.class_utils import get_model_type
from typing import Generator


class CosyVoice2:
    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False):
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        with open(f'{model_dir}/cosyvoice.yaml', 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        assert get_model_type(configs) == CosyVoice2Model, f'Do not use {model_dir} for CosyVoice2 initialization!'
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'], configs['feat_extractor'],
                                          f'{model_dir}/campplus.onnx',
                                          f'{model_dir}/speech_tokenizer_v2.onnx',
                                          f'{model_dir}/spk2info.pt',
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoice2Model(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load(f'{model_dir}/llm.pt',
                        f'{model_dir}/flow.pt',
                        f'{model_dir}/hift.pt')
        if load_jit:
            self.model.load_jit('{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))

        del configs

    def inference_instruct2(self, tts_text, instruct_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoice2Model), 'inference_instruct2 is only implemented for CosyVoice2!'

        # 流式处理优化：预处理和缓存
        normalized_texts = list(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend))

        # 预热模型以减少冷启动时间
        if stream and len(normalized_texts) > 0:
            self._warmup_model_if_needed()

        for i, text_chunk in enumerate(normalized_texts):
            model_input = self.frontend.frontend_instruct2(text_chunk, instruct_text, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(text_chunk))
            print(f'synthesis text {text_chunk}')

            # 流式处理优化：并行处理和缓冲
            for j, model_output in enumerate(self.model.tts(**model_input, stream=stream, speed=speed)):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate

                # 优化音频块处理
                if stream:
                    # 实现智能音频块大小调整
                    optimized_output = self._optimize_audio_chunk(model_output, i, j)
                    if optimized_output is not None:
                        yield optimized_output
                else:
                    yield model_output

                # 性能监控（仅在调试时启用）
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    rtf = (time.time() - start_time) / speech_len if speech_len > 0 else 0
                    logging.debug('yield speech len {}, rtf {}'.format(speech_len, rtf))

                start_time = time.time()

    def _warmup_model_if_needed(self):
        """预热模型以减少冷启动时间"""
        if not hasattr(self, '_model_warmed') or not self._model_warmed:
            print("🔥 预热模型以减少冷启动时间...")

            try:
                # 避免无限递归：直接使用模型的tts方法而不是inference_instruct2
                # 创建最小的模型输入进行预热
                warmup_text = "你好"
                warmup_instruct = "女性"

                # 使用现有的提示语音进行预热
                from cosyvoice.utils.file_utils import load_wav
                import os

                ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                try:
                    warmup_prompt = load_wav(f'{ROOT_DIR}/asset/zero_shot_prompt.wav', 16000)
                except:
                    # 如果无法加载提示语音，创建一个简单的零向量
                    warmup_prompt = torch.zeros(1, 16000)

                # 直接调用frontend和model，避免递归
                model_input = self.frontend.frontend_instruct2(
                    warmup_text, warmup_instruct, warmup_prompt, self.sample_rate
                )

                # 执行一次快速推理来预热模型（只取第一个输出）
                warmup_output = next(self.model.tts(**model_input, stream=True, speed=1.0), None)

                if warmup_output is not None:
                    self._model_warmed = True
                    print("✅ 模型预热完成")
                else:
                    print("⚠️ 预热输出为空，跳过预热")
                    self._model_warmed = False

            except Exception as e:
                print(f"⚠️ 模型预热失败，将跳过预热: {e}")
                self._model_warmed = False

    def _optimize_audio_chunk(self, model_output, chunk_index, audio_index):
        """优化音频块处理"""
        if 'tts_speech' not in model_output:
            return None

        audio_chunk = model_output['tts_speech']

        # 优化音频块大小 - 针对Apple Silicon M3优化
        min_chunk_size = 1024  # 最小音频块大小
        max_chunk_size = 8192  # 最大音频块大小，避免内存峰值

        # 如果音频块太小，缓冲后再输出
        if audio_chunk.shape[1] < min_chunk_size and audio_index == 0:
            # 对于第一个音频块，如果太小就缓冲
            if not hasattr(self, '_audio_buffer'):
                self._audio_buffer = []
            self._audio_buffer.append(audio_chunk)
            return None

        # 如果有缓冲的音频，合并后输出
        if hasattr(self, '_audio_buffer') and len(self._audio_buffer) > 0:
            buffered_audio = torch.cat(self._audio_buffer + [audio_chunk], dim=1)
            self._audio_buffer = []

            # 如果合并后的音频太大，分割输出
            if buffered_audio.shape[1] > max_chunk_size:
                # 输出前半部分，保留后半部分
                output_audio = buffered_audio[:, :max_chunk_size]
                self._audio_buffer = [buffered_audio[:, max_chunk_size:]]
                return {'tts_speech': output_audio}
            else:
                return {'tts_speech': buffered_audio}

        # 如果音频块太大，分割处理
        if audio_chunk.shape[1] > max_chunk_size:
            # 输出前半部分
            output_audio = audio_chunk[:, :max_chunk_size]
            # 保留后半部分到缓冲区
            if not hasattr(self, '_audio_buffer'):
                self._audio_buffer = []
            self._audio_buffer.append(audio_chunk[:, max_chunk_size:])
            return {'tts_speech': output_audio}

        return model_output

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()