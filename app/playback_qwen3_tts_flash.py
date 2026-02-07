import os
import time
import threading
import dashscope
import pyaudio
from dashscope.audio.qwen_tts_realtime import *
from qwen_agent.agents import Assistant
from app.tools.mcp_wrapper import AmapMapsTool, EdgeOneTool
from app.tools.custom_tools import WeatherTool, TimeTool
from dashscope import Generation
from B64PCMPlayer import B64PCMPlayer


class AudioContext:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True
        )

    def cleanup(self):
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p') and self.p:
            self.p.terminate()


class TtsCallback(QwenTtsRealtimeCallback):
    def __init__(self, player_ref, tts_ref):
        super().__init__()
        self.finish_event = threading.Event()
        self.response_done_event = threading.Event()
        self.player_ref = player_ref
        self.tts_ref = tts_ref

    def on_open(self) -> None:
        print('Connection opened, initial player...')
        pya = pyaudio.PyAudio()
        self.player_ref['pya'] = pya
        self.player_ref['b64_player'] = B64PCMPlayer(pya, save_file=True)

    def on_close(self, close_status_code, close_msg) -> None:
        print(f'Connection closed with code: {close_status_code}, msg: {close_msg}, destroy player.')
        b64_player = self.player_ref.get('b64_player')
        if b64_player:
            b64_player.wait_for_complete()
            b64_player.shutdown()
        pya = self.player_ref.get('pya')
        if pya:
            print("Terminating pyaudio")
            pya.terminate()
            self.player_ref['pya'] = None
        print('Player destroyed.')

    def on_event(self, response: str) -> None:
        try:
            response_type = response['type']
            if 'session.created' == response_type:
                print('Start Session: {}'.format(response['session']['id']))
            if 'response.audio.delta' == response_type:
                recv_audio_b64 = response['delta']
                b64_player = self.player_ref.get('b64_player')
                if b64_player:
                    b64_player.add_data(recv_audio_b64)
            if 'response.done' == response_type:
                print(f'Response done.')
                self.response_done_event.set()
            if 'session.finished' == response_type:
                print('Session finished.')
                self.finish_event.set()
        except Exception as ex:
            print(f'[Error] {ex}')
            self.finish_event.set()
            return

    def wait_for_complete(self):
        self.finish_event.wait()

    def wait_for_response_done(self):
        self.response_done_event.wait()

    def wait_for_playback_complete(self):
        b64_player = self.player_ref.get('b64_player')
        if b64_player:
            b64_player.wait_for_complete()


class SynthesizeSpeechFromLlm:
    def __init__(self, voice='Kiki'):
        self.voice = voice
        self.audio_context = AudioContext()
        self.tts_refs = {
            'pya': None,
            'b64_player': None,
            'qwen_tts_realtime': None
        }

    def llm(self, query_text: str):
        print("===================== Generating response from LLM =================")
        system_text = '你是一个闲聊型语音AI助手，来自广东广州，日常说粤语，主要任务是和用户展开日常性的友善聊天，用粤语回复。请不要回复使用任何格式化文本，回复要求口语化，不要使用markdown格式或者列表。'
        messages = [{
            'role': 'system',
            'content': system_text
        }, {
            'role': 'user',
            'content': query_text
        }]

        response = Generation.call(
            model='qwen-plus',
            messages=messages,
            result_format='message',
            stream=False,
            incremental_output=True,
        )
        print('[LLM]] Answer: ', end='')
        print(response.output.choices[0]['message']['content'])
        return response.output.choices[0]['message']['content']

    def agent_llm(self, query_text: str):
        llm_cfg = {
            "model": "qwen-plus",
            "model_server": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": os.getenv("DASHSCOPE_API_KEY"),
        }

        system = "你是一个闲聊型语音AI助手，来自广东广州，日常说粤语，主要任务是和用户展开日常性的友善聊天，用粤语回复。请不要回复使用任何格式化文本，回复要求口语化，不要使用markdown格式或者列表。"

        function_list = [
            {
                "name": "amap_weather",
                "token": os.getenv("AMAP_TOKEN"),
            },
            AmapMapsTool(),
            EdgeOneTool(),
            WeatherTool(),
            TimeTool(),
        ]

        bot = Assistant(
            llm=llm_cfg,
            name="Jarvis",
            description="一个闲聊型粤语语音AI助手",
            system_message=system,
            function_list=function_list,
        )

        messages = [{"role": "user", "content": query_text}]
        bot_response = ""
        is_tool_call = False
        tool_call_info = {}
        for response_chunk in bot.run(messages):
            new_response = response_chunk[-1]
            if "function_call" in new_response:
                is_tool_call = True
                tool_call_info = new_response["function_call"]
                print(f"正在调用工具: {tool_call_info.get('name', 'unknown')}")
            elif "function_call" not in new_response and is_tool_call:
                is_tool_call = False
                print(f"工具调用结果: {new_response.get('content', str(new_response))}")
            elif new_response.get("role") == "assistant" and "content" in new_response:
                incremental_content = new_response["content"][len(bot_response):]
                bot_response += incremental_content
        messages.extend(response_chunk)
        print('\n[Agent] Answer: ', end='')
        agent_response = messages[-1]['content']
        return agent_response

    def run(self, query_text: str):
        print("===================== Synthesizing speech from LLM output =================")
        answer = self.agent_llm(query_text)
        text_to_synthesize = str.split(answer, ",")

        callback = TtsCallback(self.tts_refs, self.tts_refs)
        qwen_tts_realtime = QwenTtsRealtime(
            model='qwen3-tts-flash-realtime',
            callback=callback,
        )
        self.tts_refs['qwen_tts_realtime'] = qwen_tts_realtime

        qwen_tts_realtime.connect()
        qwen_tts_realtime.update_session(
            voice=self.voice,
            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            mode='server_commit'
        )
        for text_chunk in text_to_synthesize:
            print(f'Speaking: {text_chunk}')
            qwen_tts_realtime.append_text(text_chunk)
            time.sleep(0.1)
        qwen_tts_realtime.finish()
        callback.wait_for_response_done()
        callback.wait_for_playback_complete()
        qwen_tts_realtime.close()
        print('Session ID: {}, first audio delay: {}'.format(
            qwen_tts_realtime.get_session_id(),
            qwen_tts_realtime.get_first_audio_delay(),
        ))

    def cleanup(self):
        self.audio_context.cleanup()
