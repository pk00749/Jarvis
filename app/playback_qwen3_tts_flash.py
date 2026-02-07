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

# 以下为北京地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

p = pyaudio.PyAudio()
# 创建音频流
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=24000,
                output=True)

voice = 'Kiki'

pya = None
b64_player: B64PCMPlayer = None
qwen_tts_realtime: QwenTtsRealtime = None


class MyCallback(QwenTtsRealtimeCallback):
    def __init__(self):
        super().__init__()
        self.finish_event = threading.Event()
        self.response_done_event = threading.Event()

    def on_open(self) -> None:
        global pya
        global b64_player
        print('Connection opened, initial player...')
        pya = pyaudio.PyAudio()
        b64_player = B64PCMPlayer(pya, save_file=True)

    def on_close(self, close_status_code, close_msg) -> None:
        print(f'Connection closed with code: {close_status_code}, msg: {close_msg}, destroy player.')
        global pya
        global b64_player
        b64_player.wait_for_complete()
        b64_player.shutdown()
        if pya:
            print("Terminating pyaudio")
            pya.terminate()
            pya = None
        print('Player destroyed.')

    def on_event(self, response: str) -> None:
        try:
            global qwen_tts_realtime
            global b64_player
            response_type = response['type']
            if 'session.created' == response_type:
                print('Start Session: {}'.format(response['session']['id']))
            if 'response.audio.delta' == response_type:
                recv_audio_b64 = response['delta']
                b64_player.add_data(recv_audio_b64)
            if 'response.done' == response_type:
                print(f'Response done.') # {qwen_tts_realtime.get_last_response_id()}
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
        global b64_player
        if b64_player:
            b64_player.wait_for_complete()


class SynthesizeSpeechFromLlm:
    @staticmethod
    def llm(query_text: str):
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
            result_format='message',  # set result format as 'message'
            stream=False,  # enable stream output
            incremental_output=True,  # enable incremental output
        )
        print('[LLM]] Answer: ', end='')
        print(response.output.choices[0]['message']['content'])
        return response.output.choices[0]['message']['content']

    @staticmethod
    def agent_llm(query_text: str):
        # LLM 配置
        llm_cfg = {
            "model": "qwen-plus",
            "model_server": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx"
            "api_key": os.getenv("DASHSCOPE_API_KEY"),
        }

        # 系统消息
        system = "你是一个闲聊型语音AI助手，来自广东广州，日常说粤语，主要任务是和用户展开日常性的友善聊天，用粤语回复。请不要回复使用任何格式化文本，回复要求口语化，不要使用markdown格式或者列表。"

        # 工具列表 - 支持MCP服务器、内置工具和自定义工具
        function_list = [
            {
                "name": "amap_weather",
                # 需要在环境变量中配置 AMAP_TOKEN
                "token": os.getenv("AMAP_TOKEN"),
            },  # qwen-agent内置工具
            AmapMapsTool(),      # MCP工具：高德地图位置搜索
            EdgeOneTool(),       # MCP工具：EdgeOne位置搜索
            WeatherTool(),       # 自定义工具：天气查询
            TimeTool(),          # 自定义工具：时间查询
        ]

        # 创建助手实例
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
        # response_chunk 是消息列表，追加到历史消息中用于多轮对话
        messages.extend(response_chunk)
        print('\n[Agent] Answer: ', end='')
        agent_response = messages[-1]['content']
        return agent_response

    def run(self, query_text: str):
        print("===================== Synthesizing speech from LLM output =================")
        # answer = self.llm(query_text)
        answer = self.agent_llm(query_text)
        text_to_synthesize = str.split(answer, ",")

        callback = MyCallback()
        qwen_tts_realtime = QwenTtsRealtime(
            model='qwen3-tts-flash-realtime',
            callback=callback,
        )

        qwen_tts_realtime.connect()
        qwen_tts_realtime.update_session(
            voice=voice,
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


if __name__ == '__main__':
    synthesizer = SynthesizeSpeechFromLlm()
    llm = synthesizer.run