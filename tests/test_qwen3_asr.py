import os
import dashscope

# Replace ABSOLUTE_PATH/welcome.mp3 with the absolute path of your local audio file.
audio_file_path = "./yue.mp3"

messages = [
    {
        "role": "system",
        "content": [
            # Configure the context for customized recognition here.
            {"text": ""},
        ]
    },
    {
        "role": "user",
        "content": [
            {"audio": audio_file_path},
        ]
    }
]
response = dashscope.MultiModalConversation.call(
    # If you have not configured the environment variable, replace the following line with your Model Studio API key: api_key = "sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen3-asr-flash",
    messages=messages,
    result_format="message",
    asr_options={
        # "language": "zh", # Optional. If you know the language of the audio, you can specify it to improve recognition accuracy.
        "enable_lid":True,
        "enable_itn":False
    }
)
print(response)