import gradio as gr
import matplotlib.pyplot as plt

def audio_to_spectrogram(audio):
    sr, data = audio
    plt.figure(figsize=(10, 5))
    plt.specgram(data, Fs=sr)
    plt.title("Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    return "hello"

demo = gr.Interface(
    fn=audio_to_spectrogram,
    inputs=[gr.Audio(sources=['microphone'], label="Input")],
    outputs=[gr.Text(label="text")],
    title="Audio to Spectrogram",
    description="Convert an input audio file to a spectrogram."
)
demo.launch()