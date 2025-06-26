import gradio as gr
import soundfile as sf
import os
from datetime import datetime
from influence.influence import Influence


def save_audio(audio):
    if audio is None:
        return "No audio recorded"

    try:
        # Create recordings directory if it doesn't exist
        os.makedirs("recordings", exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recordings/recording_{timestamp}.wav"

        # Save the audio file
        sample_rate, audio_data = audio
        sf.write(filename, audio_data, sample_rate)

        # return f"Audio saved to {filename}"
        return Influence.audio_to_text("./tests/yue.mp3")
    except Exception as e:
        print(f"Error saving audio: {e}")
        return f"Error: {str(e)}"


def ui_launch():
    with gr.Blocks() as ui:
        with gr.Row():
            gr.Interface(
                fn=save_audio,
                inputs=gr.Audio(sources=["microphone"]),
                outputs=gr.Textbox(label="Output"),
                title="Jarvis👾",
                description=""
            )
    ui.launch()


if __name__ == "__main__":
    ui_launch()
