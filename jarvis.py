import gradio as gr
from influence.influence import Influence
from listen.listen import Listen

def listener(audio):
    try:
        if audio is None:
            return "No voice to be recorded."
        filename = Listen.save_voice(audio)
        return Influence.voice_to_text(filename) #"./tests/yue.mp3"
    except Exception as e:
        return f"Fail to record voice: {e}"

def influencer(text):
    return Influence.llm(text)

def brain():
    pass

def ui_launch():
    with gr.Blocks() as ui:
        with gr.Row():
            gr.Interface(
                fn=listener,
                inputs=gr.Audio(sources=["microphone"]),
                outputs=gr.Textbox(label="Me"),
                title="Jarvis👾",
                description=""
            )
        with gr.Row():
            gr.Textbox(
                label="Jarvis"
            )

        # with gr.Row():
        #     gr.Textbox(
        #         label="Jarvis"
        #     )
    ui.launch()


if __name__ == "__main__":
    ui_launch()
