import gradio as gr
from influence.influence import Influence
from listen.listen import Listen
from speak.speak import Speak

def listener(audio):
    try:
        if audio is None:
            return "No voice to be recorded."
        filename = Listen.save_voice(audio)
        return Influence.voice_to_text(filename) #"./tests/yue.mp3"
    except Exception as e:
        return f"Fail to record voice: {e}"

def speaker(text):
    return Speak.text_to_voice(text)

def influencer(prompt):
    return Influence.llm(prompt)

def brain(audio):
    prompt_text = listener(audio)
    answer_text = influencer(prompt_text)
    return prompt_text, answer_text

def ui_launch():
    with gr.Blocks() as ui:
        with gr.Row():
            with gr.Column():
                output_me = gr.Textbox(label="You")
                output_jarvis_text = gr.Textbox(label="Jarvis")
                gr.Interface(
                    fn=brain,
                    inputs=gr.Audio(sources=["microphone"]),
                    outputs=[output_me, output_jarvis_text],
                    title="Jarvis👾",
                    description=""
                )
            with gr.Column():
                output_jarvis_text.change(fn=speaker,
                                          inputs=output_jarvis_text,
                                          outputs=gr.Audio(sources=["microphone"], autoplay=True))
    ui.launch()


if __name__ == "__main__":
    ui_launch()
