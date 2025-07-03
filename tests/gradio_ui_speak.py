import gradio as gr
demo = gr.Interface(
    fn=lambda x: x,
    inputs=None,
    outputs=gr.Audio(value="record_output.wav", autoplay=True, show_download_button=False, show_share_button=False)
)
demo.launch()
