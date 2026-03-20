import gradio as gr
import pyperclip
import numpy as np
import opencc
import librosa

from config import config, logger, LANGUAGES, load_examples
from stt import FasterWhisper
from vad import SileroVAD

stt: FasterWhisper
vad_detector: SileroVAD


# 延迟加载
def _init():
    global stt, vad_detector, SCRIPT, CSS

    stt = FasterWhisper(config=config.faster_whisper_config)
    vad_detector = SileroVAD(config=config.silero_vad_config)
    logger.info("加载 stt 模型成功")


def copy_to_clipboard(text: str):
    pyperclip.copy(text)
    gr.Info(f"已复制到剪贴板: {text}")


def process_audio(
    recording: tuple[int, np.ndarray], language: str, vad_threshold: float
) -> str:
    # return "test"
    sr, audio = recording
    audio = audio.astype(np.float32) / 32768.0
    audio = librosa.resample(audio, orig_sr=sr, target_sr=config.samplerate)

    config.silero_vad_config.threshold = vad_threshold
    timestamps = vad_detector.get_speech_timestamps(audio, config.silero_vad_config)
    if len(timestamps) == 0:
        logger.info("没有检测到语音")
        return ""
    audio = audio[timestamps[0][0] : timestamps[-1][1]]
    text = stt.transcribe(audio, language=language)
    if language == "zh":
        text = opencc.OpenCC("t2s").convert(text)
    copy_to_clipboard(text)
    return text

def update_examples_fn():
    examples = load_examples()
    examples = [[x] for x in examples]
    return gr.Dataset(samples=examples)

def create_app() -> gr.Blocks:
    with gr.Blocks(title="STT Clipboard") as app:
        with gr.Row():
            # 输入区
            with gr.Column(elem_id="left-column"):
                stt_text = gr.Textbox(
                    placeholder="(空)", show_label=False, elem_classes="text-box"
                )

                copy_button = gr.Button(value="复制", variant="primary")
                copy_button.click(fn=copy_to_clipboard, inputs=[stt_text])

                input_audio = gr.Audio(
                    sources="microphone",
                    type="numpy",
                    show_label=False,
                    elem_classes="input-audio",
                    interactive=True,
                )

                input_language = gr.Radio(
                    choices=LANGUAGES,
                    value=config.language,
                    show_label=False,
                    elem_classes="input-language",
                )

                input_threshold = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=config.silero_vad_config.threshold,
                    step=0.01,
                    label="VAD 阈值",
                )

                input_audio.start_recording(fn=lambda: None, outputs=[stt_text])

                input_audio.stop_recording(
                    fn=process_audio,
                    inputs=[input_audio, input_language, input_threshold],
                    outputs=[stt_text],
                )

                update_examples = gr.Button(value="更新预制文本", variant="primary")

            with gr.Column(elem_id="right-column"):
                edit_text = gr.Textbox(
                    placeholder="(空)", show_label=False, elem_classes="text-box"
                )

                with gr.Row():
                    copy_button = gr.Button(value="复制", variant="primary")
                    paste_button = gr.Button(value="粘贴", variant="primary")
                    clear_button = gr.Button(value="清空", variant="primary")

                    copy_button.click(
                        fn=lambda text: pyperclip.copy(text), inputs=[edit_text]
                    )

                    paste_button.click(
                        fn=lambda x: x + pyperclip.paste(),
                        inputs=[edit_text],
                        outputs=[edit_text],
                    )

                    clear_button.click(fn=lambda: "", outputs=[edit_text])

                # 预制文本
                examples = gr.Examples(
                    examples=load_examples(),
                    inputs=edit_text,
                    label="预制文本",
                    fn=lambda x: copy_to_clipboard(x),
                    run_on_click=True,
                    elem_id="examples",
                    examples_per_page=100,
                )

                update_examples.click(
                    fn=update_examples_fn,
                    outputs=examples.dataset,
                )

    return app


def main() -> None:
    _init()

    with open("script.js", "r", encoding="utf-8") as f:
        SCRIPT = f.read()
    with open("style.css", "r", encoding="utf-8") as f:
        CSS = f.read()

    app = create_app()
    app.launch(
        server_name=config.host,
        server_port=config.port,
        css=CSS,
        js=SCRIPT,
    )


if __name__ == "__main__":
    main()
