# STT Clipboard

语音转文字工具 — 录制语音，自动识别并复制到剪贴板。

## 功能

- 使用 **silero-vad** 自动去除静音片段，提升识别速度和准确性
- 支持 **简体中文**、**繁体中文**、**日语**、**英语** 及自动检测
- Gradio Web 界面，按钮大、适合 VR 操作

## 安装

1. venv

```bash
uv sync
. .venv/Scripts/activate
```

2. flash-attn (可选)

从`https://github.com/Dao-AILab/flash-attention/releases`下载预编译的 whl，然后安装

```
uv pip install /path/to/flash_attn...whl
```

3. stt 模型

```bash
hf download openai/whisper-large-v3-turbo
hf download Qwen/Qwen3-ASR-1.7B
hf download CohereLabs/cohere-transcribe-03-2026
```

## 使用

```bash
uv run app.py
```

启动后打开浏览器访问 `http://localhost:7860`。

~~推荐配合 NVIDIA Broadcast 去噪使用。~~ 由于 broadcast 输出有延迟，和 gradio 同时使用时会严重掉帧。
