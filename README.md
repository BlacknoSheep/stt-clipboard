# STT Clipboard

语音转文字剪贴板工具 — 录制语音，自动识别并复制到剪贴板。

## 功能

- 使用 **faster-whisper** (`large-v3-turbo`) 进行高速语音识别
- 使用 **silero-vad** 自动去除静音片段，提升识别速度和准确性
- 支持 **中文（自动繁体→简体）、日语、英语** 及自动检测
- **Gradio** Web 界面，按钮大、适合 VR 操作
- 识别结果自动复制到系统剪贴板

## 安装

```bash
uv sync
```

## 使用

```bash
uv run main.py
```

启动后打开浏览器访问 `http://localhost:7860`。

推荐配合 NVIDIA Broadcast 使用。
