# STT Clipboard

语音转文字工具 — 录制语音，自动识别并复制到剪贴板。

## 功能

- 使用 **silero-vad** 自动去除静音片段，提升识别速度和准确性
- 支持语言取决于模型，目前做了 **简体中文**、**日语**、**英语** 3 个按钮
- Gradio Web 界面，按钮大、适合 VR 操作

## 安装

- **wsl2**

由于依赖问题，建议使用 wsl2

```ini
# 开启 Mirrored mode，使得 wsl 和主机 localhost 互通
# 主机中配置：C:\Users\你的用户名\.wslconfig
[wsl2]
networkingMode=mirrored
```

- **venv**

下载 `https://github.com/Dao-AILab/flash-attention/releases/download/fa4-v4.0.0.beta7/flash_attn_4-4.0.0b7-py3-none-any.whl` 到 `downloads/wheels/`

```bash
uv sync
. .venv/bin/activate
```

- **安装 ffmpeg**

```bash
sudo add-apt-repository ppa:ubuntuhandbook1/ffmpeg8
sudo apt update
sudo apt install ffmpeg
```

- **安装 [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)**

- **下载 stt 模型**

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

## <font  style="color:yellow;">Tips</font >

- 推荐配合 NVIDIA Broadcast 去噪使用。
- 由于 Gradio 的录音实现问题，如果在录音的同时浏览器有音频在播放，会严重掉帧。
- Qwen/Qwen3-ASR-1.7B 官方代码未适配 transformers 5.0+，在根据 https://github.com/QwenLM/Qwen3-ASR/pull/125 ，让 AI 修改后，虽然能跑了，但是中文识别效果很差，日语和英语正常。
