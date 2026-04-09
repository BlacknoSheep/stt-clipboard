import logging
import json


LANGUAGES = ["auto", "zh", "ja", "en"]


class Config:
    log_level = logging.INFO

    # Supported models:
    # openai/whisper-large-v3-turbo - 性价比最高，支持 auto，混合语言。
    # CohereLabs/cohere-transcribe-03-2026 - 不支持 auto。
    model_name: str = "CohereLabs/cohere-transcribe-03-2026"

    language: str = "zh"  # "auto" 表示自动推断
    # whisper 和 silero-vad 原生是 16kHz
    samplerate: int = 16000
    block_size: int = 1024
    device: str = "auto"  # ["auto", "cuda", "cpu"]
    dtype: str = "float16"

    silero_vad_threshold: float = 0.2

    host: str = "127.0.0.1"
    port: int = 7860


config = Config()

logger = logging.getLogger("stt_clipboard")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(config.log_level)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def load_examples():
    with open("examples.json", "r", encoding="utf-8") as f:
        examples = list(json.load(f))
    return examples
