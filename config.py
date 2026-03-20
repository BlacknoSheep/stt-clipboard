import logging
from stt import FasterWhisperConfig
from vad import SileroVADConfig
import json


class Config:
    log_level = logging.INFO

    language: str = "zh"  # "auto" 表示自动推断
    # whisper 和 silero-vad 原生是 16kHz
    samplerate = 16000
    block_size = 1024

    faster_whisper_config = FasterWhisperConfig(
        model_size_or_path="large-v3-turbo",
        device="cuda",
        compute_type="float16",  # "float32" for cpu
        local_files_only=True,
    )

    silero_vad_config = SileroVADConfig(threshold=0.2)

    host: str = "127.0.0.1"
    port: int = 7860


config = Config()

logger = logging.getLogger("stt_clipboard")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(config.log_level)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

LANGUAGES = ["auto", "zh", "ja", "en"]

def load_examples():
    with open("examples.json", "r", encoding="utf-8") as f:
        examples = list(json.load(f))
    return examples
