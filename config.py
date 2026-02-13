import logging
from stt import FasterWhisperConfig
from vad import SileroVADConfig


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
logger.setLevel(logging.DEBUG)  # 设置你自己的代码显示 DEBUG 级别
console_handler = logging.StreamHandler()
console_handler.setLevel(config.log_level)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

LANGUAGES = ["auto", "zh", "ja", "en"]

EXAMPLES = [
    "你好",
    "我是中国人",
    "拜拜喵",
    "呼呼喵",
    "关注永雏塔菲喵",
    "晚安喵",
    "こんにちは",
    "おやすみなさい",
    "Hello",
    "Bye",
]
