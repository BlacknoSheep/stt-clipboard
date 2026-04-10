import numpy as np
from pydantic import BaseModel
from typing import Optional
import torch

from . import Qwen3ASRModel
from .inference.utils import SUPPORTED_LANGUAGES


_LANGUAGE_ALIASES = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "en": "English",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "yue": "Cantonese",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}

_CANONICAL_LANGUAGE_MAP = {
    language.lower(): language for language in SUPPORTED_LANGUAGES
}


class ModelConfig(BaseModel):
    model_name: str = "Qwen/Qwen3-ASR-1.7B"
    device: str = "auto"
    dtype: str = "bfloat16"
    attn_implementation: Optional[str] = None
    local_files_only: bool = True
    samplerate: int = 16000
    max_new_tokens: int = 256
    filter_words: list[str] = []


class Model:
    def __init__(
        self,
        config: ModelConfig,
    ):
        self.config = config
        self.model = Qwen3ASRModel.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
            device_map=config.device,
            dtype=getattr(torch, config.dtype),
            attn_implementation=config.attn_implementation,
            max_new_tokens=config.max_new_tokens,
        )

    def _map_language(self, language: Optional[str]) -> Optional[str]:
        if language is None:
            return None

        key = language.strip().lower()
        if not key:
            return None

        if key in _LANGUAGE_ALIASES:
            return _LANGUAGE_ALIASES[key]

        return _CANONICAL_LANGUAGE_MAP.get(key, language)

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> str:
        if self.config.samplerate != 16000:
            raise ValueError("qwen_asr: samplerate must be 16000")

        language = self._map_language(language)

        result = self.model.transcribe(
            (audio, self.config.samplerate), language=language
        )[0]
        text = result.text

        for word in self.config.filter_words:
            if word in text:
                return ""

        return text
