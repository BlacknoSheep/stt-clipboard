import numpy as np
from typing import Optional


class Transcriber:
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device

        self.model = None
        self.model = self.switch_model(self.model_name)

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> str:
        if self.model is None:
            raise ValueError("Model is not loaded")
        if language == "auto":
            language = None
        text = self.model.transcribe(audio, language=language)
        return text

    def switch_model(self, model_name: str):
        self.model_name = model_name

        if self.model:
            del self.model
            self.model = None

        if self.model_name in ["whisper", "openai/whisper-large-v3-turbo"]:
            self.model_name = "openai/whisper-large-v3-turbo"
            self.model_name = "openai/whisper-large-v3-turbo"
            from src.whisper.model import Model, ModelConfig

            config = ModelConfig(model_name=self.model_name, device=self.device)
            self.model = Model(config)
        elif self.model_name in ["cohere", "CohereLabs/cohere-transcribe-03-2026"]:
            from src.cohere.model import Model, ModelConfig

            config = ModelConfig(model_name=self.model_name, device=self.device)
            self.model = Model(config)
        elif self.model_name in ["qwen", "qwen_asr", "Qwen/Qwen3-ASR-1.7B"]:
            if self.model_name in ["qwen", "qwen_asr"]:
                self.model_name = "Qwen/Qwen3-ASR-1.7B"

            from src.qwen_asr.model import Model, ModelConfig

            config = ModelConfig(model_name=self.model_name, device=self.device)
            self.model = Model(config)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return self.model
