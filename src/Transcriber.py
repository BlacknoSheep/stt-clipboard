import numpy as np
from typing import Optional


class Transcriber:
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device

        self.model = self.switch_model(self.model_name)

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> str:
        if language == "auto":
            language = None
        text = self.model.transcribe(audio, language=language)
        return text

    def switch_model(self, model_name: str):
        self.model_name = model_name

        if self.model_name in ["faster-whisper", "faster-whisper-large-v3-turbo"]:
            from src.whisper.model import Model, ModelConfig

            config = ModelConfig(device=self.device)
            self.model = Model(config)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return self.model
