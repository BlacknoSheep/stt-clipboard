from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
from pydantic import BaseModel
from typing import Optional


class ModelConfig(BaseModel):
    model_size_or_path: str = "openai/whisper-large-v3-turbo"
    device: str = "auto"
    dtype: str = "float16"
    local_files_only: bool = True
    samplerate: int = 16000
    filter_words: list[str] = []


class Model:
    def __init__(
        self,
        config: ModelConfig,
    ):
        self.config = config
        self.processor: WhisperProcessor = WhisperProcessor.from_pretrained(
            config.model_size_or_path, local_files_only=config.local_files_only
        )
        self.model: WhisperForConditionalGeneration = (
            WhisperForConditionalGeneration.from_pretrained(
                config.model_size_or_path,
                local_files_only=config.local_files_only,
                device_map=config.device,
                torch_dtype=config.dtype,
                attn_implementation="flash_attention_4",
            )
        )

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> str:
        inputs = self.processor(
            audio, sampling_rate=self.config.samplerate, return_tensors="pt"
        )
        inputs = {
            k: v.to(self.model.dtype).to(self.model.device) for k, v in inputs.items()
        }

        generated_ids = self.model.generate(**inputs, language=language)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        for word in self.config.filter_words:
            if word in text:
                return ""

        return text
