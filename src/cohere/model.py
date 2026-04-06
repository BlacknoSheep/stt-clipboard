from transformers import AutoProcessor, CohereAsrForConditionalGeneration
import numpy as np
from pydantic import BaseModel
from typing import Optional
import torch


class ModelConfig(BaseModel):
    model_name: str = "CohereLabs/cohere-transcribe-03-2026"
    device: str = "auto"
    attn_implementation: Optional[str] = None # not support flash_attention_4 yet
    local_files_only: bool = True
    samplerate: int = 16000


class Model:
    def __init__(
        self,
        config: ModelConfig,
    ):
        self.config = config
        self.processor = AutoProcessor.from_pretrained(
            config.model_name, local_files_only=config.local_files_only
        )
        self.model: CohereAsrForConditionalGeneration = (
            CohereAsrForConditionalGeneration.from_pretrained(
                config.model_name,
                local_files_only=config.local_files_only,
                device_map=config.device,
                attn_implementation=config.attn_implementation,
            )
        )

    def transcribe(self, audio: np.ndarray, language: Optional[str] = "zh") -> str:
        if language is None:
            raise ValueError("cohere: language must be specified")
        inputs = self.processor(
            audio,
            sampling_rate=self.config.samplerate,
            return_tensors="pt",
            language=language,
        ).to(self.model.device, dtype=self.model.dtype)

        outputs = self.model.generate(**inputs, max_new_tokens=256)  # type: ignore
        text = self.processor.decode(outputs, skip_special_tokens=True)[0].strip()

        return text
