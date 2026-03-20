from faster_whisper import WhisperModel
import numpy as np
from pydantic import BaseModel
from typing import Union, List


class FasterWhisperConfig(BaseModel):
    model_size_or_path: str = "large-v3-turbo"
    device: str = "cuda"
    device_index: Union[int, List[int]] = 0
    compute_type: str = "default"
    cpu_threads: int = 0
    num_workers: int = 1
    local_files_only: bool = True


class FasterWhisper:
    def __init__(
        self,
        config: FasterWhisperConfig,
    ):
        self.model = WhisperModel(
            model_size_or_path=config.model_size_or_path,
            device=config.device,
            device_index=config.device_index,
            compute_type=config.compute_type,
            cpu_threads=config.cpu_threads,
            num_workers=config.num_workers,
            local_files_only=config.local_files_only,
        )

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        segments, _ = self.model.transcribe(
            audio, language=None if language.lower() == "auto" else language
        )
        text = " ".join([segment.text for segment in segments]).strip()
        return text
