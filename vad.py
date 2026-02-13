from silero_vad import load_silero_vad, get_speech_timestamps
import numpy as np
import torch
from pydantic import BaseModel


class SileroVADConfig(BaseModel):
    # For get_speech_timestamps
    threshold: float = 0.5


class SileroVAD:
    def __init__(self, config: SileroVADConfig = SileroVADConfig()):
        self.model = load_silero_vad(onnx=True)

    def get_speech_timestamps(
        self,
        audio: np.ndarray,
        config: SileroVADConfig = SileroVADConfig(),
    ) -> list[tuple[int, int]]:
        timestamps = get_speech_timestamps(
            torch.from_numpy(audio), self.model, threshold=config.threshold
        )
        t_list = []
        for t in timestamps:
            t_list.append((t["start"], t["end"]))
        return t_list
