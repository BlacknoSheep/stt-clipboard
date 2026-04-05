from silero_vad import load_silero_vad, get_speech_timestamps
import numpy as np
import torch


class SileroVAD:
    def __init__(self):
        self.model = load_silero_vad(onnx=True)

    def get_speech_timestamps(
        self,
        audio: np.ndarray,
        threshold: float = 0.2,
    ) -> list[tuple[int, int]]:
        timestamps = get_speech_timestamps(
            torch.from_numpy(audio), self.model, threshold=threshold
        )
        t_list = []
        for t in timestamps:
            t_list.append((t["start"], t["end"]))
        return t_list
