import pytest
import librosa


@pytest.fixture
def samplerate(self):
    return 16000


class TestTranscribe:
    @pytest.fixture
    def audio_data(self):
        # Load a sample audio file for testing
        audio_path = "downloads/audio/test.wav"
        audio, sr = librosa.load(audio_path, sr=16000)
        return audio, sr

    def test_whisper(self, audio_data):
        from src.Transcriber import Transcriber

        audio, sr = audio_data
        transcrber = Transcriber(model_name="whisper", device="cuda")
        text = transcrber.transcribe(audio, language="zh")
        print(f"Transcribed text: {text}")
