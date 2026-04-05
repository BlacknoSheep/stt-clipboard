from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio
from huggingface_hub import hf_hub_download

processor = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
model = CohereAsrForConditionalGeneration.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026", device_map="auto"
)

audio = load_audio("downloads/audio/test.wav", sampling_rate=16000)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language="en")
inputs.to(model.device, dtype=model.dtype)

outputs = model.generate(**inputs, max_new_tokens=256)
text = processor.decode(outputs, skip_special_tokens=True)
print(text)
