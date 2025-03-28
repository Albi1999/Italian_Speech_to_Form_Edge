import time
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from utils.data_utils import load_audio
from benchmarks.benchmarks_def import Benchmarks


class VoxPopuli:
    def __init__(self):
        self.name = "VoxPopuli Wav2Vec2 Large (IT)"
        self.model_id = "facebook/voxpopuli-wav2vec2-large-10k-it"
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
        self.model.eval()

    def _format_result(self, text, duration, ground_truth):
        base = {
            "model": self.name,
            "text": text,
            "reference": ground_truth,
            "time": duration,
        }
        metrics = Benchmarks.evaluate_stt([ground_truth], [text])
        return {**base, **metrics}

    def transcribe(self, audio_path, ground_truth):
        audio, sr = load_audio(audio_path, sr=16000)
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=16000)

        start = time.time()
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0].lower()
        duration = time.time() - start

        return self._format_result(transcription, duration, ground_truth)
