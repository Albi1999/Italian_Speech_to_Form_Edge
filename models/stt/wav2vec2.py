import time
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from utils.data_utils import load_audio
from benchmarks.benchmarks_def import Benchmarks


class Wav2Vec2Grosman:
    def __init__(self):
        self.name = "Wav2Vec2 XLSR Italian (Grosman)"
        self.model_id = "jonatasgrosman/wav2vec2-large-xlsr-53-italian"
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
        self.model.eval()

    def transcribe(self, audio_path, ground_truth):
        audio, sr = load_audio(audio_path, sr=16000)
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)

        with torch.no_grad():
            start = time.time()
            logits = self.model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0].lower()
            duration = time.time() - start

        return {
            "model": self.name,
            "text": transcription,
            "reference": ground_truth,
            "time": duration,
            **Benchmarks.evaluate_stt([ground_truth], [transcription])
        }

class Wav2Vec2DBDMG:
    def __init__(self):
        self.name = "Wav2Vec2 XLS-R 1B Italian Robust (DBDMG)"
        self.model_id = "dbdmg/wav2vec2-xls-r-1b-italian-robust"
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
        self.model.eval()

    def transcribe(self, audio_path, ground_truth):
        audio, sr = load_audio(audio_path, sr=16000)
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)

        with torch.no_grad():
            start = time.time()
            logits = self.model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0].lower()
            duration = time.time() - start

        return {
            "model": self.name,
            "text": transcription,
            "reference": ground_truth,
            "time": duration,
            **Benchmarks.evaluate_stt([ground_truth], [transcription])
        }

class Wav2Vec2Multilingual56:
    def __init__(self):
        self.name = "Wav2Vec2 XLSR Multilingual (voidful, 56 lang)"
        self.model_id = "voidful/wav2vec2-xlsr-multilingual-56"
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
        self.model.eval()

    def transcribe(self, audio_path, ground_truth):
        audio, sr = load_audio(audio_path, sr=16000)
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)

        with torch.no_grad():
            start = time.time()
            logits = self.model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0].lower()
            duration = time.time() - start

        return {
            "model": self.name,
            "text": transcription,
            "reference": ground_truth,
            "time": duration,
            **Benchmarks.evaluate_stt([ground_truth], [transcription])
        }