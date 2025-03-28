from vosk import Model as VoskModel, KaldiRecognizer
from benchmarks.benchmarks_def import Benchmarks
import os
import numpy as np
import soundfile as sf
import json
import time
import librosa

class Vosk:
    def __init__(self, model_path="models/stt/vosk-model-small-it-0.4"):
        self.name = "Vosk STT (Small IT)"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk model not found at {model_path}. Download it from https://alphacephei.com/vosk/models")
        self.model = VoskModel(model_path)

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
        audio, sr = sf.read(audio_path)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio_bytes = (audio * 32767).astype(np.int16).tobytes()

        recognizer = KaldiRecognizer(self.model, 16000)

        start = time.time()
        recognizer.AcceptWaveform(audio_bytes)
        result = json.loads(recognizer.Result())
        duration = time.time() - start

        text = result.get("text", "").lower()
        return self._format_result(text, duration, ground_truth)
