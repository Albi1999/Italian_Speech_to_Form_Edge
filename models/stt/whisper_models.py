import time
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from faster_whisper import WhisperModel
from utils.data_utils import load_audio
from benchmarks.benchmarks_def import Benchmarks


class BaseWhisperModel:
    def _format_result(self, text, duration, model_name, ground_truth):
        base = {
            "model": model_name,
            "text": text,
            "time": duration,
        }
        metrics = Benchmarks.evaluate_stt([ground_truth], [text])
        return {**base, **metrics}


class WhisperTiny(BaseWhisperModel):
    def __init__(self):
        self.name = "Whisper Tiny (OpenAI)"
        self.processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny")

    def transcribe(self, audio_path, ground_truth):
        audio, sr = load_audio(audio_path)
        inputs = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding="max_length",
            max_length=3000,
            truncation=True
        )
        start = time.time()
        generated_ids = self.model.generate(**inputs, forced_decoder_ids=None)
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].lower()
        duration = time.time() - start
        return self._format_result(transcription, duration, self.name, ground_truth)


class WhisperItaDistilled(BaseWhisperModel):
    def __init__(self):
        self.name = "Whisper Distilled IT (bofenghuang)"
        self.processor = AutoProcessor.from_pretrained("bofenghuang/whisper-large-v3-distil-it-v0.2")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained("bofenghuang/whisper-large-v3-distil-it-v0.2")

    def transcribe(self, audio_path, ground_truth):
        audio, sr = load_audio(audio_path)
        inputs = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding="max_length",
            max_length=3000,
            truncation=True
        )
        start = time.time()
        generated_ids = self.model.generate(**inputs, forced_decoder_ids=None)
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].lower()
        duration = time.time() - start
        return self._format_result(transcription, duration, self.name, ground_truth)


class FasterWhisper(BaseWhisperModel):
    def __init__(self):
        self.name = "Faster Whisper Tiny (Turbo)"
        self.model = WhisperModel("tiny", compute_type="int8")

    def transcribe(self, audio_path, ground_truth):
        start = time.time()
        segments, _ = self.model.transcribe(audio_path, language="it")
        text = " ".join([seg.text for seg in segments]).lower()
        duration = time.time() - start
        return self._format_result(text, duration, self.name, ground_truth)
