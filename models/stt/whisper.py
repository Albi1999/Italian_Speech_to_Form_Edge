import time
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from faster_whisper import WhisperModel
from utils.data_utils import load_audio
from benchmarks.benchmarks_def import Benchmarks
import torch


class BaseWhisperModel:
    def _format_result(self, text, duration, model_name, ground_truth):
        base = {
            "model": model_name,
            "text": text,
            "reference": ground_truth,
            "time": duration,
        }
        metrics = Benchmarks.evaluate_stt([ground_truth], [text])
        return {**base, **metrics}


class WhisperTiny(BaseWhisperModel):
    def __init__(self):
        self.name = "Whisper Tiny (OpenAI)"
        # Base model for multilingual support
        self.processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-tiny",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model.eval()

    def transcribe(self, audio_path, ground_truth):
        audio, sr = load_audio(audio_path)
        inputs = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_features = inputs.input_features.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="it", task="transcribe")

        start = time.time()
        generated_ids = self.model.generate(
            input_features,
            attention_mask=attention_mask,
            forced_decoder_ids=forced_decoder_ids,
            max_length=128
        )
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].lower()
        duration = time.time() - start
        return self._format_result(transcription, duration, self.name, ground_truth)



class WhisperItaDistilled(BaseWhisperModel):
    def __init__(self):
        self.name = "Whisper Distilled IT (bofenghuang)"
        self.processor = AutoProcessor.from_pretrained("bofenghuang/whisper-large-v3-distil-it-v0.2")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "bofenghuang/whisper-large-v3-distil-it-v0.2",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model.eval()
        
    def transcribe(self, audio_path, ground_truth):
        audio, sr = load_audio(audio_path)
        inputs = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_features = inputs.input_features.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="it", task="transcribe")

        start = time.time()
        generated_ids = self.model.generate(
            input_features,
            attention_mask=attention_mask,
            forced_decoder_ids=forced_decoder_ids,
            max_length=128
        )
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].lower()
        duration = time.time() - start
        return self._format_result(transcription, duration, self.name, ground_truth)



class FasterWhisper(BaseWhisperModel):
    def __init__(self):
        self.name = "Faster Whisper Tiny (Turbo)"
        self.model = WhisperModel("tiny", compute_type="int8")

    def transcribe(self, audio_path, ground_truth):
        start = time.time()
        
        segments, _ = self.model.transcribe(
            audio_path, 
            language="it",
            vad_filter=False,
            beam_size=3,
            temperature=0
        )
        
        text = " ".join([seg.text for seg in segments]).lower()
        duration = time.time() - start
        return self._format_result(text, duration, self.name, ground_truth)
