import os
import json
import librosa
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from benchmarks.benchmarks_def import Benchmarks
from utils.encode_audio import AudioEncoder

class Gemini:
    def __init__(self, model_name, audio_tokens_per_second=32, cache_dir="models/stt/cache"):
        self.model_name = model_name
        self.audio_tokens_per_second = audio_tokens_per_second
        self.cost_estimate = 0.0
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.pricing = {
            "gemini-2.0-flash": {"input": 0.70, "output": 0.40},
            "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
            "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
            "gemini-1.5-pro": {"input": 1.25, "output": 5.00}
        }

        self.model = ChatOpenAI(
            model=self.model_name,
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_BASE_URL")
        )

    def get_audio_duration(self, path):
        audio, sample_rate = librosa.load(path)
        return len(audio) / sample_rate

    def count_tokens(self, text):
        return len(text.split())

    def get_cache_path(self, dataset_name, sample_id):
        return os.path.join(self.cache_dir, f"{self.model_name}_{dataset_name}_{sample_id}.json")

    def transcribe(self, audio_path, ground_truth=None, dataset_name=None, sample_id=None):
        cache_path = self.get_cache_path(dataset_name, sample_id) if dataset_name and sample_id else None

        # Try to load cached result
        start = time.time()
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            print(f"\nLoaded cached result for {self.model_name} [{dataset_name} sample {sample_id}]")
            return cached

        audio_base64 = AudioEncoder(audio_path).encode()
        input_file = {
            "type": "input_audio",
            "input_audio": {"data": audio_base64, "format": "mp3"},
        }
        input_text = {
            "type": "text",
            "text": "Trascrivi in italiano l'audio allegato parola per parola."
        }

        try:
            response = self.model.invoke([
                HumanMessage(content=[input_file, input_text])
            ])
            transcription = response.content
        except Exception as e:
            print(f"Error with {self.model_name}: {e}")
            return {
                "model": self.model_name,
                "text": "",
                "reference": ground_truth,
                "error": str(e),
                "wer": 1.0,
                "cer": 1.0,
                "cost": 0.0
            }

        duration = time.time() - start
        audio_tokens = duration * self.audio_tokens_per_second
        input_tokens = self.count_tokens(input_text["text"])
        output_tokens = self.count_tokens(transcription)

        total_input_tokens = audio_tokens + input_tokens
        total_output_tokens = audio_tokens + output_tokens

        input_cost = (total_input_tokens / 1_000_000) * self.pricing[self.model_name]["input"]
        output_cost = (total_output_tokens / 1_000_000) * self.pricing[self.model_name]["output"]
        total_cost = input_cost + output_cost
        self.cost_estimate += total_cost

        base = {
            "model": self.model_name,
            "text": transcription,
            "reference": ground_truth,
            "time": duration,
            "cost": total_cost
        }
        metrics = Benchmarks.evaluate_stt([ground_truth], [transcription])
        result = {**base, **metrics}

        # Save to cache
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        return result