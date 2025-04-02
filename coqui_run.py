import os
import json
import pandas as pd
from tqdm import tqdm
from models import (
    WhisperTiny,
    WhisperItaDistilled,
    FasterWhisper,
    Vosk,
    Wav2Vec2Grosman,
    Wav2Vec2Multilingual56,
    Wav2Vec2DBDMG
)
from utils import save_results, STTVisualizer
import random

def run_coqui_dataset():
    BASE_DIR = "data/coqui_output"
    METADATA_PATH = os.path.join(BASE_DIR, "all_samples_coqui.json")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    random.seed(42)
    samples = random.sample(samples, 50) # Randomly sample 20 samples for testing

    models = [
        WhisperTiny(), 
        WhisperItaDistilled(), 
        FasterWhisper(),
        Vosk(),
        Wav2Vec2Grosman(),
        Wav2Vec2DBDMG(),
        Wav2Vec2Multilingual56()
    ]

    results = []
    for sample in tqdm(samples, desc="Processing Coqui samples"):
        audio_path = os.path.join(BASE_DIR, sample["coqui_audio_path"])
        reference = sample["text"]
        for model in models:
            result = model.transcribe(audio_path, reference)
            result["coqui_model"] = sample["coqui_model"]
            result["speaker_gender"] = sample["speaker_gender"]
            results.append(result)

    df = pd.DataFrame(results)

    with open("models/stt/stt_models_metadata.json", "r") as f:
        metadata = json.load(f)
    df_meta = pd.DataFrame([{"model": model, **attrs} for model, attrs in metadata.items()])

    visualizer = STTVisualizer(output_base_dir="output/stt/coqui_dataset")
    csv_dir, plot_dir = visualizer.visualize_all(df, df_meta)

    print(f"CSV saved in: {csv_dir}")
    print(f"Plots saved in: {plot_dir}")

    save_results(df, output_dir="output/stt/coqui_dataset/csv")

if __name__ == "__main__":
    run_coqui_dataset()
