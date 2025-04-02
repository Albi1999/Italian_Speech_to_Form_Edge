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

def run_synthetic_dataset():
    DATASET_DIR = "data/synthetic_dataset"
    METADATA_PATH = os.path.join(DATASET_DIR, "metadata", "samples.json")

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
    for sample in tqdm(samples, desc="Processing synthetic samples"):
        audio_path = os.path.join(DATASET_DIR, sample["audio_path"])
        reference = sample["text"]
        for model in models:
            results.append(model.transcribe(audio_path, reference))

    df = pd.DataFrame(results)

    with open("models/stt/stt_models_metadata.json", "r") as f:
        metadata = json.load(f)
    df_meta = pd.DataFrame([{"model": model, **attrs} for model, attrs in metadata.items()])

    visualizer = STTVisualizer(output_base_dir="output/stt/synthetic_dataset")
    csv_dir, plot_dir = visualizer.visualize_all(df, df_meta)

    print(f"CSV saved in: {csv_dir}")
    print(f"Plots saved in: {plot_dir}")

    save_results(df, output_dir="output/stt/synthetic_dataset/csv")


if __name__ == "__main__":
    run_synthetic_dataset()
