from datasets import load_from_disk
import random
import torchaudio
import torch
import os
from models import (
    WhisperTiny,
    WhisperItaDistilled,
    FasterWhisper,
    Vosk,
    Wav2Vec2Grosman,
    Wav2Vec2Multilingual56,
    Wav2Vec2DBDMG
)
import json
import pandas as pd
from tqdm import tqdm
from utils import save_results, STTVisualizer

def ITALIC():
    # Random seed for reproducibility
    random.seed(42)

    DATASET_NAME = "ITALIC"
    DATASET_CONFIG = "massive"
    DATASET_DIR = f"data/datasets/{DATASET_NAME}_{DATASET_CONFIG}"
    AUDIO_DIR = f"{DATASET_DIR}/audio_samples"
    SAMPLE_EXPORT_PATH = f"{DATASET_DIR}/test_samples.json"
    NUM_SAMPLES = 50

    dataset = load_from_disk(DATASET_DIR)
    sampled_dataset = dataset.shuffle(seed=42).select(range(NUM_SAMPLES))

    os.makedirs(AUDIO_DIR, exist_ok=True)

    audio_paths = []
    references = []
    exported_samples = []

    # Save audio files and prepare references
    for i, sample in enumerate(sampled_dataset):
        audio_array = sample["audio"]["array"]
        sample_rate = sample["audio"]["sampling_rate"]
        audio_path = os.path.join(AUDIO_DIR, f"audio_{i}.wav")

        # Save or convert the audio file
        waveform = torch.tensor(audio_array).unsqueeze(0)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        torchaudio.save(audio_path, waveform, 16000)

        audio_paths.append(audio_path)
        references.append(sample["utt"])
        exported_samples.append({
            "id": i,
            "text": sample["utt"],
            "audio_path": audio_path
        })

    # Save the exported samples to a JSON file for reference
    with open(SAMPLE_EXPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(exported_samples, f, ensure_ascii=False, indent=2)

    models = [WhisperTiny(), 
            WhisperItaDistilled(), 
            FasterWhisper(),
            Vosk(),
            Wav2Vec2Grosman(),
            Wav2Vec2DBDMG(),
            Wav2Vec2Multilingual56()
            ]

    results = []
    for i, audio_path in enumerate(tqdm(audio_paths, desc="Processing samples")):
        # Transcribe the audio file using the model
        for model in models:
            # Transcribe the audio file using the model
            result = model.transcribe(audio_path, references[i])
            results.append(result)

    df = pd.DataFrame(results)

    with open("models/stt/stt_models_metadata.json", "r") as f:
        metadata = json.load(f)

    # Convert to DataFrame
    df_meta = pd.DataFrame([
        {"model": model, **attrs}
        for model, attrs in metadata.items()
    ])

    # Create visualizer and generate all visualizations
    visualizer = STTVisualizer(output_base_dir="output/stt/ITALIC")
    csv_dir, plot_dir = visualizer.visualize_all(df, df_meta)
    
    print(f"Visualization complete! Results available in:")
    print(f"- CSV data: {csv_dir}")
    print(f"- Plots: {plot_dir}")

    # Save results to CSV
    # Useful to actually check the transcriptions of the different models
    save_results(df, output_dir="output/stt/ITALIC/csv")

if __name__ == "__main__":
    ITALIC()