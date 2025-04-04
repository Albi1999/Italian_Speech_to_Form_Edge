from tqdm import tqdm
import pandas as pd
import json

from models import (WhisperTiny,
                    WhisperItaDistilled,
                    FasterWhisper,
                    Vosk,
                    Wav2Vec2Grosman,
                    Wav2Vec2Multilingual56,
                    Wav2Vec2DBDMG)

from utils import (DatasetLoader,
                   save_results,
                   STTVisualizer)


def run_coqui_dataset():
    # Initialize DatasetLoader
    data_loader = DatasetLoader()

    samples = data_loader.load_dataset("coqui", samples_per_dataset=50)

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
        audio_path = sample["path"]
        reference = sample["sentence"]
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
