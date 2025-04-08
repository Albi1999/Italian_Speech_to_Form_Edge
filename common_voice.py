from weakref import ref
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

def common_voice():
    # Initialize DatasetLoader
    data_loader = DatasetLoader()

    test_samples = data_loader.load_dataset("common_voice", samples_per_dataset=100)
    models = [WhisperTiny(), 
            WhisperItaDistilled(), 
            FasterWhisper(),
            Vosk(),
            Wav2Vec2Grosman(),
            Wav2Vec2DBDMG(),
            Wav2Vec2Multilingual56()
            ]

    all_results = []

    for sample in tqdm(test_samples, "Processing samples"):
        audio_path = sample['path']
        reference = sample['sentence']
        for model in models:
            result = model.transcribe(audio_path, reference)
            all_results.append(result)

    df = pd.DataFrame(all_results)

    with open("models/stt/stt_models_metadata.json", "r") as f:
        metadata = json.load(f)

    # Convert to DataFrame
    df_meta = pd.DataFrame([
        {"model": model, **attrs}
        for model, attrs in metadata.items()
    ])

    # Create visualizer and generate all visualizations
    visualizer = STTVisualizer(output_base_dir="output/stt/common_voice")
    csv_dir, plot_dir = visualizer.visualize_all(df, df_meta)
    
    print(f"Visualization complete! Results available in:")
    print(f"- CSV data: {csv_dir}")
    print(f"- Plots: {plot_dir}")

    # Save results to CSV
    # Useful to actually check the transcriptions of the different models
    save_results(df, output_dir="output/stt/common_voice/csv")

if __name__ == "__main__":
    common_voice()