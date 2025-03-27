from models import (
    WhisperTiny,
    WhisperItaDistilled,
    FasterWhisper
)
from utils import (
    load_commonvoice_subset,
    save_results,
    plot_metrics
)
from tqdm import tqdm
import pandas as pd

def main():
    test_samples = load_commonvoice_subset(max_samples=5)
    models = [WhisperTiny(), WhisperItaDistilled(), FasterWhisper()]

    all_results = []

    for audio_path, ground_truth in tqdm(test_samples, desc="Processing samples"):
        for model in models:
            result = model.transcribe(audio_path, ground_truth)
            all_results.append(result)

    df = pd.DataFrame(all_results)
    # Save results
    save_results(df)
    # Plot metrics
    plot_files = plot_metrics(df)
    print(f"Plots saved: {', '.join(plot_files)}")

if __name__ == "__main__":
    main()