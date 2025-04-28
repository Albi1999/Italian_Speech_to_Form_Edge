import os
import json
import pandas as pd
from tqdm import tqdm

from models import Vosk, Wav2Vec2Grosman
from utils import DatasetLoader

OUTPUT_DIR = "output/stt"
NUM_SAMPLES = 100

class DataComparisonModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.output_dir = os.path.join(OUTPUT_DIR, f"{model_name}_only")
        os.makedirs(self.output_dir, exist_ok=True)
            
    def save_dataset_results(self, results, dataset_name):
        df = pd.DataFrame(results)
        dataset_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # Save raw results
        df.to_csv(os.path.join(dataset_dir, f"{dataset_name}_{self.model_name}_results.csv"), index=False)
        with open(os.path.join(dataset_dir, f"{dataset_name}_{self.model_name}_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save metrics only
        metric_cols = ["model", "wer", "cer", "bleu_avg", "rouge1_avg_f1", "levenshtein_avg", "time"]
        available_metric_cols = [col for col in metric_cols if col in df.columns]
        df_metrics = df[available_metric_cols]
        df_metrics.to_csv(os.path.join(dataset_dir, f"{dataset_name}_metrics_only.csv"), index=False)

        # Save summary
        summary = {
            "dataset": dataset_name,
            "model": df["model"].iloc[0],
            "wer": df["wer"].mean(),
            "cer": df["cer"].mean(),
            "bleu_avg": df["bleu_avg"].mean(),
            "rouge1_avg_f1": df["rouge1_avg_f1"].mean(),
            "levenshtein_avg": df["levenshtein_avg"].mean(),
            "time": df["time"].mean()
        }

        summary_path = os.path.join(self.output_dir, "summary_across_datasets.csv")
        if os.path.exists(summary_path):
            df_summary = pd.read_csv(summary_path)
            df_summary = df_summary[df_summary["dataset"] != dataset_name]  # Remove existing if present
            df_summary = pd.concat([df_summary, pd.DataFrame([summary])], ignore_index=True)
        else:
            df_summary = pd.DataFrame([summary])
        df_summary.to_csv(summary_path, index=False)

        print(f"{dataset_name} Completed and saved at {dataset_dir}")

    def run_common_voice(self, model, data_loader):
        samples = data_loader.load_dataset("common_voice", NUM_SAMPLES)
        results = [
            model.transcribe(s['path'], s['sentence'])
            for s in tqdm(samples, desc="Common Voice")
        ]
        self.save_dataset_results(results, "common_voice")

    def run_google_synthetic_dataset(self, model, data_loader):
        samples = data_loader.load_dataset("google_synthetic", NUM_SAMPLES, use_clean=True)
        results = [
            model.transcribe(s["path"], s["sentence"])
            for s in tqdm(samples, desc="Google Synthetic")
        ]
        self.save_dataset_results(results, "google_synthetic_dataset")

    def run_coqui_dataset(self, model, data_loader):
        samples = data_loader.load_dataset("coqui", NUM_SAMPLES)
        results = []
        for s in tqdm(samples, desc="Coqui"):
            result = model.transcribe(s["path"], s["sentence"])
            result["coqui_model"] = s.get("coqui_model", "unknown")
            result["speaker_gender"] = s.get("speaker_gender", "unknown")
            results.append(result)
        self.save_dataset_results(results, "coqui_dataset")

    def run_ITALIC(self, model, data_loader):
        samples = data_loader.load_dataset("ITALIC", NUM_SAMPLES)
        results = [
            model.transcribe(s["path"], s["sentence"])
            for s in tqdm(samples, desc="ITALIC")
        ]
        self.save_dataset_results(results, "ITALIC")

    def run_azure_synthetic_dataset(self, model, data_loader):
        samples = data_loader.load_dataset("azure_synthetic", NUM_SAMPLES, use_clean=True)
        results = [
            model.transcribe(s["path"], s["sentence"])
            for s in tqdm(samples, desc="Azure Synthetic")
        ]
        self.save_dataset_results(results, "azure_synthetic_dataset")

if __name__ == "__main__":
    #print("Loading Vosk Model...")
    #model = Vosk()
    print("Loading Wav2Vec2 Model...")
    model = Wav2Vec2Grosman()
    model_name = "wav2vec2"

    # Initialize DatasetLoader
    data_loader = DatasetLoader()

    # Initialize DataComparisonModel
    data_comparison_model = DataComparisonModel(model_name)

    # Run all datasets
    data_comparison_model.run_common_voice(model, data_loader)
    data_comparison_model.run_ITALIC(model, data_loader)
    data_comparison_model.run_azure_synthetic_dataset(model, data_loader)
    data_comparison_model.run_google_synthetic_dataset(model, data_loader)
    data_comparison_model.run_coqui_dataset(model, data_loader)
    print("All datasets processed successfully.")