import os
import json
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

from models import Gemini
from utils import DatasetLoader

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")

GEMINI_MODEL_NAMES = ["gemini-2.0-flash",
                      "gemini-2.0-flash-lite",
                      "gemini-1.5-flash",
                      "gemini-1.5-pro"]
models = []

DATASETS = ["coqui",
            "synthetic",
            "ITALIC",
            "common_voice"]

SAMPLES_PER_DATASET = 100
OUTPUT_DIR = "output/stt/gemini_only"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_transcription(model, dataset, dataset_name):
    """Runs transcription for a given model on a dataset."""
    results = []
    for idx, sample in enumerate(tqdm(dataset, desc=f"Running {model.model_name}")):
        path = sample["path"]
        # Check if "sentence" exists, if not, use "text"
        reference = sample.get("sentence") or sample.get("text")  # Try to load sentence, if it is not load text
        result = model.transcribe(path, reference, dataset_name=dataset_name, sample_id=idx)
        results.append(result)
    return results

def save_dataset_results(results, dataset_name, model_name):
    """Saves the results to a CSV file and a JSON file."""
    df = pd.DataFrame(results)
    dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Save raw results
    df.to_csv(os.path.join(dataset_dir, f"{dataset_name}_{model_name}_results.csv"), index=False)
    with open(os.path.join(dataset_dir, f"{dataset_name}_{model_name}_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save metrics only
    metric_cols = ["model", "wer", "cer", "bleu_avg", "rouge1_avg_f1", "levenshtein_avg", "time"]
    available_metric_cols = [col for col in metric_cols if col in df.columns]
    df_metrics = df[available_metric_cols]
    df_metrics.to_csv(os.path.join(dataset_dir, f"{dataset_name}_{model_name}_metrics_only.csv"), index=False)

    # Save summary (one row per dataset)
    summary = {
        "dataset": dataset_name,
        "model": df["model"].iloc[0],
        "wer": df["wer"].mean(),
        "cer": df["cer"].mean(),
        "time": df["time"].mean()
    }

    # Conditionally add bleu_avg, rouge1_avg_f1, levenshtein_avg to the summary if they exist
    if "bleu_avg" in df.columns:
        summary["bleu_avg"] = df["bleu_avg"].mean()
    else:
        summary["bleu_avg"] = None  # Or some default value

    if "rouge1_avg_f1" in df.columns:
        summary["rouge1_avg_f1"] = df["rouge1_avg_f1"].mean()
    else:
        summary["rouge1_avg_f1"] = None  # Or some default value

    if "levenshtein_avg" in df.columns:
        summary["levenshtein_avg"] = df["levenshtein_avg"].mean()
    else:
        summary["levenshtein_avg"] = None  # Or some default value

    summary_path = os.path.join(OUTPUT_DIR, "summary_across_datasets.csv")
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        df_summary = df_summary[df_summary["dataset"] != dataset_name]  # Remove existing if present
        df_summary = pd.concat([df_summary, pd.DataFrame([summary])], ignore_index=True)
    else:
        df_summary = pd.DataFrame([summary])
    df_summary.to_csv(summary_path, index=False)

    print(f"{dataset_name} Completed and saved at {dataset_dir}")

if __name__ == "__main__":
    # Initialize the DatasetLoader
    data_loader = DatasetLoader()

    for dataset_name in DATASETS:
        print(f"\nProcessing dataset: {dataset_name}")
        dataset = data_loader.load_dataset(dataset_name, SAMPLES_PER_DATASET)  # Use the instance

        for model_name in GEMINI_MODEL_NAMES:
            print(f"\nEvaluating model: {model_name}")
            model = Gemini(model_name)
            models.append(model)
            results = run_transcription(model, dataset, dataset_name)
            save_dataset_results(results, dataset_name, model_name)
            print(f"\nTotal cost for {model_name} on {dataset_name}: ${model.cost_estimate:.4f}")
    print("All datasets processed successfully.")
    print(f"Total cost for all models: ${sum([model.cost_estimate for model in models]):.4f}")