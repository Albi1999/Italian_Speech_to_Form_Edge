import os
import json
import pandas as pd
from tqdm import tqdm
import random
from datasets import load_from_disk
import torchaudio
import torch
import sys

from utils import load_commonvoice_subset
from models import Vosk


OUTPUT_DIR = "output/stt/vosk_only"
os.makedirs(OUTPUT_DIR, exist_ok=True)
NUM_SAMPLES = 100

def save_dataset_results(results, dataset_name):
    df = pd.DataFrame(results)
    dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Save raw results
    df.to_csv(os.path.join(dataset_dir, f"{dataset_name}_vosk_results.csv"), index=False)
    with open(os.path.join(dataset_dir, f"{dataset_name}_vosk_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save metrics only
    metric_cols = ["model", "wer", "cer", "bleu_avg", "rouge1_avg_f1", "levenshtein_avg", "time"]
    available_metric_cols = [col for col in metric_cols if col in df.columns]
    df_metrics = df[available_metric_cols]
    df_metrics.to_csv(os.path.join(dataset_dir, f"{dataset_name}_metrics_only.csv"), index=False)

    # Save summary (one row per dataset)
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

    summary_path = os.path.join(OUTPUT_DIR, "summary_across_datasets.csv")
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        df_summary = df_summary[df_summary["dataset"] != dataset_name]  # Remove existing if present
        df_summary = pd.concat([df_summary, pd.DataFrame([summary])], ignore_index=True)
    else:
        df_summary = pd.DataFrame([summary])
    df_summary.to_csv(summary_path, index=False)

    print(f"{dataset_name} Completed and saved at {dataset_dir}")





def run_common_voice(vosk_model):
    samples = load_commonvoice_subset(max_samples=NUM_SAMPLES)
    results = [vosk_model.transcribe(audio, ref) for audio, ref in tqdm(samples, desc="Common Voice")]
    save_dataset_results(results, "common_voice")



def run_synthetic_dataset(vosk_model):
    with open("data/synthetic_dataset/metadata/samples.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    random.seed(42)
    samples = random.sample(samples, NUM_SAMPLES)

    results = [
        vosk_model.transcribe(os.path.join("data/synthetic_dataset", s["audio_path"]), s["text"])
        for s in tqdm(samples, desc="Synthetic")
    ]
    save_dataset_results(results, "synthetic_dataset")



def run_coqui_dataset(vosk_model):
    with open("data/coqui_output/all_samples_coqui.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    random.seed(42)
    samples = random.sample(samples, NUM_SAMPLES)

    results = []
    for s in tqdm(samples, desc="Coqui"):
        result = vosk_model.transcribe(os.path.join("data/coqui_output", s["coqui_audio_path"]), s["text"])
        result["coqui_model"] = s["coqui_model"]
        result["speaker_gender"] = s["speaker_gender"]
        results.append(result)
    save_dataset_results(results, "coqui_dataset")



def run_ITALIC(vosk_model):

    DATASET_DIR = "data/datasets/ITALIC_massive"
    AUDIO_DIR = f"{DATASET_DIR}/audio_samples"

    dataset = load_from_disk(DATASET_DIR)
    sampled_dataset = dataset.shuffle(seed=42).select(range(min(NUM_SAMPLES, len(dataset))))
    os.makedirs(AUDIO_DIR, exist_ok=True)

    exported_samples = []

    for i, sample in enumerate(tqdm(sampled_dataset, desc="ITALIC")):
        audio_array = sample["audio"]["array"]
        sample_rate = sample["audio"]["sampling_rate"]
        audio_path = os.path.join(AUDIO_DIR, f"audio_{i}.wav")

        waveform = torch.tensor(audio_array).unsqueeze(0)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        torchaudio.save(audio_path, waveform, 16000)

        exported_samples.append({
            "id": i,
            "text": sample["utt"],
            "audio_path": audio_path
        })

    results = [
        vosk_model.transcribe(s["audio_path"], s["text"])
        for s in exported_samples
    ]
    save_dataset_results(results, "ITALIC")


if __name__ == "__main__":
    print("Loading Vosk Model...")
    model = Vosk()
    run_common_voice(model)
    run_synthetic_dataset(model)
    run_coqui_dataset(model)
    run_ITALIC(model)
    print("All datasets processed successfully.")