import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

summary_path = "output/stt/vosk_only/summary_across_datasets.csv"
output_dir = "output/stt/vosk_only/figures"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(summary_path)
metrics = ["wer", "cer", "bleu_avg", "rouge1_avg_f1", "levenshtein_avg"]

for metric in metrics:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="dataset", y=metric, palette="Set2", hue="model")
    plt.title(f"{metric.upper()} per Dataset - {df['model'].iloc[0]}")
    plt.ylabel(metric.upper())
    plt.xlabel("Dataset")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/compare_{metric}.png")
    plt.close()

print("Comparative plots saved in:", output_dir)
