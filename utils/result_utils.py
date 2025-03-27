import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt

def save_results(all_results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(all_results)
    df.to_csv(f"{output_dir}/stt_results_{timestamp}.csv", index=False)
    print(f"\nResults saved to {output_dir}/stt_results_{timestamp}.csv")

def plot_metrics(df):
    os.makedirs("results/plots", exist_ok=True)

    metrics = ["wer", "cer", "bleu_avg", "rouge1_avg_f1", "levenshtein_avg"]
    plot_files = []

    for metric in metrics:
        plt.figure()
        df.groupby("model")[metric].mean().plot(kind="bar", title=f"{metric.upper()} - Avg per Model")
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.tight_layout()

        plot_path = f"results/plots/plot_{metric}.png"
        plt.savefig(plot_path)
        plot_files.append(plot_path)
        plt.close()

    return plot_files