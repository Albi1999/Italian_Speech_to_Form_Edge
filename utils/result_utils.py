import pandas as pd
import os
import matplotlib.pyplot as plt

def save_results(all_results, output_dir="output/stt/csv"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(f"{output_dir}/stt_results.csv", index=False)
    print(f"\nResults saved to {output_dir}/stt_results.csv")

def plot_metrics(df):
    os.makedirs("output/stt/plots", exist_ok=True)

    metrics = ["wer", "cer", "bleu_avg", "rouge1_avg_f1", "levenshtein_avg"]
    plot_files = []

    for metric in metrics:
        plt.figure()
        grouped = df.groupby("model")[metric]
        means = grouped.mean()
        stds = grouped.std()

        plt.bar(means.index, means.values, yerr=stds.values, capsize=5)
        plt.title(f"{metric.upper()} - Avg ± Std per Model")
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.tight_layout()

        plot_path = f"output/stt/plots/plot_{metric}.png"
        plt.savefig(plot_path)
        plot_files.append(plot_path)
        plt.close()
    
    print(f"\nPlots saved: {', '.join(plot_files)}")

    return plot_files