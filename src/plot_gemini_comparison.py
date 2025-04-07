import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def plot_gemini_results(results_dir="output/stt/gemini_only"):
    """
    Loads Gemini benchmark results and generates plots.
    """

    summary_path = os.path.join(results_dir, "summary_across_datasets.csv")
    if not os.path.exists(summary_path):
        print(f"Error: Summary results file not found at {summary_path}")
        return

    df_all = pd.read_csv(summary_path)

    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    for dataset in df_all['dataset'].unique():
        dataset_df = df_all[df_all['dataset'] == dataset]
        
        dataset_output_dir = os.path.join(output_dir, dataset)
        os.makedirs(dataset_output_dir, exist_ok=True)

        metrics = ["wer", "cer", "bleu_avg", "rouge1_avg_f1", "levenshtein_avg", "time"]

        for metric in metrics:
            # Skip if the column does not exist
            if metric not in df_all.columns:
                print(f"Skipping {metric} plot for {dataset} because the data is missing")
                continue

            plt.figure(figsize=(10, 6))
            sns.barplot(data=dataset_df, x="model", y=metric, palette="Set2")

            plt.title(f"{metric.upper()} for {dataset.capitalize()}", fontsize=14)
            plt.ylabel(metric.upper(), fontsize=12)
            plt.xlabel("Model", fontsize=12)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            plot_path = os.path.join(dataset_output_dir, f"{metric}_comparison.png")
            plt.savefig(plot_path)
            plt.close()

    print("\nSpecific dataset plots saved in the following directory:")
    print(f"{output_dir}")


def final_comparison_between_models():
    """
    Generates a final comparison plot between all models across datasets.
    """
    results_dir = "output/stt/gemini_only"
    summary_path = os.path.join(results_dir, "summary_across_datasets.csv")
    
    if not os.path.exists(summary_path):
        print(f"Error: Summary results file not found at {summary_path}")
        return

    df_all = pd.read_csv(summary_path)

    # Create a final comparison plot for all models across datasets
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_all, x="model", y="wer", hue="dataset", palette="Set2")

    plt.title("Final Comparison of WER Across Models and Datasets", fontsize=16)
    plt.ylabel("Word Error Rate (WER)", fontsize=14)
    plt.xlabel("Model", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plot_path = os.path.join(results_dir, "final_comparison.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved final comparison plot to {plot_path}")

if __name__ == "__main__":
    plot_gemini_results()
    final_comparison_between_models()