# plot_gemini_results.py
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

    # 1. Find the summary file
    summary_path = os.path.join(results_dir, "summary_across_datasets.csv")
    if not os.path.exists(summary_path):
        print(f"Error: Summary results file not found at {summary_path}")
        return

    # 2. Load results
    df_all = pd.read_csv(summary_path)

    # 3. Create output directory
    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    # 4. Iterate over each dataset
    for dataset in df_all['dataset'].unique():
        dataset_df = df_all[df_all['dataset'] == dataset]
        
        # 5. Create a dataset-specific output directory
        dataset_output_dir = os.path.join(output_dir, dataset)
        os.makedirs(dataset_output_dir, exist_ok=True)

        # 6. Generate plots for each metric, with models on the x-axis
        metrics = ["wer", "cer", "bleu_avg", "rouge1_avg_f1", "levenshtein_avg", "time"]

        for metric in metrics:
            # Skip if the column does not exist
            if metric not in df_all.columns:
                print(f"Skipping {metric} plot for {dataset} because the data is missing")
                continue

            plt.figure(figsize=(10, 6))
            sns.barplot(data=dataset_df, x="model", y=metric, palette="Set2") #It now calls the dataset not all

            plt.title(f"{metric.upper()} for {dataset.capitalize()}", fontsize=14) #New title
            plt.ylabel(metric.upper(), fontsize=12) #y label and x label to be precise
            plt.xlabel("Model", fontsize=12)
            plt.xticks(rotation=45, ha="right") # rotation
            plt.tight_layout()

            plot_path = os.path.join(dataset_output_dir, f"{metric}_comparison.png")
            plt.savefig(plot_path)
            plt.close()

            print(f"Saved {metric} plot for {dataset} to {plot_path}")
    print("\nAll specified plots saved")