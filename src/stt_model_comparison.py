import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Configuration
root_path = "output/stt"
target_filename = "metrics_summary.csv"
output_dir = root_path
metrics_to_minimize = ['wer_mean', 'cer_mean', 'time_mean', 'size_m_first']
metrics_to_maximize = ['bleu_avg_mean', 'rouge1_avg_f1_mean']
id_column = 'model'

# Load all CSV files
all_results = []
for dirpath, _, filenames in os.walk(root_path):
    if target_filename in filenames:
        dataset_name = dirpath.split(os.sep)[-2]
        file_path = os.path.join(dirpath, target_filename)
        df = pd.read_csv(file_path)
        df['dataset'] = dataset_name
        all_results.append(df)

df_all = pd.concat(all_results, ignore_index=True)

# Normalize metrics
df_norm = df_all.copy()
scaler = MinMaxScaler()

for metric in metrics_to_minimize:
    df_norm[metric + '_norm'] = 1 - scaler.fit_transform(df_all[[metric]])

for metric in metrics_to_maximize:
    df_norm[metric + '_norm'] = scaler.fit_transform(df_all[[metric]])

norm_cols = [m + '_norm' for m in (metrics_to_minimize + metrics_to_maximize)]
df_norm['composite_score'] = df_norm[norm_cols].mean(axis=1)

# Aggregate results
agg_df = df_norm.groupby(id_column).agg({
    'composite_score': 'mean',
    'wer_mean': 'mean',
    'cer_mean': 'mean',
    'bleu_avg_mean': 'mean',
    'rouge1_avg_f1_mean': 'mean',
    'time_mean': 'mean',
    'size_m_first': 'mean',
    'dataset': 'count'
}).rename(columns={'dataset': 'num_datasets'}).sort_values('composite_score', ascending=False)

# Save the aggregated results
csv_output_path = os.path.join(output_dir, "model_comparison_summary.csv")
agg_df.to_csv(csv_output_path)
print("Model comparison summary:")
print(agg_df)

#! Since the Vosk STT (Small IT) is the second in the ranking but it has the lowest size and time, I'll select it as the best model for the STT task.

print(f"\nSaved comparison table to {csv_output_path}")

# Plot 1: Composite Score Bar Plot
plt.figure(figsize=(10, 6))
plt.barh(agg_df.index, agg_df['composite_score'], color='cornflowerblue')
plt.xlabel('Composite Score (higher = better)', fontsize=12)
plt.title('Model Comparison by Composite Score', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()

barplot_path = os.path.join(output_dir, "composite_score_plot.png")
plt.savefig(barplot_path, dpi=300)
plt.close()
print(f"Saved composite score plot to {barplot_path}")

# Plot 2: Scatter Plot (Size vs Time with WER as color)
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    agg_df['size_m_first'], 
    agg_df['time_mean'], 
    c=agg_df['wer_mean'], 
    cmap='coolwarm', 
    s=150,
    edgecolors='black'
)
plt.xlabel('Model Size (MB)', fontsize=12)
plt.ylabel('Execution Time (s)', fontsize=12)
plt.title('Model Trade-Off: Size vs Time (Color = WER)', fontsize=14)
plt.colorbar(scatter, label='WER (mean)')

# Annotate points
for i, model in enumerate(agg_df.index):
    x = agg_df['size_m_first'].iloc[i]
    y = agg_df['time_mean'].iloc[i]
    label = model
    plt.annotate(label, (x + 5, y + 0.2), fontsize=9, alpha=0.85)

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

scatterplot_path = os.path.join(output_dir, "size_time_wer_plot.png")
plt.savefig(scatterplot_path, dpi=300)
plt.close()
print(f"Saved scatter plot to {scatterplot_path}")
