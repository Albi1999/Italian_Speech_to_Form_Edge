import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_gemini_data(results_dir="output/stt/gemini_only"):
    summary_path = os.path.join(results_dir, "summary_across_datasets.csv")
    if not os.path.exists(summary_path):
        print(f"Error: Summary results file not found at {summary_path}")
        return None
    df_gemini = pd.read_csv(summary_path)
    df_gemini['source'] = 'Gemini (Google)'
    
    # Compute mean WER for Gemini models
    df_gemini_mean = df_gemini.groupby('model').agg({'wer': 'mean'}).reset_index()
    df_gemini_mean['source'] = 'Gemini (Google)'
    
    return df_gemini_mean

def load_other_models_data(results_dir="output/stt", filename="model_comparison_summary.csv"):
    summary_path = os.path.join(results_dir, filename)
    if not os.path.exists(summary_path):
        print(f"Error: Summary results file not found at {summary_path}")
        return None
    df_others = pd.read_csv(summary_path)
    
    # Assign source based on model names
    df_others['source'] = df_others['model'].apply(classify_model)
    
    return df_others

def classify_model(model_name):
    if "Wav2Vec" in model_name:
        return "Wav2VecXLSR (Meta)"
    elif "Whisper" in model_name:
        return "Whisper (OpenAI)"
    elif "Vosk" in model_name:
        return "Vosk (Alpha Cephei)"
    else:
        return "Other"

def combine_data():
    df_gemini = load_gemini_data()
    df_others = load_other_models_data()
    
    if df_gemini is None or df_others is None:
        print("Error: Could not load data.")
        return None
    
    df_gemini = df_gemini[['model', 'wer', 'source']]
    df_others = df_others[['model', 'wer_mean', 'source']]
    df_others.rename(columns={'wer_mean': 'wer'}, inplace=True)
    
    combined_df = pd.concat([df_gemini, df_others], ignore_index=True)
    combined_df = combined_df.sort_values('wer').reset_index(drop=True)
    
    return combined_df

def plot_wer_comparison(combined_df):
    if combined_df is None:
        print("Error: No data to plot.")
        return
    
    plt.figure(figsize=(14, 8))
    
    sns.barplot(data=combined_df, x='model', y='wer', hue='source', palette="Set2", errorbar=None)
    
    plt.title('WER Comparison Across Different Models', fontsize=16)
    plt.ylabel('Word Error Rate (WER)', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    plot_path = "output/stt/final_wer_comparison.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved final WER comparison plot to {plot_path}")

if __name__ == "__main__":
    combined_df = combine_data()
    plot_wer_comparison(combined_df)