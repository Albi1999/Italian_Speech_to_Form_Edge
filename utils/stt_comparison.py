import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class STTComparison:
    def __init__(self, gemini_results_dir="output/stt/gemini_only", other_results_dir="output/stt/comparison", other_filename="model_comparison_summary.csv"):
        self.gemini_results_dir = gemini_results_dir
        self.other_results_dir = other_results_dir
        self.other_filename = other_filename
        self.metrics = []
        self.metrics_to_minimize = ['wer', 'cer', 'time', 'size_m_first', 'levenshtein_avg']
        self.metrics_to_maximize = ['bleu_avg', 'rouge1_avg_f1']

    def load_gemini_data(self):
        summary_path = os.path.join(self.gemini_results_dir, "summary_across_datasets.csv")
        if not os.path.exists(summary_path):
            print(f"Error: Summary results file not found at {summary_path}")
            return None
        try:
            df_gemini = pd.read_csv(summary_path)
        except Exception as e:
            print(f"Failed to load Gemini data: {e}")
            return None
        df_gemini['source'] = 'Gemini (Google)'
        metric_columns = [col for col in df_gemini.columns if col not in ['dataset', 'model', 'source', 'time']]
        if not metric_columns:
            print("No metric columns found in Gemini data.")
            return None
        self.metrics = metric_columns
        df_gemini_mean = df_gemini.groupby('model').agg({metric: 'mean' for metric in metric_columns}).reset_index()
        df_gemini_mean['source'] = 'Gemini (Google)'
        return df_gemini_mean

    def load_other_models_data(self):
        summary_path = os.path.join(self.other_results_dir, self.other_filename)
        if not os.path.exists(summary_path):
            print(f"Error: Summary results file not found at {summary_path}")
            return None
        try:
            df_others = pd.read_csv(summary_path)
        except Exception as e:
            print(f"Failed to load other models data: {e}")
            return None
        df_others['source'] = df_others['model'].apply(self.classify_model)
        metric_columns = [col for col in df_others.columns if col not in ['model', 'source', 'size_m_first', 'num_datasets', 'composite_score']]
        df_others_filtered = df_others[['model', 'source'] + metric_columns]
        return df_others_filtered

    @staticmethod
    def classify_model(model_name):
        if "Wav2Vec" in model_name:
            return "Wav2VecXLSR (Meta)"
        elif "Whisper" in model_name:
            return "Whisper (OpenAI)"
        elif "Vosk" in model_name:
            return "Vosk (Alpha Cephei)"
        else:
            return "Other"

    def combine_data(self):
        df_gemini = self.load_gemini_data()
        df_others = self.load_other_models_data()
        if df_gemini is None or df_others is None:
            print("Error: Could not load data. Cannot combine.")
            return None
        rename_dict = {}
        for col in df_others.columns:
            if col.endswith('_mean'):
                base_name = col[:-5]
                if base_name in df_gemini.columns:
                    rename_dict[col] = base_name
        df_others = df_others.rename(columns=rename_dict)
        common_metrics = list(set(self.metrics).intersection(df_others.columns))
        if not common_metrics:
            print("Error: No common metrics found between Gemini and other models data AFTER renaming.")
            print(f"Gemini metrics: {self.metrics}")
            print(f"Other models metrics (after renaming): {df_others.columns}")
            return None
        df_gemini = df_gemini[['model', 'source'] + common_metrics]
        df_others = df_others[['model', 'source'] + common_metrics]
        combined_df = pd.concat([df_gemini, df_others], ignore_index=True)
        return combined_df

    def plot_metric_comparison(self, metric):
        combined_df = self.combine_data()
        if combined_df is None:
            print(f"Error: No data to plot for {metric}.")
            return
        
        # Sort based on metric order: Minimize -> Ascending, Maximize -> Descending
        ascending = metric in self.metrics_to_minimize
        combined_df.sort_values(by=metric, inplace=True, ascending=ascending)

        plt.figure(figsize=(14, 8))
        sns.barplot(data=combined_df, x='model', y=metric, hue='source', palette="Set2", errorbar=None)
        plt.title(f'{metric} Comparison Across Different Models', fontsize=16)
        plt.ylabel(metric, fontsize=14)
        plt.xlabel('Model', fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plot_path = f"output/stt/comparison/final_{metric}_comparison.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved final {metric} comparison plot to {plot_path}")

    def run_comparisons(self):
        self.load_gemini_data()
        if self.metrics:
            for metric in self.metrics:
                self.plot_metric_comparison(metric)
        else:
            print("No metrics found to compare.")
