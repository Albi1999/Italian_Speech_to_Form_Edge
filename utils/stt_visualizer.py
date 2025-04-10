import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

class STTVisualizer:
    def __init__(self, output_base_dir="output/stt"):
        """
        Initialize the STTVisualizer with default settings.
        
        Args:
            output_base_dir (str): Base directory for saving visualization outputs
        """
        self.output_base_dir = output_base_dir
        self.plot_dir = f"{output_base_dir}/plots"
        self.csv_dir = f"{output_base_dir}/csv"
        self.metric_names = {
            "wer": "Word Error Rate",
            "cer": "Character Error Rate",
            "bleu_avg": "BLEU Score",
            "rouge1_avg_f1": "ROUGE-1 F1 Score",
            "levenshtein_avg": "Levenshtein Distance",
            "wer_mean": "Word Error Rate",
            "cer_mean": "Character Error Rate",
            "bleu_avg_mean": "BLEU Score",
            "rouge1_avg_f1_mean": "ROUGE-1 F1 Score",
            "levenshtein_avg_mean": "Levenshtein Distance"
        }
        self.colors = {
            'italian': '#1f77b4', 
            'multilingual': '#ff7f0e',
            'english': '#2ca02c',
            'spanish': '#d62728',
            'french': '#9467bd',
            'german': '#8c564b'
        }
        
        # Create output directories
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        
        # Set default style
        self._set_plot_style()
    
    def _set_plot_style(self):
        """Set consistent style across all visualizations"""
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['axes.labelpad'] = 10
    
    def basic_metrics_plots(self, df, sort_by_performance=True):
        """
        Create basic bar charts for each metric.
        
        Args:
            df (DataFrame): DataFrame containing the results
            sort_by_performance (bool): Whether to sort models by their performance
            
        Returns:
            list: Paths to the generated plot files
        """
        plot_files = []
        metrics = ["wer", "cer", "bleu_avg", "rouge1_avg_f1", "levenshtein_avg"]
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(df['model'].unique())))
        
        for metric in metrics:
            fig, ax = plt.subplots()
            grouped = df.groupby("model")[metric]
            means = grouped.mean()
            stds = grouped.std()
            
            # Sort by performance if requested
            if sort_by_performance:
                # Sort ascending for error metrics, descending for others
                ascending = metric in ['wer', 'cer']
                sorted_indices = means.sort_values(ascending=ascending).index
                means = means[sorted_indices]
                stds = stds[sorted_indices]
            
            # Plot with better styling
            bars = ax.bar(means.index, means.values, yerr=stds.values, 
                    error_kw = dict(lw=1, capsize=4, capthick=1, color='gray', alpha=0.5),
                    capsize=5, alpha=0.8, color=colors)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Improve readability
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Better titles and labels
            plt.title(f"{self.metric_names.get(metric, metric)} by Model")
            plt.ylabel(self.metric_names.get(metric, metric))
            plt.xlabel("Model")
            
            # Add median line for reference
            median = means.median()
            plt.axhline(y=median, color='red', linestyle='--', alpha=0.7, 
                       label=f'Median: {median:.3f}')
            plt.legend()
            
            plt.tight_layout()
            plot_path = f"{self.plot_dir}/plot_{metric}.png"
            plt.savefig(plot_path)
            plot_files.append(plot_path)
            plt.close()
        
        print(f"\nBasic metric plots saved to {self.plot_dir}/")
        return plot_files
    
    def prepare_summary_data(self, df, df_meta):
        """
        Prepare summary data with metrics for each model.
        
        Args:
            df (DataFrame): Results DataFrame
            df_meta (DataFrame): Metadata for each model
            
        Returns:
            DataFrame: Processed and summarized data
        """
        # Merge results with metadata
        merged_df = df.merge(df_meta, on="model")
        
        # Calculate summary statistics
        grouped = merged_df.groupby("model").agg({
            "wer": ["mean", "std"],
            "cer": ["mean", "std"],
            "bleu_avg": "mean",
            "rouge1_avg_f1": "mean",
            "levenshtein_avg": "mean",
            "time": "mean",
            "size_m": "first",
            "language": "first"
        })
        
        # Clean up column names
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped = grouped.reset_index()
        
        # Save to CSV
        csv_path = f"{self.csv_dir}/metrics_summary.csv"
        grouped.to_csv(csv_path, index=False)
        print(f"\nSummary metrics saved to {csv_path}")
        
        return grouped
    
    def plot_scatter_relationships(self, grouped):
        """
        Plot scatter plots showing relationships between metrics and model properties.
        
        Args:
            grouped (DataFrame): Summarized metrics data
        """
        metrics = ["wer_mean", "cer_mean", "bleu_avg_mean", "rouge1_avg_f1_mean", "levenshtein_avg_mean"]
        
        for metric in metrics:
            for x_var, x_label in [("time_mean", "Inference Time (s)"), 
                                  ("size_m_first", "Model Size (M parameters)")]:
                fig, ax = plt.subplots()
                
                # Get colors based on language
                point_colors = [
                    self.colors.get(lang.lower(), '#2ca02c') 
                    for lang in grouped["language_first"]
                ]
                
                # Create scatter plot with colored points
                scatter = ax.scatter(
                    grouped[x_var], 
                    grouped[metric], 
                    c=point_colors,
                    s=80, alpha=0.7, edgecolors='black', linewidths=0.5
                )
                
                # Add model name annotations with better positioning
                texts = []
                for i, label in enumerate(grouped["model"]):
                    # Shorten very long model names
                    short_label = label if len(label) < 20 else label[:17] + "..."
                    texts.append(ax.annotate(
                        short_label, 
                        (grouped[x_var][i], grouped[metric][i]),
                        fontsize=8,
                        alpha=0.8
                    ))
                
                # Use adjust_text to prevent overlapping annotations
                adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6))
                    
                # Add trend line
                z = np.polyfit(grouped[x_var], grouped[metric], 1)
                p = np.poly1d(z)
                x_range = np.linspace(grouped[x_var].min(), grouped[x_var].max(), 100)
                ax.plot(
                    x_range, p(x_range), 
                    "r--", alpha=0.3, 
                    label=f"Trend: y={z[0]:.4f}x + {z[1]:.4f}"
                )
                
                ax.set_xlabel(x_label)
                ax.set_ylabel(self.metric_names.get(metric, metric))
                ax.set_title(f"{self.metric_names.get(metric, metric)} vs {x_label}")
                
                # Add legend for language colors
                unique_langs = grouped["language_first"].unique()
                handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=self.colors.get(lang.lower(), '#2ca02c'), 
                                     markersize=10, label=lang.capitalize())
                          for lang in unique_langs]
                
                handles.append(plt.Line2D([0], [0], color='red', alpha=0.5, 
                                        linestyle='--', label='Trend line'))
                ax.legend(handles=handles, loc='best')
                
                ax.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                plot_path = f"{self.plot_dir}/{metric}_vs_{x_var}.png"
                plt.savefig(plot_path)
                plt.close()
    
    def compare_categories(self, df, category_col, name):
        """
        Create bar charts comparing performance across different categories.
        
        Args:
            df (DataFrame): Results DataFrame
            category_col (str): Column name containing categories
            name (str): Display name for the category
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Mean WER per model and category
        grouped_box = df.groupby([category_col, "model"])["wer"].mean().reset_index()
        grouped_box = grouped_box.sort_values("wer", ascending=True)
        grouped_box[category_col] = grouped_box[category_col].str.capitalize()

        # Define distinct color palette with higher contrast
        categories = grouped_box[category_col].unique()
        palette = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]
        color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}
        colors = [color_map[cat] for cat in grouped_box[category_col]]

        # Bar plot with better styling
        bars = ax.bar(
            grouped_box["model"], 
            grouped_box["wer"], 
            color=colors, 
            edgecolor='black', 
            linewidth=0.5, 
            alpha=0.85
        )
        
        # Add value labels on top of bars
        for i, (bar, val) in enumerate(zip(bars, grouped_box["wer"])):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.005, 
                f"{val:.3f}", 
                ha='center', 
                fontsize=9,
                fontweight='bold'
            )
            # Add category as text at bottom of bar
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                0.002, 
                grouped_box[category_col].iloc[i][0],  # First letter only
                ha='center', 
                fontsize=9,
                color='black',
                fontweight='bold'
            )

        ax.set_xticks(np.arange(len(grouped_box["model"])))
        ax.set_xticklabels(grouped_box["model"], rotation=45, ha='right')
        ax.set_ylabel("Word Error Rate (WER)")
        ax.set_title(f"WER Comparison by {name}")
        ax.grid(axis="y", linestyle='--', alpha=0.7)

        # Add legend with better placement
        handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[cat]) 
                  for cat in categories]
        ax.legend(
            handles, 
            categories, 
            title=category_col.capitalize(), 
            loc="upper left", 
            framealpha=0.9
        )
        
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/wer_comparison_{category_col}.png")
        plt.close()
    
    
    def visualize_all(self, results_df, metadata_df, categorize_by_size=True):
        """
        Generate all visualizations in one call.
        
        Args:
            results_df (DataFrame): Results data
            metadata_df (DataFrame): Model metadata
            categorize_by_size (bool): Whether to categorize models by size
            
        Returns:
            tuple: Paths to CSV and plot directories
        """
        # 1. Basic metric plots
        self.basic_metrics_plots(results_df)
        
        # 2. Prepare summary data
        grouped = self.prepare_summary_data(results_df, metadata_df)
        
        # 3. Plot scatter relationships
        self.plot_scatter_relationships(grouped)
        
        # 4. Language comparisons
        self.compare_categories(
            results_df.merge(metadata_df, on="model"), 
            "language", 
            "Language Support"
        )
        
        # 5. Size comparisons if needed
        if categorize_by_size:
            grouped["size_category"] = grouped["size_m_first"].apply(
                lambda x: "small" if x < 100 else "large"
            )
            merged_df = results_df.merge(
                grouped[["model", "size_category"]], 
                on="model"
            )
            self.compare_categories(merged_df, "size_category", "Model Size")
        
        
        print(f"\nAll visualizations completed successfully!")
        return self.csv_dir, self.plot_dir