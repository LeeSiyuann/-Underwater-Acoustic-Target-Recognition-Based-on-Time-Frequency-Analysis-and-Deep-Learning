import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_DIR = "dl_comparison_results_gpu"
CSV_FILE = os.path.join(RESULTS_DIR, "final_comparison_results.csv")
FUSION_CSV_FILE = os.path.join(RESULTS_DIR, "final_fusion_results.csv")
METRIC_COLUMNS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

def load_all_results():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"{CSV_FILE} not found. Please ensure training is complete.")

    df = pd.read_csv(CSV_FILE)
    if os.path.exists(FUSION_CSV_FILE):
        fusion_df = pd.read_csv(FUSION_CSV_FILE)
        df = pd.concat([df, fusion_df], ignore_index=True)

    merged_path = os.path.join(RESULTS_DIR, "final_comparison_results_with_fusion.csv")
    df.to_csv(merged_path, index=False)
    return df

def plot_global_comparison():
    try:
        df = load_all_results()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # 1. Grouped Bar Chart for F1 Score
    plt.figure(figsize=(16, 8))
    sns.barplot(data=df, x='Model', y='f1', hue='Feature', palette='viridis')
    plt.title('Global Comparison: F1 Score across Models and Features', fontsize=16)
    plt.xlabel('Deep Learning Models', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.ylim(0, 1.0)
    plt.legend(title='Feature Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    bar_path = os.path.join(RESULTS_DIR, "global_f1_barplot")
    plt.savefig(bar_path + ".png")
    plt.savefig(bar_path + ".svg")
    plt.savefig(bar_path + ".pdf")
    plt.close()
    print(f"Saved Bar Chart to {bar_path}")

    # 2. Heatmap for F1 Score
    plt.figure(figsize=(12, 8))
    pivot_df = df.pivot(index='Model', columns='Feature', values='f1')
    sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'F1 Score'})
    plt.title('Heatmap of F1 Scores', fontsize=16)
    plt.tight_layout()
    heatmap_path = os.path.join(RESULTS_DIR, "global_f1_heatmap")
    plt.savefig(heatmap_path + ".png")
    plt.savefig(heatmap_path + ".svg")
    plt.savefig(heatmap_path + ".pdf")
    plt.close()
    print(f"Saved Heatmap to {heatmap_path}")

    # 3. Dual-dimension metric heatmaps across Model + Feature
    available_metrics = [metric for metric in METRIC_COLUMNS if metric in df.columns]
    if available_metrics:
        melted_df = df.melt(
            id_vars=['Model', 'Feature'],
            value_vars=available_metrics,
            var_name='Metric',
            value_name='Score'
        )
        melted_csv = os.path.join(RESULTS_DIR, "global_metric_model_feature_long.csv")
        melted_df.to_csv(melted_csv, index=False)

        n_metrics = len(available_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5.5 * n_rows))
        axes = axes.flatten()

        for idx, metric in enumerate(available_metrics):
            metric_df = df.pivot(index='Model', columns='Feature', values=metric)
            sns.heatmap(
                metric_df,
                annot=True,
                fmt=".4f",
                cmap="YlGnBu",
                vmin=0.0,
                vmax=1.0,
                cbar=idx == len(available_metrics) - 1,
                cbar_kws={'label': 'Score'} if idx == len(available_metrics) - 1 else None,
                ax=axes[idx]
            )
            axes[idx].set_title(f'{metric.upper()} by Model + Feature', fontsize=14)
            axes[idx].set_xlabel('Feature')
            axes[idx].set_ylabel('Model')

        for idx in range(n_metrics, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        metrics_path = os.path.join(RESULTS_DIR, "global_metric_model_feature_heatmaps")
        plt.savefig(metrics_path + ".png")
        plt.savefig(metrics_path + ".svg")
        plt.savefig(metrics_path + ".pdf")
        plt.close()
        print(f"Saved dual-dimension metric heatmaps to {metrics_path}")
        print(f"Saved long-format metric table to {melted_csv}")

if __name__ == "__main__":
    plot_global_comparison()
