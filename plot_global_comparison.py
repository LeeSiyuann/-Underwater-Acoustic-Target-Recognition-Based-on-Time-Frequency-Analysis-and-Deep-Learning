import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_DIR = "dl_comparison_results_gpu"
CSV_FILE = os.path.join(RESULTS_DIR, "final_comparison_results.csv")
FUSION_CSV_FILE = os.path.join(RESULTS_DIR, "final_fusion_results.csv")

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

    # 3. Macro comparison for Accuracy / Precision / Recall / F1 / ROC-AUC
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    available_metrics = [metric for metric in metrics if metric in df.columns]
    if available_metrics:
        melted_df = df.melt(
            id_vars=['Model', 'Feature'],
            value_vars=available_metrics,
            var_name='Metric',
            value_name='Score'
        )

        plt.figure(figsize=(18, 10))
        sns.barplot(data=melted_df, x='Model', y='Score', hue='Metric', ci=None)
        plt.title('Global Metric Comparison Across Models', fontsize=16)
        plt.xlabel('Deep Learning Models', fontsize=14)
        plt.ylabel('Metric Score', fontsize=14)
        plt.ylim(0, 1.0)
        plt.xticks(rotation=15)
        plt.tight_layout()
        metrics_path = os.path.join(RESULTS_DIR, "global_metric_barplot")
        plt.savefig(metrics_path + ".png")
        plt.savefig(metrics_path + ".svg")
        plt.savefig(metrics_path + ".pdf")
        plt.close()
        print(f"Saved Metric Comparison Chart to {metrics_path}")

if __name__ == "__main__":
    plot_global_comparison()
