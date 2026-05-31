import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import gc # 引入垃圾回收，虽然内存大，但 STFT 可能真的很大

# ======================
# 全局配置
# ======================
FEATURE_DIR = "feature_data"
SAVE_DIR = "dim_reduction_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# 随机种子，保证 t-SNE 结果可复现
RANDOM_STATE = 42
# 移除采样限制，使用全部数据
# MAX_SAMPLES = 2000 

def load_and_preprocess(feature_name, y_labels):
    """
    加载并预处理特征数据
    :param feature_name: 特征文件名 (e.g., "X_mel.npy")
    :param y_labels: 标签数据
    :return: 展平并标准化后的特征, 对应的标签
    """
    print(f"Loading {feature_name}...")
    X = np.load(os.path.join(FEATURE_DIR, feature_name))
    
    # 不再进行下采样，使用全量数据
    y = y_labels

    # 展平特征：(N, H, W) -> (N, H*W)
    # 因为 PCA/t-SNE 需要二维输入 (样本数, 特征维度)
    print(f"Original shape: {X.shape}")
    X_flat = X.reshape(len(X), -1)
    print(f"Flattened shape: {X_flat.shape}")

    # 标准化 (Standardization): 均值为0，方差为1
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    
    # 释放原始 X 内存
    del X
    del X_flat
    gc.collect()
    
    return X_scaled, y

def plot_dim_reduction(X_pca, X_tsne, y, label_map_inv, feature_name):
    """
    绘制 PCA 和 t-SNE 的对比图
    """
    # 将标签索引转换为名称
    y_names = [label_map_inv[label] for label in y]
    
    # 创建 DataFrame 方便 Seaborn 绘图
    df_pca = pd.DataFrame({'Component 1': X_pca[:, 0], 'Component 2': X_pca[:, 1], 'Class': y_names})
    df_tsne = pd.DataFrame({'Dimension 1': X_tsne[:, 0], 'Dimension 2': X_tsne[:, 1], 'Class': y_names})

    fig, axes = plt.subplots(1, 2, figsize=(24, 10)) # 增大画布尺寸
    
    # 1. PCA Plot
    sns.scatterplot(
        x='Component 1', y='Component 2', hue='Class', 
        data=df_pca, ax=axes[0], palette='viridis', s=40, alpha=0.6 # 调整点的大小和透明度
    )
    axes[0].set_title(f'PCA - {feature_name} (Full Data)', fontsize=18)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # 2. t-SNE Plot
    sns.scatterplot(
        x='Dimension 1', y='Dimension 2', hue='Class', 
        data=df_tsne, ax=axes[1], palette='viridis', s=40, alpha=0.6
    )
    axes[1].set_title(f't-SNE - {feature_name} (Full Data)', fontsize=18)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f"dr_{feature_name}_full")
    plt.savefig(save_path + ".png", dpi=300)
    plt.savefig(save_path + ".svg", dpi=300)
    plt.savefig(save_path + ".pdf", dpi=300)
    print(f"Saved visualization to {save_path}")
    plt.close()

def run_analysis():
    # 1. 加载标签
    print("Loading labels...")
    y_all = np.load(os.path.join(FEATURE_DIR, "y_labels.npy"))
    label_map_inv = {0: "Cargo", 1: "Tanker", 2: "Tug", 3: "Passengership"}

    # 定义要分析的特征列表，包含 STFT
    feature_files = [
        "X_mel.npy",
        "X_mfcc.npy", 
        "X_cqt.npy",
        "X_stft.npy" # 加入 STFT
    ]

    # 用于存储分析结果的列表
    results_data = []

    for f_name in feature_files:
        print(f"\n{'='*40}")
        print(f"Processing {f_name} (Full Dataset)...")
        print(f"{'='*40}")

        try:
            # 2. 数据预处理
            X_scaled, y = load_and_preprocess(f_name, y_all)

            # 3. PCA (主成分分析)
            print("Running PCA...")
            # 对于 STFT 这样极高维的数据，直接计算全部主成分可能很慢
            # 但这里只求前2个，svd_solver='arpack' 或 'randomized' 通常很快
            pca = PCA(n_components=2, random_state=RANDOM_STATE)
            X_pca = pca.fit_transform(X_scaled)
            
            # 获取 PCA 解释方差比
            explained_variance = pca.explained_variance_ratio_
            print(f"PCA explained variance ratio: {explained_variance}")
            
            # 记录数据
            results_data.append({
                "Feature": f_name.replace(".npy", ""),
                "PCA_Component_1_Variance": explained_variance[0],
                "PCA_Component_2_Variance": explained_variance[1],
                "Total_Explained_Variance_2C": np.sum(explained_variance)
            })

            # 4. t-SNE
            print("Running t-SNE (this will take time)...")
            # n_jobs=-1 使用所有 CPU 核心加速
            tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, 
                        random_state=RANDOM_STATE, init='pca', learning_rate='auto', n_jobs=-1)
            X_tsne = tsne.fit_transform(X_scaled)

            # 5. 可视化
            plot_dim_reduction(X_pca, X_tsne, y, label_map_inv, f_name.replace(".npy", ""))
            
            # 清理内存
            del X_scaled
            del X_pca
            del X_tsne
            gc.collect()

        except Exception as e:
            print(f"Error processing {f_name}: {e}")
            import traceback
            traceback.print_exc()

    # 保存分析结果到 CSV 和 TXT
    print("\nSaving quantitative analysis results...")
    df_results = pd.DataFrame(results_data)
    
    # 1. 保存为 CSV 表格
    csv_path = os.path.join(SAVE_DIR, "pca_variance_analysis.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Saved CSV report to {csv_path}")

    # 2. 保存为易读的 TXT 报告
    txt_path = os.path.join(SAVE_DIR, "analysis_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("DeepShip Feature Analysis Report\n")
        f.write("================================\n\n")
        for index, row in df_results.iterrows():
            f.write(f"Feature: {row['Feature']}\n")
            f.write(f"  - PCA Component 1 Explained Variance: {row['PCA_Component_1_Variance']:.4%}\n")
            f.write(f"  - PCA Component 2 Explained Variance: {row['PCA_Component_2_Variance']:.4%}\n")
            f.write(f"  - Total Explained Variance (First 2 Components): {row['Total_Explained_Variance_2C']:.4%}\n")
            f.write("-" * 40 + "\n")
    print(f"Saved TXT report to {txt_path}")

    print(f"\nFull analysis complete. Results saved in '{SAVE_DIR}' directory.")

if __name__ == "__main__":
    run_analysis()
