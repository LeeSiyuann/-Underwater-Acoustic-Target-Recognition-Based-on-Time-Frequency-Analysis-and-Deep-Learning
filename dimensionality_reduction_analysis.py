import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
import seaborn as sns
import pandas as pd
import gc
import tempfile

# ======================
# 全局配置
# ======================
FEATURE_DIR = "feature_data"
SAVE_DIR = "dim_reduction_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# 随机种子，保证 t-SNE 结果可复现
RANDOM_STATE = 42
SCALER_BATCH_SIZE = 256
PCA_BATCH_SIZE = 256
TSNE_PREP_DIM = 50


def iter_feature_batches(X_mmap, batch_size):
    """按批次遍历特征，避免一次性把超高维数据全部展开到内存。"""
    total = len(X_mmap)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = np.asarray(X_mmap[start:end], dtype=np.float32)
        batch = batch.reshape(len(batch), -1)
        yield start, end, batch


def prepare_feature_representations(feature_name, y_labels):
    """
    使用全量样本，但通过分批标准化 + 分批增量PCA，避免 STFT 触发内存错误。
    返回：
      - X_pca_2d: 用于 PCA 可视化
      - X_tsne_input: 用于 t-SNE 的低维输入
      - y: 全量标签
      - explained_variance: PCA 前两主成分解释方差比
    """
    print(f"Loading {feature_name} with memory mapping...")
    feature_path = os.path.join(FEATURE_DIR, feature_name)
    X_mmap = np.load(feature_path, mmap_mode="r")
    y = y_labels
    n_samples = len(X_mmap)
    flattened_dim = int(np.prod(X_mmap.shape[1:]))
    tsne_dim = min(TSNE_PREP_DIM, flattened_dim, n_samples - 1 if n_samples > 1 else 1)

    print(f"Original shape: {X_mmap.shape}")
    print(f"Flattened shape: ({n_samples}, {flattened_dim})")
    print("Fitting StandardScaler incrementally...")

    scaler = StandardScaler(copy=False)
    for _, _, batch in iter_feature_batches(X_mmap, SCALER_BATCH_SIZE):
        scaler.partial_fit(batch)
        del batch
    gc.collect()

    print("Fitting IncrementalPCA for 2D visualization...")
    pca_2d = IncrementalPCA(n_components=2, batch_size=PCA_BATCH_SIZE)
    for _, _, batch in iter_feature_batches(X_mmap, PCA_BATCH_SIZE):
        batch = scaler.transform(batch).astype(np.float32, copy=False)
        pca_2d.partial_fit(batch)
        del batch
    gc.collect()

    if tsne_dim >= 2:
        print(f"Fitting IncrementalPCA for t-SNE pre-reduction ({tsne_dim} dims)...")
        pca_tsne = IncrementalPCA(n_components=tsne_dim, batch_size=PCA_BATCH_SIZE)
        for _, _, batch in iter_feature_batches(X_mmap, PCA_BATCH_SIZE):
            batch = scaler.transform(batch).astype(np.float32, copy=False)
            pca_tsne.partial_fit(batch)
            del batch
        gc.collect()
    else:
        pca_tsne = None

    tmp_dir = tempfile.gettempdir()
    pca_memmap_path = os.path.join(tmp_dir, f"{feature_name}_pca2d.dat")
    tsne_memmap_path = os.path.join(tmp_dir, f"{feature_name}_tsneprep.dat")
    X_pca_2d = np.memmap(pca_memmap_path, dtype=np.float32, mode="w+", shape=(n_samples, 2))
    X_tsne_input = np.memmap(tsne_memmap_path, dtype=np.float32, mode="w+", shape=(n_samples, tsne_dim))

    print("Transforming full dataset in batches...")
    for start, end, batch in iter_feature_batches(X_mmap, PCA_BATCH_SIZE):
        batch = scaler.transform(batch).astype(np.float32, copy=False)
        X_pca_2d[start:end] = pca_2d.transform(batch).astype(np.float32, copy=False)
        if pca_tsne is not None:
            X_tsne_input[start:end] = pca_tsne.transform(batch).astype(np.float32, copy=False)
        else:
            X_tsne_input[start:end] = batch[:, :tsne_dim]
        del batch
    gc.collect()

    explained_variance = pca_2d.explained_variance_ratio_
    del X_mmap
    gc.collect()

    return X_pca_2d, X_tsne_input, y, explained_variance

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
            X_pca, X_tsne_input, y, explained_variance = prepare_feature_representations(f_name, y_all)

            # 3. PCA (主成分分析)
            print("PCA representations prepared.")
            print(f"PCA explained variance ratio: {explained_variance}")
            
            # 记录数据
            results_data.append({
                "Feature": f_name.replace(".npy", ""),
                "PCA_Component_1_Variance": explained_variance[0],
                "PCA_Component_2_Variance": explained_variance[1],
                "Total_Explained_Variance_2C": np.sum(explained_variance)
            })

            # 4. t-SNE
            print("Running t-SNE on PCA-compressed full dataset (this will take time)...")
            tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, 
                        random_state=RANDOM_STATE, init='pca', learning_rate='auto', n_jobs=-1)
            X_tsne = tsne.fit_transform(np.asarray(X_tsne_input, dtype=np.float32))

            # 5. 可视化
            plot_dim_reduction(X_pca, X_tsne, y, label_map_inv, f_name.replace(".npy", ""))
            
            # 清理内存
            del X_pca
            del X_tsne_input
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
