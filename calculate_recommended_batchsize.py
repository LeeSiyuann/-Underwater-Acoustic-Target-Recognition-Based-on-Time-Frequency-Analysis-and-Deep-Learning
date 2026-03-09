import pandas as pd
import os

# 配置
RESULTS_DIR = "dl_comparison_results_gpu"
INPUT_CSV = os.path.join(RESULTS_DIR, "gpu_memory_usage.csv")
OUTPUT_CSV = "recommend_batchsize.csv"

# 目标显存 (20GB)
TARGET_VRAM_GB = 20
TARGET_VRAM_MB = TARGET_VRAM_GB * 1024

# 原始 Batch Size (脚本中是 256)
# 注意：这里我们假设 gpu_memory_usage.csv 是基于 batch_size=256 跑出来的结果
# 如果你使用了其他 batch size，请修改这里
ORIGINAL_BATCH_SIZE = 256 

# 安全系数 (保留一部分显存给 PyTorch 上下文和其他开销)
SAFETY_FACTOR = 0.85

def calculate_recommended_batchsize():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    print(f"Loading GPU usage data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    recommendations = []

    for index, row in df.iterrows():
        feature = row['Feature']
        model = row['Model']
        max_vram_mb = row['Max_GPU_Memory_MB']
        
        # 估算单个样本的显存占用 (这是一个粗略估计，实际上显存占用包含固定开销 + 动态开销)
        # 假设显存占用与 Batch Size 线性相关：VRAM = Base_VRAM + (Per_Sample_VRAM * Batch_Size)
        # 为了安全起见，我们直接按比例缩放，并留出余量
        
        if max_vram_mb > 0:
            # 计算每 MB 显存能容纳的 Batch Size 比例
            # 推荐 Batch Size = (目标显存 / 当前显存) * 当前 Batch Size * 安全系数
            ratio = TARGET_VRAM_MB / max_vram_mb
            recommended_bs = int(ORIGINAL_BATCH_SIZE * ratio * SAFETY_FACTOR)
            
            # 向下取整到最近的 32 的倍数 (通常对 GPU 友好)
            recommended_bs = max(32, (recommended_bs // 32) * 32)
        else:
            recommended_bs = 32 # 异常情况默认值

        recommendations.append({
            "Feature": feature,
            "Model": model,
            "Measured_VRAM_MB (BS=256)": f"{max_vram_mb:.2f}",
            "Target_VRAM_MB": TARGET_VRAM_MB,
            "Recommended_BatchSize": recommended_bs
        })

    df_rec = pd.DataFrame(recommendations)
    df_rec.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully generated {OUTPUT_CSV}")
    print(df_rec)

if __name__ == "__main__":
    calculate_recommended_batchsize()
