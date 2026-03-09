import subprocess
import sys
import os
import time

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # 使用当前 Python 环境执行脚本
    # sys.executable 获取当前解释器路径
    try:
        # check=True 会在命令返回非零退出码时抛出 CalledProcessError
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError running {script_name}: {e}")
        print("Pipeline aborted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error running {script_name}: {e}")
        print("Pipeline aborted.")
        sys.exit(1)
        
    end_time = time.time()
    print(f"\n{script_name} completed in {end_time - start_time:.2f} seconds.")

def main():
    print("Starting DeepShip Analysis Pipeline...")
    pipeline_start = time.time()

    # 1. 数据预处理
    # 将原始 WAV 转换为 .npy (X_audio_32k.npy, y_labels.npy)
    if os.path.exists("preprocess_deepship_32k.py"):
        run_script("preprocess_deepship_32k.py")
    else:
        print("Warning: preprocess_deepship_32k.py not found. Skipping step 1.")

    # 2. 特征提取
    # 生成 X_mel.npy, X_mfcc.npy, X_cqt.npy, X_stft.npy
    if os.path.exists("feature_extraction_comparison.py"):
        run_script("feature_extraction_comparison.py")
    else:
        print("Warning: feature_extraction_comparison.py not found. Skipping step 2.")

    # 3. 降维与可视化分析
    # 生成 PCA/t-SNE 图表和显存占用报告
    if os.path.exists("dimensionality_reduction_analysis.py"):
        run_script("dimensionality_reduction_analysis.py")
    else:
        print("Warning: dimensionality_reduction_analysis.py not found. Skipping step 3.")

    # 4. 模型训练与对比
    # 训练 CNN/ResNet/RNN/Transformer 等模型，生成最终评估报表
    if os.path.exists("train_comparison.py"):
        run_script("train_comparison.py")
    else:
        print("Warning: train_comparison.py not found. Skipping step 4.")

    pipeline_end = time.time()
    total_time = pipeline_end - pipeline_start
    print(f"\n{'='*60}")
    print(f"Pipeline finished successfully in {total_time/60:.2f} minutes.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
