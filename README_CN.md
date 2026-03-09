# DeepShip 船舶噪声分类项目

本项目实现了一个完整的深度学习流水线，用于基于 DeepShip 数据集对船舶类型（Cargo, Tanker, Tug, Passengership）进行分类。项目涵盖了从原始音频预处理、特征提取、降维可视化分析到多模型对比训练的全过程。

## 项目结构

```
.
├── preprocess_deepship_32k.py          # 数据预处理脚本
├── feature_extraction_comparison.py    # 特征提取 (STFT, Mel, MFCC, CQT)
├── dimensionality_reduction_analysis.py # PCA & t-SNE 降维分析
├── deep_learning_models.py             # PyTorch 模型定义
├── train_comparison.py                 # 模型训练与评估主程序
├── run_pipeline.py                     # 全流程自动化运行脚本
├── DeepShip-main/                      # 原始数据   
├── processed_data/                     # 预处理后的音频数据(.npy)
├── feature_data/                       # 四种特征提取后的特征数据(.npy)和特征提取对比图
├── dim_reduction_results_full/         # 降维分析结果输出目录
└── dl_comparison_results_gpu/          # 模型训练结果输出目录
```

## 1. 数据预处理 (`preprocess_deepship_32k.py`)

将原始 WAV 音频文件转换为适合深度学习的标准格式。

*   **目标采样率**: 32 kHz
*   **切片时长**: 2.0 秒 (64,000 采样点)
*   **重叠率**: 50%
*   **滤波**: 巴特沃斯带通滤波器 (1 Hz - 12 kHz)
*   **归一化**: 最大幅值归一化
*   **输出**: `X_audio_32k.npy` 和 `y_labels.npy`

## 2. 特征提取 (`feature_extraction_comparison.py`)

从预处理后的音频中提取四种不同的时频特征：

1.  **STFT (短时傅里叶变换)**: 高分辨率线性频率图。
2.  **Mel Spectrogram (梅尔声谱图)**: 模拟人耳听觉的对数频率图 (CNN 标准输入)。
3.  **MFCC (梅尔频率倒谱系数)**: 紧凑的去相关特征。
4.  **CQT (常数 Q 变换)**: 低频高分辨率的对数频率图 (非常适合分析引擎线谱)。

## 3. 降维分析 (`dimensionality_reduction_analysis.py`)

在训练前使用无监督方法可视化特征的可分性。

*   **方法**: PCA (主成分分析) 和 t-SNE。
*   **指标**: 计算并保存 PCA 解释方差比。
*   **输出**: 高清散点图和 CSV/TXT 格式的方差报告。

## 4. 深度学习模型 (`deep_learning_models.py`)

使用 PyTorch 实现了多种神经网络架构：

*   **CNN 类**:
    *   `SimpleCNN`: 轻量级基准模型。
    *   `ResNet18`, `ResNet34`, `ResNet50`: 标准深度残差网络 (已适配单通道输入)。
*   **RNN 类**:
    *   `RNN`: 标准循环神经网络。
    *   `LSTM`: 双向长短期记忆网络。
*   **Transformer 类**:
    *   `Transformer`: 标准 Transformer Encoder (用于序列处理)。

## 5. 训练与评估 (`train_comparison.py`)

专为高性能 GPU (如 RTX 5090) 设计的核心训练脚本。

*   **优化**:
    *   **Batch Size**: 默认 2048 (针对 STFT 自动降级为 512 以管理显存)。
    *   **Num Workers**: 根据 CPU 核心数自动优化 (最高 16)。
    *   **混合精度**: 支持 CUDA 加速。
*   **评估指标**:
    *   准确率 (Accuracy), 精确率 (Precision Weighted), 召回率 (Recall Weighted), F1 分数 (F1 Score Weighted), ROC AUC。
*   **可视化**:
    *   混淆矩阵 (Confusion Matrices)。
    *   PR 曲线 (Precision-Recall Curves)。
    *   全方位的训练历史曲线 (Loss, Acc, F1 等)。
*   **日志**:
    *   仅控制台输出 (禁用文件日志)。
    *   保存训练日志到 CSV 并记录最大显存占用。

## 使用说明

### 1. 自动化流水线 (推荐)

按顺序运行整个工作流：

```bash
python run_pipeline.py
```

### 2. 手动分步执行

你也可以单独运行每个步骤：

```bash
# 步骤 1: 音频预处理
python preprocess_deepship_32k.py

# 步骤 2: 特征提取
python feature_extraction_comparison.py

# 步骤 3: 特征分析
python dimensionality_reduction_analysis.py

# 步骤 4: 模型训练
python train_comparison.py
```

## 环境要求

*   Python 3.8+
*   PyTorch (支持 CUDA)
*   Torchaudio / Torchvision
*   Librosa
*   Scikit-learn
*   Matplotlib / Seaborn
*   Pandas / NumPy
*   TQDM

## RTX Pro 6000 / 大显存 GPU 优化说明

*   训练脚本默认配置 **Batch Size 为 2048**，以充分利用RTX Pro 6000 96GB 的显存。
*   对于显存占用极高的特征 (如 **STFT**)，Batch Size 会自动限制为 **512**。
*   如果遇到 OOM (显存不足) 错误，请在 `train_comparison.py` 中减小 `BATCH_SIZE`。
