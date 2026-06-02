# DeepShip 船舶噪声分类项目

语言: [English](./README.md) | [中文](./README_CN.md)

本仓库面向本科毕业设计论文整理，提供一套完整的 DeepShip 水下声学目标识别实验链路，覆盖：

- 原始音频预处理
- STFT / Mel / MFCC / CQT 特征提取
- PCA / t-SNE 特征可视化
- 28 组单流深度学习对比实验
- 1 组 Mel+MFCC 融合实验
- 全局性能汇总与复杂度评估
- 特征层 AWGN 鲁棒性测试
- 消融与可解释性实验
- 论文图表与拼图版式生成

## 当前已核验的结果状态

当前仓库中已经存在以下主要结果目录：

- `processed_data/` 与 `feature_data/`
- `dim_reduction_results/`
- `dl_comparison_results_gpu/`
- `dl_comparison_results_gpu/ablation_results/`
- `dl_comparison_results_gpu/explainability_results/`
- `dl_comparison_results_gpu/paper_figures/`

当前全表最优结果来自 [final_comparison_results_with_fusion.csv](./dl_comparison_results_gpu/final_comparison_results_with_fusion.csv)：

- 模型：`ResNet18`
- 特征：`X_mfcc`
- `accuracy = 0.9982332155477032`
- `f1 = 0.9982335599857426`
- `roc_auc = 0.9999552089829588`

推荐在论文与汇报中统一采用以下结果表述：

- 融合模型结果处于第一梯队，适合表述为“接近最优的融合方案”
- 鲁棒性实验采用 **特征层 AWGN 干扰测试**
- 频带遮挡实验用于解释性分析和敏感频带讨论
- 可解释性部分以当前已落盘的 Grad-CAM、序列显著图和固定特征/固定模型分析结果为主

## 项目结构

```text
.
├── preprocess_deepship_32k.py
├── feature_extraction_comparison.py
├── dimensionality_reduction_analysis.py
├── deep_learning_models.py
├── train_comparison.py
├── resume_training.py
├── train_fusion.py
├── plot_global_comparison.py
├── evaluate_complexity.py
├── noise_robustness_test.py
├── ablation_experiments.py
├── explainability_experiments.py
├── generate_paper_figures.py
├── processed_data/
├── feature_data/
├── dim_reduction_results/
└── dl_comparison_results_gpu/
```

## 核心流程

### 1. 数据预处理

脚本：[preprocess_deepship_32k.py](./preprocess_deepship_32k.py)

- 目标采样率：`32 kHz`
- 切片时长：`2.0 s`
- 重叠率：`50%`
- 滤波范围：`1 Hz - 12 kHz`
- 输出：
  - [processed_data/X_audio_32k.npy](./processed_data/X_audio_32k.npy)
  - [processed_data/y_labels.npy](./processed_data/y_labels.npy)

### 2. 特征提取

脚本：[feature_extraction_comparison.py](./feature_extraction_comparison.py)

已提取特征：

- `STFT`
- `Mel`
- `MFCC`
- `CQT`

示例输出：

- [feature_data/X_stft.npy](./feature_data/X_stft.npy)
- [feature_data/X_mel.npy](./feature_data/X_mel.npy)
- [feature_data/X_mfcc.npy](./feature_data/X_mfcc.npy)
- [feature_data/X_cqt.npy](./feature_data/X_cqt.npy)
- [feature_data/comparison_Cargo.png](./feature_data/comparison_Cargo.png)

### 3. 降维分析

脚本：[dimensionality_reduction_analysis.py](./dimensionality_reduction_analysis.py)

当前真实输出：

- [dim_reduction_results/analysis_summary.txt](./dim_reduction_results/analysis_summary.txt)
- [dim_reduction_results/pca_variance_analysis.csv](./dim_reduction_results/pca_variance_analysis.csv)
- [dim_reduction_results/dr_X_mel_full.png](./dim_reduction_results/dr_X_mel_full.png)
- [dim_reduction_results/dr_X_mfcc_full.png](./dim_reduction_results/dr_X_mfcc_full.png)
- [dim_reduction_results/dr_X_cqt_full.png](./dim_reduction_results/dr_X_cqt_full.png)
- [dim_reduction_results/dr_X_stft_full.png](./dim_reduction_results/dr_X_stft_full.png)

### 4. 深度学习模型

脚本：[deep_learning_models.py](./deep_learning_models.py)

当前实现模型：

- `SimpleCNN`
- `ResNet18`
- `ResNet34`
- `ResNet50`
- `RNN`
- `LSTM`
- `Transformer`
- `DualStreamFusionModel`（定义与训练位于 [train_fusion.py](./train_fusion.py)）

### 5. 主训练与续训

脚本：

- [train_comparison.py](./train_comparison.py)
- [resume_training.py](./resume_training.py)

主要设置：

- 默认批大小：`2048`
- STFT 批大小：`512`
- `num_workers`：最高 `16`
- 每组实验输出：
  - `training_log_*.csv`
  - `history_*.png/.svg/.pdf`
  - `cm_*.png/.svg/.pdf`
  - `pr_curve_*.png/.svg/.pdf`
  - `best_*.pth`

主结果表：

- [dl_comparison_results_gpu/final_comparison_results.csv](./dl_comparison_results_gpu/final_comparison_results.csv)
- [dl_comparison_results_gpu/final_fusion_results.csv](./dl_comparison_results_gpu/final_fusion_results.csv)
- [dl_comparison_results_gpu/final_comparison_results_with_fusion.csv](./dl_comparison_results_gpu/final_comparison_results_with_fusion.csv)

### 6. 全局对比与复杂度评估

脚本：

- [plot_global_comparison.py](./plot_global_comparison.py)
- [evaluate_complexity.py](./evaluate_complexity.py)

关键输出：

- [dl_comparison_results_gpu/global_f1_barplot.png](./dl_comparison_results_gpu/global_f1_barplot.png)
- [dl_comparison_results_gpu/global_f1_heatmap.png](./dl_comparison_results_gpu/global_f1_heatmap.png)
- [dl_comparison_results_gpu/global_metric_model_feature_heatmaps.png](./dl_comparison_results_gpu/global_metric_model_feature_heatmaps.png)
- [dl_comparison_results_gpu/model_complexity.csv](./dl_comparison_results_gpu/model_complexity.csv)
- [dl_comparison_results_gpu/model_parameters_comparison.png](./dl_comparison_results_gpu/model_parameters_comparison.png)
- [dl_comparison_results_gpu/model_inference_time_comparison.png](./dl_comparison_results_gpu/model_inference_time_comparison.png)

### 7. 鲁棒性测试

脚本：[noise_robustness_test.py](./noise_robustness_test.py)

当前输出：

- [dl_comparison_results_gpu/robustness_all_models.csv](./dl_comparison_results_gpu/robustness_all_models.csv)
- [dl_comparison_results_gpu/robustness_curve_all_models.png](./dl_comparison_results_gpu/robustness_curve_all_models.png)
- [dl_comparison_results_gpu/robustness_heatmap_all_models.png](./dl_comparison_results_gpu/robustness_heatmap_all_models.png)

说明：

- 当前结果统一采用 **特征层 AWGN 干扰** 的实验设置
- 可直接用于论文中的鲁棒性对比与稳定性分析

### 8. 消融与可解释性实验

脚本：

- [ablation_experiments.py](./ablation_experiments.py)
- [explainability_experiments.py](./explainability_experiments.py)

关键输出：

- [dl_comparison_results_gpu/ablation_results/ablation_summary_results.csv](./dl_comparison_results_gpu/ablation_results/ablation_summary_results.csv)
- [dl_comparison_results_gpu/ablation_results/ablation_fusion_structure_f1.png](./dl_comparison_results_gpu/ablation_results/ablation_fusion_structure_f1.png)
- [dl_comparison_results_gpu/ablation_results/ablation_lstm_hyperparams_heatmap.png](./dl_comparison_results_gpu/ablation_results/ablation_lstm_hyperparams_heatmap.png)
- [dl_comparison_results_gpu/ablation_results/ablation_transformer_hyperparams_heatmap.png](./dl_comparison_results_gpu/ablation_results/ablation_transformer_hyperparams_heatmap.png)
- [dl_comparison_results_gpu/explainability_results/fixed_feature_model_effect.csv](./dl_comparison_results_gpu/explainability_results/fixed_feature_model_effect.csv)
- [dl_comparison_results_gpu/explainability_results/fixed_model_feature_effect.csv](./dl_comparison_results_gpu/explainability_results/fixed_model_feature_effect.csv)
- [dl_comparison_results_gpu/explainability_results/frequency_band_ablation_results.csv](./dl_comparison_results_gpu/explainability_results/frequency_band_ablation_results.csv)

### 9. 论文图表生成

脚本：[generate_paper_figures.py](./generate_paper_figures.py)

关键输出：

- [dl_comparison_results_gpu/paper_figures/workflow_overview.png](./dl_comparison_results_gpu/paper_figures/workflow_overview.png)
- [dl_comparison_results_gpu/paper_figures/feature_comparison_montage.png](./dl_comparison_results_gpu/paper_figures/feature_comparison_montage.png)
- [dl_comparison_results_gpu/paper_figures/dimensionality_reduction_montage.png](./dl_comparison_results_gpu/paper_figures/dimensionality_reduction_montage.png)
- [dl_comparison_results_gpu/paper_figures/paper_figures_index.csv](./dl_comparison_results_gpu/paper_figures/paper_figures_index.csv)
- [dl_comparison_results_gpu/paper_figures/model_architecture_index.csv](./dl_comparison_results_gpu/paper_figures/model_architecture_index.csv)

## 推荐执行顺序

```bash
python preprocess_deepship_32k.py
python feature_extraction_comparison.py
python dimensionality_reduction_analysis.py
python train_comparison.py
python resume_training.py
python train_fusion.py
python plot_global_comparison.py
python evaluate_complexity.py
python noise_robustness_test.py
python ablation_experiments.py
python explainability_experiments.py
python generate_paper_figures.py
```

## 环境要求

- Python `3.8+`
- 支持 CUDA 的 PyTorch
- Torchvision / Torchaudio
- Librosa
- Scikit-learn
- Matplotlib / Seaborn
- Pandas / NumPy

## GPU 使用说明

- 默认批大小针对大显存 GPU 设置
- STFT 会自动使用更小批大小以降低 OOM 风险
- 若仍出现显存不足，可在 `train_comparison.py` 中进一步减小批大小
