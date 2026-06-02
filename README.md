# DeepShip Classification Project

Language: [English](./README.md) | [中文](./README_CN.md)

This repository contains a complete undergraduate-thesis-oriented experiment pipeline for underwater acoustic target recognition on the DeepShip dataset. It covers:

- audio preprocessing
- time-frequency feature extraction
- PCA/t-SNE visualization
- 28 single-stream deep learning experiments
- 1 Mel+MFCC fusion experiment
- global comparison plots
- complexity and inference-time analysis
- feature-level AWGN robustness testing
- ablation and explainability experiments
- paper-ready figure generation

## Current Verified Status

The main experiment outputs are already present in the repository:

- feature data in `processed_data/` and `feature_data/`
- dimensionality reduction results in `dim_reduction_results/`
- single-stream and fusion outputs in `dl_comparison_results_gpu/`
- ablation results in `dl_comparison_results_gpu/ablation_results/`
- explainability results in `dl_comparison_results_gpu/explainability_results/`
- paper figures in `dl_comparison_results_gpu/paper_figures/`

Current best overall result from [final_comparison_results_with_fusion.csv](./dl_comparison_results_gpu/final_comparison_results_with_fusion.csv):

- model: `ResNet18`
- feature: `X_mfcc`
- `accuracy = 0.9982332155477032`
- `f1 = 0.9982335599857426`
- `roc_auc = 0.9999552089829588`

Recommended result wording for thesis writing and presentations:

- the fusion model is in the top performance tier and can be described as a near-optimal fusion solution
- robustness results are based on **feature-level AWGN perturbation**
- frequency-band ablation supports explainability and sensitive-band analysis
- the explainability section can be built around the currently saved Grad-CAM, sequence saliency, and fixed-feature/fixed-model results

## Project Structure

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

## Core Pipeline

### 1. Preprocessing

Script: [preprocess_deepship_32k.py](./preprocess_deepship_32k.py)

- target sampling rate: `32 kHz`
- segment length: `2.0 s`
- overlap: `50%`
- filtering: Butterworth bandpass `1 Hz - 12 kHz`
- outputs:
  - [processed_data/X_audio_32k.npy](./processed_data/X_audio_32k.npy)
  - [processed_data/y_labels.npy](./processed_data/y_labels.npy)

### 2. Feature Extraction

Script: [feature_extraction_comparison.py](./feature_extraction_comparison.py)

Extracted features:

- `STFT`
- `Mel`
- `MFCC`
- `CQT`

Example outputs:

- [feature_data/X_stft.npy](./feature_data/X_stft.npy)
- [feature_data/X_mel.npy](./feature_data/X_mel.npy)
- [feature_data/X_mfcc.npy](./feature_data/X_mfcc.npy)
- [feature_data/X_cqt.npy](./feature_data/X_cqt.npy)
- [feature_data/comparison_Cargo.png](./feature_data/comparison_Cargo.png)

### 3. Dimensionality Reduction

Script: [dimensionality_reduction_analysis.py](./dimensionality_reduction_analysis.py)

Verified outputs:

- [dim_reduction_results/analysis_summary.txt](./dim_reduction_results/analysis_summary.txt)
- [dim_reduction_results/pca_variance_analysis.csv](./dim_reduction_results/pca_variance_analysis.csv)
- [dim_reduction_results/dr_X_mel_full.png](./dim_reduction_results/dr_X_mel_full.png)
- [dim_reduction_results/dr_X_mfcc_full.png](./dim_reduction_results/dr_X_mfcc_full.png)
- [dim_reduction_results/dr_X_cqt_full.png](./dim_reduction_results/dr_X_cqt_full.png)
- [dim_reduction_results/dr_X_stft_full.png](./dim_reduction_results/dr_X_stft_full.png)

### 4. Models

Script: [deep_learning_models.py](./deep_learning_models.py)

Implemented models:

- `SimpleCNN`
- `ResNet18`
- `ResNet34`
- `ResNet50`
- `RNN`
- `LSTM`
- `Transformer`
- `DualStreamFusionModel` in [train_fusion.py](./train_fusion.py)

### 5. Main Training

Scripts:

- [train_comparison.py](./train_comparison.py)
- [resume_training.py](./resume_training.py)

Main settings:

- default batch size: `2048`
- STFT batch size: `512`
- `num_workers`: up to `16`
- outputs per experiment:
  - `training_log_*.csv`
  - `history_*.png/.svg/.pdf`
  - `cm_*.png/.svg/.pdf`
  - `pr_curve_*.png/.svg/.pdf`
  - `best_*.pth`

Main result tables:

- [dl_comparison_results_gpu/final_comparison_results.csv](./dl_comparison_results_gpu/final_comparison_results.csv)
- [dl_comparison_results_gpu/final_fusion_results.csv](./dl_comparison_results_gpu/final_fusion_results.csv)
- [dl_comparison_results_gpu/final_comparison_results_with_fusion.csv](./dl_comparison_results_gpu/final_comparison_results_with_fusion.csv)

### 6. Global Comparison

Scripts:

- [plot_global_comparison.py](./plot_global_comparison.py)
- [evaluate_complexity.py](./evaluate_complexity.py)

Key outputs:

- [dl_comparison_results_gpu/global_f1_barplot.png](./dl_comparison_results_gpu/global_f1_barplot.png)
- [dl_comparison_results_gpu/global_f1_heatmap.png](./dl_comparison_results_gpu/global_f1_heatmap.png)
- [dl_comparison_results_gpu/global_metric_model_feature_heatmaps.png](./dl_comparison_results_gpu/global_metric_model_feature_heatmaps.png)
- [dl_comparison_results_gpu/model_complexity.csv](./dl_comparison_results_gpu/model_complexity.csv)
- [dl_comparison_results_gpu/model_parameters_comparison.png](./dl_comparison_results_gpu/model_parameters_comparison.png)
- [dl_comparison_results_gpu/model_inference_time_comparison.png](./dl_comparison_results_gpu/model_inference_time_comparison.png)

### 7. Robustness

Script: [noise_robustness_test.py](./noise_robustness_test.py)

Verified outputs:

- [dl_comparison_results_gpu/robustness_all_models.csv](./dl_comparison_results_gpu/robustness_all_models.csv)
- [dl_comparison_results_gpu/robustness_curve_all_models.png](./dl_comparison_results_gpu/robustness_curve_all_models.png)
- [dl_comparison_results_gpu/robustness_heatmap_all_models.png](./dl_comparison_results_gpu/robustness_heatmap_all_models.png)

Note:

- this uses a **feature-level AWGN robustness setting**
- it can be used directly for robustness comparison and stability discussion in the thesis

### 8. Ablation and Explainability

Scripts:

- [ablation_experiments.py](./ablation_experiments.py)
- [explainability_experiments.py](./explainability_experiments.py)

Key outputs:

- [dl_comparison_results_gpu/ablation_results/ablation_summary_results.csv](./dl_comparison_results_gpu/ablation_results/ablation_summary_results.csv)
- [dl_comparison_results_gpu/ablation_results/ablation_fusion_structure_f1.png](./dl_comparison_results_gpu/ablation_results/ablation_fusion_structure_f1.png)
- [dl_comparison_results_gpu/ablation_results/ablation_lstm_hyperparams_heatmap.png](./dl_comparison_results_gpu/ablation_results/ablation_lstm_hyperparams_heatmap.png)
- [dl_comparison_results_gpu/ablation_results/ablation_transformer_hyperparams_heatmap.png](./dl_comparison_results_gpu/ablation_results/ablation_transformer_hyperparams_heatmap.png)
- [dl_comparison_results_gpu/explainability_results/fixed_feature_model_effect.csv](./dl_comparison_results_gpu/explainability_results/fixed_feature_model_effect.csv)
- [dl_comparison_results_gpu/explainability_results/fixed_model_feature_effect.csv](./dl_comparison_results_gpu/explainability_results/fixed_model_feature_effect.csv)
- [dl_comparison_results_gpu/explainability_results/frequency_band_ablation_results.csv](./dl_comparison_results_gpu/explainability_results/frequency_band_ablation_results.csv)

### 9. Paper Figures

Script: [generate_paper_figures.py](./generate_paper_figures.py)

Key outputs:

- [dl_comparison_results_gpu/paper_figures/workflow_overview.png](./dl_comparison_results_gpu/paper_figures/workflow_overview.png)
- [dl_comparison_results_gpu/paper_figures/feature_comparison_montage.png](./dl_comparison_results_gpu/paper_figures/feature_comparison_montage.png)
- [dl_comparison_results_gpu/paper_figures/dimensionality_reduction_montage.png](./dl_comparison_results_gpu/paper_figures/dimensionality_reduction_montage.png)
- [dl_comparison_results_gpu/paper_figures/paper_figures_index.csv](./dl_comparison_results_gpu/paper_figures/paper_figures_index.csv)
- [dl_comparison_results_gpu/paper_figures/model_architecture_index.csv](./dl_comparison_results_gpu/paper_figures/model_architecture_index.csv)

## Recommended Execution Order

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

## Environment

- Python `3.8+`
- PyTorch with CUDA
- Torchvision / Torchaudio
- Librosa
- Scikit-learn
- Matplotlib / Seaborn
- Pandas / NumPy

## GPU Notes

- default batch size is configured for high-VRAM GPUs
- STFT uses a lower batch size to avoid OOM
- if memory still overflows, reduce the batch size in `train_comparison.py`
