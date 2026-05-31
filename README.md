# DeepShip Classification Project

🌐 Language: [English](./README.md) | [中文](./README_CN.md)

This project implements a comprehensive deep learning pipeline for classifying ship types (Cargo, Tanker, Tug, Passengership) using the DeepShip dataset. It covers the entire workflow from raw audio preprocessing to feature extraction, dimensionality reduction analysis, and comparative model training.

## Project Structure

```
.
├── preprocess_deepship_32k.py          # Data preprocessing script
├── feature_extraction_comparison.py    # Feature extraction (STFT, Mel, MFCC, CQT)
├── dimensionality_reduction_analysis.py # PCA & t-SNE analysis
├── deep_learning_models.py             # PyTorch model definitions
├── train_comparison.py                 # Main training and evaluation loop
├── resume_training.py                  # Resume training and OOM protection
├── train_fusion.py                     # Multi-feature fusion (Mel+MFCC) experiment
├── ablation_experiments.py             # Structure ablation experiments
├── explainability_experiments.py       # Explainability experiments (Grad-CAM/attention/band ablation)
├── plot_global_comparison.py           # Global comparison charts generation
├── evaluate_complexity.py              # Model complexity & inference time evaluation
├── noise_robustness_test.py            # Noise robustness & SNR testing
├── generate_paper_figures.py           # Workflow/model diagrams and paper-ready montages
├── run_pipeline.py                     # Automation script for the full pipeline
├── DeepShip-main/                      # Original dataset directory
├── processed_data/                     # Preprocessed audio data (.npy)
├── feature_data/                       # Directory for preprocessed features (.npy)
├── dim_reduction_results/              # Output for dimensionality reduction analysis
└── dl_comparison_results_gpu/          # Output for model training results
```

## 1. Data Preprocessing (`preprocess_deepship_32k.py`)

Converts raw WAV audio files into a standardized format suitable for deep learning.

*   **Target Sampling Rate**: 32 kHz
*   **Segment Length**: 2.0 seconds (64,000 samples)
*   **Overlap**: 50%
*   **Filtering**: Butterworth bandpass filter (1 Hz - 12 kHz)
*   **Normalization**: Max amplitude normalization
*   **Output**: `X_audio_32k.npy` and `y_labels.npy`

## 2. Feature Extraction (`feature_extraction_comparison.py`)

Extracts four distinct time-frequency features from the preprocessed audio:

1.  **STFT (Short-Time Fourier Transform)**: High-resolution linear frequency map.
2.  **Mel Spectrogram**: Log-scale frequency map mimicking human hearing (Standard for CNNs).
3.  **MFCC (Mel-Frequency Cepstral Coefficients)**: Compact decorrelated features.
4.  **CQT (Constant-Q Transform)**: Log-frequency scale with high resolution at low frequencies (Ideal for engine line spectra).

## 3. Dimensionality Reduction Analysis (`dimensionality_reduction_analysis.py`)

Performs unsupervised analysis to visualize feature separability before training.

*   **Methods**: PCA (Principal Component Analysis) and t-SNE.
*   **Metrics**: Calculates and saves PCA explained variance ratios.
*   **Output**: High-resolution scatter plots and variance reports in CSV/TXT format.

## 4. Deep Learning Models (`deep_learning_models.py`)

Implements a variety of neural network architectures using PyTorch:

*   **CNNs**:
    *   `SimpleCNN`: Lightweight baseline.
    *   `ResNet18`, `ResNet34`, `ResNet50`: Standard deep residual networks (adapted for 1-channel input).
*   **RNNs**:
    *   `RNN`: Vanilla Recurrent Neural Network.
    *   `LSTM`: Bidirectional Long Short-Term Memory network.
*   **Transformers**:
    *   `Transformer`: Standard Transformer Encoder for sequence processing.

## 5. Training & Evaluation (`train_comparison.py`)

The core training script designed for high-performance GPUs (e.g., RTX 5090).

*   **Optimization**:
    *   **Batch Size**: 2048 (Dynamic adjustment to 512 for STFT to manage VRAM).
    *   **Num Workers**: Automatically optimized based on CPU core count (up to 16).
    *   **Mixed Precision**: Ready for CUDA acceleration.
*   **Metrics**:
    *   Accuracy, Precision (Weighted), Recall (Weighted), F1 Score (Weighted), ROC AUC.
*   **Visualization**:
    *   Confusion Matrices.
    *   Precision-Recall (PR) Curves.
    *   Comprehensive Training History plots (Loss, Acc, F1, etc.).
*   **Logging**:
    *   Console-only logging (file logging disabled).
    *   Saves training logs to CSV and max GPU memory usage stats.

## Usage

### 1. Automated Pipeline (Recommended)

To run the entire workflow sequentially:

```bash
python run_pipeline.py
```

### 2. Manual Execution

You can also run each step individually:

```bash
# Step 1: Preprocess Audio
python preprocess_deepship_32k.py

# Step 2: Extract Features
python feature_extraction_comparison.py

# Step 3: Analyze Features
python dimensionality_reduction_analysis.py

# Step 4: Train Models
python train_comparison.py

# Step 5: (Optional) Resume training or run fusion experiment
python resume_training.py
python train_fusion.py

# Step 6: Global Evaluation & Analysis
python plot_global_comparison.py
python evaluate_complexity.py
python noise_robustness_test.py

# Step 7: Structure Ablation, Explainability, and Paper Figures
python ablation_experiments.py
python explainability_experiments.py
python generate_paper_figures.py
```

## Requirements

*   Python 3.8+
*   PyTorch (CUDA supported)
*   Torchaudio / Torchvision
*   Librosa
*   Scikit-learn
*   Matplotlib / Seaborn
*   Pandas / NumPy
*   TQDM

## Notes for RTX Pro 6000 / High-VRAM GPUs

*   The training script is configured with a default **Batch Size of 2048** to fully utilize 96GB VRAM.
*   For memory-intensive features like **STFT**, the batch size automatically throttles down to **512**.
*   If you encounter OOM (Out of Memory) errors, reduce `BATCH_SIZE` in `train_comparison.py`.
