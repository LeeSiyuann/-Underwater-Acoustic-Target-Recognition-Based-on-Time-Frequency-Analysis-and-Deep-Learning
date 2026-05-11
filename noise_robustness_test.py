import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from deep_learning_models import get_model
from train_comparison import DeepShipDataset, FEATURE_DIR, RESULTS_DIR, DEVICE, TEST_SIZE, RANDOM_SEED
from train_fusion import DualStreamFusionModel, FusionDataset

FINAL_RESULTS_CSV = os.path.join(RESULTS_DIR, "final_comparison_results.csv")
FUSION_RESULTS_CSV = os.path.join(RESULTS_DIR, "final_fusion_results.csv")
SNR_LEVELS = [20, 10, 5, 0, -5, -10]

def inject_noise(features, snr_db):
    """
    在特征矩阵上注入高斯白噪声 (AWGN)
    """
    # 转换为张量以便于计算
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features)
        
    # 计算信号功率
    signal_power = torch.mean(features ** 2)
    
    # 根据 SNR (dB) 计算噪声功率
    # SNR_dB = 10 * log10(signal_power / noise_power)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # 生成噪声并叠加
    noise = torch.randn_like(features) * torch.sqrt(noise_power)
    noisy_features = features + noise
    
    return noisy_features.numpy().astype(np.float32)

def load_all_best_results():
    if not os.path.exists(FINAL_RESULTS_CSV):
        raise FileNotFoundError(f"{FINAL_RESULTS_CSV} not found.")

    results = pd.read_csv(FINAL_RESULTS_CSV)
    if os.path.exists(FUSION_RESULTS_CSV):
        fusion_results = pd.read_csv(FUSION_RESULTS_CSV)
        results = pd.concat([results, fusion_results], ignore_index=True)

    best_rows = results.sort_values("f1", ascending=False).groupby("Model", as_index=False).first()
    return best_rows

def load_and_split_single_feature(feature_name, y_all):
    feature_file = f"{feature_name}.npy" if not feature_name.endswith(".npy") else feature_name
    X = np.load(os.path.join(FEATURE_DIR, feature_file))
    X = (X - X.mean()) / (X.std() + 1e-8)
    _, X_test, _, y_test = train_test_split(
        X, y_all, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_all
    )
    return X_test.astype(np.float32), y_test

def load_and_split_fusion_features(y_all):
    X_mel = np.load(os.path.join(FEATURE_DIR, "X_mel.npy"))
    X_mfcc = np.load(os.path.join(FEATURE_DIR, "X_mfcc.npy"))
    X_mel = (X_mel - X_mel.mean()) / (X_mel.std() + 1e-8)
    X_mfcc = (X_mfcc - X_mfcc.mean()) / (X_mfcc.std() + 1e-8)

    indices = np.arange(len(y_all))
    _, test_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_all)
    return X_mel[test_idx].astype(np.float32), X_mfcc[test_idx].astype(np.float32), y_all[test_idx]

def create_model(model_name, feature_name):
    if model_name == "DualStreamFusionModel":
        return DualStreamFusionModel()
    if model_name in ['RNN', 'LSTM', 'Transformer']:
        X_sample = np.load(os.path.join(FEATURE_DIR, f"{feature_name}.npy"))
        input_size = X_sample.shape[1]
        return get_model(model_name, num_classes=4, input_size=input_size)
    return get_model(model_name, num_classes=4, input_channels=1)

def evaluate_single_stream_model(model, model_name, X_test, y_test, batch_size):
    dataset = DeepShipDataset(X_test, y_test, model_name)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return f1_score(all_labels, all_preds, average='weighted', zero_division=0)

def evaluate_fusion_stream_model(model, X1_test, X2_test, y_test, batch_size):
    dataset = FusionDataset(X1_test, X2_test, y_test)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x1, x2, labels in loader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            outputs = model(x1, x2)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return f1_score(all_labels, all_preds, average='weighted', zero_division=0)

def test_robustness():
    print("Testing Noise Robustness for all models...")
    y_all = np.load(os.path.join(FEATURE_DIR, "y_labels.npy"))
    best_rows = load_all_best_results()
    all_records = []

    for _, row in best_rows.iterrows():
        model_name = row["Model"]
        feature_name = row["Feature"]
        print(f"\nEvaluating {model_name} with best feature {feature_name}...")

        if model_name == "DualStreamFusionModel":
            X1_test, X2_test, y_test = load_and_split_fusion_features(y_all)
            model_path = os.path.join(RESULTS_DIR, "best_fusion_Mel_MFCC.pth")
            model = create_model(model_name, feature_name).to(DEVICE)
            batch_size = 256
        else:
            X_test, y_test = load_and_split_single_feature(feature_name, y_all)
            model_path = os.path.join(RESULTS_DIR, f"best_{model_name}_{feature_name}.pth")
            model = create_model(model_name, feature_name).to(DEVICE)
            batch_size = 128 if "stft" in feature_name.lower() else 256

        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        except Exception as e:
            print(f"Skipping {model_name}: failed to load weights from {model_path}. Error: {e}")
            continue

        model.eval()

        if model_name == "DualStreamFusionModel":
            clean_f1 = evaluate_fusion_stream_model(model, X1_test, X2_test, y_test, batch_size)
        else:
            clean_f1 = evaluate_single_stream_model(model, model_name, X_test, y_test, batch_size)

        all_records.append({
            "Model": model_name,
            "Feature": feature_name,
            "SNR (dB)": "Clean",
            "F1 Score": clean_f1
        })
        print(f"Clean F1: {clean_f1:.4f}")

        for snr in SNR_LEVELS:
            if model_name == "DualStreamFusionModel":
                noisy_X1 = inject_noise(X1_test, snr)
                noisy_X2 = inject_noise(X2_test, snr)
                noisy_f1 = evaluate_fusion_stream_model(model, noisy_X1, noisy_X2, y_test, batch_size)
            else:
                noisy_X = inject_noise(X_test, snr)
                noisy_f1 = evaluate_single_stream_model(model, model_name, noisy_X, y_test, batch_size)

            all_records.append({
                "Model": model_name,
                "Feature": feature_name,
                "SNR (dB)": snr,
                "F1 Score": noisy_f1
            })
            print(f"SNR {snr} dB F1: {noisy_f1:.4f}")

    df = pd.DataFrame(all_records)
    detailed_csv = os.path.join(RESULTS_DIR, "robustness_all_models.csv")
    df.to_csv(detailed_csv, index=False)

    numeric_df = df[df["SNR (dB)"] != "Clean"].copy()
    numeric_df["SNR (dB)"] = numeric_df["SNR (dB)"].astype(int)

    plt.figure(figsize=(14, 8))
    sns.lineplot(
        data=numeric_df,
        x="SNR (dB)",
        y="F1 Score",
        hue="Model",
        style="Feature",
        markers=True,
        dashes=False,
        linewidth=2.2
    )
    plt.gca().invert_xaxis()
    plt.title("Noise Robustness Comparison Across All Models")
    plt.xlabel("Signal-to-Noise Ratio (dB) - Lower is noisier")
    plt.ylabel("F1 Score")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    curve_path = os.path.join(RESULTS_DIR, "robustness_curve_all_models")
    plt.savefig(curve_path + ".png")
    plt.savefig(curve_path + ".svg")
    plt.savefig(curve_path + ".pdf")
    plt.close()

    heatmap_df = numeric_df.pivot(index="Model", columns="SNR (dB)", values="F1 Score")
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_df, annot=True, fmt=".4f", cmap="YlOrRd", cbar_kws={'label': 'F1 Score'})
    plt.title("Noise Robustness Heatmap Across All Models")
    plt.tight_layout()
    heatmap_path = os.path.join(RESULTS_DIR, "robustness_heatmap_all_models")
    plt.savefig(heatmap_path + ".png")
    plt.savefig(heatmap_path + ".svg")
    plt.savefig(heatmap_path + ".pdf")
    plt.close()

    print(f"\nSaved detailed robustness results to {detailed_csv}")
    print(f"Saved combined robustness curve to {curve_path}")
    print(f"Saved robustness heatmap to {heatmap_path}")

if __name__ == "__main__":
    test_robustness()
