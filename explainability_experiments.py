import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from deep_learning_models import get_model
from train_comparison import (
    DeepShipDataset,
    FEATURE_DIR,
    RESULTS_DIR,
    DEVICE,
    TEST_SIZE,
    RANDOM_SEED,
    evaluate_model,
)
from train_fusion import DualStreamFusionModel, FusionDataset


EXPLAIN_DIR = os.path.join(RESULTS_DIR, "explainability_results")
os.makedirs(EXPLAIN_DIR, exist_ok=True)
CLASS_NAMES = ["Cargo", "Tanker", "Tug", "Passengership"]


def load_all_results():
    final_csv = os.path.join(RESULTS_DIR, "final_comparison_results.csv")
    fusion_csv = os.path.join(RESULTS_DIR, "final_fusion_results.csv")
    results = pd.read_csv(final_csv)
    if os.path.exists(fusion_csv):
        results = pd.concat([results, pd.read_csv(fusion_csv)], ignore_index=True)
    return results


def standardize_full_array(X):
    return (X - X.mean()) / (X.std() + 1e-8)


def get_single_feature_test_split(feature_name):
    y_all = np.load(os.path.join(FEATURE_DIR, "y_labels.npy"))
    X = np.load(os.path.join(FEATURE_DIR, f"{feature_name}.npy"))
    X = standardize_full_array(X)
    _, X_test, _, y_test = train_test_split(
        X, y_all, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_all
    )
    return X_test.astype(np.float32), y_test


def get_fusion_test_split():
    y_all = np.load(os.path.join(FEATURE_DIR, "y_labels.npy"))
    X_mel = standardize_full_array(np.load(os.path.join(FEATURE_DIR, "X_mel.npy")))
    X_mfcc = standardize_full_array(np.load(os.path.join(FEATURE_DIR, "X_mfcc.npy")))
    indices = np.arange(len(y_all))
    _, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_all
    )
    return X_mel[test_idx].astype(np.float32), X_mfcc[test_idx].astype(np.float32), y_all[test_idx]


def get_target_layer(model_name, model):
    if model_name == "SimpleCNN":
        return model.conv3[0]
    if model_name.startswith("ResNet"):
        return model.model.layer4[-1]
    raise ValueError(f"Grad-CAM target layer is not defined for {model_name}.")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = normalize_map(cam)
        return cam, logits.detach()

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


def normalize_map(array):
    array = np.asarray(array, dtype=np.float32)
    min_val = array.min()
    max_val = array.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(array)
    return (array - min_val) / (max_val - min_val)


def save_overlay(base_map, heatmap, title, save_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(base_map, aspect="auto", cmap="viridis", origin="lower")
    plt.imshow(heatmap, aspect="auto", cmap="jet", origin="lower", alpha=0.45)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path + ".png")
    plt.savefig(save_path + ".svg")
    plt.savefig(save_path + ".pdf")
    plt.close()


def choose_representative_samples(model_name, feature_name, model, max_per_class=1):
    X_test, y_test = get_single_feature_test_split(feature_name)
    dataset = DeepShipDataset(X_test, y_test, model_name)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    chosen = {}
    offset = 0
    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(DEVICE))
            preds = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.numpy()
            for i, (pred, label) in enumerate(zip(preds, labels_np)):
                if pred == label and label not in chosen:
                    chosen[int(label)] = offset + i
                if len(chosen) == len(CLASS_NAMES):
                    return X_test, y_test, chosen
            offset += len(labels_np)
    return X_test, y_test, chosen


def run_fixed_effect_analysis(results_df):
    single_stream_df = results_df[results_df["Model"] != "DualStreamFusionModel"].copy()

    feature_mean = single_stream_df.groupby("Feature")["f1"].mean().sort_values(ascending=False)
    fixed_feature = feature_mean.index[0]
    feature_subset = single_stream_df[single_stream_df["Feature"] == fixed_feature].sort_values("f1", ascending=False)
    feature_csv = os.path.join(EXPLAIN_DIR, "fixed_feature_model_effect.csv")
    feature_subset.to_csv(feature_csv, index=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_subset, x="Model", y="f1", palette="viridis")
    plt.title(f"Fixed Feature Analysis: Model Impact on {fixed_feature}")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=15)
    plt.tight_layout()
    feature_plot = os.path.join(EXPLAIN_DIR, "fixed_feature_model_effect")
    plt.savefig(feature_plot + ".png")
    plt.savefig(feature_plot + ".svg")
    plt.savefig(feature_plot + ".pdf")
    plt.close()

    model_mean = single_stream_df.groupby("Model")["f1"].mean().sort_values(ascending=False)
    fixed_model = model_mean.index[0]
    model_subset = single_stream_df[single_stream_df["Model"] == fixed_model].sort_values("f1", ascending=False)
    model_csv = os.path.join(EXPLAIN_DIR, "fixed_model_feature_effect.csv")
    model_subset.to_csv(model_csv, index=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=model_subset, x="Feature", y="f1", palette="magma")
    plt.title(f"Fixed Model Analysis: Feature Impact on {fixed_model}")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    model_plot = os.path.join(EXPLAIN_DIR, "fixed_model_feature_effect")
    plt.savefig(model_plot + ".png")
    plt.savefig(model_plot + ".svg")
    plt.savefig(model_plot + ".pdf")
    plt.close()

    return fixed_feature, fixed_model


def run_gradcam_for_best_2d(results_df):
    cnn_models = ["SimpleCNN", "ResNet18", "ResNet34", "ResNet50"]
    subset = results_df[results_df["Model"].isin(cnn_models)].sort_values("f1", ascending=False)
    best_row = subset.iloc[0]
    model_name = best_row["Model"]
    feature_name = best_row["Feature"]
    X_test, y_test, chosen = choose_representative_samples(
        model_name,
        feature_name,
        load_single_model(model_name, feature_name),
    )
    model = load_single_model(model_name, feature_name)
    cam = GradCAM(model, get_target_layer(model_name, model))

    for class_idx, sample_idx in chosen.items():
        sample = torch.tensor(X_test[sample_idx], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        heatmap, logits = cam.generate(sample, class_idx)
        title = f"Grad-CAM: {model_name} on {feature_name} ({CLASS_NAMES[class_idx]})"
        save_path = os.path.join(EXPLAIN_DIR, f"gradcam_{model_name}_{feature_name}_{CLASS_NAMES[class_idx]}")
        save_overlay(X_test[sample_idx], heatmap, title, save_path)

    cam.close()


def load_single_model(model_name, feature_name):
    if model_name in ["RNN", "LSTM", "Transformer"]:
        sample = np.load(os.path.join(FEATURE_DIR, f"{feature_name}.npy"))
        input_size = sample.shape[1]
        model = get_model(model_name, num_classes=4, input_size=input_size)
    else:
        model = get_model(model_name, num_classes=4, input_channels=1)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, f"best_{model_name}_{feature_name}.pth"), map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


def run_gradcam_for_fusion():
    fusion_csv = os.path.join(RESULTS_DIR, "final_fusion_results.csv")
    if not os.path.exists(fusion_csv):
        return

    model = DualStreamFusionModel().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_fusion_Mel_MFCC.pth"), map_location=DEVICE))
    model.eval()

    X_mel_test, X_mfcc_test, y_test = get_fusion_test_split()
    chosen = {}
    dataset = FusionDataset(X_mel_test, X_mfcc_test, y_test)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    offset = 0
    with torch.no_grad():
        for x1, x2, labels in loader:
            logits = model(x1.to(DEVICE), x2.to(DEVICE))
            preds = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.numpy()
            for i, (pred, label) in enumerate(zip(preds, labels_np)):
                if pred == label and label not in chosen:
                    chosen[int(label)] = offset + i
                if len(chosen) == len(CLASS_NAMES):
                    break
            offset += len(labels_np)
            if len(chosen) == len(CLASS_NAMES):
                break

    cam_mel = GradCAM(model, model.stream1[4])
    cam_mfcc = GradCAM(model, model.stream2[4])

    for class_idx, sample_idx in chosen.items():
        x1 = torch.tensor(X_mel_test[sample_idx], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        x2 = torch.tensor(X_mfcc_test[sample_idx], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        def fusion_forward_with_mel(inp):
            return model(inp, x2)

        def fusion_forward_with_mfcc(inp):
            return model(x1, inp)

        # 临时替换 model.forward 以便重用 Grad-CAM
        original_forward = model.forward
        model.forward = lambda a, b=None: fusion_forward_with_mel(a)
        heatmap_mel, _ = cam_mel.generate(x1, class_idx)
        model.forward = lambda a=None, b=None: fusion_forward_with_mfcc(b if b is not None else a)
        heatmap_mfcc, _ = cam_mfcc.generate(x2, class_idx)
        model.forward = original_forward

        save_overlay(
            X_mel_test[sample_idx],
            heatmap_mel,
            f"Fusion Grad-CAM (Mel branch) - {CLASS_NAMES[class_idx]}",
            os.path.join(EXPLAIN_DIR, f"gradcam_fusion_mel_{CLASS_NAMES[class_idx]}"),
        )
        save_overlay(
            X_mfcc_test[sample_idx],
            heatmap_mfcc,
            f"Fusion Grad-CAM (MFCC branch) - {CLASS_NAMES[class_idx]}",
            os.path.join(EXPLAIN_DIR, f"gradcam_fusion_mfcc_{CLASS_NAMES[class_idx]}"),
        )

    cam_mel.close()
    cam_mfcc.close()


def run_sequence_saliency(results_df):
    subset = results_df[results_df["Model"].isin(["RNN", "LSTM", "Transformer"])].sort_values("f1", ascending=False)
    best_row = subset.iloc[0]
    model_name = best_row["Model"]
    feature_name = best_row["Feature"]
    model = load_single_model(model_name, feature_name)
    X_test, y_test = get_single_feature_test_split(feature_name)

    saved_classes = set()
    for sample_idx in range(len(X_test)):
        if len(saved_classes) == len(CLASS_NAMES):
            break
        label = int(y_test[sample_idx])
        if label in saved_classes:
            continue

        dataset = DeepShipDataset(X_test[sample_idx:sample_idx + 1], y_test[sample_idx:sample_idx + 1], model_name)
        sample, _ = dataset[0]
        sample = sample.unsqueeze(0).to(DEVICE)
        sample.requires_grad_(True)
        logits = model(sample)
        pred = logits.argmax(dim=1).item()
        if pred != label:
            continue

        model.zero_grad()
        logits[0, pred].backward()
        saliency = sample.grad.abs().detach().cpu().numpy()[0]
        if model_name in ["RNN", "LSTM", "Transformer"]:
            saliency = saliency.T

        saliency = normalize_map(saliency)
        save_overlay(
            X_test[sample_idx],
            saliency,
            f"Input Saliency: {model_name} on {feature_name} ({CLASS_NAMES[label]})",
            os.path.join(EXPLAIN_DIR, f"saliency_{model_name}_{feature_name}_{CLASS_NAMES[label]}"),
        )
        saved_classes.add(label)


def run_transformer_attention(results_df):
    subset = results_df[results_df["Model"] == "Transformer"].sort_values("f1", ascending=False)
    if subset.empty:
        return

    row = subset.iloc[0]
    feature_name = row["Feature"]
    model = load_single_model("Transformer", feature_name)
    X_test, y_test = get_single_feature_test_split(feature_name)
    dataset = DeepShipDataset(X_test, y_test, "Transformer")

    selected_sample = None
    for sample_idx in range(len(dataset)):
        sample, label = dataset[sample_idx]
        candidate = sample.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(candidate).argmax(dim=1).item()
        if pred == int(label):
            selected_sample = candidate
            break

    if selected_sample is None:
        return

    attention_maps = []
    original_forwards = []

    for layer in model.transformer_encoder.layers:
        attn_module = layer.self_attn
        original_forward = attn_module.forward

        def wrapped_forward(query, key, value, original_forward=original_forward, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            attn_output, attn_weights = original_forward(query, key, value, **kwargs)
            attention_maps.append(attn_weights.detach().cpu())
            return attn_output, attn_weights

        attn_module.forward = wrapped_forward
        original_forwards.append((attn_module, original_forward))

    with torch.no_grad():
        _ = model(selected_sample)

    averaged_maps = []
    for attn in attention_maps:
        averaged_maps.append(attn[0].mean(dim=0).numpy())

    for attn_module, original_forward in original_forwards:
        attn_module.forward = original_forward

    if not averaged_maps:
        return

    attention_map = np.mean(np.stack(averaged_maps, axis=0), axis=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(attention_map, cmap="mako")
    plt.title(f"Transformer Attention Map on {feature_name}")
    plt.xlabel("Source Time Step")
    plt.ylabel("Target Time Step")
    plt.tight_layout()
    save_path = os.path.join(EXPLAIN_DIR, f"attention_transformer_{feature_name}")
    plt.savefig(save_path + ".png")
    plt.savefig(save_path + ".svg")
    plt.savefig(save_path + ".pdf")
    plt.close()


def evaluate_f1_for_feature_bands(model_name, feature_name, bands):
    model = load_single_model(model_name, feature_name)
    X_test, y_test = get_single_feature_test_split(feature_name)
    dataset = DeepShipDataset(X_test, y_test, model_name)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    baseline_metrics, _, _, _ = evaluate_model(model, loader, criterion, DEVICE)
    rows = []

    for band_name, start_idx, end_idx in bands:
        X_masked = X_test.copy()
        X_masked[:, start_idx:end_idx, :] = 0.0
        masked_dataset = DeepShipDataset(X_masked, y_test, model_name)
        masked_loader = DataLoader(masked_dataset, batch_size=128, shuffle=False)
        masked_metrics, _, _, _ = evaluate_model(model, masked_loader, criterion, DEVICE)
        rows.append({
            "Model": model_name,
            "Feature": feature_name,
            "Band": band_name,
            "Baseline_F1": baseline_metrics["f1"],
            "Masked_F1": masked_metrics["f1"],
            "F1_Drop": baseline_metrics["f1"] - masked_metrics["f1"],
        })
    return pd.DataFrame(rows)


def run_frequency_band_ablation(results_df):
    cnn_subset = results_df[results_df["Model"].isin(["SimpleCNN", "ResNet18", "ResNet34", "ResNet50"])].sort_values("f1", ascending=False)
    seq_subset = results_df[results_df["Model"].isin(["RNN", "LSTM", "Transformer"])].sort_values("f1", ascending=False)
    selected = [cnn_subset.iloc[0], seq_subset.iloc[0]]
    all_rows = []

    for row in selected:
        feature_name = row["Feature"]
        sample = np.load(os.path.join(FEATURE_DIR, f"{feature_name}.npy"))
        freq_bins = sample.shape[1]
        step = max(freq_bins // 6, 1)
        bands = []
        for idx, start in enumerate(range(0, freq_bins, step)):
            end = min(start + step, freq_bins)
            bands.append((f"Band_{idx + 1}", start, end))
            if end == freq_bins:
                break

        band_df = evaluate_f1_for_feature_bands(row["Model"], feature_name, bands)
        all_rows.append(band_df)

    result_df = pd.concat(all_rows, ignore_index=True)
    csv_path = os.path.join(EXPLAIN_DIR, "frequency_band_ablation_results.csv")
    result_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=result_df, x="Band", y="F1_Drop", hue="Model")
    plt.title("Frequency-band Ablation (Feature-level Band Masking)")
    plt.tight_layout()
    plot_path = os.path.join(EXPLAIN_DIR, "frequency_band_ablation")
    plt.savefig(plot_path + ".png")
    plt.savefig(plot_path + ".svg")
    plt.savefig(plot_path + ".pdf")
    plt.close()


def main():
    results_df = load_all_results()
    run_fixed_effect_analysis(results_df)
    run_gradcam_for_best_2d(results_df)
    run_gradcam_for_fusion()
    run_sequence_saliency(results_df)
    run_transformer_attention(results_df)
    run_frequency_band_ablation(results_df)
    print(f"Explainability outputs saved to {EXPLAIN_DIR}")


if __name__ == "__main__":
    main()
