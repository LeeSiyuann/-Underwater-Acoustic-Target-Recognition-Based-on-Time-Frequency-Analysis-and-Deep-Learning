import argparse
import os
import gc
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from deep_learning_models import LSTMClassifier, TransformerClassifier
from train_comparison import (
    DeepShipDataset,
    FEATURE_DIR,
    RESULTS_DIR,
    DEVICE,
    TEST_SIZE,
    RANDOM_SEED,
    train_model,
    evaluate_model,
    plot_metrics_history,
)
from train_fusion import FusionDataset, DualStreamFusionModel


ABLATION_DIR = os.path.join(RESULTS_DIR, "ablation_results")
os.makedirs(ABLATION_DIR, exist_ok=True)


class SingleBranchMelModel(nn.Module):
    """严格消融：去掉 MFCC 支路，仅保留 Mel 支路。"""

    def __init__(self, num_classes=4):
        super().__init__()
        self.stream = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x1, x2=None):
        out = self.stream(x1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class SingleBranchMFCCModel(nn.Module):
    """严格消融：去掉 Mel 支路，仅保留 MFCC 支路。"""

    def __init__(self, num_classes=4):
        super().__init__()
        self.stream = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x1=None, x2=None):
        out = self.stream(x2)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class ShallowDualStreamFusionModel(nn.Module):
    """严格消融：去掉每个支路的第二个卷积模块，仅保留浅层双流。"""

    def __init__(self, num_classes=4):
        super().__init__()
        self.stream1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.stream2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x1, x2):
        out1 = self.stream1(x1).view(x1.size(0), -1)
        out2 = self.stream2(x2).view(x2.size(0), -1)
        fused = torch.cat((out1, out2), dim=1)
        return self.fc(fused)


def evaluate_fusion_like_model(model, data_loader, criterion, device, mode, num_classes=4):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x1, x2, labels in data_loader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            if mode == "mel_only":
                outputs = model(x1, None)
            elif mode == "mfcc_only":
                outputs = model(None, x2)
            else:
                outputs = model(x1, x2)

            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            running_loss += loss.item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(data_loader)
    all_probs = np.array(all_probs)

    try:
        y_test_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        roc_auc = roc_auc_score(y_test_bin, all_probs, multi_class="ovr", average="weighted")
    except Exception:
        roc_auc = 0.0

    metrics = {
        "loss": epoch_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "roc_auc": roc_auc,
    }
    return metrics


def train_fusion_like_model(model, train_loader, criterion, optimizer, device, mode):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for x1, x2, labels in train_loader:
        x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
        optimizer.zero_grad()
        if mode == "mel_only":
            outputs = model(x1, None)
        elif mode == "mfcc_only":
            outputs = model(None, x2)
        else:
            outputs = model(x1, x2)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def load_final_results():
    final_csv = os.path.join(RESULTS_DIR, "final_comparison_results.csv")
    if not os.path.exists(final_csv):
        raise FileNotFoundError(f"{final_csv} not found.")
    return pd.read_csv(final_csv)


def get_best_feature_for_model(results_df, model_name, fallback_feature):
    subset = results_df[results_df["Model"] == model_name]
    if subset.empty:
        return fallback_feature
    return subset.sort_values("f1", ascending=False).iloc[0]["Feature"]


def standardize_full_array(X):
    return (X - X.mean()) / (X.std() + 1e-8)


def save_history(history, save_stem):
    pd.DataFrame(history).to_csv(save_stem + ".csv", index=False)
    plot_metrics_history(history, save_stem)


def plot_bar(df, x, y, hue, title, save_stem):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x=x, y=y, hue=hue)
    plt.title(title)
    plt.xticks(rotation=15)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(save_stem + ".png")
    plt.savefig(save_stem + ".svg")
    plt.savefig(save_stem + ".pdf")
    plt.close()


def plot_heatmap(df, index_col, column_col, value_col, title, save_stem):
    pivot_df = df.pivot(index=index_col, columns=column_col, values=value_col)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_stem + ".png")
    plt.savefig(save_stem + ".svg")
    plt.savefig(save_stem + ".pdf")
    plt.close()


def run_fusion_ablation(epochs, batch_size, learning_rate):
    print("Running strict fusion structure ablation...")
    y_all = np.load(os.path.join(FEATURE_DIR, "y_labels.npy"))
    X_mel = standardize_full_array(np.load(os.path.join(FEATURE_DIR, "X_mel.npy")))
    X_mfcc = standardize_full_array(np.load(os.path.join(FEATURE_DIR, "X_mfcc.npy")))

    indices = np.arange(len(y_all))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_all
    )

    train_dataset = FusionDataset(X_mel[train_idx], X_mfcc[train_idx], y_all[train_idx])
    test_dataset = FusionDataset(X_mel[test_idx], X_mfcc[test_idx], y_all[test_idx])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    variant_builders = {
        "DualStreamFusion": (DualStreamFusionModel, "dual"),
        "MelOnlyBranch": (SingleBranchMelModel, "mel_only"),
        "MFCCOnlyBranch": (SingleBranchMFCCModel, "mfcc_only"),
        "ShallowDualStream": (ShallowDualStreamFusionModel, "dual"),
    }

    rows = []
    for variant_name, (builder, mode) in variant_builders.items():
        print(f"Training fusion ablation variant: {variant_name}")
        model = builder().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_metrics = None
        best_f1 = -1.0
        history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "test_precision": [],
            "test_recall": [],
            "test_f1": [],
            "test_roc_auc": [],
        }

        for _ in tqdm(range(epochs), desc=f"FusionAblation-{variant_name}", leave=False):
            train_loss, train_acc = train_fusion_like_model(
                model, train_loader, criterion, optimizer, DEVICE, mode
            )
            metrics = evaluate_fusion_like_model(model, test_loader, criterion, DEVICE, mode)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_loss"].append(metrics["loss"])
            history["test_acc"].append(metrics["accuracy"])
            history["test_precision"].append(metrics["precision"])
            history["test_recall"].append(metrics["recall"])
            history["test_f1"].append(metrics["f1"])
            history["test_roc_auc"].append(metrics["roc_auc"])

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_metrics = copy.deepcopy(metrics)

        save_stem = os.path.join(ABLATION_DIR, f"history_ablation_fusion_{variant_name}")
        save_history(history, save_stem)
        rows.append({
            "Experiment Group": "FusionStructure",
            "Variant": variant_name,
            "Feature": "Fusion_Mel_MFCC",
            **best_metrics,
        })

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    csv_path = os.path.join(ABLATION_DIR, "ablation_fusion_structure_results.csv")
    df.to_csv(csv_path, index=False)
    plot_bar(
        df,
        x="Variant",
        y="f1",
        hue="Experiment Group",
        title="Strict Fusion Structure Ablation (Weighted F1)",
        save_stem=os.path.join(ABLATION_DIR, "ablation_fusion_structure_f1"),
    )
    return df


def run_sequence_ablation(epochs, batch_size, learning_rate):
    print("Running sequence hyper-parameter ablation...")
    results_df = load_final_results()
    y_all = np.load(os.path.join(FEATURE_DIR, "y_labels.npy"))

    configs = {
        "LSTM": {
            "feature": get_best_feature_for_model(results_df, "LSTM", "X_cqt"),
            "variants": [
                {"hidden_size": 128, "num_layers": 1},
                {"hidden_size": 128, "num_layers": 2},
                {"hidden_size": 256, "num_layers": 2},
                {"hidden_size": 256, "num_layers": 3},
                {"hidden_size": 512, "num_layers": 3},
            ],
        },
        "Transformer": {
            "feature": get_best_feature_for_model(results_df, "Transformer", "X_cqt"),
            "variants": [
                {"d_model": 128, "num_layers": 2, "nhead": 4},
                {"d_model": 128, "num_layers": 4, "nhead": 4},
                {"d_model": 256, "num_layers": 4, "nhead": 8},
                {"d_model": 256, "num_layers": 6, "nhead": 8},
                {"d_model": 384, "num_layers": 6, "nhead": 8},
            ],
        },
    }

    all_rows = []
    for model_name, config in configs.items():
        feature_name = config["feature"]
        X = standardize_full_array(np.load(os.path.join(FEATURE_DIR, f"{feature_name}.npy")))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_all, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_all
        )
        train_dataset = DeepShipDataset(X_train, y_train, model_name)
        test_dataset = DeepShipDataset(X_test, y_test, model_name)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        input_size = X_train.shape[1]

        for variant in config["variants"]:
            if model_name == "LSTM":
                variant_name = f"hs{variant['hidden_size']}_layers{variant['num_layers']}"
                model = LSTMClassifier(
                    input_size=input_size,
                    hidden_size=variant["hidden_size"],
                    num_layers=variant["num_layers"],
                    num_classes=4,
                )
            else:
                variant_name = f"dm{variant['d_model']}_layers{variant['num_layers']}"
                model = TransformerClassifier(
                    input_size=input_size,
                    d_model=variant["d_model"],
                    nhead=variant["nhead"],
                    num_layers=variant["num_layers"],
                    num_classes=4,
                )

            model = model.to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            best_metrics = None
            best_f1 = -1.0
            history = {
                "train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": [],
                "test_precision": [],
                "test_recall": [],
                "test_f1": [],
                "test_roc_auc": [],
            }

            for _ in tqdm(range(epochs), desc=f"{model_name}Ablation-{variant_name}", leave=False):
                train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, DEVICE)
                metrics, _, _, _ = evaluate_model(model, test_loader, criterion, DEVICE)

                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                history["test_loss"].append(metrics["loss"])
                history["test_acc"].append(metrics["accuracy"])
                history["test_precision"].append(metrics["precision"])
                history["test_recall"].append(metrics["recall"])
                history["test_f1"].append(metrics["f1"])
                history["test_roc_auc"].append(metrics["roc_auc"])

                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    best_metrics = copy.deepcopy(metrics)

            save_stem = os.path.join(ABLATION_DIR, f"history_ablation_{model_name}_{feature_name}_{variant_name}")
            save_history(history, save_stem)
            all_rows.append({
                "Experiment Group": f"{model_name}HyperParams",
                "Model": model_name,
                "Feature": feature_name,
                "Variant": variant_name,
                **variant,
                **best_metrics,
            })

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(ABLATION_DIR, "ablation_sequence_hyperparams_results.csv")
    df.to_csv(csv_path, index=False)

    for model_name in ["LSTM", "Transformer"]:
        subset = df[df["Model"] == model_name].copy()
        if subset.empty:
            continue
        if model_name == "LSTM":
            subset["ConfigY"] = subset["num_layers"].astype(int).astype(str)
            subset["ConfigX"] = subset["hidden_size"].astype(int).astype(str)
        else:
            subset["ConfigY"] = subset["num_layers"].astype(int).astype(str)
            subset["ConfigX"] = subset["d_model"].astype(int).astype(str)

        plot_heatmap(
            subset,
            index_col="ConfigY",
            column_col="ConfigX",
            value_col="f1",
            title=f"{model_name} Hyper-parameter Ablation (Weighted F1)",
            save_stem=os.path.join(ABLATION_DIR, f"ablation_{model_name.lower()}_hyperparams_heatmap"),
        )

    return df


def main():
    parser = argparse.ArgumentParser(description="Run strict structure ablation experiments.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs for each ablation variant.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for ablation experiments.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    args = parser.parse_args()

    fusion_df = run_fusion_ablation(args.epochs, args.batch_size, args.learning_rate)
    sequence_df = run_sequence_ablation(args.epochs, args.batch_size, args.learning_rate)

    combined_df = pd.concat([fusion_df, sequence_df], ignore_index=True, sort=False)
    combined_csv = os.path.join(ABLATION_DIR, "ablation_summary_results.csv")
    combined_df.to_csv(combined_csv, index=False)
    print(f"Saved ablation summary to {combined_csv}")


if __name__ == "__main__":
    main()
