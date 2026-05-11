import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from train_comparison import plot_pr_curve, plot_metrics_history

FEATURE_DIR = "feature_data"
RESULTS_DIR = "dl_comparison_results_gpu"
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# 双特征融合网络设计
# =====================================================================
class DualStreamFusionModel(nn.Module):
    def __init__(self, num_classes=4):
        super(DualStreamFusionModel, self).__init__()
        
        # Stream 1: 用于处理 Mel 频谱 (2D 卷积)
        self.stream1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1) # 输出 shape: (Batch, 64, 1, 1)
        )
        
        # Stream 2: 用于处理 MFCC (2D 卷积)
        self.stream2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1) # 输出 shape: (Batch, 64, 1, 1)
        )
        
        # 融合层 (Concat 后的全连接)
        self.fc = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x1, x2):
        # x1: Mel, x2: MFCC
        out1 = self.stream1(x1)
        out1 = out1.view(out1.size(0), -1) # Flatten -> (Batch, 64)
        
        out2 = self.stream2(x2)
        out2 = out2.view(out2.size(0), -1) # Flatten -> (Batch, 64)
        
        # 拼接特征
        fused = torch.cat((out1, out2), dim=1) # (Batch, 128)
        
        # 分类
        out = self.fc(fused)
        return out

class FusionDataset(Dataset):
    def __init__(self, X1, X2, y):
        self.X1 = torch.tensor(X1, dtype=torch.float32).unsqueeze(1)
        self.X2 = torch.tensor(X2, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]

def train_fusion_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for x1, x2, labels in tqdm(train_loader, desc="Training", leave=False):
        x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)

        optimizer.zero_grad()
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

def evaluate_fusion_model(model, test_loader, criterion, device, num_classes=4):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x1, x2, labels in test_loader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            outputs = model(x1, x2)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(test_loader)
    all_probs = np.array(all_probs)

    try:
        y_test_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        roc_auc = roc_auc_score(y_test_bin, all_probs, multi_class='ovr', average='weighted')
    except Exception:
        roc_auc = 0.0

    metrics = {
        'loss': epoch_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'roc_auc': roc_auc,
    }

    return metrics, all_preds, all_labels, all_probs

def train_fusion():
    print("Preparing Multi-Feature Fusion Training (Mel + MFCC)...")
    
    # 1. 加载数据
    y_all = np.load(os.path.join(FEATURE_DIR, "y_labels.npy"))
    
    # 载入 Mel 和 MFCC
    X_mel = np.load(os.path.join(FEATURE_DIR, "X_mel.npy"))
    X_mfcc = np.load(os.path.join(FEATURE_DIR, "X_mfcc.npy"))
    
    # 预处理
    X_mel = (X_mel - X_mel.mean()) / (X_mel.std() + 1e-8)
    X_mfcc = (X_mfcc - X_mfcc.mean()) / (X_mfcc.std() + 1e-8)
    
    # 2. 划分数据集 (必须使用相同的 seed 保证数据对齐)
    indices = np.arange(len(y_all))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_all)
    
    train_dataset = FusionDataset(X_mel[train_idx], X_mfcc[train_idx], y_all[train_idx])
    test_dataset = FusionDataset(X_mel[test_idx], X_mfcc[test_idx], y_all[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # 3. 初始化模型
    model = DualStreamFusionModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. 训练
    best_f1 = 0
    epochs = 200
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'test_precision': [], 'test_recall': [],
        'test_f1': [], 'test_roc_auc': []
    }
    
    for epoch in range(epochs):
        train_loss, train_acc = train_fusion_epoch(model, train_loader, criterion, optimizer, DEVICE)
        metrics, _, _, _ = evaluate_fusion_model(model, test_loader, criterion, DEVICE)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(metrics['loss'])
        history['test_acc'].append(metrics['accuracy'])
        history['test_precision'].append(metrics['precision'])
        history['test_recall'].append(metrics['recall'])
        history['test_f1'].append(metrics['f1'])
        history['test_roc_auc'].append(metrics['roc_auc'])

        print(
            f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} - "
            f"Test Acc: {metrics['accuracy']:.4f} - Test F1: {metrics['f1']:.4f}"
        )

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_fusion_Mel_MFCC.pth"))

    history_path = os.path.join(RESULTS_DIR, "training_log_fusion_Mel_MFCC.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    plot_metrics_history(history, os.path.join(RESULTS_DIR, "history_fusion_Mel_MFCC"))

    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_fusion_Mel_MFCC.pth"), map_location=DEVICE))
    final_metrics, all_preds, all_labels, all_probs = evaluate_fusion_model(model, test_loader, criterion, DEVICE)

    final_results = pd.DataFrame([{
        "Feature": "Fusion_Mel_MFCC",
        "Model": "DualStreamFusionModel",
        "loss": final_metrics['loss'],
        "accuracy": final_metrics['accuracy'],
        "precision": final_metrics['precision'],
        "recall": final_metrics['recall'],
        "f1": final_metrics['f1'],
        "roc_auc": final_metrics['roc_auc'],
    }])
    final_results.to_csv(os.path.join(RESULTS_DIR, "final_fusion_results.csv"), index=False)

    label_map_inv = {0: "Cargo", 1: "Tanker", 2: "Tug", 3: "Passengership"}
    class_names = [label_map_inv[i] for i in range(4)]

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: Fusion_Mel_MFCC\nF1: {final_metrics["f1"]:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(RESULTS_DIR, "cm_fusion_Mel_MFCC.png"))
    plt.savefig(os.path.join(RESULTS_DIR, "cm_fusion_Mel_MFCC.svg"))
    plt.savefig(os.path.join(RESULTS_DIR, "cm_fusion_Mel_MFCC.pdf"))
    plt.close()

    plot_pr_curve(all_labels, all_probs, 4, class_names, os.path.join(RESULTS_DIR, "pr_curve_fusion_Mel_MFCC"))

    print(f"\nFusion Training Complete! Best Val F1: {best_f1:.4f}")
    print(f"Saved history CSV to {history_path}")
    print("Saved history plots, confusion matrix, PR curve, and final metrics CSV.")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    train_fusion()
