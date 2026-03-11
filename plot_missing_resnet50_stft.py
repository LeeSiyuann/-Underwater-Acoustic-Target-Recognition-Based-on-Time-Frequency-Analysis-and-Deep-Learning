import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import pandas as pd
import gc

# 导入配置和函数
# 确保当前目录有 deep_learning_models.py 和 train_comparison.py
from deep_learning_models import get_model
from train_comparison import DeepShipDataset, plot_pr_curve, FEATURE_DIR, RESULTS_DIR, DEVICE, TEST_SIZE, RANDOM_SEED

# 配置
MODEL_NAME = "ResNet50"
FEATURE_FILE = "X_stft.npy"
FEATURE_NAME = "X_stft"
BATCH_SIZE = 128  # 针对 STFT 使用较小的 Batch Size

def plot_missing_results():
    print(f"Generating missing plots for {MODEL_NAME} on {FEATURE_NAME}...")
    
    # 1. 加载标签
    print("Loading labels...")
    try:
        y_all = np.load(os.path.join(FEATURE_DIR, "y_labels.npy"))
    except FileNotFoundError:
        print("Error: y_labels.npy not found.")
        return

    # 2. 加载特征数据
    print(f"Loading feature data: {FEATURE_FILE}...")
    try:
        X = np.load(os.path.join(FEATURE_DIR, FEATURE_FILE))
    except FileNotFoundError:
        print(f"Error: {FEATURE_FILE} not found.")
        return

    # 3. 数据预处理
    mean = X.mean()
    std = X.std()
    X = (X - mean) / (std + 1e-8)
    
    # 4. 数据集切分 (必须使用相同的随机种子 42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_all, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_all
    )
    
    # 释放训练集内存，只保留测试集
    del X
    del X_train
    del y_train
    gc.collect()

    # 5. 创建测试集 DataLoader
    test_dataset = DeepShipDataset(X_test, y_test, MODEL_NAME)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # 6. 初始化模型并加载权重
    print(f"Loading model weights...")
    model_path = os.path.join(RESULTS_DIR, f"best_{MODEL_NAME}_{FEATURE_NAME}.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    try:
        # ResNet50 是 2D 模型，输入通道为 1
        model = get_model(MODEL_NAME, num_classes=4, input_channels=1)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 7. 运行推理
    print("Running inference on test set...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_probs = np.array(all_probs)
    
    # 8. 计算 F1 分数 (用于标题)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    print(f"Test F1 Score: {f1:.4f}")

    # 9. 绘制并保存混淆矩阵
    print("Plotting Confusion Matrix...")
    label_map_inv = {0: "Cargo", 1: "Tanker", 2: "Tug", 3: "Passengership"}
    class_names = [label_map_inv[i] for i in range(4)]
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {MODEL_NAME} on {FEATURE_NAME}\nF1: {f1:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(RESULTS_DIR, f"cm_{MODEL_NAME}_{FEATURE_NAME}")
    plt.savefig(cm_path + ".png")
    plt.savefig(cm_path + ".svg")
    plt.savefig(cm_path + ".pdf")
    plt.close()
    print(f"Confusion Matrix saved to {cm_path}.png")

    # 10. 绘制并保存 PR 曲线
    print("Plotting PR Curve...")
    pr_path = os.path.join(RESULTS_DIR, f"pr_curve_{MODEL_NAME}_{FEATURE_NAME}")
    plot_pr_curve(all_labels, all_probs, 4, class_names, pr_path)
    print(f"PR Curve saved to {pr_path}.png")
    
    print("Done!")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    plot_missing_results()
