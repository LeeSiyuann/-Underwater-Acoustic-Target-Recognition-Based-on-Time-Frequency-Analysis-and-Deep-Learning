import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gc
from datetime import datetime
import multiprocessing

# 导入模型
from deep_learning_models import get_model

# ======================
# 全局配置
# ======================
FEATURE_DIR = "feature_data"
RESULTS_DIR = "dl_comparison_results_gpu" # 新的保存目录
os.makedirs(RESULTS_DIR, exist_ok=True)

# 日志配置 - 仅保留控制台输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# 针对 5090 GPU 的训练超参数
DEFAULT_BATCH_SIZE = 2048 # 默认 Batch Size
STFT_BATCH_SIZE = 512     # STFT 特有 Batch Size (避免显存溢出)
EPOCHS = 50      
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
RANDOM_SEED = 42

# 设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 只在开始时打印一次设备信息
logging.info(f"Using Device: {DEVICE}")
if torch.cuda.is_available():
    logging.info(f"GPU Name: {torch.cuda.get_device_name(0)}")

# =================================================================================================
# 自定义数据集类
# =================================================================================================
class DeepShipDataset(Dataset):
    def __init__(self, X, y, model_type):
        self.y = torch.tensor(y, dtype=torch.long)
        
        # 2D 模型 (CNN, ResNet) 需要 (N, Channel, Height, Width)
        if model_type in ['SimpleCNN', 'ResNet18', 'ResNet34', 'ResNet50']:
            self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) 
            
        # 序列模型 (RNN, LSTM, Transformer) 需要 (N, Seq_Len, Features)
        elif model_type in ['RNN', 'LSTM', 'Transformer']:
            X_transposed = np.transpose(X, (0, 2, 1)) 
            self.X = torch.tensor(X_transposed, dtype=torch.float32)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =================================================================================================
# 训练与评估函数
# =================================================================================================
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # 减少 tqdm 的刷新频率以减少日志干扰
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def evaluate_model(model, test_loader, criterion, device, num_classes=4):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = [] 
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            probs = torch.softmax(outputs, dim=1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    epoch_loss = running_loss / len(test_loader)
    all_probs = np.array(all_probs)
    
    # 计算多维度指标
    try:
        # ROC AUC 需要 one-hot 编码的标签
        y_test_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        roc_auc = roc_auc_score(y_test_bin, all_probs, multi_class='ovr', average='weighted')
    except:
        roc_auc = 0.0 # 异常处理
    
    metrics = {
        'loss': epoch_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'roc_auc': roc_auc
    }
    
    return metrics, all_preds, all_labels, all_probs

# =================================================================================================
# 绘图辅助函数
# =================================================================================================
def plot_pr_curve(y_test, y_score, num_classes, class_names, save_path):
    y_test_bin = label_binarize(y_test, classes=list(range(num_classes)))
    precision = dict()
    recall = dict()
    
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        plt.plot(recall[i], precision[i], lw=2, label=f'Class {class_names[i]}')
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(save_path + ".png")
    plt.savefig(save_path + ".svg")
    plt.savefig(save_path + ".pdf")
    plt.close()

def plot_metrics_history(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0, 0].plot(epochs, history['test_loss'], label='Test Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid()

    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], label='Train Acc')
    axes[0, 1].plot(epochs, history['test_acc'], label='Test Acc')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid()

    # Precision
    axes[0, 2].plot(epochs, history['test_precision'], label='Test Precision', color='green')
    axes[0, 2].set_title('Precision')
    axes[0, 2].grid()

    # Recall
    axes[1, 0].plot(epochs, history['test_recall'], label='Test Recall', color='orange')
    axes[1, 0].set_title('Recall')
    axes[1, 0].grid()

    # F1 Score
    axes[1, 1].plot(epochs, history['test_f1'], label='Test F1', color='red')
    axes[1, 1].set_title('F1 Score')
    axes[1, 1].grid()

    # ROC AUC
    axes[1, 2].plot(epochs, history['test_roc_auc'], label='Test ROC AUC', color='purple')
    axes[1, 2].set_title('ROC AUC')
    axes[1, 2].grid()

    plt.tight_layout()
    plt.savefig(save_path + ".png")
    plt.savefig(save_path + ".svg")
    plt.savefig(save_path + ".pdf")
    plt.close()

# =================================================================================================
# 主流程
# =================================================================================================
def run_comparison():
    # 动态设置 num_workers
    # 你的 CPU 有 22 vCPU，建议设置为核心数的一半或略少，留有余量
    # 对于 22 核，设置为 16 是个不错的激进值，或者保守点 8-12
    cpu_count = multiprocessing.cpu_count()
    num_workers = min(16, cpu_count) 
    logging.info(f"Setting num_workers = {num_workers}")

    label_map_inv = {0: "Cargo", 1: "Tanker", 2: "Tug", 3: "Passengership"}
    class_names = [label_map_inv[i] for i in range(4)]
    
    logging.info("Loading labels...")
    y_all = np.load(os.path.join(FEATURE_DIR, "y_labels.npy"))

    # 特征列表
    features_to_test = ["X_mel.npy", "X_mfcc.npy", "X_cqt.npy", "X_stft.npy"]
    
    # 模型列表 (新增 ResNet34/50, RNN)
    models_to_test = [
        "SimpleCNN", "ResNet18", "ResNet34", "ResNet50", 
        "RNN", "LSTM", "Transformer"
    ]
    
    final_results = []
    gpu_usage_records = [] # 用于记录显存占用

    for feature_file in features_to_test:
        feature_name = feature_file.replace(".npy", "")
        logging.info(f"\n{'='*60}\nProcessing Feature: {feature_name}\n{'='*60}")
        
        # 动态设置 Batch Size
        current_batch_size = STFT_BATCH_SIZE if "stft" in feature_name.lower() else DEFAULT_BATCH_SIZE
        logging.info(f"Using Batch Size: {current_batch_size}")

        try:
            X = np.load(os.path.join(FEATURE_DIR, feature_file))
        except Exception as e:
            logging.error(f"Failed to load {feature_file}: {e}")
            continue

        mean = X.mean()
        std = X.std()
        X = (X - mean) / (std + 1e-8)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_all, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_all
        )
        
        del X
        gc.collect()

        for model_name in models_to_test:
            logging.info(f"\nTraining Model: {model_name} on {feature_name}")
            
            # 清空显存缓存以获得准确的测量
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            try:
                train_dataset = DeepShipDataset(X_train, y_train, model_name)
                test_dataset = DeepShipDataset(X_test, y_test, model_name)
            except Exception as e:
                logging.error(f"Dataset creation failed for {model_name}: {e}")
                continue
            
            train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=current_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            
            input_freq_bins = X_train.shape[1] 
            
            try:
                if model_name in ['RNN', 'LSTM', 'Transformer']:
                    model = get_model(model_name, num_classes=4, input_size=input_freq_bins)
                else:
                    model = get_model(model_name, num_classes=4, input_channels=1)
            except Exception as e:
                logging.error(f"Model initialization failed for {model_name}: {e}")
                continue
            
            model = model.to(DEVICE)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            
            best_f1 = 0.0
            history = {
                'train_loss': [], 'train_acc': [], 
                'test_loss': [], 'test_acc': [], 
                'test_precision': [], 'test_recall': [], 
                'test_f1': [], 'test_roc_auc': []
            }
            
            pbar = tqdm(range(EPOCHS), desc=f"{model_name}-{feature_name}", unit="epoch")
            for epoch in pbar:
                train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, DEVICE)
                metrics, _, _, _ = evaluate_model(model, test_loader, criterion, DEVICE)
                
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['test_loss'].append(metrics['loss'])
                history['test_acc'].append(metrics['accuracy'])
                history['test_precision'].append(metrics['precision'])
                history['test_recall'].append(metrics['recall'])
                history['test_f1'].append(metrics['f1'])
                history['test_roc_auc'].append(metrics['roc_auc'])
                
                pbar.set_postfix({'Acc': f"{metrics['accuracy']:.4f}", 'F1': f"{metrics['f1']:.4f}"})
                
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"best_{model_name}_{feature_name}.pth"))
            
            # 记录显存峰值
            max_memory = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2) # MB
            gpu_usage_records.append({
                "Feature": feature_name,
                "Model": model_name,
                "Max_GPU_Memory_MB": max_memory
            })
            logging.info(f"Max GPU Memory Used: {max_memory:.2f} MB")

            df_history = pd.DataFrame(history)
            df_history.to_csv(os.path.join(RESULTS_DIR, f"training_log_{model_name}_{feature_name}.csv"), index=False)
            
            plot_metrics_history(history, os.path.join(RESULTS_DIR, f"history_{model_name}_{feature_name}"))

            model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, f"best_{model_name}_{feature_name}.pth")))
            final_metrics, all_preds, all_labels, all_probs = evaluate_model(model, test_loader, criterion, DEVICE)
            
            logging.info(f"Best Test F1: {final_metrics['f1']:.4f}")
            final_metrics['Feature'] = feature_name
            final_metrics['Model'] = model_name
            final_results.append(final_metrics)
            
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Confusion Matrix: {model_name} on {feature_name}\nF1: {final_metrics["f1"]:.4f}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(RESULTS_DIR, f"cm_{model_name}_{feature_name}.png"))
            plt.savefig(os.path.join(RESULTS_DIR, f"cm_{model_name}_{feature_name}.svg"))
            plt.savefig(os.path.join(RESULTS_DIR, f"cm_{model_name}_{feature_name}.pdf"))
            plt.close()
            
            plot_pr_curve(all_labels, all_probs, 4, class_names, os.path.join(RESULTS_DIR, f"pr_curve_{model_name}_{feature_name}"))
            
            del model
            del train_loader
            del test_loader
            gc.collect()
            torch.cuda.empty_cache()

    # 保存最终结果
    df_results = pd.DataFrame(final_results)
    df_results.to_csv(os.path.join(RESULTS_DIR, "final_comparison_results.csv"), index=False)
    
    # 保存显存占用记录
    df_gpu = pd.DataFrame(gpu_usage_records)
    df_gpu.to_csv(os.path.join(RESULTS_DIR, "gpu_memory_usage.csv"), index=False)
    
    logging.info("\nComparison complete! Summaries saved.")
    logging.info(df_results)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    run_comparison()
