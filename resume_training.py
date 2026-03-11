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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gc
import multiprocessing

# Import necessary components from the original script
# Ensure deep_learning_models.py is in the same directory
from deep_learning_models import get_model
from train_comparison import DeepShipDataset, train_model, evaluate_model, plot_pr_curve, plot_metrics_history
from train_comparison import FEATURE_DIR, RESULTS_DIR, DEVICE, EPOCHS, LEARNING_RATE, TEST_SIZE, RANDOM_SEED

# ======================
# Resume Configuration
# ======================
# Reduced batch size for STFT to prevent OOM (Out Of Memory)
# User reported OOM with ResNet50 on STFT (likely with batch size 512)
# We reduce it to 128 to be safe for the heavy ResNet50 + large STFT features
DEFAULT_BATCH_SIZE = 2048 
STFT_BATCH_SIZE = 128     # Significantly reduced from 512

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def recover_previous_results(features, models):
    """
    Scans the results directory for existing training logs to recover results
    for tasks that have already completed.
    """
    recovered_results = []
    completed_tasks = set()
    
    logging.info("Scanning for completed tasks...")
    
    for feature_file in features:
        feature_name = feature_file.replace(".npy", "")
        for model_name in models:
            log_file = os.path.join(RESULTS_DIR, f"training_log_{model_name}_{feature_name}.csv")
            
            if os.path.exists(log_file):
                try:
                    df = pd.read_csv(log_file)
                    if not df.empty:
                        # Find the row with the best Test F1 Score
                        best_row = df.loc[df['test_f1'].idxmax()]
                        
                        # Reconstruct the metrics dictionary
                        metrics = {
                            'loss': best_row['test_loss'],
                            'accuracy': best_row['test_acc'],
                            'precision': best_row['test_precision'],
                            'recall': best_row['test_recall'],
                            'f1': best_row['test_f1'],
                            'roc_auc': best_row['test_roc_auc'],
                            'Feature': feature_name,
                            'Model': model_name
                        }
                        recovered_results.append(metrics)
                        completed_tasks.add((feature_name, model_name))
                        logging.info(f"Recovered results for {model_name} on {feature_name} (Best F1: {metrics['f1']:.4f})")
                except Exception as e:
                    logging.warning(f"Found log file for {model_name} on {feature_name} but failed to read it: {e}")
    
    return recovered_results, completed_tasks

def run_resume():
    # Set num_workers
    cpu_count = multiprocessing.cpu_count()
    num_workers = min(16, cpu_count)
    logging.info(f"Setting num_workers = {num_workers}")

    label_map_inv = {0: "Cargo", 1: "Tanker", 2: "Tug", 3: "Passengership"}
    class_names = [label_map_inv[i] for i in range(4)]
    
    logging.info("Loading labels...")
    try:
        y_all = np.load(os.path.join(FEATURE_DIR, "y_labels.npy"))
    except FileNotFoundError:
        logging.error("y_labels.npy not found. Please ensure feature data exists.")
        return

    # Full list of tasks
    features_to_test = ["X_mel.npy", "X_mfcc.npy", "X_cqt.npy", "X_stft.npy"]
    models_to_test = [
        "SimpleCNN", "ResNet18", "ResNet34", "ResNet50", 
        "RNN", "LSTM", "Transformer"
    ]
    
    # Recover previous results
    final_results, completed_tasks = recover_previous_results(features_to_test, models_to_test)
    gpu_usage_records = [] 
    
    # Identify what's left to run
    # Outer loop: Feature, Inner loop: Model (same as original script)
    
    for feature_file in features_to_test:
        feature_name = feature_file.replace(".npy", "")
        
        # Check if all models for this feature are already done
        all_models_done = all((feature_name, m) in completed_tasks for m in models_to_test)
        if all_models_done:
            logging.info(f"Skipping feature {feature_name} (all models completed)")
            continue

        logging.info(f"\n{'='*60}\nProcessing Feature: {feature_name}\n{'='*60}")
        
        # Set Batch Size (using the reduced STFT_BATCH_SIZE)
        current_batch_size = STFT_BATCH_SIZE if "stft" in feature_name.lower() else DEFAULT_BATCH_SIZE
        logging.info(f"Using Batch Size: {current_batch_size}")

        # Load Data
        try:
            X = np.load(os.path.join(FEATURE_DIR, feature_file))
        except Exception as e:
            logging.error(f"Failed to load {feature_file}: {e}")
            continue

        # Preprocessing
        mean = X.mean()
        std = X.std()
        X = (X - mean) / (std + 1e-8)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_all, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_all
        )
        
        del X
        gc.collect()

        for model_name in models_to_test:
            # Skip if already done
            if (feature_name, model_name) in completed_tasks:
                logging.info(f"Skipping {model_name} on {feature_name} (Already completed)")
                continue

            logging.info(f"\nTraining Model: {model_name} on {feature_name}")
            
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
            
            # Record GPU usage
            max_memory = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2) # MB
            gpu_usage_records.append({
                "Feature": feature_name,
                "Model": model_name,
                "Max_GPU_Memory_MB": max_memory
            })
            logging.info(f"Max GPU Memory Used: {max_memory:.2f} MB")

            # Save history
            df_history = pd.DataFrame(history)
            df_history.to_csv(os.path.join(RESULTS_DIR, f"training_log_{model_name}_{feature_name}.csv"), index=False)
            
            plot_metrics_history(history, os.path.join(RESULTS_DIR, f"history_{model_name}_{feature_name}"))

            # Final Evaluation of Best Model
            model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, f"best_{model_name}_{feature_name}.pth")))
            final_metrics, all_preds, all_labels, all_probs = evaluate_model(model, test_loader, criterion, DEVICE)
            
            logging.info(f"Best Test F1: {final_metrics['f1']:.4f}")
            final_metrics['Feature'] = feature_name
            final_metrics['Model'] = model_name
            final_results.append(final_metrics)
            
            # Confusion Matrix
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
            
            # PR Curve
            plot_pr_curve(all_labels, all_probs, 4, class_names, os.path.join(RESULTS_DIR, f"pr_curve_{model_name}_{feature_name}"))
            
            del model
            del train_loader
            del test_loader
            gc.collect()
            torch.cuda.empty_cache()

    # Save aggregated results (merged with recovered ones)
    df_results = pd.DataFrame(final_results)
    output_path = os.path.join(RESULTS_DIR, "final_comparison_results.csv")
    df_results.to_csv(output_path, index=False)
    logging.info(f"Final results saved to {output_path}")
    
    # Note: gpu_memory_usage.csv will only contain records from THIS run, 
    # as we can't easily reconstruct previous usage from logs.
    if gpu_usage_records:
        df_gpu = pd.DataFrame(gpu_usage_records)
        gpu_path = os.path.join(RESULTS_DIR, "gpu_memory_usage_resume.csv")
        df_gpu.to_csv(gpu_path, index=False)
        logging.info(f"New GPU usage records saved to {gpu_path}")
    
    logging.info("\nResume complete!")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    run_resume()
