import torch
import time
import pandas as pd
import os
from deep_learning_models import get_model
from train_fusion import DualStreamFusionModel
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "dl_comparison_results_gpu"
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_complexity():
    models_to_test = [
        "SimpleCNN", "ResNet18", "ResNet34", "ResNet50",
        "RNN", "LSTM", "Transformer", "DualStreamFusionModel"
    ]
    
    # 假设特征维度 (Batch=1, Channels=1, Freq=128, Time=126) 类似 Mel 谱图
    freq_bins = 128
    time_steps = 126
    
    results = []
    
    print(f"Evaluating Model Complexity on {DEVICE}...")
    
    for model_name in models_to_test:
        print(f"Evaluating {model_name}...")
        
        # 1. 初始化模型
        if model_name == "DualStreamFusionModel":
            model = DualStreamFusionModel()
            dummy_input = (
                torch.randn(1, 1, freq_bins, time_steps).to(DEVICE),
                torch.randn(1, 1, 20, time_steps).to(DEVICE)
            )
        elif model_name in ['RNN', 'LSTM', 'Transformer']:
            model = get_model(model_name, num_classes=4, input_size=freq_bins)
            dummy_input = torch.randn(1, time_steps, freq_bins).to(DEVICE)
        else:
            model = get_model(model_name, num_classes=4, input_channels=1)
            dummy_input = torch.randn(1, 1, freq_bins, time_steps).to(DEVICE)
            
        model = model.to(DEVICE)
        model.eval()
        
        # 2. 计算参数量 (Parameters)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 3. 测量推理时间 (Inference Time)
        # 预热 GPU
        with torch.no_grad():
            for _ in range(10):
                if model_name == "DualStreamFusionModel":
                    _ = model(dummy_input[0], dummy_input[1])
                else:
                    _ = model(dummy_input)
                
        # 正式测量 (运行 100 次取平均)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                if model_name == "DualStreamFusionModel":
                    _ = model(dummy_input[0], dummy_input[1])
                else:
                    _ = model(dummy_input)
        
        # 确保 GPU 运算完成
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.time()
        avg_inference_time_ms = ((end_time - start_time) / 100) * 1000
        
        results.append({
            "Model": model_name,
            "Parameters (Millions)": params / 1e6,
            "Inference Time (ms/sample)": avg_inference_time_ms
        })
        
        del model
        del dummy_input
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. 保存结果
    df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "model_complexity.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Model", y="Parameters (Millions)", palette="Blues_d")
    plt.title("Model Parameter Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    params_path = os.path.join(RESULTS_DIR, "model_parameters_comparison")
    plt.savefig(params_path + ".png")
    plt.savefig(params_path + ".svg")
    plt.savefig(params_path + ".pdf")
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Model", y="Inference Time (ms/sample)", palette="Oranges_d")
    plt.title("Model Inference Time Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    time_path = os.path.join(RESULTS_DIR, "model_inference_time_comparison")
    plt.savefig(time_path + ".png")
    plt.savefig(time_path + ".svg")
    plt.savefig(time_path + ".pdf")
    plt.close()

    print("\nComplexity Evaluation Complete!")
    print(df.to_string(index=False))
    print(f"Saved results to {csv_path}")

if __name__ == "__main__":
    evaluate_complexity()
