import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# ======================
# 全局参数（与 preprocess_deepship_32k.py 保持一致）
# ======================
SR = 32000          # 采样率
N_FFT = 1024        # FFT 窗口大小
HOP_LENGTH = 512    # 步长 (约 16ms)
WIN_LENGTH = 1024   # 窗长
N_MELS = 128        # Mel 滤波器组数量
N_MFCC = 20         # MFCC 系数数量

DATA_DIR = "processed_data"
SAVE_DIR = "feature_data"
os.makedirs(SAVE_DIR, exist_ok=True)

def extract_features(signal, sr=SR):
    """
    提取多种时频特征
    """
    # 1. STFT (取幅值并转换为分贝)
    stft = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    # 2. Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=N_FFT, 
                                              hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 3. MFCC
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC, 
                                n_fft=N_FFT, hop_length=HOP_LENGTH)

    # 4. CQT (常数 Q 变换)
    # CQT 对低频分辨率更高，适合分析船舶噪声
    cqt = librosa.cqt(signal, sr=sr, hop_length=HOP_LENGTH, n_bins=84)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    return stft_db, mel_db, mfcc, cqt_db

def plot_comparison(stft, mel, mfcc, cqt, label_name):
    """
    可视化对比不同的特征图
    """
    plt.figure(figsize=(15, 10))

    # STFT
    plt.subplot(2, 2, 1)
    librosa.display.specshow(stft, sr=SR, hop_length=HOP_LENGTH, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'STFT Spectrogram ({label_name})')

    # Mel
    plt.subplot(2, 2, 2)
    librosa.display.specshow(mel, sr=SR, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')

    # MFCC
    plt.subplot(2, 2, 3)
    librosa.display.specshow(mfcc, sr=SR, hop_length=HOP_LENGTH, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')

    # CQT
    plt.subplot(2, 2, 4)
    librosa.display.specshow(cqt, sr=SR, hop_length=HOP_LENGTH, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q Transform (CQT)')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"comparison_{label_name}.png"))
    plt.close()

def main():
    # 1. 加载数据
    print("Loading preprocessed data...")
    X = np.load(os.path.join(DATA_DIR, "X_audio_32k.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_labels.npy"))
    
    label_map_inv = {0: "Cargo", 1: "Tanker", 2: "Tug", 3: "Passengership"}

    # 用于保存所有特征的列表
    all_stft = []
    all_mel = []
    all_mfcc = []
    all_cqt = []

    print(f"Total samples to process: {len(X)}")
    
    # 2. 遍历样本提取特征
    # 为了节省时间/空间，这里展示前 1000 个或者全部。如果数据量极大，建议分批处理。
    # 这里我们处理全部数据，并为每个类别选一个样本进行可视化对比。
    visualized_labels = set()

    for i in tqdm(range(len(X))):
        signal = X[i]
        label = y[i]
        label_name = label_map_inv[label]

        stft_db, mel_db, mfcc, cqt_db = extract_features(signal)

        # 为每个类别保存一个可视化对比图
        if label_name not in visualized_labels:
            plot_comparison(stft_db, mel_db, mfcc, cqt_db, label_name)
            visualized_labels.add(label_name)

        all_stft.append(stft_db)
        all_mel.append(mel_db)
        all_mfcc.append(mfcc)
        all_cqt.append(cqt_db)

    # 3. 转换为 numpy 数组并保存
    print("Saving features...")
    np.save(os.path.join(SAVE_DIR, "X_stft.npy"), np.array(all_stft))
    np.save(os.path.join(SAVE_DIR, "X_mel.npy"), np.array(all_mel))
    np.save(os.path.join(SAVE_DIR, "X_mfcc.npy"), np.array(all_mfcc))
    np.save(os.path.join(SAVE_DIR, "X_cqt.npy"), np.array(all_cqt))
    # 标签保持不变，但也拷贝一份方便后续使用
    np.save(os.path.join(SAVE_DIR, "y_labels.npy"), y)

    print("All features extracted and saved in 'feature_data' directory.")

if __name__ == "__main__":
    main()
