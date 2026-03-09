import os
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

# ======================
# 全局参数（严格遵守）
# ======================
TARGET_SR = 32000          # 目标采样率，这里设置为 32kHz，不做降采样
SEGMENT_SEC = 2.0          # 每段音频的时长，设置为 2 秒
OVERLAP = 0.5              # 重叠率，50% 重叠

# 计算分段长度和跳步长度
# SEGMENT_LEN = 32000 * 2 = 64000 采样点
SEGMENT_LEN = int(TARGET_SR * SEGMENT_SEC)   
# HOP_LEN = 64000 * (1 - 0.5) = 32000 采样点
HOP_LEN = int(SEGMENT_LEN * (1 - OVERLAP))   

# 带通滤波的频率范围
LOW_FREQ = 1.0             # 低频截止频率 1Hz
HIGH_FREQ = 12000.0        # 高频截止频率 12kHz

# 数据集路径和保存路径
DATASET_ROOT = "DeepShip-main"  # 原始数据集根目录
SAVE_DIR = "processed_data"     # 处理后数据保存目录
os.makedirs(SAVE_DIR, exist_ok=True) # 创建保存目录，如果存在则忽略

def bandpass_filter(signal, sr, low_freq, high_freq, order=4):
    """
    对信号进行带通滤波
    :param signal: 输入信号
    :param sr: 采样率
    :param low_freq: 低频截止
    :param high_freq: 高频截止
    :param order: 滤波器阶数
    :return: 滤波后的信号
    """
    nyq = 0.5 * sr # 奈奎斯特频率 (采样率的一半)
    low = low_freq / nyq # 归一化低频
    high = high_freq / nyq # 归一化高频

    # 设计巴特沃斯带通滤波器，输出为 SOS (Second-Order Sections) 格式以保证稳定性
    sos = butter(order, [low, high], btype="band", output="sos")
    # 使用 sosfilt 进行滤波
    return sosfilt(sos, signal)

def segment_signal(signal, segment_len, hop_len):
    """
    对信号进行切片/分段
    :param signal: 输入信号
    :param segment_len: 每段长度
    :param hop_len: 步长
    :return: 分段后的信号列表
    """
    segments = []

    # 情况 1：音频长度不足 1 个窗口 → 补零
    if len(signal) < segment_len:
        padded = np.zeros(segment_len, dtype=np.float32) # 创建全零数组
        padded[:len(signal)] = signal # 将信号填充进去
        segments.append(padded)
        return segments

    # 情况 2：正常滑窗
    # 从 0 开始，步长为 hop_len，直到剩下的长度不足 segment_len
    for start in range(0, len(signal) - segment_len + 1, hop_len):
        seg = signal[start:start + segment_len] # 切片
        segments.append(seg)

    return segments

def preprocess_wav(wav_path):
    """
    预处理单个 WAV 文件
    :param wav_path: WAV 文件路径
    :return: 处理后的分段列表
    """
    # 读取音频文件，返回信号和采样率
    signal, sr = sf.read(wav_path)

    # 只处理单通道，如果是多通道则取第一个通道
    if signal.ndim > 1:
        signal = signal[:, 0]

    # 强制检查采样率，确保符合预期的 TARGET_SR
    if sr != TARGET_SR:
        raise ValueError(f"{wav_path} sampling rate {sr} != 32000")

    # 幅值归一化：除以最大绝对值，防止削波，+1e-8 防止除零
    max_val = np.max(np.abs(signal)) + 1e-8
    signal = signal / max_val

    # 带通滤波
    signal = bandpass_filter(
        signal,
        sr=TARGET_SR,
        low_freq=LOW_FREQ,
        high_freq=HIGH_FREQ
    )

    # 分段
    segments = segment_signal(
        signal,
        segment_len=SEGMENT_LEN,
        hop_len=HOP_LEN
    )

    return segments

# 类别标签映射
label_map = {
    "Cargo": 0,          # 货船
    "Tanker": 1,         # 油轮
    "Tug": 2,            # 拖船
    "Passengership": 3   # 客轮
}

X = [] # 用于存储特征（音频片段）
y = [] # 用于存储标签

# 遍历每个类别
for class_name, label in label_map.items():
    class_dir = os.path.join(DATASET_ROOT, class_name) # 拼接类别目录路径

    # 遍历类别目录下的所有文件
    for file in os.listdir(class_dir):
        # 只处理 .wav 文件
        if not file.lower().endswith(".wav"):
            continue

        wav_path = os.path.join(class_dir, file)

        try:
            # 预处理音频文件，获取分段
            segments = preprocess_wav(wav_path)

            # 将分段和对应的标签加入列表
            for seg in segments:
                X.append(seg)
                y.append(label)

        except Exception as e:
            # 如果处理出错，打印错误信息并跳过
            print(f"Skip {wav_path}: {e}")

# 转换为 numpy 数组
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print("Preprocessing finished.")
print("X shape:", X.shape)
print("y shape:", y.shape)

# 保存处理后的数据
np.save(os.path.join(SAVE_DIR, "X_audio_32k.npy"), X)
np.save(os.path.join(SAVE_DIR, "y_labels.npy"), y)

print("Saved:")
print(" - X_audio_32k.npy")
print(" - y_labels.npy")
