import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# =================================================================================================
# 1. 简单 CNN (SimpleCNN)
# -------------------------------------------------------------------------------------------------
# 这是一个轻量级的卷积神经网络，作为 Baseline 模型。
# 适用于初步验证数据和特征的有效性，计算成本低。
# =================================================================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4, input_channels=1):
        super(SimpleCNN, self).__init__()
        
        # 第一层卷积块：Conv -> BatchNorm -> ReLU -> MaxPool
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第二层卷积块
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第三层卷积块
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 全局平均池化 (Global Average Pooling)
        # 将特征图尺寸 (C, H, W) 压缩为 (C, 1, 1)，从而适应任意尺寸输入
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 全连接分类层
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (Batch, 1, Freq, Time)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x

# =================================================================================================
# 2. ResNet 系列 - 使用 torchvision
# -------------------------------------------------------------------------------------------------
class ResNetBase(nn.Module):
    def __init__(self, model_func, num_classes=4, input_channels=1):
        super(ResNetBase, self).__init__()
        self.model = model_func(weights=None)
        # 使用 torchvision 官方实现
        # weights=None 表示不使用预训练权重（因为声谱图与自然图像差异大，且通道数不同）
        # 修改第一层卷积以适应单通道输入 (1, 64, 7, 7)
        self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 修改全连接层以适应我们的类别数 (4)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

class ResNet18(ResNetBase):
    def __init__(self, num_classes=4, input_channels=1):
        super().__init__(models.resnet18, num_classes, input_channels)

class ResNet34(ResNetBase):
    def __init__(self, num_classes=4, input_channels=1):
        super().__init__(models.resnet34, num_classes, input_channels)

class ResNet50(ResNetBase):
    def __init__(self, num_classes=4, input_channels=1):
        super().__init__(models.resnet50, num_classes, input_channels)

# =================================================================================================
# 3. RNN (Vanilla RNN)
# -------------------------------------------------------------------------------------------------
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, num_classes=4, dropout=0.3):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (Batch, Time, Freq)
        output, _ = self.rnn(x)
        out = output[:, -1, :]
        out = self.fc(out)
        return out

# =================================================================================================
# 4. LSTM循环神经网络
# -------------------------------------------------------------------------------------------------
# 将声谱图视为时间序列。
# 输入形状通常为 (Batch, Time_Steps, Features)。
# 这里的 Features 对应频谱图的频率维度 (Freq Bins)。
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, num_classes=4, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        
        # 针对 5090 显卡，增加了 hidden_size 和 layers，提升模型容量
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 层
        # batch_first=True 意味着输入维度是 (Batch, Seq_Len, Features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        
        # 全连接层
        # 因为是双向 LSTM (bidirectional=True)，所以输入维度是 hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (Batch, Time, Freq) -> 我们把 Freq 作为 input_size
        
        # LSTM 输出: output, _
        # output shape: (Batch, Seq_Len, Hidden_Size * 2)
        
        # 我们取最后一个时间步的输出用于分类
        # 或者可以使用 Global Average Pooling on output
        # 这里使用最后时间步 (Bi-LSTM 需要拼接前向和后向的最后状态)
        
        # 简单取 output 的最后一个时间步
        output, _ = self.lstm(x)
        out = output[:, -1, :] 
        
        out = self.fc(out)
        return out

# =================================================================================================
# 5. Standard Transformer Encoder (用于序列数据)
# -------------------------------------------------------------------------------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, input_size, d_model=256, nhead=8, num_layers=4, num_classes=4, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        # 特征嵌入层：将输入的频率特征维度映射到 d_model 维度
        # 针对 5090 显卡，增加了 d_model, nhead, num_layers
        self.embedding = nn.Linear(input_size, d_model)
        
        # Positional Encoding (Learnable)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, d_model)) # 假设最大长度 1000
        
        # Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):

        # 1. Embedding
        # x: (Batch, Time, Freq)
        x = self.embedding(x)
        
        # 2. Add Positional Encoding
        seq_len = x.size(1)
        # 截取对应长度的位置编码
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # 3. Transformer Encoder
        x = self.transformer_encoder(x)
        
        # 4. Global Average Pooling (将序列维度聚合)
        x = x.mean(dim=1) # (Batch, d_model)
        
        # 5. Classification
        x = self.fc(x)
        return x

def get_model(model_name, num_classes=4, input_channels=1, input_size=None):
    """
    模型工厂函数，根据名称返回模型实例
    :param model_name: 模型名称 (SimpleCNN, ResNet18, LSTM, Transformer)
    :param num_classes: 类别数量
    :param input_channels: 输入通道数 (仅用于 CNN/ResNet)
    :param input_size: 输入特征维度 (仅用于 LSTM/Transformer，对应 Freq bins)
    """
    if model_name == 'SimpleCNN':
        return SimpleCNN(num_classes, input_channels)
    elif model_name == 'ResNet18':
        return ResNet18(num_classes, input_channels)
    elif model_name == 'ResNet34':
        return ResNet34(num_classes, input_channels)
    elif model_name == 'ResNet50':
        return ResNet50(num_classes, input_channels)
    elif model_name == 'RNN':
        if input_size is None: raise ValueError("RNN requires input_size")
        return RNNClassifier(input_size=input_size, num_classes=num_classes)
    elif model_name == 'LSTM':
        if input_size is None: raise ValueError("LSTM requires input_size")
        return LSTMClassifier(input_size=input_size, num_classes=num_classes)
    elif model_name == 'Transformer': 
        if input_size is None: raise ValueError("Transformer requires input_size")
        return TransformerClassifier(input_size=input_size, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
