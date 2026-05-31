# 会话历史时间线归档

本文档用于按时间线整理自本次长会话开始以来的主要沟通内容、需求变更、代码撰写与修改场景，以及文档与实验产物的演进过程。文档遵循以下原则：

- 只记录在会话中真实出现过的需求、问题、决策和产物。
- 代码与结果尽量链接到当前仓库中可核验的文件。
- 对于“对话中曾提出、但当前仓库不一定仍保留对应文件或结果”的内容，单独注明为“历史对话信息”。
- 不编造未出现过的实验结果、参数、引用或图表。

## 1. 会话起点与初始任务定义

本次项目最初围绕一个明确目标展开：基于 DeepShip 数据集，构建一条完整的“原始音频预处理 -> 多种时频特征提取 -> 降维分析 -> 深度学习分类 -> 结果对比分析”的水下声学目标识别流水线。初始重点并不只是“训练一个模型”，而是：

- 对四种特征统一开展实验：STFT、Mel、MFCC、CQT。
- 构建并比较多类深度学习模型：CNN、ResNet、RNN、LSTM、Transformer。
- 支持高性能 GPU 环境运行。
- 输出足够完整的评价指标与图表，满足毕业设计论文撰写需要。

与这一目标直接对应的核心代码后来集中在：

- [preprocess_deepship_32k.py](./preprocess_deepship_32k.py)
- [feature_extraction_comparison.py](./feature_extraction_comparison.py)
- [dimensionality_reduction_analysis.py](./dimensionality_reduction_analysis.py)
- [deep_learning_models.py](./deep_learning_models.py)
- [train_comparison.py](./train_comparison.py)

## 2. 阶段一：深度学习主干能力建设

### 2.1 扩展模型种类

会话前期，用户首先明确要求继续撰写代码，构建多种先进深度学习模型作为分类器，包括但不限于：

- 卷积神经网络
- ResNet
- 循环神经网络
- Transformer

随后围绕这一要求，项目的模型定义逐步集中到 [deep_learning_models.py](./deep_learning_models.py)。当前仓库中可核验到的模型包括：

- `SimpleCNN`
- `ResNet18`
- `ResNet34`
- `ResNet50`
- `RNNClassifier`
- `LSTMClassifier`
- `TransformerClassifier`

其实现特点包括：

- ResNet 使用 `torchvision.models`，并改写首层卷积以适配单通道时频图输入。
- RNN/LSTM/Transformer 接收转置后的 `(Batch, Time, Freq)` 序列输入。
- 通过统一的 `get_model()` 工厂函数管理模型实例化。

对应代码见：

- [deep_learning_models.py](./deep_learning_models.py)

### 2.2 训练主脚本初步成型

为了对所有特征与模型组合做统一训练和评估，会话中同步构建了主训练脚本 [train_comparison.py](./train_comparison.py)。从当前代码看，该脚本已经承担了以下职责：

- 加载四种特征 `X_mel.npy`、`X_mfcc.npy`、`X_cqt.npy`、`X_stft.npy`
- 构建统一数据集 `DeepShipDataset`
- 自动区分二维卷积模型与序列模型的数据张量组织方式
- 对 4 种特征 × 7 种模型做矩阵式训练
- 输出训练日志、最佳模型权重、混淆矩阵、PR 曲线、训练历史图、汇总结果表

当前脚本中的特征与模型列表可直接追溯到：

- [train_comparison.py](./train_comparison.py)

## 3. 阶段二：环境与依赖路线调整

### 3.1 从手工 ResNet 回退到 torchvision

会话中曾出现一次关键环境决策变更：早期因为 `torchvision` 导入报错，曾短暂采取手动实现 ResNet 的替代方案；随后用户明确说明实际训练环境具备符合要求的 `torch`、`cuda` 和 `torchvision`，要求回退到官方 `torchvision` 版本而不是手工实现。

这个变更的结果是：

- 项目最终以 `torchvision.models.resnet18/resnet34/resnet50` 为基础实现 ResNet。
- 手工 ResNet 没有成为当前仓库的最终保留实现。

当前可核验代码：

- [deep_learning_models.py](./deep_learning_models.py)

### 3.2 GPU 训练与硬件参数逐步明确

会话中用户先后提到过不同的高性能环境信息，包括：

- 5090 / 24GB 或 32GB 显存环境
- Linux + 22 vCPU + RTX Pro 6000 + 大内存环境

围绕这些条件，训练脚本逐步形成了高吞吐设置：

- 全局 `DEFAULT_BATCH_SIZE = 2048`
- STFT 特征单独设置较小批量
- `num_workers` 根据 CPU 核心数自动设置上限 16
- 控制台只在开始阶段打印一次设备信息，避免训练日志反复刷屏

相关代码见：

- [train_comparison.py](./train_comparison.py)
- [resume_training.py](./resume_training.py)

## 4. 阶段三：评价指标与可视化要求持续加码

在模型主干搭建完成后，用户进一步把需求从“能训练”扩展到“能完整做论文级评估”。会话中明确要求增加：

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR 曲线
- Loss / Accuracy / Precision / Recall / F1 / ROC-AUC 历史曲线
- 每一步指标保存为 CSV
- 图像保存为 PNG / SVG / PDF

这些要求最终落实到了 [train_comparison.py](./train_comparison.py) 的以下能力上：

- `evaluate_model()` 统一返回多项测试指标
- `plot_metrics_history()` 生成 2×3 多指标训练历史图
- `plot_pr_curve()` 生成多类别 PR 曲线
- 每个组合输出 `training_log_*.csv`
- 每个组合输出 `history_*.png/.svg/.pdf`
- 每个组合输出 `cm_*.png/.svg/.pdf`
- 每个组合输出 `pr_curve_*.png/.svg/.pdf`

当前可核验产物目录：

- [dl_comparison_results_gpu](./dl_comparison_results_gpu)

## 5. 阶段四：模型名单多次变更

会话中模型清单不是一次性定稿，而是经历过数轮变更：

### 5.1 中途扩充

用户曾要求继续增加：

- `ResNet34`
- `ResNet50`
- `RNN`
- `Swin Transformer`
- 如果没有 Vision Transformer，则再补 `ViT`

### 5.2 随后裁剪

在后续讨论中，用户又明确要求删除：

- `ViT`
- `SwinTransformer`

最终当前仓库中的主实验模型稳定为 7 个单流模型：

- SimpleCNN
- ResNet18
- ResNet34
- ResNet50
- RNN
- LSTM
- Transformer

这一点在当前代码中可直接确认：

- [train_comparison.py](./train_comparison.py)
- [deep_learning_models.py](./deep_learning_models.py)

## 6. 阶段五：日志策略与训练体验优化

会话中用户对训练期间的输出形式提出过两类具体改动：

### 6.1 去掉文件日志

用户不希望每次训练自动生成 `training_log_YYYYMMDD_HHMMSS.txt` 一类的日志文本文件，只希望：

- 控制台输出保留
- 真正有分析价值的结构化指标保存到 CSV

因此当前 [train_comparison.py](./train_comparison.py) 中只保留了：

- `logging.StreamHandler(sys.stdout)`

没有再挂接 `FileHandler`。

### 6.2 减少冗余设备信息

用户指出“Using GPU: NVIDIA ...”一类信息反复出现会干扰训练输出，于是训练脚本改为：

- 只在程序开始时打印一次设备信息
- 训练迭代阶段不反复输出相同 GPU 名称

## 7. 阶段六：显存占用、吞吐率与批量大小优化

会话中，显存与吞吐率是最频繁的工程问题之一。

### 7.1 全局 Batch Size 放大

由于用户环境显存较大，会话中批量大小从较小值一路上调，最终主训练脚本采用：

- `DEFAULT_BATCH_SIZE = 2048`

位置：

- [train_comparison.py](./train_comparison.py)

### 7.2 STFT 单独限流

因为 STFT 特征维度大、显存压力显著高于 Mel/MFCC/CQT，用户提出对 STFT 施加特殊批量限制，最终形成：

- 训练主脚本中 `STFT_BATCH_SIZE = 512`
- 断点续训脚本中，为避免 `ResNet50 + STFT` 再次 OOM，将 `STFT_BATCH_SIZE` 进一步降到 `128`

对应代码：

- [train_comparison.py](./train_comparison.py)
- [resume_training.py](./resume_training.py)

### 7.3 num_workers 调优

用户明确告知 Linux 22 vCPU 环境，希望利用 CPU 并行加载数据来缓解 GPU 饥饿。于是当前代码中统一采取：

- `num_workers = min(16, multiprocessing.cpu_count())`

对应文件：

- [train_comparison.py](./train_comparison.py)
- [resume_training.py](./resume_training.py)

### 7.4 显存监控与推荐批量

会话中围绕“显存占用很高但核心运算占用低”的问题，先做了概念讨论，后又落到了两个实际产物：

- 训练过程记录显存峰值：`gpu_memory_usage.csv`
- 依据目标显存推导推荐批量：`recommend_batchsize.csv`

可核验文件：

- [dl_comparison_results_gpu/gpu_memory_usage.csv](./dl_comparison_results_gpu/gpu_memory_usage.csv)
- [recommend_batchsize.csv](./recommend_batchsize.csv)
- [calculate_recommended_batchsize.py](./calculate_recommended_batchsize.py)

## 8. 阶段七：数据与特征链路补齐

为了让训练实验真正落地，会话中也同步建设了前置数据链路。

### 8.1 原始音频预处理

当前 [preprocess_deepship_32k.py](./preprocess_deepship_32k.py) 完成了：

- 32 kHz 采样率强校验
- 2 秒滑窗切片
- 50% 重叠
- 1 Hz 到 12 kHz 巴特沃斯带通滤波
- 幅值归一化

当前可核验输出：

- [processed_data/X_audio_32k.npy](./processed_data/X_audio_32k.npy)
- [processed_data/y_labels.npy](./processed_data/y_labels.npy)

### 8.2 四种特征提取与类别示例图

当前 [feature_extraction_comparison.py](./feature_extraction_comparison.py) 完成了：

- STFT
- Mel Spectrogram
- MFCC
- CQT

并为四个类别各生成一张四特征对比图：

- [feature_data/comparison_Cargo.png](./feature_data/comparison_Cargo.png)
- [feature_data/comparison_Tanker.png](./feature_data/comparison_Tanker.png)
- [feature_data/comparison_Tug.png](./feature_data/comparison_Tug.png)
- [feature_data/comparison_Passengership.png](./feature_data/comparison_Passengership.png)

对应特征矩阵：

- [feature_data/X_stft.npy](./feature_data/X_stft.npy)
- [feature_data/X_mel.npy](./feature_data/X_mel.npy)
- [feature_data/X_mfcc.npy](./feature_data/X_mfcc.npy)
- [feature_data/X_cqt.npy](./feature_data/X_cqt.npy)

## 9. 阶段八：关于采样率改为 16kHz 的设计讨论

会话中用户曾明确提出一个“不要修改代码、只做分析”的问题：如果把采样率从 32kHz 改为 16kHz，五个主要代码文件需要如何修改。

当时讨论的重点包括：

- [preprocess_deepship_32k.py](./preprocess_deepship_32k.py) 中 `TARGET_SR` 和高频截止频率的联动变化
- [feature_extraction_comparison.py](./feature_extraction_comparison.py) 中 `SR`、`HOP_LENGTH`、`N_FFT` 的适配
- 训练与模型定义部分基本可以复用，因为模型输入维度具备一定自适应性

这次讨论没有直接改动当前仓库代码，但形成了后续论文中“参数改动影响分析”的基础素材。

## 10. 阶段九：ResNet50 + STFT 爆显存与断点续训

这是一轮非常关键的工程问题处理过程。

### 10.1 问题背景

用户明确反馈：

- `train_comparison.py` 在运行到 `STFT + ResNet50` 时因显存爆掉而终止
- 希望从这里继续跑
- 不修改原脚本
- 尽可能保持最终输出文件形式不变

### 10.2 解决方案

会话中随后围绕这个问题构建了 [resume_training.py](./resume_training.py)，它具备：

- 扫描已有 `training_log_*.csv`
- 识别已经完成的“特征 + 模型”任务
- 从历史日志中恢复最佳指标
- 跳过已完成任务
- 仅继续未完成部分
- 对 STFT 特征采用更保守的 `batch_size = 128`
- 最后重新拼合生成 [dl_comparison_results_gpu/final_comparison_results.csv](./dl_comparison_results_gpu/final_comparison_results.csv)

### 10.3 绘图遗漏与补救

会话中用户还指出，由于中断与脚本缺陷，`ResNet50_X_stft` 一度缺少 PR 曲线和混淆矩阵。围绕这个问题有两轮处理：

- 先提出通过单独脚本补画缺失图像的思路
- 后续又修复了 `resume_training.py` 中缺失 `matplotlib.pyplot as plt` 导入的问题

当前仓库中 `ResNet50_X_stft` 的关键产物已可核验：

- [dl_comparison_results_gpu/history_ResNet50_X_stft.png](./dl_comparison_results_gpu/history_ResNet50_X_stft.png)
- [dl_comparison_results_gpu/cm_ResNet50_X_stft.png](./dl_comparison_results_gpu/cm_ResNet50_X_stft.png)
- [dl_comparison_results_gpu/pr_curve_ResNet50_X_stft.png](./dl_comparison_results_gpu/pr_curve_ResNet50_X_stft.png)

相关脚本：

- [resume_training.py](./resume_training.py)
- [plot_missing_resnet50_stft.py](./plot_missing_resnet50_stft.py)

## 11. 阶段十：矩阵训练结果逐步完善

随着主训练脚本、断点恢复脚本和融合脚本逐步补齐，仓库中最终形成了较完整的实验结果集合。

当前可核验到：

- 28 组单流模型训练日志：`training_log_{Model}_{Feature}.csv`
- 对应的历史图、混淆矩阵、PR 曲线
- 28 组单流最优权重
- 1 组融合实验日志、图表和最优权重
- 汇总表：
  - [dl_comparison_results_gpu/final_comparison_results.csv](./dl_comparison_results_gpu/final_comparison_results.csv)
  - [dl_comparison_results_gpu/final_fusion_results.csv](./dl_comparison_results_gpu/final_fusion_results.csv)
  - [dl_comparison_results_gpu/final_comparison_results_with_fusion.csv](./dl_comparison_results_gpu/final_comparison_results_with_fusion.csv)

从当前 [final_comparison_results_with_fusion.csv](./dl_comparison_results_gpu/final_comparison_results_with_fusion.csv) 看，最突出的现象包括：

- 最优单项是 `ResNet18 + X_mfcc`，`f1 = 0.9982335599857426`
- 融合模型 `DualStreamFusionModel + Fusion_Mel_MFCC` 表现很高，但并不是当前表中的全局最优，`f1 = 0.9955809781067444`
- 表现较弱的组合主要出现在 `Transformer + X_mfcc`、`Transformer + X_stft`

## 12. 阶段十一：多特征融合实验

会话中用户明确要求加入“多特征融合实验”这一加分项。随后围绕 Mel + MFCC 双流结构，形成了 [train_fusion.py](./train_fusion.py)。

### 12.1 初版问题

用户后来发现，融合实验最初只输出了模型权重，没有同步保存：

- 训练日志 CSV
- 历史曲线
- 混淆矩阵
- PR 曲线
- 最终汇总指标

### 12.2 随后补齐

对应地，融合脚本后续补上了：

- `training_log_fusion_Mel_MFCC.csv`
- `history_fusion_Mel_MFCC.*`
- `cm_fusion_Mel_MFCC.*`
- `pr_curve_fusion_Mel_MFCC.*`
- `final_fusion_results.csv`

当前可核验结果：

- [train_fusion.py](./train_fusion.py)
- [dl_comparison_results_gpu/training_log_fusion_Mel_MFCC.csv](./dl_comparison_results_gpu/training_log_fusion_Mel_MFCC.csv)
- [dl_comparison_results_gpu/history_fusion_Mel_MFCC.png](./dl_comparison_results_gpu/history_fusion_Mel_MFCC.png)
- [dl_comparison_results_gpu/cm_fusion_Mel_MFCC.png](./dl_comparison_results_gpu/cm_fusion_Mel_MFCC.png)
- [dl_comparison_results_gpu/pr_curve_fusion_Mel_MFCC.png](./dl_comparison_results_gpu/pr_curve_fusion_Mel_MFCC.png)

## 13. 阶段十二：全局对比、复杂度评估与鲁棒性测试

在项目后期，用户希望把“单项实验”升级成“论文化的综合评估模块”，因此又增加了三类脚本。

### 13.1 全局对比图表

[plot_global_comparison.py](./plot_global_comparison.py) 负责：

- 合并单流与融合实验结果
- 输出 [final_comparison_results_with_fusion.csv](./dl_comparison_results_gpu/final_comparison_results_with_fusion.csv)
- 生成：
  - [global_f1_barplot.png](./dl_comparison_results_gpu/global_f1_barplot.png)
  - [global_f1_heatmap.png](./dl_comparison_results_gpu/global_f1_heatmap.png)
  - [global_metric_model_feature_heatmaps.png](./dl_comparison_results_gpu/global_metric_model_feature_heatmaps.png)

### 13.2 模型复杂度与实时性评估

[evaluate_complexity.py](./evaluate_complexity.py) 负责输出：

- [model_complexity.csv](./dl_comparison_results_gpu/model_complexity.csv)
- [model_parameters_comparison.png](./dl_comparison_results_gpu/model_parameters_comparison.png)
- [model_inference_time_comparison.png](./dl_comparison_results_gpu/model_inference_time_comparison.png)

### 13.3 鲁棒性与抗噪测试

用户要求“全部模型进行测试，把所有结果绘制到一张大图进行对比”，随后形成了 [noise_robustness_test.py](./noise_robustness_test.py)。

当前脚本实际做法是：

- 读取 [final_comparison_results.csv](./dl_comparison_results_gpu/final_comparison_results.csv) 和 [final_fusion_results.csv](./dl_comparison_results_gpu/final_fusion_results.csv)
- 对每个 `Model` 选取当前 `f1` 最高的那一条结果作为“该模型最佳特征”
- 在测试集特征矩阵上直接注入高斯白噪声
- 计算不同 SNR 下的 F1
- 绘制一张总曲线图和一张总热力图

对应产物：

- [robustness_all_models.csv](./dl_comparison_results_gpu/robustness_all_models.csv)
- [robustness_curve_all_models.png](./dl_comparison_results_gpu/robustness_curve_all_models.png)
- [robustness_heatmap_all_models.png](./dl_comparison_results_gpu/robustness_heatmap_all_models.png)

## 14. 阶段十三：用户对“是否已经完成系统对比实验”的追问

会话中用户专门问过一句：  
“在测试集上进行全面的对比实验，系统分析各模型的识别精度、收敛性与泛化能力；这一步咱们之前完成了吗？”

围绕这一问题，讨论实际上完成了三件事：

- 明确指出当前已有的 28 组单流实验和 1 组融合实验已经构成“系统对比实验”的主体
- 将“识别精度”对应到 `final_comparison_results*.csv`
- 将“收敛性”对应到 `training_log_*.csv` 与 `history_*.png`
- 将“泛化能力”对应到测试集指标以及后续补充的抗噪实验

这次对话的意义在于，它把“项目代码已做什么”与“论文措辞该如何对应”正式连接起来。

## 15. 阶段十四：项目总结文档集的逐步形成

随着实验工作接近收尾，会话转入“写论文材料准备”阶段，围绕项目历史与实验结果，逐渐生成了多类辅助文档。

### 15.1 工作总结文档

用户要求把“任务书对应目标 -> 实际内容 -> 代码引用 -> 实验产物”系统整理出来，于是形成了：

- [work.md](./work.md)

其后用户又指出其中遗漏了训练阶段生成的大量图表与表格，于是又补充列出了：

- 全局对比图
- 单模型训练日志
- 历史图
- 混淆矩阵
- PR 曲线
- 复杂度对比图
- 鲁棒性曲线与热力图

### 15.2 论文结果分析教程

为指导论文写作，又形成了：

- [analysis_tutorial.md](./analysis_tutorial.md)

其后用户继续指出教程中遗漏了：

- `feature_data/comparison_*.png`
- `training_log_*.csv`
- `global_metric_model_feature_heatmaps.png`
- `model_parameters_comparison.png`
- `model_inference_time_comparison.png`
- `robustness_heatmap_all_models.png`

随后教程再做补充。

### 15.3 中期检查报告

围绕开题报告、中期检查和论文阶段汇总，还形成了：

- [report.md](./report.md)

需要说明的是，`report.md` 中部分表述是基于当时对话进展撰写的阶段性总结；在现在撰写正式论文时，应优先以仓库现存文件重新核验，尤其是鲁棒性结果与降维输出目录状态。

## 16. 阶段十五：会话归档 HTML 与 PDF 打印版

用户后来又要求将“整个会话全过程”整理成 HTML 文档，并支持浏览器直接打印导出 PDF。于是形成了：

- [conversation_history_summary.html](./conversation_history_summary.html)

当前可核验到该 HTML 已具备：

- A4 页面设置
- 页面边距
- 清晰标题层级
- 正文字体与代码块样式
- 打印按钮与 PDF 导出适配

这一步的意义主要是为后续论文写作和项目过程追溯提供归档底稿。

## 17. 阶段十六：LaTeX 本科论文模板梳理

在实验与总结文档阶段之后，会话又进入论文正式撰写准备阶段。用户要求完整浏览 [nwputhesis-main](./nwputhesis-main) 模板，并输出一份可执行的使用说明书。

围绕这一需求，会话中进一步梳理了：

- 本科论文模板的主入口
- `content/thesis/undergraduate` 下内容层文件
- `nwputhesis.cls` 与 `infra/*.def` 的底层格式职责
- XeLaTeX / Biber 的建议编译流程
- 章节填充位置和引用写法
- 常见报错与定稿前自查项

相关说明文档：

- [nwputhesis-main/本科毕业论文模板使用说明书.md](./nwputhesis-main/本科毕业论文模板使用说明书.md)

## 18. 阶段十七：论文撰写规则收束

在最新阶段，会话中又引入了更严格的论文规则：

- 必须遵守 `nwputhesis-main` 模板
- 论文必须逐章节、逐小节推进
- 每一大章节都要先做 plan 并留档
- 每个小节完成后都应检查内容并编译查看
- 图表必须优先使用真实实验生成图，不允许用伪造图表替代

这组规则为接下来正式写论文设定了边界，也使当前项目从“实验工程”正式转入“论文章节化落地”的阶段。

## 19. 当前会话的最新任务

本轮最新需求不是继续改模型，而是进入资料统筹阶段。用户要求：

1. 汇总并整理项目范围内全部上下文问答记录，生成标准 Markdown 时间线文档。
2. 全面查阅相关支撑文档、历史信息和核心代码，输出：
   - 项目目的总结
   - 实验状态与关键结果总结
   - 毕业论文完成计划参考稿
   - 缺失图表、数据核对项和必须补做实验清单
3. 严格遵守“不编造结果、不修改问题定义、所有数值可追溯到仓库文件”的约束。

当前这份 [conversation_timeline.md](./conversation_timeline.md) 就是对第 1 项要求的正式交付。

## 20. 当前可直接追溯的关键文件索引

### 20.1 核心代码

- [preprocess_deepship_32k.py](./preprocess_deepship_32k.py)
- [feature_extraction_comparison.py](./feature_extraction_comparison.py)
- [dimensionality_reduction_analysis.py](./dimensionality_reduction_analysis.py)
- [deep_learning_models.py](./deep_learning_models.py)
- [train_comparison.py](./train_comparison.py)
- [resume_training.py](./resume_training.py)
- [train_fusion.py](./train_fusion.py)
- [plot_global_comparison.py](./plot_global_comparison.py)
- [evaluate_complexity.py](./evaluate_complexity.py)
- [noise_robustness_test.py](./noise_robustness_test.py)

### 20.2 关键结果表

- [final_comparison_results.csv](./dl_comparison_results_gpu/final_comparison_results.csv)
- [final_fusion_results.csv](./dl_comparison_results_gpu/final_fusion_results.csv)
- [final_comparison_results_with_fusion.csv](./dl_comparison_results_gpu/final_comparison_results_with_fusion.csv)
- [model_complexity.csv](./dl_comparison_results_gpu/model_complexity.csv)
- [robustness_all_models.csv](./dl_comparison_results_gpu/robustness_all_models.csv)
- [gpu_memory_usage.csv](./dl_comparison_results_gpu/gpu_memory_usage.csv)
- [recommend_batchsize.csv](./recommend_batchsize.csv)

### 20.3 现有说明与总结文档

- [README_CN.md](./README_CN.md)
- [README.md](./README.md)
- [work.md](./work.md)
- [analysis_tutorial.md](./analysis_tutorial.md)
- [report.md](./report.md)
- [conversation_history_summary.html](./conversation_history_summary.html)

## 21. 时间线归档结论

从会话全程看，项目经历了以下主线演进：

1. 从“先把模型搭起来”扩展到“完整特征 × 模型矩阵实验”。
2. 从“能跑通”扩展到“能输出论文级图表和指标”。
3. 从“单次训练脚本”扩展到“断点续训、全局比较、复杂度评估、鲁棒性测试、融合实验”。
4. 从“项目实现”过渡到“工作总结、中期报告、结果分析教程、HTML 归档、LaTeX 论文模板准备”。

因此，当前项目不再只是一个实验脚本集合，而已经形成了一套较完整的毕业设计研究资产库。接下来的重点不再是无边界扩展功能，而是：

- 用现有真实结果校正论文论述
- 补齐仍缺失或仍需复核的结果项
- 严格按本科论文模板逐章、逐节推进正式撰写
