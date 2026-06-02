# 会话历史时间线归档

本文档按时间顺序记录本次长会话中出现的主要需求、代码修改方向、实验扩展和文档建设过程。内容以“当前仓库中能够核验的文件”作为最终落点，不对不存在的结果做补写。

## 1. 起点：明确研究目标与实验主线

会话最初围绕一个清晰目标展开：基于 DeepShip 数据集，建立“预处理 -> 时频特征提取 -> 特征可视化 -> 深度学习分类 -> 结果对比分析”的完整水下声学目标识别流程。用户最早提出的核心诉求包括：

- 四类特征统一开展实验：`STFT`、`Mel`、`MFCC`、`CQT`
- 多类深度学习模型统一比较：CNN、ResNet、RNN、LSTM、Transformer
- 支持 GPU 训练
- 输出足够完整的指标、日志和图表，满足毕业设计论文使用

对应主干代码最终集中到：

- [preprocess_deepship_32k.py](./preprocess_deepship_32k.py)
- [feature_extraction_comparison.py](./feature_extraction_comparison.py)
- [dimensionality_reduction_analysis.py](./dimensionality_reduction_analysis.py)
- [deep_learning_models.py](./deep_learning_models.py)
- [train_comparison.py](./train_comparison.py)

## 2. 早期模型扩展：从基础 CNN 到多范式模型

用户最早要求“继续撰写代码，构建多种先进深度学习模型作为分类器”，并强调：

- 至少覆盖 CNN、ResNet、循环网络、Transformer
- 支持全部四种特征输入
- 提供超参数接口
- 代码需有充分注释

围绕这一要求，当前仓库中形成了如下稳定模型集合：

- `SimpleCNN`
- `ResNet18`
- `ResNet34`
- `ResNet50`
- `RNN`
- `LSTM`
- `Transformer`

相关实现见：

- [deep_learning_models.py](./deep_learning_models.py)

## 3. 环境路线调整：回归 torchvision，明确 GPU 训练

会话中曾出现一次重要路线调整：

- 早期因环境问题，短暂考虑手写 ResNet
- 随后用户明确说明实际 GPU 环境支持 `torchvision`
- 因此最终回归官方 `torchvision.models` 路线

同时，用户多次给出硬件条件，希望提高吞吐率与显存利用率，于是训练脚本逐步形成：

- 默认大批量训练
- STFT 单独限流
- `num_workers` 自动上限 16
- 仅开始时打印一次设备信息

相关代码：

- [deep_learning_models.py](./deep_learning_models.py)
- [train_comparison.py](./train_comparison.py)
- [resume_training.py](./resume_training.py)

## 4. 指标与可视化要求升级到论文级

在模型框架成型后，用户明确要求将实验输出从“能跑通”提升为“可直接支撑论文分析”，要求加入：

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR 曲线
- 多指标训练历史曲线
- CSV 日志
- PNG/SVG/PDF 输出

这些要求最终落实到了 [train_comparison.py](./train_comparison.py)，并在 [dl_comparison_results_gpu](./dl_comparison_results_gpu) 中形成：

- `training_log_*.csv`
- `history_*.png/.svg/.pdf`
- `cm_*.png/.svg/.pdf`
- `pr_curve_*.png/.svg/.pdf`
- `best_*.pth`
- `final_comparison_results.csv`

## 5. 模型名单多次变更：加入又删除 ViT/Swin

会话中模型集合并非一次定稿，而是经历过一次扩充和一次收缩：

- 中途曾要求加入 `ResNet34`、`ResNet50`、`RNN`、`Swin Transformer`、`ViT`
- 随后用户又明确要求删除 `ViT` 和 `SwinTransformer`

最终当前仓库保留的是 7 个单流模型，不再包含 ViT/Swin。

## 6. 工程化训练优化：日志、显存与批大小

围绕训练效率和稳定性，会话中又做了多轮工程优化：

- 去掉 `training_log_YYYYMMDD_HHMMSS.txt` 这类低价值文本日志
- 仅保留控制台日志和结构化 CSV
- 全局批大小提升到 `2048`
- STFT 特征单独使用较小批量
- 记录显存峰值并生成推荐批大小表

当前相关产物包括：

- [dl_comparison_results_gpu/gpu_memory_usage.csv](./dl_comparison_results_gpu/gpu_memory_usage.csv)
- [recommend_batchsize.csv](./recommend_batchsize.csv)
- [calculate_recommended_batchsize.py](./calculate_recommended_batchsize.py)

## 7. 数据前链路建设：预处理与四类特征全部落盘

为了让分类实验真正具备输入基础，会话中同步补齐了数据前链路：

- 原始音频预处理
- 四类特征提取
- 四类船舶样本时频图对比

当前可核验结果：

- [processed_data/X_audio_32k.npy](./processed_data/X_audio_32k.npy)
- [processed_data/y_labels.npy](./processed_data/y_labels.npy)
- [feature_data/X_stft.npy](./feature_data/X_stft.npy)
- [feature_data/X_mel.npy](./feature_data/X_mel.npy)
- [feature_data/X_mfcc.npy](./feature_data/X_mfcc.npy)
- [feature_data/X_cqt.npy](./feature_data/X_cqt.npy)
- [feature_data/comparison_Cargo.png](./feature_data/comparison_Cargo.png)
- [feature_data/comparison_Tanker.png](./feature_data/comparison_Tanker.png)
- [feature_data/comparison_Tug.png](./feature_data/comparison_Tug.png)
- [feature_data/comparison_Passengership.png](./feature_data/comparison_Passengership.png)

## 8. 降维分析阶段：PCA/t-SNE 与 STFT 内存问题修复

用户在中后期要求加入 PCA 与 t-SNE 的可视化分析，以便从特征空间角度理解可分性。之后又反馈：

- `dimensionality_reduction_analysis.py` 运行到 STFT 时出现 Linux 内存错误
- 不能减少样本
- 必须保留全量样本

围绕这一问题，脚本被改造成更稳健的增量式处理方案，当前目录中已存在真实结果：

- [dim_reduction_results/analysis_summary.txt](./dim_reduction_results/analysis_summary.txt)
- [dim_reduction_results/pca_variance_analysis.csv](./dim_reduction_results/pca_variance_analysis.csv)
- [dim_reduction_results/dr_X_mel_full.png](./dim_reduction_results/dr_X_mel_full.png)
- [dim_reduction_results/dr_X_mfcc_full.png](./dim_reduction_results/dr_X_mfcc_full.png)
- [dim_reduction_results/dr_X_cqt_full.png](./dim_reduction_results/dr_X_cqt_full.png)
- [dim_reduction_results/dr_X_stft_full.png](./dim_reduction_results/dr_X_stft_full.png)

这意味着早期一些文档中“降维结果缺失”的说法已经过时。

## 9. 关键工程故障：ResNet50 + STFT OOM 与断点续训

主训练阶段曾发生一次关键中断：

- `train_comparison.py` 在 `ResNet50 + STFT` 处显存溢出终止
- 用户要求不修改原主脚本，但要从中断处继续跑，并尽量保持最终输出格式不变

为解决这一问题，会话中新增并修正了：

- [resume_training.py](./resume_training.py)
- [plot_missing_resnet50_stft.py](./plot_missing_resnet50_stft.py)

同时修复了 `resume_training.py` 中缺失 `matplotlib.pyplot as plt` 导入的问题。当前缺失图已经补齐：

- [dl_comparison_results_gpu/history_ResNet50_X_stft.png](./dl_comparison_results_gpu/history_ResNet50_X_stft.png)
- [dl_comparison_results_gpu/cm_ResNet50_X_stft.png](./dl_comparison_results_gpu/cm_ResNet50_X_stft.png)
- [dl_comparison_results_gpu/pr_curve_ResNet50_X_stft.png](./dl_comparison_results_gpu/pr_curve_ResNet50_X_stft.png)

## 10. 主实验闭环：28 组单流 + 1 组融合

随着主训练、续训和补图完成，仓库中形成了完整的主实验矩阵：

- 28 组单流模型实验
- 1 组 `Mel + MFCC` 融合实验
- 统一汇总结果表

当前主结果的事实基线来自：

- [dl_comparison_results_gpu/final_comparison_results.csv](./dl_comparison_results_gpu/final_comparison_results.csv)
- [dl_comparison_results_gpu/final_fusion_results.csv](./dl_comparison_results_gpu/final_fusion_results.csv)
- [dl_comparison_results_gpu/final_comparison_results_with_fusion.csv](./dl_comparison_results_gpu/final_comparison_results_with_fusion.csv)

当前最关键的主结果结论是：

- 全表最佳为 `ResNet18 + X_mfcc`
- `f1 = 0.9982335599857426`
- 融合模型结果很高，但不是当前全局最佳

## 11. 全局图、复杂度与鲁棒性实验

在主实验完成后，用户继续要求将结果提升到更适合论文分析的层级，于是增加了三个方向：

### 11.1 全局比较

- [plot_global_comparison.py](./plot_global_comparison.py)
- 输出 [global_f1_barplot.png](./dl_comparison_results_gpu/global_f1_barplot.png)
- 输出 [global_f1_heatmap.png](./dl_comparison_results_gpu/global_f1_heatmap.png)
- 输出 [global_metric_model_feature_heatmaps.png](./dl_comparison_results_gpu/global_metric_model_feature_heatmaps.png)

### 11.2 复杂度与实时性

- [evaluate_complexity.py](./evaluate_complexity.py)
- 输出 [model_complexity.csv](./dl_comparison_results_gpu/model_complexity.csv)
- 输出 [model_parameters_comparison.png](./dl_comparison_results_gpu/model_parameters_comparison.png)
- 输出 [model_inference_time_comparison.png](./dl_comparison_results_gpu/model_inference_time_comparison.png)

### 11.3 鲁棒性测试

- [noise_robustness_test.py](./noise_robustness_test.py)
- 输出 [robustness_all_models.csv](./dl_comparison_results_gpu/robustness_all_models.csv)
- 输出 [robustness_curve_all_models.png](./dl_comparison_results_gpu/robustness_curve_all_models.png)
- 输出 [robustness_heatmap_all_models.png](./dl_comparison_results_gpu/robustness_heatmap_all_models.png)

会话后期又进一步明确：鲁棒性实验的准确表述应为“特征层 AWGN 干扰测试”。

## 12. 消融实验：从“计划”变成“已完成”

在后续阶段，用户明确要求补充严格意义上的消融实验，包括：

- 融合支路去除比较
- 浅层与完整融合结构比较
- 不同 `hidden_size / num_layers / d_model` 的系统性比较

随后形成：

- [ablation_experiments.py](./ablation_experiments.py)
- [dl_comparison_results_gpu/ablation_results/ablation_summary_results.csv](./dl_comparison_results_gpu/ablation_results/ablation_summary_results.csv)
- [dl_comparison_results_gpu/ablation_results/ablation_fusion_structure_results.csv](./dl_comparison_results_gpu/ablation_results/ablation_fusion_structure_results.csv)
- [dl_comparison_results_gpu/ablation_results/ablation_sequence_hyperparams_results.csv](./dl_comparison_results_gpu/ablation_results/ablation_sequence_hyperparams_results.csv)

当前可确认：

- `DualStreamFusion` 在结构消融内部结果中优于单支路与浅层双流
- `LSTM` 内部最优配置为 `hs512_layers3`
- `Transformer` 内部最优配置为 `dm128_layers2`

因此，消融实验已经从“计划项”转变为“已落盘结果项”。

## 13. 可解释性实验：从思路提出到真实结果落盘

用户还要求解释“为什么某些特征 + 模型组合更好”，并提出：

- 固定特征比较模型
- 固定模型比较特征
- Grad-CAM
- attention 可解释性
- SHAP 思想 / 频带消融

随后形成：

- [explainability_experiments.py](./explainability_experiments.py)

中途又修复了两类错误：

- `RecursionError: maximum recursion depth exceeded`
- `RuntimeError: cudnn RNN backward can only be called in training mode`

当前已确认存在的结果包括：

- [fixed_feature_model_effect.csv](./dl_comparison_results_gpu/explainability_results/fixed_feature_model_effect.csv)
- [fixed_model_feature_effect.csv](./dl_comparison_results_gpu/explainability_results/fixed_model_feature_effect.csv)
- [frequency_band_ablation_results.csv](./dl_comparison_results_gpu/explainability_results/frequency_band_ablation_results.csv)
- `gradcam_ResNet18_X_mfcc_*`
- `gradcam_fusion_mel_*`
- `gradcam_fusion_mfcc_*`
- `saliency_LSTM_X_cqt_*`

当前已落盘的可解释性结果主要覆盖固定特征/固定模型分析、Grad-CAM、融合支路 Grad-CAM、序列显著图与频带遮挡实验。

## 14. 论文图表自动整理与拼图版式

为了服务论文撰写，用户要求将流程图、结构图、特征图、降维图、训练结果图都整理为论文可直接引用的版式。围绕这一需求形成了：

- [generate_paper_figures.py](./generate_paper_figures.py)

当前 `paper_figures` 目录中已存在：

- [workflow_overview.png](./dl_comparison_results_gpu/paper_figures/workflow_overview.png)
- [feature_comparison_montage.png](./dl_comparison_results_gpu/paper_figures/feature_comparison_montage.png)
- [dimensionality_reduction_montage.png](./dl_comparison_results_gpu/paper_figures/dimensionality_reduction_montage.png)
- `cm_montage_*`
- `history_montage_*`
- `pr_curve_montage_*`
- 模型结构图索引和论文图索引

因此，当前绝大多数论文引用图已经具备，可直接进入论文排版与正文分析。

## 15. 文档资产建设：从项目说明到论文模板说明

随着实验逐步跑完，会话进入文档整理阶段，先后形成了：

- [work.md](./work.md)
- [README.md](./README.md)
- [README_CN.md](./README_CN.md)
- [analysis_tutorial.md](./analysis_tutorial.md)
- [report.md](./report.md)
- [conversation_history_summary.html](./conversation_history_summary.html)
- [project_summary_and_thesis_plan.md](./project_summary_and_thesis_plan.md)
- [nwputhesis-main/本科毕业论文模板使用说明书.md](./nwputhesis-main/本科毕业论文模板使用说明书.md)

用户随后又多次要求对这些文档做“以真实结果为准”的回写修正，使文档表述与当前落盘结果保持一致。

## 16. 论文撰写规则收束

会话后期引入了更严格的论文执行规则：

- 必须遵守 `nwputhesis-main` 模板
- 必须逐章、逐小节推进
- 每章先做 plan 并留档
- 每个小节完成后检查并编译
- 图表必须使用真实实验图，不得编造

这标志着项目目标从“继续扩功能”切换为“用真实结果支撑正式论文写作”。

## 17. 当前最新阶段：五个实验脚本全部运行完毕后的状态同步

在最新阶段，用户明确说明“五个实验脚本已全部运行完毕”，要求：

1. 同步更新以下文件中的实验相关内容，确保与实际输出完全一致：
   - [conversation_timeline.md](./conversation_timeline.md)
   - [project_summary_and_thesis_plan.md](./project_summary_and_thesis_plan.md)
   - [README.md](./README.md)
   - [README_CN.md](./README_CN.md)
   - [work.md](./work.md)
2. 对照 [李思源开题报告最终.md](./李思源开题报告最终.md) 形成正式评估结论，并统一论文写作中的结果表述口径。

本次修订即针对这一最新要求展开。

## 18. 当前时间线归档结论

从整个会话看，项目主线经历了以下演进：

1. 从“搭出基础模型”扩展到“完整特征 × 模型矩阵实验”。
2. 从“能训练”扩展到“能保存论文级日志、曲线、表格与总图”。
3. 从“主实验”扩展到“融合、复杂度、鲁棒性、消融、可解释性和论文图版式”。
4. 从“代码工程”进入“文档统筹、开题对照、论文模板准备与正式写作”阶段。

当前项目已经不只是一个实验脚本集合，而是一套较完整的本科毕业设计研究资产库。接下来的重点已经不是继续无边界扩模型，而是：

- 用真实结果统一论文措辞与结果口径
- 以现有完整实验链路支撑正文分析
- 严格按论文模板逐章逐节落稿
