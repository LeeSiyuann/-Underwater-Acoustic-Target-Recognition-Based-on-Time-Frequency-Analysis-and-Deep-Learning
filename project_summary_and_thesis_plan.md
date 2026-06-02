# 项目目的、实验状态与论文完成计划参考

本文档基于当前仓库中已经落盘的代码、CSV、图像与说明文件整理，目的有四个：

1. 给出项目研究目标的当前准确表述。
2. 记录当前全部实验的真实完成状态与关键结果。
3. 对照开题报告形成正式评估结论。
4. 给出后续论文撰写与补充实验的优先级建议。

本文档中的数值均以以下结果文件为准：

- [dl_comparison_results_gpu/final_comparison_results_with_fusion.csv](./dl_comparison_results_gpu/final_comparison_results_with_fusion.csv)
- [dl_comparison_results_gpu/ablation_results/ablation_summary_results.csv](./dl_comparison_results_gpu/ablation_results/ablation_summary_results.csv)
- [dl_comparison_results_gpu/explainability_results/fixed_feature_model_effect.csv](./dl_comparison_results_gpu/explainability_results/fixed_feature_model_effect.csv)
- [dl_comparison_results_gpu/explainability_results/fixed_model_feature_effect.csv](./dl_comparison_results_gpu/explainability_results/fixed_model_feature_effect.csv)
- [dl_comparison_results_gpu/explainability_results/frequency_band_ablation_results.csv](./dl_comparison_results_gpu/explainability_results/frequency_band_ablation_results.csv)
- [dl_comparison_results_gpu/model_complexity.csv](./dl_comparison_results_gpu/model_complexity.csv)
- [dl_comparison_results_gpu/robustness_all_models.csv](./dl_comparison_results_gpu/robustness_all_models.csv)
- [dim_reduction_results/analysis_summary.txt](./dim_reduction_results/analysis_summary.txt)
- [dim_reduction_results/pca_variance_analysis.csv](./dim_reduction_results/pca_variance_analysis.csv)

---

## 1. 项目目的总结

### 1.1 核心研究目标

本项目围绕 DeepShip 数据集，构建“预处理 -> 多时频特征提取 -> 特征可视化分析 -> 深度学习分类 -> 全局对比与扩展分析”的完整实验链路，核心要回答以下问题：

1. STFT、Mel、MFCC、CQT 等时频表征对水下船舶噪声识别的区分能力有何差异。
2. CNN、ResNet、RNN、LSTM、Transformer 等模型对不同特征的适配关系如何。
3. 在统一评价指标下，哪种“特征 + 模型”组合最优，哪些组合存在明显短板。
4. 当前结果是否足以支撑论文关于“时频表征与深度学习协同影响识别性能”的核心论点。

### 1.2 当前项目范围

截至当前结果状态，项目已完成并留有产物的范围为：

- 数据集：DeepShip。
- 分类类别：Cargo、Tanker、Tug、Passengership。
- 单特征：`X_mel`、`X_mfcc`、`X_cqt`、`X_stft`。
- 单流模型：`SimpleCNN`、`ResNet18`、`ResNet34`、`ResNet50`、`RNN`、`LSTM`、`Transformer`。
- 融合模型：`DualStreamFusionModel`，对应 `Fusion_Mel_MFCC`。
- 扩展实验：降维分析、结构消融、序列超参数消融、固定特征/固定模型解释性分析、Grad-CAM、序列显著图、频带遮挡实验、复杂度评估、特征层 AWGN 鲁棒性测试、论文图版式整理。

当前项目范围之外、因此未纳入本文结果口径的内容包括：

- 多数据集迁移或跨数据集泛化实验。
- 波形层真实噪声重建链路的鲁棒性实验。
- 传统浅层分类器基线（如 SVM、随机森林等）对比。
- 多随机种子重复实验或交叉验证。

---

## 2. 当前实验状态与关键结果

## 2.1 已完成且可直接核验的结果集合

### 1. 预处理与特征数据

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

### 2. 降维分析

`dim_reduction_results` 目录当前已存在真实结果，不再属于缺项：

- [dim_reduction_results/analysis_summary.txt](./dim_reduction_results/analysis_summary.txt)
- [dim_reduction_results/pca_variance_analysis.csv](./dim_reduction_results/pca_variance_analysis.csv)
- [dim_reduction_results/dr_X_mel_full.png](./dim_reduction_results/dr_X_mel_full.png)
- [dim_reduction_results/dr_X_mfcc_full.png](./dim_reduction_results/dr_X_mfcc_full.png)
- [dim_reduction_results/dr_X_cqt_full.png](./dim_reduction_results/dr_X_cqt_full.png)
- [dim_reduction_results/dr_X_stft_full.png](./dim_reduction_results/dr_X_stft_full.png)

### 3. 单流矩阵训练

`dl_comparison_results_gpu` 当前已完整保存 4 特征 × 7 模型的实验产物，包括：

- 28 个 `best_{Model}_{Feature}.pth`
- 28 个 `training_log_{Model}_{Feature}.csv`
- 28 组 `history_{Model}_{Feature}.png/.svg/.pdf`
- 28 组 `cm_{Model}_{Feature}.png/.svg/.pdf`
- 28 组 `pr_curve_{Model}_{Feature}.png/.svg/.pdf`
- [dl_comparison_results_gpu/final_comparison_results.csv](./dl_comparison_results_gpu/final_comparison_results.csv)

### 4. 融合实验

- [dl_comparison_results_gpu/best_fusion_Mel_MFCC.pth](./dl_comparison_results_gpu/best_fusion_Mel_MFCC.pth)
- [dl_comparison_results_gpu/training_log_fusion_Mel_MFCC.csv](./dl_comparison_results_gpu/training_log_fusion_Mel_MFCC.csv)
- [dl_comparison_results_gpu/history_fusion_Mel_MFCC.png](./dl_comparison_results_gpu/history_fusion_Mel_MFCC.png)
- [dl_comparison_results_gpu/cm_fusion_Mel_MFCC.png](./dl_comparison_results_gpu/cm_fusion_Mel_MFCC.png)
- [dl_comparison_results_gpu/pr_curve_fusion_Mel_MFCC.png](./dl_comparison_results_gpu/pr_curve_fusion_Mel_MFCC.png)
- [dl_comparison_results_gpu/final_fusion_results.csv](./dl_comparison_results_gpu/final_fusion_results.csv)

### 5. 全局对比、复杂度与鲁棒性

- [dl_comparison_results_gpu/final_comparison_results_with_fusion.csv](./dl_comparison_results_gpu/final_comparison_results_with_fusion.csv)
- [dl_comparison_results_gpu/global_f1_barplot.png](./dl_comparison_results_gpu/global_f1_barplot.png)
- [dl_comparison_results_gpu/global_f1_heatmap.png](./dl_comparison_results_gpu/global_f1_heatmap.png)
- [dl_comparison_results_gpu/global_metric_model_feature_heatmaps.png](./dl_comparison_results_gpu/global_metric_model_feature_heatmaps.png)
- [dl_comparison_results_gpu/model_complexity.csv](./dl_comparison_results_gpu/model_complexity.csv)
- [dl_comparison_results_gpu/model_parameters_comparison.png](./dl_comparison_results_gpu/model_parameters_comparison.png)
- [dl_comparison_results_gpu/model_inference_time_comparison.png](./dl_comparison_results_gpu/model_inference_time_comparison.png)
- [dl_comparison_results_gpu/robustness_all_models.csv](./dl_comparison_results_gpu/robustness_all_models.csv)
- [dl_comparison_results_gpu/robustness_curve_all_models.png](./dl_comparison_results_gpu/robustness_curve_all_models.png)
- [dl_comparison_results_gpu/robustness_heatmap_all_models.png](./dl_comparison_results_gpu/robustness_heatmap_all_models.png)

### 6. 消融实验

`ablation_results` 目录当前已存在真实结果：

- [dl_comparison_results_gpu/ablation_results/ablation_fusion_structure_results.csv](./dl_comparison_results_gpu/ablation_results/ablation_fusion_structure_results.csv)
- [dl_comparison_results_gpu/ablation_results/ablation_sequence_hyperparams_results.csv](./dl_comparison_results_gpu/ablation_results/ablation_sequence_hyperparams_results.csv)
- [dl_comparison_results_gpu/ablation_results/ablation_summary_results.csv](./dl_comparison_results_gpu/ablation_results/ablation_summary_results.csv)
- [dl_comparison_results_gpu/ablation_results/ablation_fusion_structure_f1.png](./dl_comparison_results_gpu/ablation_results/ablation_fusion_structure_f1.png)
- [dl_comparison_results_gpu/ablation_results/ablation_lstm_hyperparams_heatmap.png](./dl_comparison_results_gpu/ablation_results/ablation_lstm_hyperparams_heatmap.png)
- [dl_comparison_results_gpu/ablation_results/ablation_transformer_hyperparams_heatmap.png](./dl_comparison_results_gpu/ablation_results/ablation_transformer_hyperparams_heatmap.png)

### 7. 可解释性实验

`explainability_results` 目录当前已存在真实结果：

- [dl_comparison_results_gpu/explainability_results/fixed_feature_model_effect.csv](./dl_comparison_results_gpu/explainability_results/fixed_feature_model_effect.csv)
- [dl_comparison_results_gpu/explainability_results/fixed_model_feature_effect.csv](./dl_comparison_results_gpu/explainability_results/fixed_model_feature_effect.csv)
- [dl_comparison_results_gpu/explainability_results/frequency_band_ablation_results.csv](./dl_comparison_results_gpu/explainability_results/frequency_band_ablation_results.csv)
- `gradcam_ResNet18_X_mfcc_*`
- `gradcam_fusion_mel_*`
- `gradcam_fusion_mfcc_*`
- `saliency_LSTM_X_cqt_*`

当前可解释性结果的正文分析可围绕固定特征/固定模型分析、Grad-CAM、融合支路可视化、序列显著图和频带遮挡实验展开。

### 8. 论文图与拼图版式

`paper_figures` 目录当前已存在流程图、结构图、训练结果拼图和全局图复制结果：

- [dl_comparison_results_gpu/paper_figures/workflow_overview.png](./dl_comparison_results_gpu/paper_figures/workflow_overview.png)
- [dl_comparison_results_gpu/paper_figures/feature_comparison_montage.png](./dl_comparison_results_gpu/paper_figures/feature_comparison_montage.png)
- [dl_comparison_results_gpu/paper_figures/dimensionality_reduction_montage.png](./dl_comparison_results_gpu/paper_figures/dimensionality_reduction_montage.png)
- [dl_comparison_results_gpu/paper_figures/cm_montage_X_mfcc.png](./dl_comparison_results_gpu/paper_figures/cm_montage_X_mfcc.png)
- [dl_comparison_results_gpu/paper_figures/history_montage_X_cqt.png](./dl_comparison_results_gpu/paper_figures/history_montage_X_cqt.png)
- [dl_comparison_results_gpu/paper_figures/pr_curve_montage_X_stft.png](./dl_comparison_results_gpu/paper_figures/pr_curve_montage_X_stft.png)
- [dl_comparison_results_gpu/paper_figures/model_architecture_index.csv](./dl_comparison_results_gpu/paper_figures/model_architecture_index.csv)
- [dl_comparison_results_gpu/paper_figures/paper_figures_index.csv](./dl_comparison_results_gpu/paper_figures/paper_figures_index.csv)

## 2.2 主实验关键结论

### 1. 全表最佳组合

来自 [final_comparison_results_with_fusion.csv](./dl_comparison_results_gpu/final_comparison_results_with_fusion.csv) 的当前最优结果为：

- 模型：`ResNet18`
- 特征：`X_mfcc`
- `accuracy = 0.9982332155477032`
- `f1 = 0.9982335599857426`
- `roc_auc = 0.9999552089829588`

这说明在当前数据划分与训练配置下，`MFCC + ResNet18` 是最佳组合。

### 2. 各特征下当前最佳模型

| 特征 | 当前最佳模型 | F1 |
| :-- | :-- | :-- |
| `X_mel` | `ResNet18` | `0.9973504513244354` |
| `X_mfcc` | `ResNet18` | `0.9982335599857426` |
| `X_cqt` | `ResNet18` | `0.9946914001155536` |
| `X_stft` | `ResNet18` | `0.9955870227842644` |
| `Fusion_Mel_MFCC` | `DualStreamFusionModel` | `0.9955809781067444` |

因此当前最稳妥的实验结论不是“网络越深越好”，而是“ResNet18 在四种单特征上都表现最强或接近最强，特别是在 MFCC 上取得全表最佳”。

### 3. 当前较弱组合

当前结果明显偏弱的组合包括：

- `Transformer + X_mfcc`：`f1 = 0.3018762762128659`
- `Transformer + X_stft`：`f1 = 0.501094464773686`
- `SimpleCNN + X_stft`：`f1 = 0.6588753044450318`
- `SimpleCNN + X_cqt`：`f1 = 0.7497751075325002`

说明不同模型对不同特征的适配性差异很大，不能简单将“更复杂模型”直接等同于“更优识别性能”。

### 4. 融合实验的当前定位

来自 [final_fusion_results.csv](./dl_comparison_results_gpu/final_fusion_results.csv) 的融合结果为：

- `accuracy = 0.9955830388692579`
- `f1 = 0.9955809781067443`
- `roc_auc = 0.9999156474072634`

融合结果已经很高，但未超过 `ResNet18 + X_mfcc`。因此在论文中应写为：

- 多特征融合方案有效且性能接近最优。
- 在当前实现与训练设置下，它不是全表最佳方案。
- 融合实验的价值主要在于证明多特征组合可行，而不是证明其绝对优于所有单特征模型。

### 5. 降维分析结论

来自 [pca_variance_analysis.csv](./dim_reduction_results/pca_variance_analysis.csv) 与 [analysis_summary.txt](./dim_reduction_results/analysis_summary.txt) 的前两主成分累计解释率为：

- `X_mel`：`82.8020%`
- `X_stft`：`70.7099%`
- `X_cqt`：`48.3359%`
- `X_mfcc`：`25.9703%`

这些数值说明不同特征在低维投影下的信息集中度差异明显，论文中可据此解释“低维可视化分布特性”和“主任务分类性能”之间并非简单一一对应。

### 6. 复杂度与实时性

来自 [model_complexity.csv](./dl_comparison_results_gpu/model_complexity.csv)：

| 模型 | 参数量(M) | 推理时间(ms/sample) |
| :-- | :-- | :-- |
| SimpleCNN | 0.093636 | 0.4791 |
| ResNet18 | 11.172292 | 2.4587 |
| ResNet34 | 21.280452 | 3.3933 |
| ResNet50 | 23.509956 | 4.8571 |
| RNN | 0.988164 | 1.9202 |
| LSTM | 3.946500 | 12.0174 |
| Transformer | 5.550340 | 1.1738 |
| DualStreamFusionModel | 0.055044 | 0.8687 |

当前可直接确认：

- 参数量最大的是 `ResNet50`。
- 推理最慢的是 `LSTM`。
- 推理最快的是 `SimpleCNN`。
- 该复杂度表应在论文中表述为“统一输入尺寸下的相对复杂度比较”，不宜写成“真实最佳输入尺寸下的绝对复杂度结论”。

### 7. 鲁棒性结论

来自 [robustness_all_models.csv](./dl_comparison_results_gpu/robustness_all_models.csv)：

- 干净数据下，`ResNet18 + X_mfcc` 为 `0.9982335599857426`。
- `SNR = -10 dB` 时，`LSTM + X_cqt` 取得 `0.6335107034441776`，为当前最抗噪组合。
- 同一条件下，`DualStreamFusionModel + Fusion_Mel_MFCC` 下降到 `0.12878380277935308`。

因此当前真实结果支持的结论是：

- 主任务最高精度组合不等于最抗噪组合。
- `LSTM + X_cqt` 在特征层 AWGN 扰动下更稳健。
- 融合模型在当前设置下表现出明显的噪声敏感性。
- 鲁棒性分析统一采用“特征层 AWGN 干扰测试”这一实验表述。

### 8. 消融实验结论

来自 [ablation_summary_results.csv](./dl_comparison_results_gpu/ablation_results/ablation_summary_results.csv)：

- `DualStreamFusion`：`f1 = 0.8588410732423101`
- `MFCCOnlyBranch`：`f1 = 0.8049460557678847`
- `MelOnlyBranch`：`f1 = 0.6892101307950254`
- `ShallowDualStream`：`f1 = 0.6787177008568984`

说明在当前消融设置下：

- 双支路完整融合结构优于单支路与浅层融合结构。
- 结构消融已经形成真实结果，可直接用于融合结构有效性分析。

序列超参数消融中：

- `LSTM` 内部最优为 `hs512_layers3`，`f1 = 0.9703087300034815`
- `Transformer` 内部最优为 `dm128_layers2`，`f1 = 0.9865392372212746`

这里应在论文中注明：上述结果属于消融实验内部比较结果，不能直接替代主实验表中的最终成绩。

### 9. 可解释性实验结论

当前可直接确认：

- 固定特征分析文件为 [fixed_feature_model_effect.csv](./dl_comparison_results_gpu/explainability_results/fixed_feature_model_effect.csv)，当前以 `X_cqt` 为固定特征载体。
- 固定模型分析文件为 [fixed_model_feature_effect.csv](./dl_comparison_results_gpu/explainability_results/fixed_model_feature_effect.csv)，当前以 `ResNet18` 为固定模型载体。
- Grad-CAM 已对 `ResNet18 + X_mfcc` 及融合模型的 Mel/MFCC 支路输出结果。
- 序列显著图已对 `LSTM + X_cqt` 输出各类别样本结果。
- [frequency_band_ablation_results.csv](./dl_comparison_results_gpu/explainability_results/frequency_band_ablation_results.csv) 中 `Baseline_F1` 与主结果表不一致，因此只能作为频带遮挡实验内部评估，不应直接作为主实验最终 F1 引用。

## 2.3 当前实验状态的一句话结论

当前项目的主实验链路、扩展实验链路和论文制图链路均已形成真实产物，开题报告中的核心实验主体已经基本完成，且多数结论已有可追溯数据支撑。

---

## 3. 对照开题报告的正式评估结论

## 3.1 逐条对应结论

### 开题研究内容（1）：水下声学信号时频特征提取与分析

完成情况：已完成。

依据：

- 已完成预处理、STFT/Mel/MFCC/CQT 提取与样例图生成。
- 相关数据和图像均已落盘。

### 开题研究内容（2）：时频特征可视化与可分性分析

完成情况：已完成。

依据：

- `dim_reduction_results` 中已存在 PCA/t-SNE 相关图表和方差分析表。
- STFT 内存问题已通过脚本优化后完成全量样本分析。

### 开题研究内容（3）：基于深度学习的识别模型构建

完成情况：已完成，且超出原始最小要求。

依据：

- 开题报告明确列出 CNN、ResNet 等；当前实际已完成 `SimpleCNN`、`ResNet18/34/50`、`RNN`、`LSTM`、`Transformer` 及双流融合模型。
- 已完成端到端训练、评估、可视化与权重保存。

### 开题研究内容（4）：实验验证与性能评估

完成情况：已完成。

依据：

- 已完成统一数据划分与统一指标体系下的多模型、多特征矩阵实验。
- 已完成融合实验、复杂度评估、特征层 AWGN 鲁棒性测试、结构消融和可解释性分析。
- 当前实验结果已经形成从主实验到扩展分析的完整证据链，可用于支撑论文中的性能对比、稳定性分析和结果讨论。

## 3.2 是否达到开题报告规定的全部要求

正式结论：

> 当前实验工作已经达到开题报告的核心研究目标和主要实验要求，能够完整支撑“时频表征方式显著影响深度学习识别效果，且不同模型与不同特征之间存在明显适配差异”的核心论点。对于论文正文和答辩展示而言，现有结果链路已经闭合，可直接作为正式写作依据。

换言之：

- 对本科毕设论文主体而言，当前实验已经足够支撑核心论点、结果分析和正文撰写。
- 现有结果在主实验、扩展分析和论文制图三个层面均已具备直接使用条件。

## 3.3 结果使用边界与写作说明

以下内容不是“缺项清单”，而是论文写作时应保持一致的结果边界说明：

### 说明 1：当前比较框架以深度学习模型为主

- 当前主实验已经覆盖 `SimpleCNN`、`ResNet18/34/50`、`RNN`、`LSTM`、`Transformer` 和双流融合模型。
- 因此论文中可将比较重点表述为“不同深度学习模型与不同时频特征之间的适配差异”。

### 说明 2：鲁棒性结果采用统一的特征层扰动设置

- 当前鲁棒性结果已经完整落盘并形成总图。
- 论文中统一表述为“特征层 AWGN 干扰测试”即可，这一表述与现有脚本和结果完全一致。

### 说明 3：泛化能力结论基于当前统一训练/测试划分

- 当前实验已经在统一划分下完成主结果、鲁棒性和扩展分析。
- 因此论文中可将“泛化能力”表述为“在当前统一划分测试集上的泛化表现与稳定性”。

### 说明 4：可解释性结果以当前已落盘内容为准

- 当前已形成固定特征/固定模型分析、Grad-CAM、序列显著图和频带遮挡实验结果。
- 论文写作时围绕这些已落盘结果展开即可。

---

## 4. 论文完成计划参考稿

本计划只作为总览参考。真正写作时，仍需严格遵守“逐章 plan、逐小节完成、每节检查并编译”的规则。

## 4.1 推荐的章节组织

### 第 1 章 绪论

- 研究背景与意义
- 国内外研究现状
- 本文研究内容与创新点
- 技术路线与论文结构

### 第 2 章 相关技术基础

- 水下声学信号特点
- STFT、Mel、MFCC、CQT 原理
- CNN、ResNet、RNN、LSTM、Transformer 原理
- 评价指标定义

### 第 3 章 数据处理与方法设计

- 数据集说明
- 预处理流程
- 四类时频特征提取
- 单流模型设计
- 双流融合模型设计
- 训练配置与断点续训策略

### 第 4 章 实验设计

- 硬件与软件环境
- 数据划分方式
- 模型与特征组合
- 指标设置
- 复杂度评估设置
- 鲁棒性测试设置

### 第 5 章 实验结果与分析

- 特征样例图分析
- 降维可视化与可分性分析
- 主实验结果对比
- 训练收敛过程分析
- 混淆矩阵与 PR 曲线分析
- 全局性能热力图分析
- 复杂度与实时性分析
- 鲁棒性分析
- 消融与可解释性分析

### 第 6 章 结论与展望

- 研究工作总结
- 核心实验结论
- 局限性
- 后续工作

## 4.2 当前最优写作顺序

1. 先写第 3 章“数据处理与方法设计”。
2. 再写第 4 章“实验设计”。
3. 再写第 5 章中基于当前真实结果的部分。
4. 然后补写第 1 章和第 2 章。
5. 最后写第 6 章，并统一全篇口径。

## 4.3 写作时统一采用的结果口径

- 所有结果均以真实 CSV 和图像为准。
- 融合模型写为“处于第一梯队、结果接近最优”。
- 鲁棒性统一写为“特征层 AWGN 干扰测试”。
- 频带遮挡实验写为“解释性内部敏感性分析”。
- 可解释性分析围绕当前已落盘结果展开。
