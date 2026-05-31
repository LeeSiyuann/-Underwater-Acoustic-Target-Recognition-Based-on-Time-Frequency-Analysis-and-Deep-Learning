# 项目目的、实验状态与论文完成计划参考

本文档基于当前仓库中可核验的代码、结果文件和已有总结材料，输出以下四部分内容：

1. 项目目的总结
2. 实验状态与关键结果总结
3. 毕业论文完成计划参考稿
4. 缺失图表、数据核对项与必须补做实验清单

文档严格遵守以下边界：

- 不修改问题定义，不擅自引入新的研究假设。
- 不编造未在仓库文件中出现的实验数值。
- 所有数值优先追溯到现有 CSV、脚本和结果目录。
- 对“已有脚本但当前目录未检索到产物”的部分，明确标注为“待补跑/待确认”。

***

## 1. 项目目的总结

### 1.1 核心研究目的

本项目的核心目标，是围绕 DeepShip 数据集构建一套“基于时频分析与深度学习的水下声学目标识别”实验体系，系统研究不同声学特征表示与不同深度学习模型之间的适配关系。更具体地说，项目不是只追求一个最高分模型，而是同时回答四类问题：

1. 对船舶噪声这类非平稳水下声学信号，哪种时频表征更有效。
2. 二维卷积网络、残差网络、循环网络和 Transformer 等不同范式模型，对这些特征的适配性有何差异。
3. 在高性能 GPU 环境下，如何把大规模矩阵实验稳定跑完，并留下完整、可追溯的图表和日志。
4. 在论文撰写层面，如何把实验过程、结果与分析组织成可直接进入毕业设计论文的材料。

这一研究目标与以下代码直接对应：

- 数据预处理：[preprocess\_deepship\_32k.py](./preprocess_deepship_32k.py)
- 特征提取：[feature\_extraction\_comparison.py](./feature_extraction_comparison.py)
- 模型定义：[deep\_learning\_models.py](./deep_learning_models.py)
- 矩阵训练：[train\_comparison.py](./train_comparison.py)
- 断点续训：[resume\_training.py](./resume_training.py)
- 融合实验：[train\_fusion.py](./train_fusion.py)
- 全局分析：[plot\_global\_comparison.py](./plot_global_comparison.py)
- 复杂度评估：[evaluate\_complexity.py](./evaluate_complexity.py)
- 鲁棒性评估：[noise\_robustness\_test.py](./noise_robustness_test.py)

### 1.2 当前项目范围边界

依据当前仓库和会话历史，项目当前已经明确落在以下范围内：

- 数据集：DeepShip
- 分类对象：Cargo、Tanker、Tug、Passengership 四类船舶
- 特征类型：STFT、Mel、MFCC、CQT，加上一个 `Mel + MFCC` 双流融合实验
- 单流模型：SimpleCNN、ResNet18、ResNet34、ResNet50、RNN、LSTM、Transformer
- 融合模型：DualStreamFusionModel
- 评估维度：Accuracy、Precision、Recall、F1、ROC-AUC、PR 曲线、混淆矩阵、训练历史、模型复杂度、推理时间、抗噪鲁棒性

当前并没有证据表明项目已正式纳入以下内容，因此论文中不应擅自写成“已完成”：

- 波形级别端到端模型
- 多数据集迁移实验
- 海洋真实背景噪声库的系统评估
- 消融实验的完整矩阵
- 交叉验证或多随机种子重复实验

### 1.3 项目的工程目标

从工程角度看，本项目还承担了一个非常明确的目的：把大规模实验真正跑稳定，而不是只在概念层面讨论。当前仓库中可以验证到的工程化能力包括：

- 大批量训练与数据并行加载
- STFT 特征单独限流，降低 OOM 风险
- 断点续训与结果恢复
- 显存占用统计与批量推荐
- 结果自动绘图与统一汇总

对应结果：

- [gpu\_memory\_usage.csv](./dl_comparison_results_gpu/gpu_memory_usage.csv)
- [recommend\_batchsize.csv](./recommend_batchsize.csv)
- [final\_comparison\_results.csv](./dl_comparison_results_gpu/final_comparison_results.csv)

***

## 2. 实验状态与关键结果总结

## 2.1 当前实验完成状态

### 已有且可核验的实验产物

当前仓库中已经可以确认存在以下结果集合：

### 1. 预处理与特征数据

- [processed\_data/X\_audio\_32k.npy](./processed_data/X_audio_32k.npy)
- [processed\_data/y\_labels.npy](./processed_data/y_labels.npy)
- [feature\_data/X\_stft.npy](./feature_data/X_stft.npy)
- [feature\_data/X\_mel.npy](./feature_data/X_mel.npy)
- [feature\_data/X\_mfcc.npy](./feature_data/X_mfcc.npy)
- [feature\_data/X\_cqt.npy](./feature_data/X_cqt.npy)
- 四类样本特征对比图：
  - [comparison\_Cargo.png](./feature_data/comparison_Cargo.png)
  - [comparison\_Tanker.png](./feature_data/comparison_Tanker.png)
  - [comparison\_Tug.png](./feature_data/comparison_Tug.png)
  - [comparison\_Passengership.png](./feature_data/comparison_Passengership.png)

### 2. 单流矩阵训练结果

当前 [dl\_comparison\_results\_gpu](./dl_comparison_results_gpu) 中可以确认：

- 28 组 `best_{Model}_{Feature}.pth` 权重文件
- 28 组 `training_log_{Model}_{Feature}.csv`
- 28 组 `history_{Model}_{Feature}.*`
- 28 组 `cm_{Model}_{Feature}.*`
- 28 组 `pr_curve_{Model}_{Feature}.*`
- 汇总表 [final\_comparison\_results.csv](./dl_comparison_results_gpu/final_comparison_results.csv)

### 3. 融合实验结果

- [best\_fusion\_Mel\_MFCC.pth](./dl_comparison_results_gpu/best_fusion_Mel_MFCC.pth)
- [training\_log\_fusion\_Mel\_MFCC.csv](./dl_comparison_results_gpu/training_log_fusion_Mel_MFCC.csv)
- [history\_fusion\_Mel\_MFCC.png](./dl_comparison_results_gpu/history_fusion_Mel_MFCC.png)
- [cm\_fusion\_Mel\_MFCC.png](./dl_comparison_results_gpu/cm_fusion_Mel_MFCC.png)
- [pr\_curve\_fusion\_Mel\_MFCC.png](./dl_comparison_results_gpu/pr_curve_fusion_Mel_MFCC.png)
- [final\_fusion\_results.csv](./dl_comparison_results_gpu/final_fusion_results.csv)

### 4. 全局分析、复杂度与鲁棒性结果

- [final\_comparison\_results\_with\_fusion.csv](./dl_comparison_results_gpu/final_comparison_results_with_fusion.csv)
- [global\_f1\_barplot.png](./dl_comparison_results_gpu/global_f1_barplot.png)
- [global\_f1\_heatmap.png](./dl_comparison_results_gpu/global_f1_heatmap.png)
- [global\_metric\_model\_feature\_heatmaps.png](./dl_comparison_results_gpu/global_metric_model_feature_heatmaps.png)
- [model\_complexity.csv](./dl_comparison_results_gpu/model_complexity.csv)
- [model\_parameters\_comparison.png](./dl_comparison_results_gpu/model_parameters_comparison.png)
- [model\_inference\_time\_comparison.png](./dl_comparison_results_gpu/model_inference_time_comparison.png)
- [robustness\_all\_models.csv](./dl_comparison_results_gpu/robustness_all_models.csv)
- [robustness\_curve\_all\_models.png](./dl_comparison_results_gpu/robustness_curve_all_models.png)
- [robustness\_heatmap\_all\_models.png](./dl_comparison_results_gpu/robustness_heatmap_all_models.png)

### 5. 当前已确认存在的降维分析结果

当前已确认目录 [dim\_reduction\_results](./dim_reduction_results) 中存在以下文件：

- [analysis\_summary.txt](./dim_reduction_results/analysis_summary.txt)
- [dr\_X\_mel\_full.png](./dim_reduction_results/dr_X_mel_full.png)
- [dr\_X\_mfcc\_full.png](./dim_reduction_results/dr_X_mfcc_full.png)
- [dr\_X\_cqt\_full.png](./dim_reduction_results/dr_X_cqt_full.png)
- [dr\_X\_stft\_full.png](./dim_reduction_results/dr_X_stft_full.png)
- [pca\_variance\_analysis.csv](./dim_reduction_results/pca_variance_analysis.csv)

这意味着 PCA / t-SNE 图和方差分析表已经具备论文引用基础，但建议后续统一用更新后的 [dimensionality\_reduction\_analysis.py](./dimensionality_reduction_analysis.py) 继续输出到同一目录，并补齐 `.svg/.pdf` 版本。

## 2.2 关键结果总结

以下数值直接来自 [final\_comparison\_results\_with\_fusion.csv](./dl_comparison_results_gpu/final_comparison_results_with_fusion.csv)。

### 1. 最佳总体结果

按 `f1` 排序，当前全表最高结果是：

- `ResNet18 + X_mfcc`
- `f1 = 0.9982335599857426`
- `accuracy = 0.9982332155477032`
- `roc_auc = 0.9999552089829588`

这说明在当前这组训练与测试划分下，MFCC 与 ResNet18 的组合表现最佳。

### 2. 各特征下的最佳模型

根据当前汇总表，可归纳为：

| 特征                | 当前最佳模型                  | F1                   |
| :---------------- | :---------------------- | :------------------- |
| `X_mel`           | `ResNet18`              | `0.9973504513244354` |
| `X_mfcc`          | `ResNet18`              | `0.9982335599857426` |
| `X_cqt`           | `ResNet18`              | `0.9946914001155536` |
| `X_stft`          | `ResNet18`              | `0.9955870227842644` |
| `Fusion_Mel_MFCC` | `DualStreamFusionModel` | `0.9955809781067444` |

目前最稳定的结论不是“ResNet50 最好”，而是“ResNet18 在四种单特征上均给出非常强的结果，尤其在 MFCC 上拿到全表最高分”。论文中应据此修正文案，不宜机械强调更深网络一定更优。

### 3. 相对较弱的组合

当前较弱结果主要集中在 Transformer 的某些组合：

- `Transformer + X_mfcc`：`f1 = 0.3018762762128659`
- `Transformer + X_stft`：`f1 = 0.501094464773686`
- `SimpleCNN + X_stft`：`f1 = 0.6588753044450318`
- `SimpleCNN + X_cqt`：`f1 = 0.7497751075325002`

这说明：

- 并不是所有“更先进”的序列模型都天然适配所有特征表示。
- STFT 和 MFCC 在当前实现下，对 Transformer 的适配并不理想。

### 4. 融合模型的客观定位

融合模型 `DualStreamFusionModel` 的结果为：

- `accuracy = 0.9955830388692579`
- `f1 = 0.9955809781067444`
- `roc_auc = 0.9999156474072634`

这个结果非常高，但**不是当前全表最高**。因此在论文中更稳妥的说法应是：

- 融合模型取得了接近最优的结果；
- 但在当前实验条件下，尚未超过 `ResNet18 + MFCC`；
- 融合实验的价值更多体现在“证明多特征融合可行”，而不是“已经全面超越所有单特征方案”。

### 5. 模型复杂度与实时性

以下结果来自 [model\_complexity.csv](./dl_comparison_results_gpu/model_complexity.csv)：

| 模型                    | 参数量(M)    | 推理时间(ms/sample) |
| :-------------------- | :-------- | :-------------- |
| SimpleCNN             | 0.093636  | 0.4791          |
| ResNet18              | 11.172292 | 2.4587          |
| ResNet34              | 21.280452 | 3.3933          |
| ResNet50              | 23.509956 | 4.8571          |
| RNN                   | 0.988164  | 1.9202          |
| LSTM                  | 3.946500  | 12.0174         |
| Transformer           | 5.550340  | 1.1738          |
| DualStreamFusionModel | 0.055044  | 0.8687          |

可以直接得出几条有数据支撑的观察：

- 参数量最大的是 `ResNet50`，为 `23.509956M`
- 推理最慢的是 `LSTM`，为 `12.0174 ms/sample`
- 推理最快的是 `SimpleCNN`
- 当前复杂度表中，`DualStreamFusionModel` 参数量甚至低于 `SimpleCNN`，这一点应在论文中保守表述，并建议复核其统计方式

### 6. 鲁棒性结果

以下结果来自 [robustness\_all\_models.csv](./dl_comparison_results_gpu/robustness_all_models.csv)。

#### 1. 干净数据下

干净数据下各模型的 F1 都比较高，例如：

- `DualStreamFusionModel`：`0.9955809781067443`
- `ResNet18 + X_mfcc`：`0.9982335599857426`
- `LSTM + X_cqt`：`0.9840640669184093`
- `Transformer + X_cqt`：`0.9814890544581805`

#### 2. 强噪声下

在 `SNR = -10 dB` 时：

- `LSTM + X_cqt`：`0.6335107034441776`
- `RNN + X_cqt`：`0.4577372604904366`
- `Transformer + X_cqt`：`0.2663224156672989`
- `ResNet34 + X_mel`：`0.23745693809054358`
- `DualStreamFusionModel`：`0.12878380277935308`

因此，当前鲁棒性结果与某些早期乐观表述并不一致。就仓库现有数据看：

- `LSTM + X_cqt` 是当前最抗噪的一组
- 多个基于 MFCC 的 CNN/ResNet 模型在低 SNR 下会迅速塌缩到接近同一低值
- 融合模型在当前噪声注入方式下表现很脆弱，不能在论文中写成“最鲁棒”

这正是后续论文撰写必须谨慎核对的重点。

## 2.3 当前实验状态一句话结论

如果只用一句话概括当前状态，可以写成：

> 项目主体实验链路已经完成，单流矩阵训练、融合实验、全局对比、复杂度评估和抗噪测试均已有可追溯结果文件；但降维结果文件当前缺失，且鲁棒性、复杂度和部分汇总图的解释口径仍需在论文中谨慎核对。

***

## 3. 毕业论文完成计划参考稿

这份计划仅作为总览参考，真正执行时请遵守“逐章、逐小节规划并留档”的规则，不要把这份总计划直接当成最终执行清单。

## 3.1 建议的论文章节结构

结合当前项目材料，建议本科论文主体采用以下结构：

### 第 1 章 绪论

建议包含：

- 研究背景与意义
- 水下声学目标识别的工程应用场景
- 国内外研究现状
- 本文研究内容与技术路线
- 本文结构安排

素材来源：

- [李思源开题报告最终.md](./李思源开题报告最终.md)
- [report.md](./report.md)
- 当前实验结果与项目 README

### 第 2 章 相关技术基础

建议包含：

- 水下声学信号基本特点
- STFT、Mel、MFCC、CQT 的理论基础与差异
- CNN / ResNet / RNN / LSTM / Transformer 的基本原理
- 评价指标定义：Accuracy、Precision、Recall、F1、ROC-AUC

素材来源：

- [feature\_extraction\_comparison.py](./feature_extraction_comparison.py)
- [deep\_learning\_models.py](./deep_learning_models.py)

### 第 3 章 系统设计与研究方法

建议包含：

- 数据集与类别说明
- 数据预处理流程
- 四种特征提取流程
- 单流模型设计
- 融合模型设计
- 训练配置、批量调度与断点续训机制

素材来源：

- [preprocess\_deepship\_32k.py](./preprocess_deepship_32k.py)
- [feature\_extraction\_comparison.py](./feature_extraction_comparison.py)
- [train\_comparison.py](./train_comparison.py)
- [resume\_training.py](./resume_training.py)
- [train\_fusion.py](./train_fusion.py)

### 第 4 章 实验设计

建议包含：

- 实验环境
- 数据划分方式
- 训练超参数设置
- 模型与特征组合设计
- 复杂度评估方法
- 鲁棒性评估方法

素材来源：

- [train\_comparison.py](./train_comparison.py)
- [evaluate\_complexity.py](./evaluate_complexity.py)
- [noise\_robustness\_test.py](./noise_robustness_test.py)

### 第 5 章 实验结果与分析

建议包含：

- 特征图视觉对比分析
- 矩阵训练总体结果分析
- 典型训练收敛过程分析
- 混淆矩阵与 PR 曲线分析
- 全局性能图表分析
- 模型复杂度与实时性分析
- 鲁棒性结果分析

素材来源：

- [feature\_data](./feature_data)
- [dl\_comparison\_results\_gpu](./dl_comparison_results_gpu)
- [analysis\_tutorial.md](./analysis_tutorial.md)

### 第 6 章 结论与展望

建议包含：

- 本文工作总结
- 主要实验结论
- 局限性
- 后续工作展望

## 3.2 撰写优先级建议

按当前项目状态，论文撰写优先级建议如下：

### 优先级 A：先写能直接由现有结果支撑的章节

1. 第 3 章 系统设计与研究方法
2. 第 4 章 实验设计
3. 第 5 章 中“矩阵训练结果”“复杂度分析”“鲁棒性分析”的初稿

原因：

- 这些内容高度依赖现有代码和现有结果表
- 当前证据最完整
- 写作风险最小

### 优先级 B：再写理论与综述章节

1. 第 1 章 绪论
2. 第 2 章 相关技术基础

原因：

- 这些章节可基于开题报告、文献与代码反推整理
- 对最终结论依赖较小

### 优先级 C：最后写总结与展望

1. 第 6 章 结论与展望

原因：

- 它必须建立在前述分析已经定稿的基础上
- 否则容易出现与前文结果不一致的问题

## 3.3 推荐的执行顺序

建议按以下顺序推进：

1. 先核对现有结果文件是否足够支撑论文
2. 补跑缺失结果或补做必要实验
3. 先写第 3 章和第 4 章
4. 基于真实图表写第 5 章
5. 再回填第 1 章和第 2 章
6. 最后写第 6 章并统一格式

## 3.4 章节内的小节推进建议

为符合逐小节推进的规则，每章内部建议采用这种粒度：

- 先列本章小节目录
- 逐小节写
- 每写完一个小节即检查：
  - 论述是否与真实结果一致
  - 图表是否已存在并可引用
  - 代码与公式是否有出处
  - 如在 LaTeX 中已录入，及时编译检查版式

***

## 4. 缺失图表、数据核对项与必须补做实验

本节是当前最重要的风险控制区。

## 4.1 当前缺失或待确认的图表/数据

### 1. 降维分析结果文件缺失

当前问题：

- [dimensionality\_reduction\_analysis.py](./dimensionality_reduction_analysis.py) 存在
- 但目录 [dim\_reduction\_results\_full](./dim_reduction_results_full) 当前未检索到文件

影响：

- 论文中如果写 PCA / t-SNE 结果，当前缺乏可直接插入的真实图表和数据表

建议动作：

- 先确认结果是否被移动到其他目录
- 若确实不存在，则重新运行脚本并保存输出

### 2. `global_metric_model_feature_heatmaps.png` 已替代旧的聚合多指标图

当前处理：

- 旧的 `global_metric_barplot.*` 已删除
- [plot\_global\_comparison.py](./plot_global_comparison.py) 已改为输出 `Model + Feature` 双维度多指标热力图
- 新输出文件为 [global\_metric\_model\_feature\_heatmaps.png](./dl_comparison_results_gpu/global_metric_model_feature_heatmaps.png)

当前建议：

- 论文中统一引用新的双维度热力图
- 不再使用跨特征求均值的旧聚合柱状图

### 3. 复杂度结果的测量口径需确认

当前问题：

- [evaluate\_complexity.py](./evaluate_complexity.py) 使用固定 dummy input 尺寸
- 它不是基于每个模型真实最佳特征的实际尺寸逐项测量

影响：

- 如果论文里要写“最佳模型在真实输入下的复杂度”，当前表述会不够严谨

建议动作：

- 保留现有表作为“统一输入尺寸下的相对复杂度比较”
- 若论文需要更严谨口径，可补做“按最佳特征实际尺寸”的复杂度测量

### 4. 鲁棒性实验的噪声注入位置需确认

当前问题：

- [noise\_robustness\_test.py](./noise_robustness_test.py) 目前是在特征矩阵层面直接注入噪声
- 不是在原始音频上加噪后重新走预处理与特征提取链路

影响：

- 如果论文要宣称“模拟真实海洋环境噪声条件”，当前证据链还不够完整

建议动作：

- 在论文中诚实表述为“特征层 AWGN 干扰测试”
- 若需要更真实的环境扰动结论，应补做“波形层加噪 + 重新提特征”的实验

### 5. 融合模型复杂度结果需复核

当前问题：

- `DualStreamFusionModel` 参数量仅 `0.055044M`
- 这一数值低于 `SimpleCNN`

影响：

- 虽然从代码结构看它确实是轻量双流网络，但这个结论在论文里仍应谨慎解释

建议动作：

- 写论文前再复核一次参数统计
- 必要时在正文注明“当前统计采用可训练参数总数”

## 4.2 当前缺少的必要图表

如果以本科毕业论文正式提交为目标，当前仍建议补齐或确认以下图表：

### 必须补齐或找回

1. PCA 方差比图/表
2. t-SNE 可视化图
3. 若论文需要展示“数据分布”，还应补一个类别样本数量统计表或柱状图

### 建议补充

1. 模型结构示意图
2. 整体实验流程图
3. 论文中引用的关键样本时频图拼图版式

其中第 1、2 项如果当前没有现成图，应单独规划，不要在论文中用虚构示意图替代真实实验图。

## 4.3 当前必须补做或至少需要确认的实验

### 1. 降维分析结果复跑或找回

这是最明确的缺项，因为相关输出文件当前不在仓库中。

### 2. 鲁棒性实验口径确认

要先决定论文最终采用哪种表述：

- 如果接受“特征层加噪测试”，现有结果可用，但要如实写明
- 如果要上升到“环境噪声下的系统鲁棒性”，则必须补做波形层加噪实验

### 3. 若论文需要“消融实验”，当前证据仍不充分

当前仓库已有：

- 单流与融合模型对比
- 多特征、多模型矩阵结果
- 复杂度和鲁棒性测试

但还没有严格意义上的“结构消融”，例如：

- 去掉融合支路后的比较
- 去掉某层模块后的比较
- 不同 hidden size / 层数的系统消融

如果导师或答辩要求“消融实验”，这部分目前仍属未完成。

## 4.4 当前不建议再盲目扩展的内容

为了避免论文写作阶段失控，当前不建议再随意扩展以下方向，除非有明确指导要求：

- 再新增更多模型
- 再引入新数据集
- 再追加大量无结果支撑的新理论模块
- 在尚未核对当前结果口径前，贸然做大规模参数搜索

当前最重要的是把已有真实结果整理干净、核对清楚并落进论文章节。

***

## 5. 本文档的使用建议

这份文档建议在后续论文撰写中这样使用：

- “项目目的总结”可作为绪论和研究内容部分的提纲底稿
- “实验状态与关键结果总结”可作为实验章节的事实基线
- “毕业论文完成计划参考稿”只作为总计划，不替代逐章逐节 plan
- “缺失图表、数据核对项与必须补做实验清单”应在正式写实验分析前逐项核对

如果后续开始按论文模板正式写作，最推荐的下一步是：

1. 先确认降维结果是否存在；不存在则补跑
2. 明确鲁棒性实验在论文中的表述口径
3. 先从“第 3 章 系统设计与研究方法”开始逐小节写作
