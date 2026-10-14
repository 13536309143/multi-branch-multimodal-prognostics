# 方法与实现细节 / Method and implementation

本项目以多分支神经网络预测下一次计划检查退化。输入为信号图像、检修文本与四个数值变量；其他标识字段仅用于审计、划分和可追溯输出。

## 分支与融合

- 图像：Conv2D通道1→4→4→8，核4，ReLU；每层池化4。卷积补零左/上1、右/下2；池化右/下补零取ceil。325×194→82×49→21×13→6×4，展平192→4、sigmoid。
- 文本：原版GloVe6B/100d，最多27词。Conv1D窗口2/3/4各32滤波器、ReLU、全局最大池化，96→4、sigmoid。词向量默认冻结，PAD为0，预训练缺失词按种子初始化。
- 数值：训练集均值/标准差变换后4→4、sigmoid。
- 融合：所选分支4维特征拼接，线性输出标量。itn可训练参数30,937；冻结词表大小会影响总参数数，不改变文本卷积结构。
- 消融：七种模态组合从头独立训练，不共享已训练三分支权重。

## 数据契约

四个数值变量依次为化学清洗次数、上次化学清洗后的机械清洗次数、距上次维修时间、距下一次计划检查时间。标签已对齐下一次检查，既不shift，也不构造RUL。

图像按PIL宽194、高325读取为L并除以255；无信号模板默认保留坐标轴。`zero_template`是可选敏感性配置，未纳入当前结果。文本分词为 `[a-z0-9]+(?:'[a-z0-9]+)?`，小写、保留缩写撇号、不移除停用词；训练词表外词映射UNK。两套已完成划分的测试词汇均出现在相应训练词表中。

数值缩放、目标缩放和词表只拟合训练行；GloVe是固定外部查表，不拟合测试标签。所有测试指标在目标反变换后计算，不裁剪预测。

## 训练与选择

Adam，lr=0.001，batch64，MSE，weight_decay=0，最多500轮。卷积/线性层Xavier初始化、bias0。验证MSE下降超过1e-5时更新最佳状态，40轮无更新早停。记录训练损失是批次更新期间平均，验证损失是轮末模型评价；最佳权重重放用于计算严格同权重的训练/测试差距。

种子同时设置Python、NumPy、Torch和CUDA；使用确定性Torch操作、num_workers=0。单次训练保存最佳权重，不保存可在任意epoch续训的完整优化器状态。

SVR：RBF，C100，gamma=scale，epsilon0.1；输入训练标准化，目标原单位。RF：100棵树，scikit-learn默认其他参数，n_jobs=1。配置`validation_grid`可做训练/验证选优，但未用于当前快照。

加权集成独立i/t/n预测，以验证MAE倒数归一化选权重；零验证误差专门处理，不用测试误差。

## 划分与统计

目录级协议：外层五折各10目录，内层随机5%行。来源隔离：用完整时间/文本/数值输入历史哈希识别同源组，不使用目标；外层每折10目录/6来源，剩余24来源的2组作为验证。

五个训练种子共享固定划分。主表先每种子等权平均五折，再求种子均值和样本SD。合并OOF、来源等权和子群指标另列，避免改变主表口径。复制来源仍被保留，不能将50目录解释为50个独立来源。

## English implementation notes

Image padding/pooling, tokenizer, frozen embeddings, train-only scaling, initialization, Adam settings, early stopping, linear fusion, SVR gamma/epsilon, and inverse-MAE ensemble weighting are explicitly recorded choices. Selected checkpoints use validation MSE only; test data do not determine stopping or weights. The source-isolated protocol groups identical input histories without targets, retains duplicates, and changes inner validation as well as outer grouping.

Training-seed SD is conditional on fixed folds; it is not a population confidence interval. The versioned metrics predate publication-oriented naming and portability changes; the release retains the historical training source hashes. The architecture follows the method cited in the README; implementation choices should not be represented as undisclosed original-author settings.
