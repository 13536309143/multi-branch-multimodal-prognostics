# 验证与结果完整性 / Validation and integrity

## 已完成快照

| 检查 | 范围 | 状态 |
|---|---:|---|
| 方法×种子×折结果 | 两协议各250，共500 | 完整 |
| 原始数据 | 7,977文件、7,626行 | SHA-256及对齐通过 |
| 测试/验证指标 | 500项保存结果 | 已由保存预测与真实标签重算 |
| OOF覆盖 | 每方法、每种子 | 每行恰好作为外层测试一次 |
| 单分支集成 | 50项 | 权重有效且可重建预测 |
| DNN权重 | 350项 | 严格加载、参数有限 |
| 传统模型 | 100项 | 模型加载、4输入特征 |
| 代表性权重预测重放 | 10项DNN | 验证/测试最大差值0 |

完整模型文件加载与代表性预测重放是不同检查；没有声称全部350项神经网络都重新推理。CPU/CUDA以及软件版本改变时，重放允许小浮点误差。

## 使用流程

1. `scripts/verify_environment.py`：依赖版本、设备可用性和矩阵运算，无优化更新。
2. `scripts/download_dataset.py --verify-only`：与发布哈希逐文件比较。
3. `python -m mbdnn ... prepare`：审计、固定划分、来源交集。
4. `python -m mbdnn ... verify`：全部七种模态网络真实输入前向，不训练。
5. `python -m pytest -q checks`：27项必要结构、指标、划分与流程检查；合成夹具只验证程序行为，不写为实验成绩。
6. `scripts/evaluate_snapshot.py <run_root>`：重算保存测试指标。
7. 追加`--replay`：加载指定最佳神经/传统模型并重新预测原测试行，无训练。

## English

The released snapshot has 500 complete result records. Test/validation scores were recomputed, OOF row coverage and ensemble composition were checked, all 350 DNN checkpoints and 100 classical models loaded, and ten representative DNN checkpoints reproduced saved validation/test predictions exactly. This is internal consistency evidence, not a claim that every checkpoint was inference-replayed or that fixed-fold SD captures deployment uncertainty.

Essential contract checks remain under `checks/` because the one-click workflow invokes them before training. Disposable recovery scripts and caches are kept out of the publication. No experiments are run merely by downloading data or checking environments.
