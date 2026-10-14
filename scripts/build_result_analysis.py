"""Render a Chinese interpretation and scientific plots from saved diagnostics."""
import base64
import html
from html.parser import HTMLParser
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis_20261014"
RUNS = {"paper": ROOT / "results/paper/0d4ecd526f5caf0a",
        "clean_sources": ROOT / "results/clean_sources/aaaa63a00b5260a1"}
LABELS = {"itn":"图像＋文本＋数值", "in":"图像＋数值", "tn":"文本＋数值", "it":"图像＋文本",
          "i":"仅图像", "t":"仅文本", "n":"仅数值 DNN", "ensemble":"单分支加权集成", "svr":"数值 SVR", "rf":"数值 RF"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=180)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join(["---"]*len(headers)) + "|"] + ["| " + " | ".join(map(str,row)) + " |" for row in rows])


def main():
    d = read(OUT / "diagnostics.json")
    replay = read(OUT / "checkpoint_replay.json")
    assert replay["status"] == "passed" and replay["selected_cases"] == 10 and replay["optimizer_updates"] == 0
    p, c = d["paper"], d["clean_sources"]
    ps = {r["model"]:r for r in p["summary"]}
    cs = {r["model"]:r for r in c["summary"]}
    models = list(LABELS)
    assert all(r["complete_five_folds"] and r["fold_runs"]==25 and r["seeds"]==5 for data in (ps,cs) for r in data.values())
    colors = {"itn":"#c34b3a", "in":"#2678b2", "tn":"#9467bd", "t":"#d59c1a", "svr":"#22845b", "rf":"#747474"}
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    x=np.arange(len(models))
    axes[0].bar(x-.18,[ps[m]["mae_mean"] for m in models],.36,yerr=[ps[m]["mae_std_across_seeds"] for m in models],capsize=2,label="Paper-style split")
    axes[0].bar(x+.18,[cs[m]["mae_mean"] for m in models],.36,yerr=[cs[m]["mae_std_across_seeds"] for m in models],capsize=2,label="Source-isolated split")
    axes[0].set(xticks=x,xticklabels=models,ylabel="MAE (equal-fold mean)",title="Protocols change model ranking")
    axes[0].tick_params(axis="x",rotation=45)
    axes[0].legend(fontsize=8)
    selected=["itn","tn","t","in","svr","rf"]
    x=np.arange(len(selected))
    for offset,key,label in [(-.18,"source_exposed_to_training","Repeated source present in training"),(.18,"source_not_exposed_to_training","Source absent from training")]:
        axes[1].bar(x+offset,[p["models"][m]["subgroups"][key]["mae"] for m in selected],.36,label=label)
    axes[1].set(xticks=x,xticklabels=selected,ylabel="MAE (pooled row/seed predictions)",title="Paper-style test subsets; descriptive, not causal")
    axes[1].legend(fontsize=7)
    for ax in axes: ax.grid(axis="y",alpha=.2)
    save(fig,"protocol_and_exposure")

    bins=["target_0_20","target_20_40","target_40_60","target_60_80","target_80_100","target_100_200"]
    labels=["<20","20–40","40–60","60–80","80–100","≥100"]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for m in ("itn","in","svr"):
        for ax,key in zip(axes,("mae","bias")):
            ax.plot(labels,[c["models"][m]["subgroups"][b][key] for b in bins],"o-",label=m,color=colors[m])
    axes[0].set(title="Source-isolated errors by true target",xlabel="Next-inspection degradation",ylabel="MAE")
    axes[1].set(title="Signed error: prediction minus target",xlabel="Next-inspection degradation",ylabel="Mean signed error")
    axes[1].axhline(0,color="black",lw=.8)
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    save(fig,"severity_diagnostics")

    fig, axes=plt.subplots(2,2,figsize=(12,7))
    for ax,(protocol,m) in zip(axes.flat,[("paper","itn"),("clean_sources","itn"),("paper","t"),("clean_sources","t")]):
        folder=RUNS[protocol]/f"seed_2021/fold_0/{m}"
        h=read(folder/"history.json")
        r=read(folder/"result.json")
        scale=read(folder.parent/"preprocessing.json")["target_scaler"]["scale"]
        for key,label in [("train_mse_scaled","Train: within-epoch average"),("validation_mse_scaled","Validation: end-of-epoch")]:
            ax.plot([row["epoch"] for row in h],[row[key]*scale**2 for row in h],label=label)
        ax.axvline(r["best_epoch"],color="black",ls="--",lw=.8,label=f"Selected epoch {r['best_epoch']}")
        ax.set(title=f"{protocol}; {m}; seed 2021, fold 0",xlabel="Epoch",ylabel="MSE (original target units squared)",yscale="log")
        ax.legend(fontsize=7); ax.grid(alpha=.2)
    save(fig,"train_validation_diagnostics")

    fig,axes=plt.subplots(2,1,figsize=(13,7))
    for ax,trajectory in zip(axes,(19,16)):
        for mi,m in enumerate(("itn","in","svr")):
            rows=[]
            for f in range(5):
                r=read(RUNS["clean_sources"]/f"seed_2021/fold_{f}/{m}/result.json")
                manifest=read(ROOT/"artifacts/clean_sources/manifest.json")
                rows.extend((manifest[i],v) for i,v in zip(r["test_ids"],r["test_predictions"]) if manifest[i]["trajectory"]==trajectory)
            rows.sort(key=lambda item:item[0]["sample"])
            time=[row["time"] for row,v in rows]
            if mi==0: ax.plot(time,[row["target"] for row,v in rows],color="black",lw=1.5,label="Ground truth")
            ax.plot(time,[v for row,v in rows],color=colors[m],alpha=.85,lw=1,label=m)
        ax.set(title=f"Source-isolated trajectory {trajectory}; diagnostic difficult source; fixed seed 2021",xlabel="Time (atu)",ylabel="Next-inspection degradation")
        ax.legend(ncol=4,fontsize=8); ax.grid(alpha=.2)
    save(fig,"difficult_source_trajectories")

    def relative(a,b): return (a/b-1)*100
    lines=["# 已完成实验的完整结果分析", "", "分析日期：2026-10-14。仅读取既有结果及最佳权重做推理；本次没有训练、调参或改写原实验产物。", "",
           "## 1. 总体判断与完成范围", "",
           "两套协议各完成 10 方法 × 5 外层折 × 5 训练种子 = 250 项，共 500 项结果。其中 450 项模型拟合、50 项单分支组合。全部保存指标已由预测重算核对；350 个 DNN 权重和 100 个传统模型在恢复验收中均可加载。此次另外重放 10 个代表性最佳 DNN 权重，其验证、测试预测与已保存文件的最大差值均为 0。重放只验证这 10 项，不声称所有权重都已逐预测重放。", "",
           "结论：工程执行已完成，数值复现尚未达到论文；论文式划分中三分支具有 MAE 优势，但数据来源隔离后该优势消失。现有数据更支持优先解决文本泛化与模态融合问题，而非直接增加网络复杂度。", "",
           "## 2. 指标含义及汇总口径", "",
           "预测目标是下一次计划检查时的退化程度（nextinslevel），不是剩余寿命 RUL，也不是故障分类准确率。MAE 与目标同单位；MSE 单位平方且对大误差更敏感；MRE = 100 × mean(|预测−真实|/真实)，是相对误差百分比，不是准确率。三者均越低越好。", "",
           "主表先对每个训练种子的五折等权平均，再对五个种子求均值及样本标准差。以下 ± 是训练种子 SD，不是置信区间。两个协议各只有一组固定外层划分；五个种子没有重新抽取外层折。下文子群、偏差及来源等权分析合并五个种子的样本预测，按行加权，因此与主表的等权折均值略有差别。这里没有把五次预测平均后作为一个新集成模型评估。", "",
           "## 3. 十种方法的完整对照", ""]
    lines.append(markdown_table(["模型","论文 MAE","论文式 MAE ± SD","论文式 MSE","论文式 MRE%","来源隔离 MAE ± SD","来源隔离 MSE","来源隔离 MRE%"],
                               [[f"{m}：{LABELS[m]}",f"{ps[m]['paper_mae']:.2f}",f"{ps[m]['mae_mean']:.4f} ± {ps[m]['mae_std_across_seeds']:.4f}",f"{ps[m]['mse_mean']:.2f}",f"{ps[m]['mre_percent_mean']:.2f}",f"{cs[m]['mae_mean']:.4f} ± {cs[m]['mae_std_across_seeds']:.4f}",f"{cs[m]['mse_mean']:.2f}",f"{cs[m]['mre_percent_mean']:.2f}"] for m in models]))
    lines.extend(["",f"论文式：itn 的 MAE/MRE 最好，但 MSE 最好的是 RF（{ps['rf']['mse_mean']:.2f}），in 的 MSE（{ps['in']['mse_mean']:.2f}）也优于 itn（{ps['itn']['mse_mean']:.2f}）。所以不能说三分支在全部指标上全面领先。itn 相对 tn 的 MAE 改善仅 {-relative(ps['itn']['mae_mean'],ps['tn']['mae_mean']):.2f}%，相对 RF 改善 {-relative(ps['itn']['mae_mean'],ps['rf']['mae_mean']):.2f}%。", "",
                  f"来源隔离：SVR 的 MAE/MRE 最好；in 的 MSE 最好，MAE 排第二。相对 in，加入文本的 itn MAE 恶化 {relative(cs['itn']['mae_mean'],cs['in']['mae_mean']):.2f}%；相对 SVR 恶化 {relative(cs['itn']['mae_mean'],cs['svr']['mae_mean']):.2f}%。itn 在 25 个匹配折/种子实验中没有一次 MAE 优于 in、RF 或 SVR。来源等权后，对 in 的 30 个来源也全部落后，对 SVR 仅 1/30 来源更好。这些是描述性的胜负计数，不是 25 次独立显著性检验。", "",
                  f"图像并非完全无用：来源隔离时 in 相对 n 的 MAE 改善 {-relative(cs['in']['mae_mean'],cs['n']['mae_mean']):.2f}%，25 项中 24 项更好，30 来源中 26 个更好。但 in 相对 SVR 的平均 MAE 仍高 {relative(cs['in']['mae_mean'],cs['svr']['mae_mean']):.2f}%，没有证明多模态必然优于强数值基线。", "",
                  "![协议与来源重叠对照](protocol_and_exposure.png)", "", "## 4. 与原论文的差距", "",
                  markdown_table(["itn 指标","论文 Table 3","论文式复现","相对论文恶化"],[[key, f"{ps['itn']['paper_'+field]:.2f}",f"{ps['itn'][field+'_mean']:.4f}",f"{relative(ps['itn'][field+'_mean'],ps['itn']['paper_'+field]):.2f}%"] for key,field in [("MAE","mae"),("MSE","mse"),("MRE%","mre_percent")]]), "",
                  "原论文 itn/tn/t/RF 的 MAE 分别为 3.20/3.66/4.91/4.80。本次分别为 4.86/5.09/7.06/5.76，尤其文本相关方法没有达到原数值。原论文 in/n/i 的 MAE 为 7.74/8.31/16.74，本次则为 6.43/7.48/15.09；偏差不是所有方法统一变差，不能用一个统一缩放或指标单位错误解释。", "",
                  "没有取得作者原始代码、原折列表及全部训练参数。优化器、学习率、初始化、目标标准化、嵌入冻结、文本末端激活、早停、SVR gamma/epsilon 和具体集成权重映射等均有公开记录的工程假设。当前能称为基于公开方法与数据的可审计复现，不能称为与作者实现完全一致。来源隔离是额外泛化协议，不能把其 12.51 与原论文 3.20 直接视为同条件复现差距。", "",
                  "## 5. 重复历史与文本重用：最大的解释因素", "",
                  "50 个文件夹只有 30 个不同的时间/文本/数值输入历史分组。1–20 与 31–50 一一成对，全部目标也相同；17 对图像完全相同，另外 3 对各只有一张图不同。来源分组不使用目标，也不作为模型输入。这些重复证据来自本地发布数据；不能据此断言作者的未知原始折存在相同重叠。", "",
                  f"本次论文式外层划分中 {p['test_rows_exposed_to_source_in_training']}/7626（{p['test_rows_exposed_to_source_in_training']/7626*100:.2f}%）测试行的重复来源出现在训练集里。itn 在已见来源 MAE={p['models']['itn']['subgroups']['source_exposed_to_training']['mae']:.2f}，未见来源 MAE={p['models']['itn']['subgroups']['source_not_exposed_to_training']['mae']:.2f}；差异很大。因此 4.86 不能直接解读为完全新设备/历史的泛化误差。", "",
                  f"完全相同的整条文本，在论文式测试中 {p['test_rows_exact_text_seen']}/7626（{p['test_rows_exact_text_seen']/7626*100:.2f}%）行已在训练中出现，来源隔离时仅 {c['test_rows_exact_text_seen']}/7626（{c['test_rows_exact_text_seen']/7626*100:.2f}%）。来源隔离 itn 的已见文本 MAE={c['models']['itn']['subgroups']['exact_text_seen_in_training']['mae']:.2f}，未见文本 MAE={c['models']['itn']['subgroups']['exact_text_unseen_in_training']['mae']:.2f}。两个协议所有测试行的分词词汇都在相应训练词表中出现，故这次下降不能解释为大量测试单词被映射为 UNK。GloVe 本身仍有两个无预训练向量词，这是另一概念。", "",
                  "这些现象与模型依赖训练中重复文本/历史的解释一致，但子群难度和目标分布也不同，不能仅由描述性比较估计重叠的独立因果效应。来源隔离还同时改变外层折、内部验证分组和实际验证大小：论文式各折约 5.01%，来源隔离五折分别为 " + "/".join(f"{s['validation_fraction_of_non_test']*100:.2f}%" for s in c["split_diagnostics"]) + "。需要控制对照才能拆分影响。", "",
                  "## 6. 训练是否充分，是否过拟合", "",
                  markdown_table(["模型","论文式最佳 epoch 中位数","来源隔离最佳 epoch 中位数","来源隔离最佳≤10 epoch 次数/25","来源隔离跑满500次/25"],[[m, int(p['models'][m]['training']['best_epoch_median']),int(c['models'][m]['training']['best_epoch_median']),c['models'][m]['training']['best_epoch_at_most_10_count'],c['models'][m]['training']['max_epochs_reached_count']] for m in ("itn","tn","it","t","in","n")]), "",
                  "来源隔离 it/t 分别在 24/25、22/25 项中于前十轮取得最佳验证点；itn 也有 14/25 项。许多曲线随后验证误差升高，patience=40 后停止。最佳 epoch 很早不表示漏跑或只训练了几轮：例如最佳第 6 轮的实验通常仍运行到第 46 轮，并恢复最佳权重。对这些情况增加最大 epoch 本身不会改变已选最佳模型，需要改善泛化或验证/选择方案。", "",
                  "最佳权重的精确重放例子（种子 2021）：", ""])
    selected=[r for r in replay["records"] if (r["protocol"],r["fold"],r["model"]) in (("paper",0,"itn"),("clean_sources",0,"itn"),("clean_sources",0,"tn"),("clean_sources",0,"in"),("clean_sources",3,"itn"))]
    lines.append(markdown_table(["协议/折/模型","最佳 epoch","训练 MAE","验证 MAE","测试 MAE"],[[f"{r['protocol']}/{r['fold']}/{r['model']}",r['best_epoch'],*[f"{r[k+'_metrics_replayed']['mae']:.3f}" for k in ('train','validation','test')]] for r in selected]))
    lines.extend(["", "来源隔离 fold 3 的 itn 训练 MAE 已达 1.58，测试仍为 12.58；fold 0 的 tn 训练 2.79、测试 10.12。这是明显的泛化差距，不能将所有不佳表现归为优化没收敛。另一方面 n 在论文式 19/25、来源隔离 4/25 项达到 500 上限，有些仍在缓慢改善，训练预算对该小数值网络可能影响结果；该假设需要验证集上的学习率/预算对照，而非从测试表现延长训练。", "",
                  f"来源隔离的训练均值常数预测 sanity reference（各折只取本折训练目标均值）合并 OOF MAE 为 {c['train_mean_constant_sanity_reference']['mae']:.2f}，高于文本单分支的约16.73。因此文本模型并非完全没有学习信号；当前问题是其可迁移信号不足，尚不如数值基线。这个常数参考只用于诊断，没有纳入原论文主表。", "",
                  "图中训练 MSE 为各批次更新期间的平均，验证 MSE 为轮末模型计算，两者不严格同一权重；精确差距以上面的最佳权重重放为依据。各折目标尺度不同，诊断曲线已还原为目标原单位平方。", "",
                  "![训练与验证诊断](train_validation_diagnostics.png)", "", "## 7. 折间差异、随机性与重复加权", "",
                  markdown_table(["模型","论文式 fold0/1/2/3/4 MAE","来源隔离 fold0/1/2/3/4 MAE"],[[m," / ".join(f"{r['mae_mean']:.2f}" for r in p['models'][m]['folds'])," / ".join(f"{r['mae_mean']:.2f}" for r in c['models'][m]['folds'])] for m in ('itn','tn','in','svr','rf')]), "",
                  "论文式 itn 的种子 SD 为 0.214，但折均值从 3.21 到 7.69：设备/来源构成的差异明显大于初始化波动。来源隔离 itn 种子 SD 为 0.973，相对均值约 7.77%，高于 in 的约 3.31%。SVR 在固定同一划分下为确定性方法，所以种子 SD=0；这不表示新设备不确定性为零。RF 的很小种子 SD 也只表示当前树随机性影响小。", "",
                  "发布样本未去重，双份历史在训练与按行评估中仍有双份权重。补充来源等权 MAE（对每来源先求其全部行/种子的 MAE，再对30来源平均）如下；它修正评估权重，不是去重后重新训练的结果：", "",
                  markdown_table(["模型","论文式主表 MAE","论文式来源等权 MAE","来源隔离主表 MAE","来源隔离来源等权 MAE"],[[m,f"{ps[m]['mae_mean']:.3f}",f"{p['models'][m]['source_macro_mae']:.3f}",f"{cs[m]['mae_mean']:.3f}",f"{c['models'][m]['source_macro_mae']:.3f}"] for m in ('itn','tn','in','svr','rf')]), "",
                  "论文式 itn 相对 RF 的主表优势为 0.90，而来源等权后仅约 0.26；来源隔离后 SVR/in 的领先顺序仍然成立。7626 行不是 7626 个独立设备，重复目标区间和重复历史有相关性；25 折/种子也不是25个独立数据集。现阶段没有报告独立总体置信区间或 p 值。正式比较需增加来源分组重划分，并按来源而非按行处理不确定性。", "",
                  "## 8. 哪些目标范围及设备最难", "",
                  "来源隔离、合并五种子的逐行误差：", "",
                  markdown_table(["真实退化范围","原始行数","itn MAE","itn 偏差(预测−真值)","itn MRE%","in MAE","SVR MAE"],[[label,c['models']['itn']['subgroups'][b]['unique_rows'],f"{c['models']['itn']['subgroups'][b]['mae']:.2f}",f"{c['models']['itn']['subgroups'][b]['bias']:+.2f}",f"{c['models']['itn']['subgroups'][b]['mre_percent']:.2f}",f"{c['models']['in']['subgroups'][b]['mae']:.2f}",f"{c['models']['svr']['subgroups'][b]['mae']:.2f}"] for label,b in zip(labels,bins)]), "",
                  "itn 对低退化系统性高估、对高退化系统性低估，表现出向中间值收缩。总体偏差约 −0.68 看起来不大，但正负偏差相互抵消，不能据此称模型校准良好。低值区 MRE 放大是分母较小的自然结果；高退化区的绝对误差则更大。对维护风险而言，高值低估值得优先检查，但当前实验没有故障阈值与决策成本，不能直接换算漏报率或实际危险事件。真实目标分区仅用于事后诊断；预测时不能根据未知真实目标选择模型。", "",
                  f"来源隔离 itn 的绝对误差第95百分位为 {c['models']['itn']['pooled_all_seed_predictions']['p95_absolute_error']:.2f}，最大值 {c['models']['itn']['pooled_all_seed_predictions']['max_absolute_error']:.2f}；最大的5%预测误差贡献了 {c['models']['itn']['worst_5_percent_squared_error_share_percent']:.2f}% 平方误差。论文式对应 P95={p['models']['itn']['pooled_all_seed_predictions']['p95_absolute_error']:.2f}。这些统计合并五种子预测，不能当作某个部署模型的单次误差保证。", "",
                  "![退化程度分区诊断](severity_diagnostics.png)", "",
                  "最难的来源隔离 itn 来源（先合并同来源目录，防止把复制件计作独立案例）：", "",
                  markdown_table(["来源组","文件夹","itn MAE","itn 平均偏差","in MAE","SVR MAE"],[[r['source_group'],','.join(map(str,r['trajectories'])),f"{r['mae']:.2f}",f"{r['bias']:+.2f}",f"{next(z['mae'] for z in c['models']['in']['sources'] if z['source_group']==r['source_group']):.2f}",f"{next(z['mae'] for z in c['models']['svr']['sources'] if z['source_group']==r['source_group']):.2f}"] for r in sorted(c['models']['itn']['sources'],key=lambda row:row['mae'],reverse=True)[:5]]), "",
                  "19/49 是同一来源，16/46 是同一来源，应分别只看作一个困难来源。图中使用固定首种子2021展示两个困难来源，未选最佳测试种子：19 在后期高退化区存在明显低估，16 的误差也集中在部分区间。这些目标本来表示下次检查退化，会有平台和维修后的下降；不能简单施加整条序列单调递增约束，也不能把平台当作标签对齐错误。", "",
                  "![困难来源预测轨迹](difficult_source_trajectories.png)", "", "## 9. 空白图像与集成为什么没有改善", "",
                  f"有 {1534}/7626（{1534/7626*100:.2f}%）图像属于无信号模板，仍保留坐标轴。来源隔离 itn 在这些行 MAE={c['models']['itn']['subgroups']['blank_image']['mae']:.2f}，有信号图像行为 {c['models']['itn']['subgroups']['signal_image']['mae']:.2f}。这个差异支持单独报告模态缺失场景，但发生时点、文本、目标分布也不同，不能断言替换为零矩阵就能改善。zero_template 是尚未运行的敏感性设置。", "",
                  "集成用验证 MAE 倒数确定 i/t/n 权重。平均图像/文本/数值权重，论文式为 " + "/".join(f"{v*100:.2f}%" for v in p['ensemble_weights_mean_image_text_numeric']) + "，来源隔离为 " + "/".join(f"{v*100:.2f}%" for v in c['ensemble_weights_mean_image_text_numeric']) + "。论文式验证文本误差很低，所以文本权重大；来源隔离虽降低了文本权重，图像与文本仍合计近49%，其单分支泛化误差较大。最终集成 MAE 9.52，高于数值 DNN 8.20，且25项匹配实验全部不如 n。因此固定正权重平均没有保护较强数值分支。不能据此证明任何动态门控都会成功，需与数值保底、只用验证选权重、残差融合等实测对照。", "",
                  "当前融合头为拼接后的线性标量层，输出等于各模态分支贡献的加和；融合头本身没有可学习的乘法/非线性交互，分支特征又压缩到4维。它是现有实现的结构特点，可能限制条件依赖融合；尚无证据证明这就是性能差距的唯一原因。", "", "## 10. 计算成本与实现校验边界", "",
                  markdown_table(["方法","论文式平均训练秒/折","来源隔离平均训练秒/折"],[[m,f"{ps[m]['training_seconds_mean_per_fold']:.2f}",f"{cs[m]['training_seconds_mean_per_fold']:.2f}"] for m in ('itn','in','tn','i','t','n','svr','rf')]), "",
                  "计时包括每轮验证，停止轮数不同；本机GPU与论文CPU环境不同，不能把这些数值宣称为论文级加速倍数。集成记录时间是三单分支时间之和，统计总拟合成本时不能再次累加集成。", "",
                  "已确认：原始数据校验、预测/标签行对齐、目标还原、测试与验证指标重算、集成权重重建、模型可加载及此次代表性权重预测重放。没有观察到 NaN、漏折或恢复导致的旧结果改写。上述校验证明产物内部一致，不证明未公开的作者实现完全一致，也不自动证明评估划分具备真实工业代表性。", "",
                  "## 11. 下一步的研究顺序（尚未运行）", "",
                  "1. 先补控制实验拆分来源重叠和内部验证的影响。固定当前外层目录折，运行设备级内部验证（已有 group_validation 配置，但不能隔离重复来源）；固定当前来源外层折，再比较样本级与来源级验证。尽量对齐验证样本预算并记录来源数，随机样本验证包含同源信息本身也应作为研究因素。这样才能判断文本下降有多少来自外层重叠、有多少来自模型选择变化。", "",
                  "2. 使用完全来源隔离的训练/内部验证做预处理与超参敏感性：嵌入冻结/微调、文本末端激活、标准化、空白图像处理、正则化及合理学习率/预算；SVR 与 RF 也需在同等内部验证下选择参数。主表早停按 MSE 选而集成/一些比较按 MAE，需要预先明确主指标与选择指标。不能按已看过的测试曲线选择最优超参后仍把原测试称为未经使用的最终检验。", "",
                  "3. 在30来源层面做去重/来源平衡的训练与评估敏感性；增加来源分组的外层重划分，把训练随机性与划分随机性分别报告。当前测试结果已经用于诊断，后续开发优先使用嵌套来源分组交叉验证或另取独立数据作为最终确认，不把五种子SD当总体置信区间。", "",
                  "4. 创新优先方向是‘文本与图像不可靠时，融合如何保留强数值预测’：以 SVR 和 in 为必须比较的强基线，验证学习残差、模态置信度/缺失掩码与动态权重。针对维修事件/检查区间引入时间建模时，只使用预测时已知历史及计划检查间隔，避免未来信息。是否优于基线必须实测；增加注意力层本身不是创新有效性的证据。", "",
                  "5. 若希望形成高水平论文，需要至少一个独立多模态数据源验证和按来源报告的不确定性。此处30种输入历史以及高度相关行样本不足以单独支持普遍适用于真实工业新设备的结论。当前最可研究的问题来自评估重叠、文本迁移、缺失模态和负迁移，而非将论文式分数继续压低。", "",
                  "## 12. 可引用结论与证据文件", "",
                  "可以陈述：在公开数据及明确工程假设下完成两套固定划分的全方法比较；论文式三分支 MAE=4.8567，来源隔离=12.5125；来源隔离 SVR MAE=6.8405，图像＋数值 MSE=87.4415；文本相关模型存在明显泛化差距，简单多模态融合尚未优于强数值基线。不能陈述：已精确复现作者所有数值、原论文必然泄漏、三分支在新设备上普遍最优，或某个尚未运行的创新已经有效。", "",
                  "- 原论文：[DOI](https://doi.org/10.1016/j.jmsy.2021.01.007)，[作者上传全文](https://www.researchgate.net/publication/349102644_A_multi-branch_deep_neural_network_model_for_failure_prognostics_based_on_multimodal_data)。论文数值依据已核对的 Table 3，工程差异见 [复现说明](../../docs/reproduction.md)。", "",
                  "- 全部诊断数据：[diagnostics.json](diagnostics.json)；10项重放证据：[checkpoint_replay.json](checkpoint_replay.json)；生成脚本：[分析](../../scripts/analyze_completed_results.py)、[重放](../../scripts/replay_analysis_checkpoints.py)、[报告](../../scripts/build_result_analysis.py)。", "",
                  "- 原始完整报告：[论文式](../../results/paper/0d4ecd526f5caf0a/full_report.html)、[来源隔离](../../results/clean_sources/aaaa63a00b5260a1/full_report.html)；恢复完整性：[验收记录](../recovery_20261014/completion_receipt.json)。", ""])
    markdown="\n".join(lines)
    (OUT/"result_analysis.md").write_text(markdown,encoding="utf-8")
    # Minimal Markdown rendering for this generated document; embed scientific figures.
    blocks=[]
    i=0
    def inline(text):
        text=html.escape(text)
        return re.sub(r"\[([^\]]+)\]\(([^)]+)\)",r'<a href="\2">\1</a>',text)
    while i<len(lines):
        line=lines[i]
        if line.startswith("|"):
            table_lines=line.splitlines()
            body=[]
            for ti,tline in enumerate(table_lines):
                if ti==1: continue
                tag="th" if ti==0 else "td"
                body.append("<tr>"+"".join(f"<{tag}>{inline(cell.strip())}</{tag}>" for cell in tline.strip("|").split("|"))+"</tr>")
            blocks.append('<div class="table"><table>'+"".join(body)+"</table></div>")
        elif line.startswith("!["):
            match=re.fullmatch(r"!\[([^]]+)\]\(([^)]+)\)",line)
            encoded=base64.b64encode((OUT/match[2]).read_bytes()).decode()
            blocks.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(match[1])}"><figcaption>{html.escape(match[1])}</figcaption></figure>')
        elif line.startswith("# ") or line.startswith("## "):
            level=1 if line.startswith("# ") else 2
            blocks.append(f"<h{level}>{inline(line.lstrip('# '))}</h{level}>")
        elif line:
            blocks.append("<p>"+inline(line)+"</p>")
        i+=1
    document='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>完整实验结果分析</title><style>body{font-family:"Microsoft YaHei",system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 24px;color:#23313c;line-height:1.8}h1{font-size:28px}h2{font-size:21px;margin-top:36px;border-bottom:1px solid #dae2e8;padding-bottom:8px}p{margin:12px 0}a{color:#1667a6}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border:1px solid #dce3e8;text-align:left;white-space:nowrap}th{background:#eaf0f6}.table{overflow:auto}figure{margin:24px 0}img{width:100%;height:auto}figcaption{font-size:13px;color:#666;text-align:center}@media print{body{max-width:none;font-size:10pt}figure,table{break-inside:avoid}h2{break-after:avoid}}</style></head><body>'''+"\n".join(blocks)+"</body></html>"
    (OUT/"result_analysis.html").write_text(document,encoding="utf-8")
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []
        def handle_starttag(self,tag,attrs):
            if tag == "a":
                self.links.extend(value for key,value in attrs if key == "href")
    parser = Links()
    parser.feed(document)
    missing = [link for link in parser.links if not link.startswith("http") and not (OUT/link).exists()]
    assert not missing, missing
    assert document.count("data:image/png;base64,") == 4
    assert all(r[part+"_max_prediction_difference_from_saved"] == 0 for r in replay["records"] for part in ("validation", "test"))
    (OUT/"analysis_verification.json").write_text(json.dumps({"status":"passed", "completed_results_read":500,
                                                          "selected_checkpoints_replayed":10,
                                                          "maximum_replayed_prediction_difference":0,
                                                          "embedded_figures":4,"missing_local_links":missing,
                                                          "optimizer_updates":0},indent=2),encoding="utf-8")
    print("Wrote",OUT/"result_analysis.html")
    for protocol, data in d.items():
        print(protocol, "constant train mean",data['train_mean_constant_sanity_reference'])
        for m in ('itn','in','svr','t','tn','rf'):
            z=data['models'][m]
            print(m,"tail share",z['worst_5_percent_squared_error_share_percent'],"severity",[(b,round(z['subgroups'][b]['mae'],3),round(z['subgroups'][b]['bias'],3)) for b in bins])


if __name__=="__main__":
    main()
