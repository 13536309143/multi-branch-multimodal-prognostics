"""Regenerate release figures from both experiment ZIPs and frozen diagnostics.

Extract the v1.0.0 experiment archives into the project root first. No training.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager
from matplotlib.text import Text
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FIGURES=ROOT/"docs/figures"
FIGURES.mkdir(parents=True,exist_ok=True)
RUNS={"paper":ROOT/"results/paper/0d4ecd526f5caf0a","clean_sources":ROOT/"results/clean_sources/aaaa63a00b5260a1"}
NAMES={"paper":"Directory-level protocol","clean_sources":"Source-isolated protocol"}
MODELS=["itn","in","tn","it","i","t","n","ensemble","svr","rf"]
COLORS={"itn":"#bd4937","in":"#257bb0","tn":"#895eaf","n":"#cc961f","svr":"#258459","rf":"#777777"}
LANGUAGE="en"
FONT="DejaVu Sans"
MODEL_ZH={"itn":"三模态(itn)","in":"图像＋数值(in)","tn":"文本＋数值(tn)","it":"图像＋文本(it)",
          "i":"仅图像(i)","t":"仅文本(t)","n":"数值网络(n)","ensemble":"加权集成","svr":"支持向量回归","rf":"随机森林"}
TRANSLATIONS={
    "Signal image\n1 × 325 × 194":"信号图像\n1 × 325 × 194",
    "3 × Conv2D + ReLU + Pool\nchannels 4 / 4 / 8\nFlatten 192 → Dense 4 + sigmoid":"三层二维卷积＋激活＋池化\n通道数 4 / 4 / 8\n展平 192 → 全连接 4 ＋ S形激活",
    "Inspection text\n≤27 tokens":"检修文本\n最多27个词",
    "Frozen GloVe 6B, 100d\nConv1D windows 2 / 3 / 4, 32 each\nGlobal max pool → Dense 4 + sigmoid":"冻结 GloVe 6B，100维词向量\n一维卷积窗口 2 / 3 / 4，各32个滤波器\n全局最大池化 → 全连接 4 ＋ S形激活",
    "Numerical data\n4 variables":"数值历史\n4个变量",
    "Train-only standardization\nDense 4 → 4 + sigmoid":"仅用训练集拟合标准化\n全连接 4 → 4 ＋ S形激活",
    "4d":"4维", "Concat\n12d":"特征拼接\n12维",
    "Linear regression\nNext-inspection\ndegradation":"线性回归\n预测下一次检查\n退化程度",
    "MB-DNN: multi-branch multimodal degradation prediction":"MB-DNN：多分支神经网络的多模态退化预测",
    "Train-only scaling and vocabulary  •  MSE training  •  Validation-based checkpoint selection":"训练集拟合缩放与词表  ·  均方误差训练  ·  验证集选择最佳权重",
    "Target distribution; 7,626 aligned rows":"目标分布：7,626条对齐观测",
    "Next-inspection degradation":"下一次检查退化程度", "Rows":"观测数",
    "50 folders; 30 distinct input histories":"50个目录，30种不同输入历史",
    "Original trajectory folder":"原始设备目录", "Signal image":"有信号图像",
    "No-signal template":"无信号模板", "Image availability":"图像可用性",
    "MAE":"平均绝对误差(MAE)", "MSE":"均方误差(MSE)", "MRE (%)":"平均相对误差(%)",
    "Protocol sensitivity":"评估协议敏感性", "MAE; equal-fold mean":"平均绝对误差：五折等权均值",
    "Repeated source present in training":"训练集中出现重复来源", "Source absent from training":"训练集中未见来源",
    "Directory protocol: source-exposure subsets":"目录级协议：按来源重叠分组",
    "MAE; pooled row/seed predictions":"平均绝对误差：合并样本与种子预测",
    "Training: batch-update average":"训练：批次更新期间的平均",
    "Validation: end-of-epoch":"验证：每轮结束时计算",
    "Epoch":"训练轮次", "MSE, original units squared":"均方误差：原始目标单位平方",
    "Source-isolated errors by true target":"来源隔离：按真实退化程度分区",
    "Signed error: prediction minus target":"预测偏差：预测值减真实值",
    "Mean signed error":"平均预测偏差", "Ground truth":"真实值", "Time (atu)":"时间(任意时间单位)",
    "Model":"模型", "Distinct source group":"不同输入历史来源组",
    "Source-isolated MAE by source; five-seed mean":"来源隔离：逐来源绝对误差，五种子均值",
    "Signal image available":"有信号图像", "No-signal image template":"无信号模板图像",
    "Source-isolated missing-image subgroups; descriptive comparison":"来源隔离：图像可用性分组的描述性对照",
    "Absolute prediction error":"绝对预测误差", "Cumulative fraction":"累计比例",
    "Error distributions; all five seed predictions, correlated observations":"误差分布：合并五种子预测，观测之间存在相关性",
    "True next-inspection degradation":"真实下一次检查退化程度", "OOF prediction":"折外测试预测值",
    "Directory-level protocol":"目录级协议", "Source-isolated protocol":"来源隔离协议",
}


def translate(text):
    if re.search(r"[\u4e00-\u9fff]",text):
        return text
    if text in MODEL_ZH:
        return MODEL_ZH[text]
    if text in TRANSLATIONS:
        return TRANSLATIONS[text]
    match=re.fullmatch(r"(Directory-level protocol|Source-isolated protocol); (\w+); seed (\d+) / fold (\d+)",text)
    if match:
        protocol,model,seed,fold=match.groups()
        return f"{TRANSLATIONS[protocol]}；{MODEL_ZH[model]}；种子{seed}／折{fold}"
    match=re.fullmatch(r"Selected epoch (\d+)",text)
    if match:
        return f"最佳轮次：{match[1]}"
    match=re.fullmatch(r"Source-isolated trajectory (\d+); difficult source; fixed seed (\d+)",text)
    if match:
        return f"来源隔离：设备{match[1]}（困难来源）；固定种子{match[2]}"
    match=re.fullmatch(r"(\w+); fixed seed (\d+)",text)
    if match and match[1] in MODEL_ZH:
        return f"{MODEL_ZH[match[1]]}；固定种子{match[2]}"
    for protocol in NAMES.values():
        if text==protocol+"; equal-fold mean and training-seed SD":
            return TRANSLATIONS[protocol]+"：五折等权均值与训练种子标准差"
    if re.search(r"[A-Za-z]{3,}",text) and text.strip():
        raise ValueError(f"Untranslated figure label: {text!r}")
    return text


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig,name):
    if LANGUAGE=="zh":
        for axis in fig.axes:
            ticks=axis.get_xticks()
            labels=axis.get_xticklabels()
            original=[label.get_text() for label in labels]
            translated=[translate(label) for label in original]
            if translated!=original:
                rotation=labels[0].get_rotation() if labels else 0
                if any(label in MODEL_ZH for label in original) and rotation==0:
                    rotation=30
                axis.set_xticks(ticks,translated,rotation=rotation)
        for text in fig.findobj(match=Text):
            text.set_text(translate(text.get_text()))
            text.set_fontfamily(FONT)
        # Chinese labels are longer than model codes; keep comparison panels legible.
        for axis in fig.axes:
            if len(axis.get_xticks())==10:
                axis.tick_params(axis="x",labelsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES/(name+".png"),dpi=170,bbox_inches="tight")
    fig.savefig(FIGURES/(name+".pdf"),bbox_inches="tight")
    plt.close(fig)


def main():
    rows=read(ROOT/"artifacts/paper/manifest.json")
    diagnostics=read(ROOT/"experiments/analysis/diagnostics.json")
    plt.rcParams.update({"font.family":FONT,"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.unicode_minus":False})

    fig,ax=plt.subplots(figsize=(13,6))
    ax.set(xlim=(0,13),ylim=(0,6)); ax.axis("off")
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.1",fc=color,ec="#738797",lw=.8))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=10)
    def arrow(a,b):
        ax.annotate("",xy=b,xytext=a,arrowprops={"arrowstyle":"->","color":"#456477","lw":1.5})
    for y,text,branch,color in [(4.3,"Signal image\n1 × 325 × 194","3 × Conv2D + ReLU + Pool\nchannels 4 / 4 / 8\nFlatten 192 → Dense 4 + sigmoid","#e3eef8"),
                                (2.5,"Inspection text\n≤27 tokens","Frozen GloVe 6B, 100d\nConv1D windows 2 / 3 / 4, 32 each\nGlobal max pool → Dense 4 + sigmoid","#e9e2f3"),
                                (.7,"Numerical data\n4 variables","Train-only standardization\nDense 4 → 4 + sigmoid","#e3f0e8")]:
        box(.2,y,1.8,1.05,text,color); box(2.5,y,4.25,1.05,branch,color)
        box(7.25,y,.65,1.05,"4d",color)
        arrow((2,y+.5),(2.4,y+.5)); arrow((6.85,y+.5),(7.15,y+.5)); arrow((8,y+.5),(8.85,3))
    box(8.9,2.35,1.35,1.2,"Concat\n12d","#f9eed6")
    box(10.8,2.35,1.9,1.2,"Linear regression\nNext-inspection\ndegradation","#f4e0d9")
    arrow((10.35,2.95),(10.7,2.95))
    ax.text(6.5,5.8,"MB-DNN: multi-branch multimodal degradation prediction",ha="center",fontsize=16,weight="bold")
    ax.text(6.5,.12,"Train-only scaling and vocabulary  •  MSE training  •  Validation-based checkpoint selection",ha="center",fontsize=10,color="#456477")
    save(fig,"architecture")

    fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].hist([r["target"] for r in rows],bins=25,color="#3c80af",edgecolor="white")
    axes[0].set(title="Target distribution; 7,626 aligned rows",xlabel="Next-inspection degradation",ylabel="Rows")
    counts=[sum(r["trajectory"]==t for r in rows) for t in range(1,51)]
    axes[1].bar(range(1,51),counts,color=["#3c80af" if t<=30 else "#d19738" for t in range(1,51)])
    axes[1].set(title="50 folders; 30 distinct input histories",xlabel="Original trajectory folder",ylabel="Rows",ylim=(0,190))
    axes[2].bar(["Signal image","No-signal template"],[6092,1534],color=["#3c80af","#d19738"])
    axes[2].set(title="Image availability",ylabel="Rows")
    for i,v in enumerate((6092,1534)):axes[2].text(i,v+100,str(v),ha="center")
    for ax in axes:ax.grid(axis="y",alpha=.2)
    save(fig,"dataset_overview")

    for protocol in RUNS:
        summary={r["model"]:r for r in diagnostics[protocol]["summary"]}
        fig,axes=plt.subplots(1,3,figsize=(15,4.7))
        for ax,key,label in zip(axes,("mae","mse","mre_percent"),("MAE","MSE","MRE (%)")):
            ax.bar(MODELS,[summary[m][key+"_mean"] for m in MODELS],yerr=[summary[m][key+"_std_across_seeds"] for m in MODELS],capsize=2,color=[COLORS.get(m,"#7392a3") for m in MODELS])
            ax.set(title=label,ylabel=label); ax.tick_params(axis="x",rotation=55);ax.grid(axis="y",alpha=.2)
        fig.suptitle(NAMES[protocol]+"; equal-fold mean and training-seed SD",fontsize=14)
        save(fig,"benchmark_"+protocol)

    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    x=np.arange(10)
    for offset,protocol in [(-.18,"paper"),(.18,"clean_sources")]:
        summary={r["model"]:r for r in diagnostics[protocol]["summary"]}
        axes[0].bar(x+offset,[summary[m]["mae_mean"] for m in MODELS],.36,label=NAMES[protocol])
    axes[0].set(xticks=x,xticklabels=MODELS,title="Protocol sensitivity",ylabel="MAE; equal-fold mean")
    axes[0].tick_params(axis="x",rotation=55); axes[0].legend(fontsize=8)
    selected=["itn","tn","t","in","svr","rf"];x=np.arange(len(selected))
    for offset,key,label in [(-.18,"source_exposed_to_training","Repeated source present in training"),(.18,"source_not_exposed_to_training","Source absent from training")]:
        axes[1].bar(x+offset,[diagnostics['paper']['models'][m]['subgroups'][key]['mae'] for m in selected],.36,label=label)
    axes[1].set(xticks=x,xticklabels=selected,title="Directory protocol: source-exposure subsets",ylabel="MAE; pooled row/seed predictions")
    axes[1].legend(fontsize=7)
    for ax in axes:ax.grid(axis="y",alpha=.2)
    save(fig,"protocol_sensitivity")

    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for ax,(protocol,m) in zip(axes.flat,[("paper","itn"),("clean_sources","itn"),("paper","t"),("clean_sources","t")]):
        folder=RUNS[protocol]/f"seed_2021/fold_0/{m}"
        history=read(folder/"history.json");result=read(folder/"result.json")
        scale=read(folder.parent/"preprocessing.json")["target_scaler"]["scale"]
        ax.plot([r['epoch'] for r in history],[r['train_mse_scaled']*scale**2 for r in history],label="Training: batch-update average")
        ax.plot([r['epoch'] for r in history],[r['validation_mse_scaled']*scale**2 for r in history],label="Validation: end-of-epoch")
        ax.axvline(result['best_epoch'],color='black',ls='--',lw=.8,label=f"Selected epoch {result['best_epoch']}")
        ax.set(title=f"{NAMES[protocol]}; {m}; seed 2021 / fold 0",yscale='log',xlabel='Epoch',ylabel='MSE, original units squared');ax.legend(fontsize=7);ax.grid(alpha=.2)
    save(fig,"training_curves")

    bins=["target_0_20","target_20_40","target_40_60","target_60_80","target_80_100","target_100_200"]
    labels=["<20","20–40","40–60","60–80","80–100","≥100"]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for m in ("itn","in","svr"):
        for ax,key in zip(axes,("mae","bias")):
            ax.plot(labels,[diagnostics["clean_sources"]["models"][m]["subgroups"][b][key] for b in bins],"o-",label=m,color=COLORS[m])
    axes[0].set(title="Source-isolated errors by true target",xlabel="Next-inspection degradation",ylabel="MAE")
    axes[1].set(title="Signed error: prediction minus target",xlabel="Next-inspection degradation",ylabel="Mean signed error")
    axes[1].axhline(0,color="black",lw=.8)
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    save(fig,"degradation_ranges")
    fig,axes=plt.subplots(2,1,figsize=(13,7))
    for ax,trajectory in zip(axes,(19,16)):
        for mi,m in enumerate(("itn","in","svr")):
            selected_rows=[]
            for f in range(5):
                result=read(RUNS["clean_sources"]/f"seed_2021/fold_{f}/{m}/result.json")
                selected_rows.extend((rows[i],v) for i,v in zip(result["test_ids"],result["test_predictions"]) if rows[i]["trajectory"]==trajectory)
            selected_rows.sort(key=lambda item:item[0]["sample"])
            time=[row["time"] for row,v in selected_rows]
            if mi==0:ax.plot(time,[row["target"] for row,v in selected_rows],color="black",lw=1.5,label="Ground truth")
            ax.plot(time,[v for row,v in selected_rows],color=COLORS[m],alpha=.85,lw=1,label=m)
        ax.set(title=f"Source-isolated trajectory {trajectory}; difficult source; fixed seed 2021",xlabel="Time (atu)",ylabel="Next-inspection degradation")
        ax.legend(ncol=4,fontsize=8);ax.grid(alpha=.2)
    save(fig,"trajectory_diagnostics")
    selected=["itn","in","tn","n","svr","rf"]
    sources=sorted(r['source_group'] for r in diagnostics['clean_sources']['models']['itn']['sources'])
    matrix=np.array([[next(r['mae'] for r in diagnostics['clean_sources']['models'][m]['sources'] if r['source_group']==g) for m in selected] for g in sources])
    fig,ax=plt.subplots(figsize=(8,9))
    im=ax.imshow(matrix,aspect='auto',cmap='YlOrRd',vmin=0,vmax=25)
    ax.set(xticks=range(len(selected)),xticklabels=selected,yticks=range(30),yticklabels=sources,xlabel='Model',ylabel='Distinct source group',title='Source-isolated MAE by source; five-seed mean')
    for i in range(30):
        for j in range(len(selected)):ax.text(j,i,f"{matrix[i,j]:.1f}",ha='center',va='center',fontsize=7,color='white' if matrix[i,j]>17 else '#333333')
    fig.colorbar(im,ax=ax,label='MAE')
    save(fig,"source_errors")

    fig,ax=plt.subplots(figsize=(9,4.2));selected=["itn","in","tn","n","i"]
    x=np.arange(len(selected))
    for offset,key,label in [(-.18,'signal_image','Signal image available'),(.18,'blank_image','No-signal image template')]:
        ax.bar(x+offset,[diagnostics['clean_sources']['models'][m]['subgroups'][key]['mae'] for m in selected],.36,label=label)
    ax.set(xticks=x,xticklabels=selected,ylabel='MAE',title='Source-isolated missing-image subgroups; descriptive comparison')
    ax.legend();ax.grid(axis='y',alpha=.2);save(fig,"image_availability")

    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    for ax,protocol in zip(axes,RUNS):
        for m in ('itn','in','svr','rf'):
            errors=[]
            for path in sorted(RUNS[protocol].glob(f'seed_*/fold_*/{m}/predictions.csv')):
                with path.open(encoding='utf-8') as stream:
                    errors.extend(abs(float(r['prediction'])-float(r['target'])) for r in csv.DictReader(stream))
            values=np.sort(errors)
            ax.plot(values,np.arange(1,len(values)+1)/len(values),label=m,color=COLORS[m])
        ax.set(title=NAMES[protocol],xlabel='Absolute prediction error',ylabel='Cumulative fraction',xlim=(0,45),ylim=(0,1));ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Error distributions; all five seed predictions, correlated observations',fontsize=12)
    save(fig,"error_distributions")

    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,m in zip(axes,('itn','in','svr')):
        y=[];pred=[]
        for path in sorted(RUNS['clean_sources'].glob(f'seed_2021/fold_*/{m}/predictions.csv')):
            with path.open(encoding='utf-8') as stream:
                for r in csv.DictReader(stream):y.append(float(r['target']));pred.append(float(r['prediction']))
        ax.scatter(y,pred,s=4,alpha=.15,color=COLORS[m],rasterized=True)
        ax.plot([0,140],[0,140],'k--',lw=1)
        ax.set(title=f'{m}; fixed seed 2021',xlabel='True next-inspection degradation',ylabel='OOF prediction',xlim=(0,140),ylim=(0,140));ax.grid(alpha=.2)
    save(fig,"prediction_scatter")
    print('Generated',len(list(FIGURES.glob('*.pdf'))),'publication figure pairs')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language",choices=("en","zh","both"),default="both")
    parser.add_argument("--font-file",type=Path,help="Optional CJK font file for Chinese figure rendering")
    args=parser.parse_args()
    if args.font_file:
        font_manager.fontManager.addfont(args.font_file)
        chinese_font=font_manager.FontProperties(fname=args.font_file).get_name()
    else:
        available={f.name for f in font_manager.fontManager.ttflist}
        chinese_font=next((f for f in ("Microsoft YaHei","Noto Sans CJK SC","Source Han Sans SC","SimHei","WenQuanYi Zen Hei") if f in available),None)
    for language in (("en","zh") if args.language=="both" else (args.language,)):
        if language=="zh" and chinese_font is None:
            raise RuntimeError("Chinese figures need a CJK font: install Noto Sans CJK or supply --font-file.")
        LANGUAGE=language
        FONT=chinese_font if language=="zh" else "DejaVu Sans"
        FIGURES=ROOT/"docs/figures"/("zh-CN" if language=="zh" else "")
        FIGURES.mkdir(parents=True,exist_ok=True)
        main()
