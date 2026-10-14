"""Regenerate release figures from both experiment ZIPs and frozen diagnostics.

Extract the v1.0.0 experiment archives into the project root first. No training.
"""
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FIGURES=ROOT/"docs/figures"
FIGURES.mkdir(parents=True,exist_ok=True)
RUNS={"paper":ROOT/"results/paper/0d4ecd526f5caf0a","clean_sources":ROOT/"results/clean_sources/aaaa63a00b5260a1"}
NAMES={"paper":"Directory-level protocol","clean_sources":"Source-isolated protocol"}
MODELS=["itn","in","tn","it","i","t","n","ensemble","svr","rf"]
COLORS={"itn":"#bd4937","in":"#257bb0","tn":"#895eaf","n":"#cc961f","svr":"#258459","rf":"#777777"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig,name):
    fig.tight_layout()
    fig.savefig(FIGURES/(name+".png"),dpi=170,bbox_inches="tight")
    fig.savefig(FIGURES/(name+".pdf"),bbox_inches="tight")
    plt.close(fig)


def main():
    rows=read(ROOT/"artifacts/paper/manifest.json")
    diagnostics=read(ROOT/"experiments/analysis/diagnostics.json")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})

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


if __name__=='__main__':main()
