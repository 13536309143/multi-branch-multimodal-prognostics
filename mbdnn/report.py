import csv
from pathlib import Path

import numpy as np

from .experiment import ALL_MODELS
from .metrics import regression_metrics
from .utils import read_json, sha256, write_json

# Numeric comparison targets from Table 3, never substituted for actual results.
PAPER_RESULTS = {
    "itn": (3.20, 45.57, 6.76, 17912), "ensemble": (6.15, 107.13, 18.26, 17607),
    "in": (7.74, 93.76, 13.24, 17071), "tn": (3.66, 54.14, 7.67, 543),
    "it": (6.90, 156.80, 16.06, 17548), "i": (16.74, 414.73, 40.33, 16988),
    "t": (4.91, 106.43, 10.68, 589), "n": (8.31, 110.48, 15.50, 30),
    "svr": (6.93, 78.58, 13.10, 7), "rf": (4.80, 61.78, 9.82, 6)
}


def summarize(run_root, allow_partial=False):
    run_root = Path(run_root)
    run = read_json(run_root / "run.json")
    records = [read_json(p) for p in sorted(run_root.glob("seed_*/fold_*/*/result.json"))]
    if not records:
        raise ValueError("No experiments have been completed; cannot generate experimental results")
    summary = []
    for name in ALL_MODELS:
        selected = [r for r in records if r["model"] == name]
        if not selected:
            continue
        seed_metrics, pooled_metrics = [], []
        present_seeds = sorted({r["seed"] for r in selected})
        all_seeds = sorted(set(run["requested_seeds"]) | set(present_seeds))
        for seed in all_seeds:
            group = sorted((r for r in selected if r["seed"] == seed), key=lambda r: r["fold"])
            if [r["fold"] for r in group] != list(range(5)) and not allow_partial:
                raise ValueError(f"Incomplete five-fold results: {name}, seed={seed}; partial results are not Table 3 reproductions")
            if not group:
                continue
            ids = [i for r in group for i in r["test_ids"]]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate out-of-fold sample IDs")
            seed_metrics.append({key: float(np.mean([r["metrics"][key] for r in group])) for key in ("mae", "mse", "mre_percent")})
            # Rebuild targets from individual CSVs, checking alignment.
            target, pred = [], []
            for r in group:
                file = run_root / f"seed_{seed}" / f"fold_{r['fold']}" / name / "predictions.csv"
                with file.open(encoding="utf-8") as stream:
                    data = list(csv.DictReader(stream))
                if [int(row["row_id"]) for row in data] != r["test_ids"]:
                    raise ValueError("Prediction CSV alignment failure")
                target.extend(float(row["target"]) for row in data)
                pred.extend(float(row["prediction"]) for row in data)
            pooled_metrics.append({"seed": seed, **regression_metrics(target, pred)})
        complete = all([r["fold"] for r in sorted((r for r in selected if r["seed"] == seed), key=lambda r: r["fold"])] == list(range(5)) for seed in all_seeds)
        item = {"model": name, "complete_five_folds": complete, "seeds": len(seed_metrics), "fold_runs": len(selected),
                "training_seconds_mean_per_fold": float(np.mean([r["training_seconds"] for r in selected]))}
        for key, paper_value in zip(("mae", "mse", "mre_percent"), PAPER_RESULTS[name][:3]):
            values = [r[key] for r in seed_metrics]
            item[key + "_mean"] = float(np.mean(values))
            item[key + "_std_across_seeds"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            item["paper_" + key] = paper_value
            item["delta_" + key] = item[key + "_mean"] - paper_value
        item["pooled_oof_metrics_separate_from_paper_average"] = pooled_metrics
        summary.append(item)
    label = "partial_summary" if allow_partial else "table3_comparison"
    write_json(run_root / f"{label}.json", summary)
    scalar_keys = [k for k in summary[0] if k != "pooled_oof_metrics_separate_from_paper_average"]
    with (run_root / f"{label}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys)
        writer.writeheader()
        for row in summary:
            writer.writerow({k: row[k] for k in scalar_keys})
    protocol = run["config"]["outer_grouping"]
    lines = ["# Multimodal prognostics results", "", f"Outer grouping: {protocol}; inner validation: {run['config']['validation']}. Interpret each evaluation protocol separately.", "",
             "Fold metrics are averaged equally; seed SD is reported separately. Training times are not directly comparable to the paper CPU.", "",
             "| Model | MAE | MSE | MRE (%) | Paper MAE | Δ MAE | Complete 5 folds |", "|---|---:|---:|---:|---:|---:|:---:|"]
    for row in summary:
        lines.append(f"| {row['model']} | {row['mae_mean']:.4f} | {row['mse_mean']:.4f} | {row['mre_percent_mean']:.4f} | {row['paper_mae']:.2f} | {row['delta_mae']:+.4f} | {row['complete_five_folds']} |")
    (run_root / f"{label}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def plot_trajectory(run_root, seed, trajectory, models=("itn", "tn", "rf")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run_root = Path(run_root)
    fig, axis = plt.subplots(figsize=(11, 4.5))
    reference = None
    for name in models:
        rows = []
        for file in run_root.glob(f"seed_{seed}/fold_*/{name}/predictions.csv"):
            with file.open(encoding="utf-8") as stream:
                rows.extend(r for r in csv.DictReader(stream) if int(r["trajectory"]) == trajectory)
        rows.sort(key=lambda row: int(row["sample"]))
        if not rows:
            plt.close(fig)
            raise ValueError(f"No OOF predictions for trajectory={trajectory}, seed={seed}, model={name}")
        key = [(r["row_id"], r["target"]) for r in rows]
        if reference is None:
            reference = key
            axis.plot([float(r["time"]) for r in rows], [float(r["target"]) for r in rows], color="black", label="Ground truth", linewidth=1.8)
        elif key != reference:
            plt.close(fig)
            raise ValueError("Models have different OOF samples")
        axis.plot([float(r["time"]) for r in rows], [float(r["prediction"]) for r in rows], label=name, alpha=0.8)
    axis.set(xlabel="Time (atu)", ylabel="Next-inspection degradation", title=f"Trajectory {trajectory}; out-of-fold predictions; seed {seed}")
    axis.legend(ncol=4)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    destination = run_root / "figures"
    destination.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(destination / f"trajectory_{trajectory}_seed_{seed}.{extension}", dpi=200)
    plt.close(fig)
    return destination


def validate_completed_run(run_root, models, seeds):
    """Recompute test/validation scores; require every expected model/seed/fold."""
    run_root = Path(run_root)
    run = read_json(run_root / "run.json")
    artifact = Path(run["config"]["artifact_root"])
    for name, key in (("manifest.json", "manifest_sha256"), ("splits.json", "splits_sha256")):
        if sha256(artifact / name) != run[key]:
            raise ValueError(f"Preparation changed since training: {name}")
    rows, splits = read_json(artifact / "manifest.json"), read_json(artifact / "splits.json")
    records, predictions = [], {}
    for seed in seeds:
        for model in models:
            combined = []
            for fold in range(5):
                folder = run_root / f"seed_{seed}" / f"fold_{fold}" / model
                result = read_json(folder / "result.json")
                if (result["seed"], result["fold"], result["model"]) != (seed, fold, model):
                    raise ValueError(f"Result identity mismatch: {folder}")
                if result["test_ids"] != splits[fold]["test"] or result["validation_ids"] != splits[fold]["validation"]:
                    raise ValueError(f"Result split mismatch: {folder}")
                with (folder / "predictions.csv").open(encoding="utf-8") as stream:
                    csv_rows = list(csv.DictReader(stream))
                if [int(r["row_id"]) for r in csv_rows] != result["test_ids"]:
                    raise ValueError(f"CSV test sample mismatch: {folder}")
                target, prediction = [], []
                for data, index in zip(csv_rows, result["test_ids"]):
                    original = rows[index]
                    if int(data["trajectory"]) != original["trajectory"] or int(data["sample"]) != original["sample"] or not np.isclose(float(data["target"]), original["target"], rtol=0, atol=1e-9) or not np.isclose(float(data["time"]), original["time"], rtol=0, atol=1e-9):
                        raise ValueError(f"CSV target/identity mismatch: {folder}, row={index}")
                    target.append(float(data["target"]))
                    prediction.append(float(data["prediction"]))
                    combined.append({**original, "prediction": float(data["prediction"])})
                if not np.allclose(prediction, result["test_predictions"], rtol=0, atol=1e-9):
                    raise ValueError(f"CSV/JSON prediction mismatch: {folder}")
                computed = regression_metrics(target, prediction)
                val = regression_metrics([rows[i]["target"] for i in result["validation_ids"]], result["validation_predictions"])
                for key in ("mae", "mse", "mre_percent", "rmse", "n"):
                    if not np.isclose(computed[key], result["metrics"][key], rtol=1e-8, atol=1e-8) or not np.isclose(val[key], result["validation_metrics"][key], rtol=1e-8, atol=1e-8):
                        raise ValueError(f"Saved metric mismatch: {folder}, metric={key}")
                if model in ("svr", "rf"):
                    required = ["model.joblib", "selection.json"]
                elif model == "ensemble":
                    required = []
                    weights = np.asarray(result["weights_image_text_numeric"])
                    if weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
                        raise ValueError("Invalid ensemble weights")
                    components = [read_json(folder.parent / m / "result.json") for m in ("i", "t", "n")]
                    for part in ("test", "validation"):
                        if any(c[part + "_ids"] != result[part + "_ids"] for c in components):
                            raise ValueError("Ensemble component row alignment mismatch")
                        expected = weights @ np.array([c[part + "_predictions"] for c in components])
                        if not np.allclose(expected, result[part + "_predictions"], rtol=1e-8, atol=1e-8):
                            raise ValueError("Ensemble predictions do not match saved weights/components")
                else:
                    required = ["checkpoint.pt", "history.json"]
                if any(not (folder / name).is_file() or (folder / name).stat().st_size == 0 for name in required):
                    raise ValueError(f"Missing model/history artifact: {folder}")
                records.append(result)
            if sorted(r["row_id"] for r in combined) != list(range(len(rows))):
                raise ValueError(f"Incomplete or duplicated OOF coverage: {model}, seed={seed}")
            predictions[(seed, model)] = sorted(combined, key=lambda r: r["row_id"])
    write_json(run_root / "result_validation.json", {"status": "passed", "expected_model_seed_fold_runs": len(models) * len(seeds) * 5,
                                                    "checked_runs": len(records), "test_and_validation_metrics_recomputed": True,
                                                    "every_sample_tested_once_per_seed_model": True,
                                                    "model_files_present": True, "checkpoint_weights_replayed": False})
    return run, rows, splits, records, predictions


def _csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def complete_report(run_root, models=None, seeds=None):
    """Create HTML/Markdown and standalone figures from completed experiments only."""
    import base64
    import html
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .experiment import DNN_MODELS
    run_root = Path(run_root)
    info = read_json(run_root / "run.json")
    models = list(models or info["config"]["models"])
    seeds = list(info["config"]["seeds"] if seeds is None else seeds)
    run, rows, splits, records, predictions = validate_completed_run(run_root, models, seeds)
    summary = [r for r in summarize(run_root) if r["model"] in models]
    figures = run_root / "figures"
    figures.mkdir(exist_ok=True)
    fold_rows = [{"model": r["model"], "seed": r["seed"], "fold": r["fold"], **r["metrics"],
                  "validation_mae": r["validation_metrics"]["mae"], "training_seconds": r["training_seconds"],
                  "best_epoch": r.get("best_epoch"), "epochs": r.get("epochs")} for r in records]
    _csv(run_root / "fold_metrics.csv", fold_rows)
    trajectory_rows = []
    for (seed, model), data in predictions.items():
        for trajectory in sorted({r["trajectory"] for r in rows}):
            group = [r for r in data if r["trajectory"] == trajectory]
            trajectory_rows.append({"seed": seed, "model": model, "trajectory": trajectory,
                                    **regression_metrics([r["target"] for r in group], [r["prediction"] for r in group])})
    _csv(run_root / "trajectory_metrics.csv", trajectory_rows)
    bins = []
    for model in models:
        data = [r for seed in seeds for r in predictions[(seed, model)]]
        for lower, upper in ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, float("inf"))):
            group = [r for r in data if lower <= r["target"] < upper]
            if group:
                bins.append({"model": model, "target_bin": f"[{lower},{upper})", "seed_sample_observations": len(group),
                             **regression_metrics([r["target"] for r in group], [r["prediction"] for r in group])})
    _csv(run_root / "degradation_bin_metrics.csv", bins)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, key, label in zip(axes, ("mae", "mse", "mre_percent"), ("MAE", "MSE", "MRE (%)")):
        x = np.arange(len(summary))
        axis.bar(x, [r[key + "_mean"] for r in summary], yerr=[r[key + "_std_across_seeds"] or 0 for r in summary], capsize=2, color="#367ab5")
        axis.scatter(x, [r["paper_" + key] for r in summary], marker="x", color="#cc5b32", label="Paper reference")
        axis.set_xticks(x, [r["model"] for r in summary], rotation=60)
        axis.set_title(label + "; equal-fold mean, seed SD")
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "model_comparison.png", dpi=160)
    fig.savefig(figures / "model_comparison.pdf")
    plt.close(fig)

    primary = "itn" if "itn" in models else models[0]
    # Fixed first seed for illustrative figures; never choose a best test seed.
    data = predictions[(seeds[0], primary)]
    true = np.array([r["target"] for r in data])
    pred = np.array([r["prediction"] for r in data])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(true, pred, s=5, alpha=0.3)
    limits = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    axes[0].plot(limits, limits, "k--", linewidth=1)
    axes[0].set(xlabel="True next-inspection degradation", ylabel="Prediction", title=f"OOF predictions: {primary}, seed {seeds[0]}")
    axes[1].hist(pred - true, bins=50, color="#367ab5")
    axes[1].set(xlabel="Prediction - truth", ylabel="Samples", title="OOF residuals")
    fig.tight_layout()
    fig.savefig(figures / "oof_diagnostics.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for axis, model in zip(axes.flat, DNN_MODELS):
        for fold in range(5):
            history = read_json(run_root / f"seed_{seeds[0]}" / f"fold_{fold}" / model / "history.json") if model in models else []
            if history:
                axis.plot([r["epoch"] for r in history], [r["validation_mse_scaled"] for r in history], alpha=0.6, label=f"fold {fold}")
        axis.set(title=model, xlabel="Epoch", ylabel="Validation MSE (scaled)")
        axis.set_yscale("symlog", linthresh=0.01)
    axes.flat[-1].axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"Validation curves, fixed seed {seeds[0]}")
    fig.tight_layout()
    fig.savefig(figures / "validation_curves.png", dpi=160)
    plt.close(fig)
    plot_models = [m for m in ("itn", "tn", "rf") if m in models] or [primary]
    for trajectory in sorted({r["trajectory"] for r in rows}):
        plot_trajectory(run_root, seeds[0], trajectory, plot_models)
        if trajectory % 10 == 0:
            print(f"Report: trajectory plots {trajectory}/50", flush=True)

    artifact = Path(run["config"]["artifact_root"])
    audit = read_json(artifact / "data_audit.json")
    overlaps = read_json(artifact / "source_overlap.json")
    zero_overlap = all(not x[k] for x in overlaps for k in ("train_test_source_overlap", "train_validation_source_overlap", "validation_test_source_overlap"))
    protocol = run["config"]["outer_grouping"]
    lines = ["# MB-DNN 完整实验报告", "", f"运行指纹：`{run['fingerprint']}`；协议：`{protocol}`；内部验证：`{run['config']['validation']}`。", "",
             f"已完成 {len(models)} 种方法 × {len(seeds)} 个种子 × 5 折，共 {len(records)} 个结果。所有测试/验证指标重算通过，每个种子、每种方法的每条样本恰好测试一次。", "",
             f"数据：{len(rows)} 条样本、{audit['trajectories']} 个目录、{audit['distinct_input_history_groups']} 个不同输入历史。三套集合的复制来源交集均为空：**{zero_overlap}**。", "",
             "目录级协议可能有复制来源跨集合；source_history 隔离复制来源。保留复制样本仍会影响样本权重；协议之间还改变内部验证方式。", "",
             "## 方法对比与种子波动", "", "下表为等权五折平均后的种子均值 ± 种子样本标准差；只有一个种子时不估计标准差。", "",
             "| 方法 | MAE | MSE | MRE (%) | 每折训练秒数 |", "|---|---:|---:|---:|---:|"]
    def display(row, key):
        sd = row[key + "_std_across_seeds"]
        return f"{row[key + '_mean']:.4f}" + (f" ± {sd:.4f}" if sd is not None else " (SD 未估计)")
    table_cells = []
    for row in summary:
        cells = [row["model"], display(row, "mae"), display(row, "mse"), display(row, "mre_percent"), f"{row['training_seconds_mean_per_fold']:.2f}"]
        table_cells.append(cells)
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "各种子的合并 OOF 指标单独见 table3_comparison.json。耗时包含验证及最佳权重保存前的训练阶段，不直接与原论文硬件比较；集成时间是三个单分支的训练时间之和。", "",
              "## 数据划分与一致性验证", "", "| 折 | 训练样本 | 验证样本 | 测试样本 | 测试设备目录 |", "|---|---:|---:|---:|---|"]
    for s in splits:
        lines.append(f"| {s['fold']} | {len(s['train'])} | {len(s['validation'])} | {len(s['test'])} | {s['test_trajectories']} |")
    lines += ["", "result_validation.json 记录结果检查。检查覆盖指标、CSV/JSON 预测对应、原始标签对应、完整测试覆盖以及模型与历史文件存在性；未重新加载全部权重做第二次推断。", "",
              "## 图表", "", "![方法对照](figures/model_comparison.png)", "", "![测试诊断](figures/oof_diagnostics.png)", "", "![验证曲线](figures/validation_curves.png)", "",
              "设备轨迹图使用固定的第一个种子，没有根据测试误差挑选种子或设备。全部 50 台设备均生成 PNG/PDF。", "",
              "## 可追溯文件与解释限制", "",
              "- fold_metrics.csv：每方法、每种子、每折的测试与验证指标及停止轮次。",
              "- trajectory_metrics.csv：每设备、每方法、每种子的测试指标。",
              "- degradation_bin_metrics.csv：退化程度区间误差；观测数包含多个种子的重复评价，不能当成独立设备数量。",
              "- 各 seed/fold/model 目录：测试预测、结果；DNN 的训练历史与最佳权重，SVR/RF 的模型与选择记录。",
              "- run.json：完整参数、环境、数据/划分/词向量指纹与源码哈希。", "",
              "优化、激活、缩放与分词是明确记录的实现选择。超参数与停止规则只由训练/验证决定；测试误差仅用于最终评价。相关架构文献的参考数值另存table3_comparison.json，不主张不同划分与未公开设置下逐数值一致。", "",
              "## 实际参数", "", "```json", __import__("json").dumps(run["config"], ensure_ascii=False, indent=2), "```", "",
              "## 实际环境", "", "```json", __import__("json").dumps(run["environment"], ensure_ascii=False, indent=2), "```", ""]
    (run_root / "full_report.md").write_text("\n".join(lines), encoding="utf-8")
    # Standalone HTML embeds every figure so the report can be shared as one file.
    def embedded_image(path, title):
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<figure><img alt="{html.escape(title)}" src="data:image/png;base64,{encoded}"><figcaption>{html.escape(title)}</figcaption></figure>'
    # Render our generated Markdown subset without another runtime dependency.
    import re
    def inline(text):
        text = html.escape(text)
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    rendered, index = [], 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("```"):
            code = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code.append(lines[index])
                index += 1
            rendered.append('<pre><code>' + html.escape("\n".join(code)) + '</code></pre>')
        elif line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].startswith("|"):
                cells = [c.strip() for c in lines[index].strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", c) for c in cells):
                    tag = "th" if not table_lines else "td"
                    table_lines.append('<tr>' + ''.join(f'<{tag}>{inline(c)}</{tag}>' for c in cells) + '</tr>')
                index += 1
            rendered.append('<div class="table-scroll"><table>' + ''.join(table_lines) + '</table></div>')
            continue
        elif line.startswith("- "):
            items = []
            while index < len(lines) and lines[index].startswith("- "):
                items.append('<li>' + inline(lines[index][2:]) + '</li>')
                index += 1
            rendered.append('<ul>' + ''.join(items) + '</ul>')
            continue
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line)
            if match:
                rendered.append(embedded_image(run_root / match[2], match[1]))
        elif line.startswith("## "):
            rendered.append('<h2>' + inline(line[3:]) + '</h2>')
        elif line.startswith("### "):
            rendered.append('<h3>' + inline(line[4:]) + '</h3>')
        elif line.startswith("# "):
            rendered.append('<h1>' + inline(line[2:]) + '</h1>')
        elif line:
            rendered.append('<p>' + inline(line) + '</p>')
        index += 1
    document = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MB-DNN 完整实验报告</title>',
                '<style>body{font:16px system-ui;max-width:1150px;margin:40px auto;padding:20px;line-height:1.7;color:#192f40}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ccd8e1;text-align:right}th:first-child,td:first-child{text-align:left}.table-scroll{overflow-x:auto}img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px ui-monospace,monospace;background:#f4f7fa;padding:22px}details{margin-top:20px}h1,h2{color:#175fa4}figcaption{color:#587080}code{background:#f4f7fa}@media print{body{margin:0;max-width:none}details{break-inside:avoid}}</style>',
                ''.join(rendered), '<h2>全部设备的测试轨迹</h2>']
    for trajectory in sorted({r["trajectory"] for r in rows}):
        document.append(f'<details><summary>设备 {trajectory}</summary>' + embedded_image(figures / f"trajectory_{trajectory}_seed_{seeds[0]}.png", f"设备 {trajectory}，种子 {seeds[0]}") + '</details>')
    document.append('</html>')
    destination = run_root / "full_report.html"
    destination.write_text("\n".join(document), encoding="utf-8")
    return destination
