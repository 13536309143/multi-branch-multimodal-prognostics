"""Read-only diagnostics for the two completed fixed-split experiments.

Writes analysis artifacts separately; never trains or rewrites experiment results.
"""
import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "analysis_20261014"
RUNS = {"paper": ROOT / "results/paper/0d4ecd526f5caf0a",
        "clean_sources": ROOT / "results/clean_sources/aaaa63a00b5260a1"}
MODELS = ["itn", "in", "tn", "it", "i", "t", "n", "ensemble", "svr", "rf"]
SEEDS = list(range(2021, 2026))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def score(y, p):
    e = np.asarray(p) - np.asarray(y)
    return {"n": len(e), "mae": float(np.abs(e).mean()), "mse": float((e * e).mean()),
            "mre_percent": float((np.abs(e) / y).mean() * 100), "bias": float(e.mean()),
            "p90_absolute_error": float(np.quantile(np.abs(e), .90)),
            "p95_absolute_error": float(np.quantile(np.abs(e), .95)),
            "max_absolute_error": float(np.abs(e).max())}


def tokens(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    output = {}
    for protocol, root in RUNS.items():
        rows = read(ROOT / "artifacts" / protocol / "manifest.json")
        splits = read(ROOT / "artifacts" / protocol / "splits.json")
        y = np.array([r["target"] for r in rows])
        groups = np.array([r["source_group"] for r in rows])
        trajectories = np.array([r["trajectory"] for r in rows])
        blank = np.array([r["blank_image"] for r in rows])
        pred = {m: np.empty((5, len(rows))) for m in MODELS}
        result = {m: [] for m in MODELS}
        fold_info = []
        exposed = np.zeros(len(rows), dtype=bool)
        text_seen = np.zeros(len(rows), dtype=bool)
        token_unknown = np.zeros(len(rows), dtype=bool)
        for f, split in enumerate(splits):
            train_ids, val_ids, test_ids = (split[k] for k in ("train", "validation", "test"))
            train_groups = set(groups[train_ids])
            train_text = {rows[i]["text"] for i in train_ids}
            vocab = {t for i in train_ids for t in tokens(rows[i]["text"])}
            for i in test_ids:
                exposed[i] = groups[i] in train_groups
                text_seen[i] = rows[i]["text"] in train_text
                token_unknown[i] = any(t not in vocab for t in tokens(rows[i]["text"]))
            fold_info.append({"fold": f, "train_rows": len(train_ids), "validation_rows": len(val_ids),
                              "test_rows": len(test_ids), "validation_fraction_of_non_test": len(val_ids)/(len(train_ids)+len(val_ids)),
                              "train_sources": len(train_groups), "validation_sources": len(set(groups[val_ids])),
                              "test_sources": len(set(groups[test_ids])),
                              "exposed_test_rows": int(exposed[test_ids].sum()),
                              "exact_text_seen_test_rows": int(text_seen[test_ids].sum()),
                              "unknown_token_test_rows": int(token_unknown[test_ids].sum())})
        for si, seed in enumerate(SEEDS):
            for f in range(5):
                for m in MODELS:
                    folder = root / f"seed_{seed}/fold_{f}/{m}"
                    r = read(folder / "result.json")
                    ids = r["test_ids"]
                    pred[m][si, ids] = r["test_predictions"]
                    extra = {"seed": seed, "fold": f, "test_mae": r["metrics"]["mae"],
                             "test_mse": r["metrics"]["mse"], "validation_mae": r["validation_metrics"]["mae"],
                             "validation_mse": r["validation_metrics"]["mse"]}
                    if "epochs" in r:
                        h = read(folder / "history.json")
                        scale = read(folder.parent / "preprocessing.json")["target_scaler"]["scale"]
                        b = h[r["best_epoch"]-1]
                        extra.update(epochs=r["epochs"], best_epoch=r["best_epoch"],
                                     best_epoch_train_mse_approx_original=b["train_mse_scaled"]*scale*scale,
                                     best_epoch_validation_mse_original=b["validation_mse_scaled"]*scale*scale,
                                     last_epoch_train_mse_approx_original=h[-1]["train_mse_scaled"]*scale*scale,
                                     last_epoch_validation_mse_original=h[-1]["validation_mse_scaled"]*scale*scale)
                    if "weights_image_text_numeric" in r:
                        extra["weights_image_text_numeric"] = r["weights_image_text_numeric"]
                    result[m].append(extra)
        summary = read(root / "table3_comparison.json")
        diagnostics = {}
        for m in MODELS:
            p = pred[m]
            repeated_y = np.tile(y, 5)
            e = np.abs(p-y)
            subgroups = {}
            masks = {"blank_image": blank, "signal_image": ~blank,
                     "source_exposed_to_training": exposed, "source_not_exposed_to_training": ~exposed,
                     "exact_text_seen_in_training": text_seen, "exact_text_unseen_in_training": ~text_seen,
                     "contains_unseen_token": token_unknown, "all_tokens_seen": ~token_unknown}
            masks.update({f"target_{lo}_{hi}": (y >= lo) & (y < hi) for lo, hi in [(0,20),(20,40),(40,60),(60,80),(80,100),(100,200)]})
            for key, mask in masks.items():
                if mask.any():
                    subgroups[key] = {"unique_rows": int(mask.sum()), **score(np.tile(y[mask], 5), p[:,mask].reshape(-1))}
            source_stats = []
            for g in sorted(set(groups)):
                mask = groups == g
                source_stats.append({"source_group": int(g), "trajectories": sorted(set(trajectories[mask].tolist())),
                                     "unique_rows": int(mask.sum()), **score(np.tile(y[mask], 5), p[:,mask].reshape(-1))})
            trajectory_stats = []
            for t in sorted(set(trajectories)):
                mask = trajectories == t
                trajectory_stats.append({"trajectory": int(t), "source_group": int(groups[mask][0]),
                                         **score(np.tile(y[mask], 5), p[:,mask].reshape(-1))})
            squared = ((p-y)**2).reshape(-1)
            ae = e.reshape(-1)
            worst = np.argsort(ae)[-int(np.ceil(.05*len(ae))):]
            d = {"pooled_all_seed_predictions": score(repeated_y, p.reshape(-1)),
                 "source_macro_mae": float(np.mean([s["mae"] for s in source_stats])),
                 "trajectory_macro_mae": float(np.mean([s["mae"] for s in trajectory_stats])),
                 "worst_5_percent_squared_error_share_percent": float(squared[worst].sum()/squared.sum()*100),
                 "subgroups": subgroups, "sources": source_stats,
                 "worst_trajectories": sorted(trajectory_stats, key=lambda r:r["mae"], reverse=True)[:10],
                 "folds": [{"fold": f, "mae_mean": float(np.mean([r["test_mae"] for r in result[m] if r["fold"]==f])),
                             "mae_seed_sd": float(np.std([r["test_mae"] for r in result[m] if r["fold"]==f],ddof=1))} for f in range(5)],
                 "runs": result[m]}
            if "epochs" in result[m][0]:
                d["training"] = {"epochs_median": float(np.median([r["epochs"] for r in result[m]])),
                                 "best_epoch_median": float(np.median([r["best_epoch"] for r in result[m]])),
                                 "best_epoch_min": min(r["best_epoch"] for r in result[m]),
                                 "best_epoch_max": max(r["best_epoch"] for r in result[m]),
                                 "best_epoch_at_most_10_count": sum(r["best_epoch"]<=10 for r in result[m]),
                                 "max_epochs_reached_count": sum(r["epochs"]==500 for r in result[m]),
                                 "validation_mae_mean": float(np.mean([r["validation_mae"] for r in result[m]])),
                                 "best_train_mse_approx_mean": float(np.mean([r["best_epoch_train_mse_approx_original"] for r in result[m]])),
                                 "best_validation_mse_mean": float(np.mean([r["best_epoch_validation_mse_original"] for r in result[m]]))}
            diagnostics[m] = d
        paired = []
        for a,b in [("itn","tn"),("itn","in"),("itn","rf"),("itn","svr"),("in","svr"),("in","n"),("ensemble","n")]:
            sources_a = {r["source_group"]: r["mae"] for r in diagnostics[a]["sources"]}
            sources_b = {r["source_group"]: r["mae"] for r in diagnostics[b]["sources"]}
            paired.append({"a": a, "b": b, "runs_a_better_count": sum(x["test_mae"]<z["test_mae"] for x,z in zip(result[a],result[b])),
                           "source_groups_a_better_count": sum(sources_a[g]<sources_b[g] for g in sources_a),
                           "source_macro_mae_difference_a_minus_b": float(np.mean([sources_a[g]-sources_b[g] for g in sources_a]))})
        constant = np.empty(len(rows))
        for split in splits:
            constant[split["test"]] = y[split["train"]].mean()
        output[protocol] = {"summary": summary, "split_diagnostics": fold_info,
                            "train_mean_constant_sanity_reference": score(y,constant),
                            "test_rows_exposed_to_source_in_training": int(exposed.sum()),
                            "test_rows_exact_text_seen": int(text_seen.sum()),
                            "test_rows_with_unseen_tokens": int(token_unknown.sum()),
                            "models": diagnostics, "paired_comparisons": paired,
                            "ensemble_weights_mean_image_text_numeric": np.mean([r["weights_image_text_numeric"] for r in result["ensemble"]],axis=0).tolist()}
    (OUT / "diagnostics.json").write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Wrote", OUT / "diagnostics.json")
    for protocol, data in output.items():
        print("\nPROTOCOL",protocol)
        print("EXPOSURE",data["test_rows_exposed_to_source_in_training"],"TEXT SEEN",data["test_rows_exact_text_seen"],"UNSEEN TOKEN ROWS",data["test_rows_with_unseen_tokens"])
        print("VALIDATION FRACTIONS",[round(r["validation_fraction_of_non_test"],4) for r in data["split_diagnostics"]])
        for m in MODELS:
            d=data["models"][m]
            print(m,"MAE",next(r["mae_mean"] for r in data["summary"] if r["model"]==m),"source_macro",round(d["source_macro_mae"],4),"folds",[round(r["mae_mean"],3) for r in d["folds"]],"training",d.get("training"))
        print("PAIRED",data["paired_comparisons"])
        print("ITN SUBGROUPS",data["models"]["itn"]["subgroups"])
        print("ITN WORST",data["models"]["itn"]["worst_trajectories"][:6])
        print("ENSEMBLE WEIGHTS",data["ensemble_weights_mean_image_text_numeric"])


if __name__ == "__main__":
    main()
