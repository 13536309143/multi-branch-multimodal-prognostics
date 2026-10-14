"""Training is only reachable from the explicit CLI train subcommand."""
import csv
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from torch.utils.data import DataLoader

from .data import MultimodalDataset, Standardizer, embedding_matrix, vocabulary
from .metrics import ensemble_weights, regression_metrics
from .splits import make_splits, validate_split
from .utils import environment, read_json, seed_everything, sha256, write_json

DNN_MODELS = ("itn", "in", "tn", "it", "i", "t", "n")
ALL_MODELS = (*DNN_MODELS, "ensemble", "svr", "rf")


def prepared(cfg, check_files=True):
    folder = Path(cfg["artifact_root"])
    rows = read_json(folder / "manifest.json")
    splits = read_json(folder / "splits.json")
    expected = make_splits(rows, cfg["split_seed"], cfg["validation"], cfg["validation_fraction"], cfg["outer_grouping"])
    if splits != expected:
        raise ValueError("Split/config mismatch; run prepare with this config")
    for split in splits:
        validate_split(rows, split)
    checksums = read_json(folder / "checksums.json")
    if check_files:
        for record in checksums:
            if sha256(Path(cfg["data_root"]) / record["path"]) != record["sha256"]:
                raise ValueError(f"Raw data changed: {record['path']}; run prepare again")
    if [r["row_id"] for r in rows] != list(range(len(rows))):
        raise ValueError("Manifest row IDs must be contiguous")
    return rows, splits


def preprocessing(rows, split, cfg, seed, asset=None):
    train = split["train"]
    numeric = Standardizer(cfg["numeric_scaling"]).fit([rows[i]["numeric"] for i in train])
    target = Standardizer(cfg["target_scaling"]).fit([rows[i]["target"] for i in train])
    vocab = vocabulary(rows, train)
    embeddings, coverage = embedding_matrix(vocab, asset, seed) if asset is not None else (None, None)
    return numeric, target, vocab, embeddings, coverage


def loader(cfg, rows, indices, prep, modalities, shuffle=False, seed=0):
    numeric, target, vocab, _, _ = prep
    dataset = MultimodalDataset(cfg["data_root"], rows, indices, vocab, numeric, target, modalities, cfg["blank_images"])
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=shuffle,
                      num_workers=cfg["num_workers"], generator=generator)


def select_device(cfg):
    device = cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items() if k != "row_id"}


@torch.no_grad()
def predict(model, batches, device, target_scaler):
    model.eval()
    ids, values = [], []
    for batch in batches:
        ids.extend(batch["row_id"].tolist())
        values.extend(model(move(batch, device)).cpu().numpy().tolist())
    return ids, target_scaler.inverse(np.asarray(values))


def train_dnn(name, cfg, rows, split, prep, seed, destination):
    from .models import MultiBranchDNN
    seed_everything(seed, cfg["deterministic"])
    device = select_device(cfg)
    model = MultiBranchDNN(name, prep[3], cfg["embedding_trainable"], cfg["text_dense_activation"]).to(device)
    train_loader = loader(cfg, rows, split["train"], prep, name, True, seed)
    val_loader = loader(cfg, rows, split["validation"], prep, name)
    test_loader = loader(cfg, rows, split["test"], prep, name)
    if cfg["optimizer"] != "adam":
        raise ValueError("Only the documented Adam reconstruction is implemented")
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best_loss, bad_epochs, history = float("inf"), 0, []
    best_state, best_epoch = None, None
    start = time.perf_counter()
    for epoch in range(cfg["max_epochs"]):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(batch), batch["target"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch["target"])
            count += len(batch["target"])
        val_ids, val_pred = predict(model, val_loader, device, prep[1])
        val_target = np.array([rows[i]["target"] for i in val_ids])
        val_loss = float(np.mean((prep[1].transform(val_pred) - prep[1].transform(val_target))**2))
        if not np.isfinite(val_loss):
            raise FloatingPointError("Nonfinite validation loss")
        history.append({"epoch": epoch + 1, "train_mse_scaled": total / count, "validation_mse_scaled": val_loss})
        if val_loss < best_loss - cfg["min_delta"]:
            best_loss, bad_epochs, best_epoch = val_loss, 0, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"seed={seed} fold={split['fold']} model={name} epoch={epoch + 1} val_mse={val_loss:.6g}", flush=True)
        if bad_epochs >= cfg["patience"]:
            break
    if best_state is None:
        raise RuntimeError("No checkpoint; max_epochs must be positive")
    model.load_state_dict(best_state)
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start
    val_ids, val_pred = predict(model, val_loader, device, prep[1])
    test_ids, test_pred = predict(model, test_loader, device, prep[1])
    destination.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "modalities": name, "numeric_scaler": prep[0].state(),
                "target_scaler": prep[1].state(), "vocabulary": prep[2], "config": cfg,
                "seed": seed, "fold": split["fold"], "best_epoch": best_epoch}, destination / "checkpoint.pt")
    write_json(destination / "history.json", history)
    return {"model": name, "seed": seed, "fold": split["fold"], "training_seconds": training_seconds,
            "epochs": len(history), "best_epoch": best_epoch, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "validation_ids": val_ids, "validation_predictions": val_pred.tolist(),
            "test_ids": test_ids, "test_predictions": test_pred.tolist()}


def train_baseline(name, cfg, rows, split, prep, seed, destination):
    x = prep[0].transform([r["numeric"] for r in rows])
    y = np.array([r["target"] for r in rows])
    # Classical baselines use targets in original units: epsilon=0.1 is explicit.
    values = ([10.0**i for i in range(-5, 6)] if name == "svr" else list(range(10, 301, 10))) if cfg["baseline_selection"] == "validation_grid" else [cfg["svr_c"] if name == "svr" else cfg["rf_estimators"]]
    best, best_score, best_value, search = None, float("inf"), None, []
    start = time.perf_counter()
    for value in values:
        model = SVR(kernel="rbf", C=value, gamma=cfg["svr_gamma"], epsilon=cfg["svr_epsilon"]) if name == "svr" else RandomForestRegressor(n_estimators=value, random_state=seed, n_jobs=1)
        model.fit(x[split["train"]], y[split["train"]])
        score = regression_metrics(y[split["validation"]], model.predict(x[split["validation"]]))["mae"]
        search.append({"value": value, "validation_mae": score})
        if score < best_score:
            best, best_score, best_value = model, score, value
    training_seconds = time.perf_counter() - start
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best, "numeric_scaler": prep[0].state(), "config": cfg}, destination / "model.joblib")
    write_json(destination / "selection.json", search)
    return {"model": name, "seed": seed, "fold": split["fold"], "training_seconds": training_seconds,
            "selected_parameter": best_value, "validation_ids": split["validation"],
            "validation_predictions": best.predict(x[split["validation"]]).tolist(),
            "test_ids": split["test"], "test_predictions": best.predict(x[split["test"]]).tolist()}


def make_ensemble(results, cfg, rows, split, seed):
    components = [results[name] for name in ("i", "t", "n")]
    for result in components:
        if result["test_ids"] != split["test"] or result["validation_ids"] != split["validation"]:
            raise ValueError("Ensemble row alignment failure")
    target = [rows[i]["target"] for i in split["validation"]]
    errors = [regression_metrics(target, r["validation_predictions"])["mae"] for r in components]
    weights = ensemble_weights(errors, cfg["ensemble_weighting"])
    return {"model": "ensemble", "seed": seed, "fold": split["fold"],
            "weights_image_text_numeric": weights.tolist(), "weight_source": "validation_only",
            "training_seconds": sum(r["training_seconds"] for r in components),
            "validation_ids": split["validation"], "test_ids": split["test"],
            "validation_predictions": (weights @ np.array([r["validation_predictions"] for r in components])).tolist(),
            "test_predictions": (weights @ np.array([r["test_predictions"] for r in components])).tolist()}


def store_result(result, rows, destination):
    result["metrics"] = regression_metrics([rows[i]["target"] for i in result["test_ids"]], result["test_predictions"])
    result["validation_metrics"] = regression_metrics([rows[i]["target"] for i in result["validation_ids"]], result["validation_predictions"])
    with (destination / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row_id", "trajectory", "sample", "time", "target", "prediction", "blank_image"])
        writer.writeheader()
        for i, prediction in zip(result["test_ids"], result["test_predictions"]):
            writer.writerow({**{k: rows[i][k] for k in ("row_id", "trajectory", "sample", "time", "target", "blank_image")}, "prediction": prediction})
    write_json(destination / "result.json", result)


def run_experiments(cfg, models=None, seeds=None, folds=None, resume=False):
    requested = list(models or cfg["models"])
    if not set(requested) <= set(ALL_MODELS):
        raise ValueError("Unknown model")
    schedule = [name for name in ALL_MODELS if name in requested or ("ensemble" in requested and name in ("i", "t", "n"))]
    seeds = list(cfg["seeds"] if seeds is None else seeds)
    folds = list(range(5) if folds is None else folds)
    if not seeds or not folds or not set(folds) <= set(range(5)):
        raise ValueError("Need nonempty seeds and fold indices 0..4")
    if len(set(seeds)) != len(seeds) or len(set(folds)) != len(folds):
        raise ValueError("Duplicate seed or fold")
    rows, splits = prepared(cfg)
    needs_text = any("t" in name for name in schedule if name in DNN_MODELS)
    asset_path = Path(cfg["embedding_file"])
    if needs_text and not asset_path.exists():
        raise FileNotFoundError("Original GloVe missing; run fetch-embeddings before training")
    asset = read_json(asset_path) if needs_text else None
    identity = {"config": cfg, "manifest_sha256": sha256(Path(cfg["artifact_root"]) / "manifest.json"),
                "checksums_sha256": sha256(Path(cfg["artifact_root"]) / "checksums.json"),
                "splits_sha256": sha256(Path(cfg["artifact_root"]) / "splits.json"),
                "embeddings_sha256": sha256(asset_path) if needs_text else None,
                "source_hashes": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    run_root = Path(cfg["results_root"]) / fingerprint
    if run_root.exists() and not resume:
        raise FileExistsError(f"Run exists: {run_root}; use --resume to skip completed models")
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "run.json", {**identity, "fingerprint": fingerprint, "environment": environment(),
                                     "requested_models": requested, "scheduled_models": schedule,
                                     "requested_seeds": seeds, "requested_folds": folds})
    for seed in seeds:
        for fold in folds:
            split, results = splits[fold], {}
            prep = preprocessing(rows, split, cfg, seed, asset)
            fold_root = run_root / f"seed_{seed}" / f"fold_{fold}"
            write_json(fold_root / "preprocessing.json", {"numeric_scaler": prep[0].state(), "target_scaler": prep[1].state(),
                                                        "vocabulary": prep[2], "embedding_coverage": prep[4], "fit_row_ids": split["train"]})
            for name in schedule:
                destination = fold_root / name
                result_file = destination / "result.json"
                if resume and result_file.exists():
                    result = read_json(result_file)
                    if result["test_ids"] != split["test"] or result["validation_ids"] != split["validation"]:
                        raise ValueError("Cached result row IDs do not match")
                else:
                    if name in DNN_MODELS:
                        result = train_dnn(name, cfg, rows, split, prep, seed, destination)
                    elif name in ("svr", "rf"):
                        result = train_baseline(name, cfg, rows, split, prep, seed, destination)
                    else:
                        result = make_ensemble(results, cfg, rows, split, seed)
                        destination.mkdir(parents=True, exist_ok=True)
                    store_result(result, rows, destination)
                results[name] = result
                print(f"Completed seed={seed} fold={fold} model={name}: {result['metrics']}", flush=True)
    return run_root
