"""Replay selected saved checkpoints for diagnostics, without optimization."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from mbdnn.data import Standardizer
from mbdnn.experiment import loader, predict
from mbdnn.metrics import regression_metrics
from mbdnn.models import MultiBranchDNN
from mbdnn.utils import seed_everything


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = [(p, 0, m) for p in ("paper", "clean_sources") for m in ("itn", "in", "tn", "t")]
    cases.extend([("clean_sources", 3, "itn"), ("clean_sources", 3, "t")])
    records = []
    roots = {"paper": ROOT / "results/paper/0d4ecd526f5caf0a",
             "clean_sources": ROOT / "results/clean_sources/aaaa63a00b5260a1"}
    for protocol, fold, name in cases:
        folder = roots[protocol] / f"seed_2021/fold_{fold}/{name}"
        checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        seed_everything(2021, cfg["deterministic"])
        scalers = []
        for key in ("numeric_scaler", "target_scaler"):
            state = checkpoint[key]
            scaler = Standardizer(state["mode"])
            scaler.mean, scaler.scale = np.asarray(state["mean"]), np.asarray(state["scale"])
            scalers.append(scaler)
        vocab = checkpoint["vocabulary"]
        embedding = np.zeros((len(vocab), 100), dtype=np.float32) if "t" in name else None
        model = MultiBranchDNN(name, embedding, cfg["embedding_trainable"], cfg["text_dense_activation"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device)
        prep = (*scalers, vocab, embedding, None)
        rows = read(ROOT / "artifacts" / protocol / "manifest.json")
        split = read(ROOT / "artifacts" / protocol / "splits.json")[fold]
        result = read(folder / "result.json")
        record = {"protocol": protocol, "fold": fold, "seed": 2021, "model": name,
                  "best_epoch": checkpoint["best_epoch"], "device": str(device)}
        for part in ("train", "validation", "test"):
            ids, prediction = predict(model, loader(cfg, rows, split[part], prep, name), device, prep[1])
            truth = np.asarray([rows[i]["target"] for i in ids])
            record[part + "_metrics_replayed"] = regression_metrics(truth, prediction)
            if part != "train":
                assert ids == result[part + "_ids"]
                delta = float(np.max(np.abs(prediction - result[part + "_predictions"])))
                record[part + "_max_prediction_difference_from_saved"] = delta
                if not np.allclose(prediction, result[part + "_predictions"], rtol=1e-5, atol=1e-4):
                    raise AssertionError(f"Checkpoint replay differs: {folder}, {part}, {delta}")
        records.append(record)
        print(json.dumps(record), flush=True)
    out = ROOT / "artifacts/analysis_20261014/checkpoint_replay.json"
    out.write_text(json.dumps({"status": "passed", "selected_cases": len(records),
                              "optimizer_updates": 0, "records": records}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
