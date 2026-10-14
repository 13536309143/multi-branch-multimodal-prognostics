"""Evaluate released saved predictions or replay a selected trained checkpoint."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from mbdnn.metrics import regression_metrics


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--replay", action="store_true", help="Replay one selected trained DNN or baseline; no optimization")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--model", choices=("itn","in","tn","it","i","t","n","svr","rf"), default="itn")
    parser.add_argument("--device", choices=("auto","cpu","cuda"), default="auto")
    args = parser.parse_args()
    checked = 0
    for path in sorted(args.run_root.glob("seed_*/fold_*/*/result.json")):
        record = read(path)
        with path.with_name("predictions.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert [int(row["row_id"]) for row in rows] == record["test_ids"]
        target = np.asarray([float(row["target"]) for row in rows])
        prediction = np.asarray([float(row["prediction"]) for row in rows])
        np.testing.assert_allclose(prediction, record["test_predictions"], rtol=0, atol=1e-9)
        metrics = regression_metrics(target, prediction)
        for key in metrics:
            np.testing.assert_allclose(metrics[key], record["metrics"][key], rtol=1e-8, atol=1e-8)
        checked += 1
    if checked == 0:
        raise ValueError("No saved experiment predictions found")
    print(f"Verified saved test metrics for {checked} result records. No training executed.")
    if not args.replay:
        return
    import torch
    from mbdnn.data import Standardizer, read_rows
    from mbdnn.experiment import loader, predict
    from mbdnn.models import MultiBranchDNN
    from mbdnn.utils import seed_everything
    folder = args.run_root / f"seed_{args.seed}/fold_{args.fold}/{args.model}"
    result = read(folder / "result.json")
    rows = read_rows(ROOT / "Multimodal data for prognostics")
    if args.model in ("svr","rf"):
        import joblib
        bundle = joblib.load(folder / "model.joblib")
        state = bundle["numeric_scaler"]
        x = np.asarray([rows[i]["numeric"] for i in result["test_ids"]])
        prediction = bundle["model"].predict((x - state["mean"]) / state["scale"])
    else:
        bundle = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
        cfg = {**bundle["config"], "data_root": str(ROOT / "Multimodal data for prognostics")}
        seed_everything(args.seed, cfg["deterministic"])
        scalers = []
        for key in ("numeric_scaler","target_scaler"):
            state = bundle[key]
            scaler = Standardizer(state["mode"])
            scaler.mean, scaler.scale = np.asarray(state["mean"]), np.asarray(state["scale"])
            scalers.append(scaler)
        vocab = bundle["vocabulary"]
        embeddings = np.zeros((len(vocab),100), dtype=np.float32) if "t" in args.model else None
        model = MultiBranchDNN(args.model, embeddings, cfg["embedding_trainable"], cfg["text_dense_activation"])
        model.load_state_dict(bundle["state_dict"], strict=True)
        device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device)
        model.to(device)
        ids, prediction = predict(model, loader(cfg, rows, result["test_ids"], (*scalers,vocab,embeddings,None), args.model), device, scalers[1])
        assert ids == result["test_ids"]
    delta = float(np.max(np.abs(prediction - result["test_predictions"])))
    np.testing.assert_allclose(prediction, result["test_predictions"], rtol=1e-5, atol=1e-4)
    print(json.dumps({"model":args.model,"seed":args.seed,"fold":args.fold,"max_prediction_difference":delta,
                      "metrics":regression_metrics([rows[i]["target"] for i in result["test_ids"]], prediction),
                      "optimizer_updates":0},indent=2))


if __name__=="__main__":
    main()
