import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import audit
from .experiment import ALL_MODELS, DNN_MODELS, loader, move, prepared, preprocessing, select_device
from .models import MultiBranchDNN
from .splits import make_splits
from .utils import environment, load_config, read_json, seed_everything, sha256, write_json


def prepare(cfg):
    rows, report = audit(cfg["data_root"], cfg["artifact_root"])
    splits = make_splits(rows, cfg["split_seed"], cfg["validation"], cfg["validation_fraction"], cfg["outer_grouping"])
    overlap = []
    for split in splits:
        sources = [{rows[i]["source_group"] for i in split[name]} for name in ("train", "validation", "test")]
        overlap.append({"fold": split["fold"], "train_test_source_overlap": sorted(sources[0] & sources[2]),
                        "train_validation_source_overlap": sorted(sources[0] & sources[1]),
                        "validation_test_source_overlap": sorted(sources[1] & sources[2])})
    write_json(Path(cfg["artifact_root"]) / "source_overlap.json", overlap)
    write_json(Path(cfg["artifact_root"]) / "splits.json", splits)
    write_json(Path(cfg["artifact_root"]) / "preparation.json", {"config": cfg, "environment": environment(),
                                                               "training_executed": False})
    print(json.dumps({"samples": report["samples"], "trajectories": report["trajectories"],
                      "images": report["image_counts"], "distinct_input_history_groups": report["distinct_input_history_groups"],
                      "outer_grouping": cfg["outer_grouping"], "source_overlap": overlap,
                      "split_test_trajectories": [s["test_trajectories"] for s in splits]}, ensure_ascii=False, indent=2))


def verify(cfg):
    rows, splits = prepared(cfg)
    path = Path(cfg["embedding_file"])
    if not path.exists():
        raise FileNotFoundError("GloVe missing; run fetch-embeddings")
    seed_everything(cfg["seeds"][0], cfg["deterministic"])
    prep = preprocessing(rows, splits[0], cfg, cfg["seeds"][0], read_json(path))
    batch = next(iter(loader(cfg, rows, splits[0]["validation"][:4], prep, "itn")))
    device = select_device(cfg)
    model_input = move(batch, device)
    checks = []
    with torch.no_grad():
        for name in DNN_MODELS:
            model = MultiBranchDNN(name, prep[3], cfg["embedding_trainable"], cfg["text_dense_activation"]).to(device).eval()
            output = model(model_input)
            if output.shape != (len(batch["row_id"]),) or not torch.isfinite(output).all():
                raise ValueError(f"Forward verification failed: {name}")
            checks.append({"model": name, "output_shape": list(output.shape), "parameters": sum(p.numel() for p in model.parameters()),
                           "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    result = {"training_executed": False, "backpropagation_executed": False, "data_samples": len(rows),
              "embedding_sha256": sha256(path), "coverage_fold_0": prep[4], "forward_checks": checks,
              "device_checked": str(device), "environment": environment()}
    write_json(Path(cfg["artifact_root"]) / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Multi-branch multimodal prognostics. Training requires explicit 'train'.")
    parser.add_argument("--config", default="configs/paper.json", help="Config path (before subcommand)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Audit raw data and persist fixed splits; NO training")
    fetch = sub.add_parser("fetch-embeddings", help="Download original pretrained vectors; NO training")
    fetch.add_argument("--local-file", help="Use an existing original glove.6B.100d.txt")
    sub.add_parser("verify", help="Data, split and inference-only checks; NO training")
    sub.add_parser("plan", help="Show experiment schedule and missing inputs; NO training")
    train = sub.add_parser("train", help="Explicitly launch experiments")
    train.add_argument("--models", nargs="+", choices=ALL_MODELS)
    train.add_argument("--seeds", nargs="+", type=int)
    train.add_argument("--folds", nargs="+", type=int, choices=range(5))
    train.add_argument("--resume", action="store_true", help="Skip completed models; incomplete models restart")
    summary = sub.add_parser("summarize", help="Summarize existing results only")
    summary.add_argument("run_root")
    summary.add_argument("--allow-partial", action="store_true")
    plot = sub.add_parser("plot", help="Plot existing OOF predictions only")
    plot.add_argument("run_root")
    plot.add_argument("--seed", type=int, default=2021)
    plot.add_argument("--trajectory", type=int, default=1)
    plot.add_argument("--models", nargs="+", choices=ALL_MODELS, default=["itn", "tn", "rf"])
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "prepare":
        prepare(cfg)
    elif args.command == "fetch-embeddings":
        from .embeddings import fetch_embeddings
        if Path(cfg["embedding_file"]).exists():
            print(f"Existing asset preserved: {cfg['embedding_file']}")
        else:
            fetch_embeddings(cfg["data_root"], cfg["embedding_file"], args.local_file)
    elif args.command == "verify":
        verify(cfg)
    elif args.command == "plan":
        print(json.dumps({"training_executed": False, "models": cfg["models"], "seeds": cfg["seeds"],
                          "outer_folds": 5, "validation_policy": cfg["validation"],
                          "raw_data_exists": Path(cfg["data_root"]).is_dir(),
                          "prepared": (Path(cfg["artifact_root"]) / "splits.json").exists(),
                          "original_glove_ready": Path(cfg["embedding_file"]).exists(),
                          "config": cfg}, ensure_ascii=False, indent=2))
    elif args.command == "train":
        from .experiment import run_experiments
        print(run_experiments(cfg, args.models, args.seeds, args.folds, args.resume))
    elif args.command == "summarize":
        from .report import summarize
        print(json.dumps(summarize(args.run_root, args.allow_partial), ensure_ascii=False, indent=2))
    elif args.command == "plot":
        from .report import plot_trajectory
        print(plot_trajectory(args.run_root, args.seed, args.trajectory, args.models))


if __name__ == "__main__":
    main()
