import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path):
    path = Path(path).resolve()
    cfg = read_json(path)
    base = cfg.pop("extends", None)
    if base:
        cfg = {**load_config(path.parent / base), **cfg}
    cfg["config_file"] = str(path)
    for key in ("data_root", "artifact_root", "results_root", "embedding_file"):
        if not Path(cfg[key]).is_absolute():
            cfg[key] = str(PROJECT_ROOT / cfg[key])
    if cfg["validation"] not in ("sample", "trajectory"):
        raise ValueError("validation must be sample or trajectory")
    if cfg["outer_grouping"] not in ("trajectory", "source_history"):
        raise ValueError("Unknown outer_grouping")
    if cfg["blank_images"] not in ("preserve", "zero_template"):
        raise ValueError("Unknown blank image policy")
    if cfg["baseline_selection"] not in ("paper_fixed", "validation_grid"):
        raise ValueError("Unknown baseline selection policy")
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Need distinct, nonempty random seeds")
    for key in ("batch_size", "max_epochs", "patience", "rf_estimators"):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 < cfg["validation_fraction"] < 1 or cfg["learning_rate"] <= 0 or cfg["weight_decay"] < 0 or cfg["min_delta"] < 0:
        raise ValueError("Invalid optimization or validation settings")
    if cfg["num_workers"] < 0:
        raise ValueError("num_workers must be nonnegative")
    return cfg


def seed_everything(seed, deterministic=True):
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def environment():
    import PIL
    import sklearn
    import matplotlib
    return {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "torch": torch.__version__, "pillow": PIL.__version__,
            "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__,
            "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
