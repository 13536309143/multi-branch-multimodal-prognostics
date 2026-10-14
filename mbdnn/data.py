"""Read the released, already aligned rows. Never shift the supplied target."""
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import sha256, write_json

FEATURE_FILES = ("Numberofchemicalcleaning.txt", "Numberofmechanicalcleaning.txt",
                 "timefromlastrepair.txt", "nextinstime.txt")
FEATURE_NAMES = ("chemical_cleanings", "mechanical_cleanings_since_chemical", "time_since_repair", "time_to_next_inspection")
IMAGE_HW = (325, 194)
MAX_WORDS = 27


def tokenize(text):
    # Lowercase, strip punctuation, retain contractions as a single word.
    # This is an explicit reconstruction assumption, not an author tokenizer.
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def image_array(path):
    with Image.open(path) as image:
        if image.size != (IMAGE_HW[1], IMAGE_HW[0]):
            raise ValueError(f"Unexpected image size at {path}: {image.size}")
        return np.asarray(image.convert("L"), dtype=np.uint8)


def pixel_hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def read_rows(root):
    root = Path(root)
    folders = sorted((d for d in root.iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name))
    if [int(d.name) for d in folders] != list(range(1, 51)):
        raise ValueError("Expected exactly trajectories 1..50")
    rows = []
    for folder in folders:
        vectors = [np.loadtxt(folder / "time.txt", ndmin=1), np.loadtxt(folder / "nextinslevel.txt", ndmin=1)]
        vectors += [np.loadtxt(folder / "Numerical" / name, ndmin=1) for name in FEATURE_FILES]
        texts = (folder / "workorder.txt").read_text(encoding="utf-8-sig").splitlines()
        n = len(vectors[0])
        if any(v.ndim != 1 or len(v) != n or not np.isfinite(v).all() for v in vectors) or len(texts) != n:
            raise ValueError(f"Misaligned, nonfinite or non-vector data in {folder}")
        if np.any(np.diff(vectors[0]) <= 0):
            raise ValueError(f"Non-increasing times in {folder}")
        if np.any(vectors[1] <= 0) or any(np.any(v < 0) for v in vectors[2:]):
            raise ValueError(f"Nonpositive target or negative numerical features in {folder}")
        expected = {f"{i}.tiff" for i in range(1, n + 1)}
        actual = {p.name for p in (folder / "Image").iterdir() if p.is_file()}
        if expected != actual:
            raise ValueError(f"Missing or extra images in {folder}")
        for i in range(n):
            if not texts[i].strip() or len(tokenize(texts[i])) > MAX_WORDS:
                raise ValueError(f"Empty or overlength text in {folder}, row {i + 1}")
            rows.append({"row_id": len(rows), "trajectory": int(folder.name), "sample": i + 1,
                         "time": float(vectors[0][i]), "target": float(vectors[1][i]),
                         "numeric": [float(v[i]) for v in vectors[2:]], "text": texts[i],
                         "image": str((folder / "Image" / f"{i + 1}.tiff").relative_to(root))})
    # Conservative duplicate grouping uses input history only (never targets).
    signatures = {}
    for folder in folders:
        trajectory = int(folder.name)
        history = [{k: row[k] for k in ("time", "numeric", "text")} for row in rows if row["trajectory"] == trajectory]
        import json
        signature = hashlib.sha256(json.dumps(history, sort_keys=True).encode()).hexdigest()
        signatures.setdefault(signature, []).append(trajectory)
    source_groups = {trajectory: min(group) for group in signatures.values() for trajectory in group}
    for row in rows:
        row["source_group"] = source_groups[row["trajectory"]]
    return rows


def audit(root, destination):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rows = read_rows(root)
    template = image_array(root / "1" / "Image" / "1.tiff")
    template_hash = pixel_hash(template)
    image_counts, modes, shapes = Counter(), Counter(), Counter()
    records = []
    trajectory_pixels = {}
    for i, row in enumerate(rows):
        path = root / row["image"]
        array = image_array(path)
        row["blank_image"] = pixel_hash(array) == template_hash
        trajectory_pixels.setdefault(row["trajectory"], []).append(pixel_hash(array))
        image_counts["blank" if row["blank_image"] else "signal"] += 1
        with Image.open(path) as image:
            modes[image.mode] += 1
            shapes[str(image.size)] += 1
        records.append({"path": row["image"], "sha256": sha256(path)})
        if (i + 1) % 1500 == 0:
            print(f"Audited {i + 1}/{len(rows)} images", flush=True)
    for path in sorted(root.rglob("*.txt")):
        records.append({"path": str(path.relative_to(root)), "sha256": sha256(path)})
    records.append({"path": "Readme.docx", "sha256": sha256(root / "Readme.docx")})
    counts = Counter(row["trajectory"] for row in rows)
    target = np.array([r["target"] for r in rows])
    numeric = np.array([r["numeric"] for r in rows])
    source_members = {}
    for row in rows:
        source_members.setdefault(row["source_group"], set()).add(row["trajectory"])
    duplicates = [sorted(group) for group in source_members.values() if len(group) > 1]
    duplicate_evidence = []
    for group in duplicates:
        left, right = group
        a, b = [r for r in rows if r["trajectory"] == left], [r for r in rows if r["trajectory"] == right]
        duplicate_evidence.append({"trajectories": group, "samples": len(a),
                                   "different_image_rows": sum(x != y for x, y in zip(trajectory_pixels[left], trajectory_pixels[right])),
                                   "different_target_rows": sum(x["target"] != y["target"] for x, y in zip(a, b)),
                                   "input_time_text_numeric_identical": True})
    report = {"trajectories": len(counts), "samples": len(rows), "samples_per_trajectory": dict(counts),
              "image_counts": dict(image_counts), "image_modes": dict(modes), "image_sizes_width_height": dict(shapes),
              "blank_template_pixel_sha256": template_hash, "blank_template_has_axes": True,
              "max_tokenized_words": max(len(tokenize(r["text"])) for r in rows),
              "unique_texts": len({r["text"] for r in rows}),
              "target_min": float(target.min()), "target_max": float(target.max()),
              "feature_order": FEATURE_NAMES, "feature_min": numeric.min(axis=0).tolist(),
              "feature_max": numeric.max(axis=0).tolist(), "training_executed": False,
              "distinct_input_history_groups": len(source_members), "duplicate_history_groups": duplicates,
              "duplicate_evidence": duplicate_evidence}
    write_json(destination / "manifest.json", rows)
    write_json(destination / "data_audit.json", report)
    write_json(destination / "checksums.json", records)
    with (destination / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row_id", "trajectory", "source_group", "sample", "time", "target", *FEATURE_NAMES, "blank_image", "image", "text"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: v for k, v in row.items() if k != "numeric"}, **dict(zip(FEATURE_NAMES, row["numeric"]))})
    return rows, report


class Standardizer:
    def __init__(self, mode="standard"):
        if mode not in ("standard", "none"):
            raise ValueError("Scaling must be standard or none")
        self.mode = mode

    def fit(self, values):
        values = np.asarray(values, dtype=np.float64)
        self.mean = values.mean(axis=0) if self.mode == "standard" else np.zeros(values.shape[1:])
        self.scale = values.std(axis=0) if self.mode == "standard" else np.ones(values.shape[1:])
        self.scale = np.where(self.scale < 1e-12, 1.0, self.scale)
        return self

    def transform(self, values):
        return (np.asarray(values) - self.mean) / self.scale

    def inverse(self, values):
        return np.asarray(values) * self.scale + self.mean

    def state(self):
        return {"mode": self.mode, "mean": np.asarray(self.mean).tolist(), "scale": np.asarray(self.scale).tolist()}


def vocabulary(rows, train_ids):
    words = sorted({word for i in train_ids for word in tokenize(rows[i]["text"])})
    return {"<pad>": 0, "<unk>": 1, **{word: i + 2 for i, word in enumerate(words)}}


def embedding_matrix(vocab, asset, seed):
    if asset.get("source") != "glove.6B.100d" or asset.get("dimension") != 100:
        raise ValueError("Expected the original GloVe 6B 100d asset")
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(-0.25, 0.25, (len(vocab), 100)).astype(np.float32)
    found = []
    for word, index in vocab.items():
        if word in asset["vectors"]:
            vector = np.asarray(asset["vectors"][word], dtype=np.float32)
            if vector.shape != (100,) or not np.isfinite(vector).all():
                raise ValueError(f"Invalid GloVe vector for {word}")
            matrix[index] = vector
            found.append(word)
    matrix[0] = 0
    return matrix, {"pretrained": len(found), "lexical_words": len(vocab) - 2,
                    "oov": sorted(set(vocab) - set(found) - {"<pad>", "<unk>"})}


class MultimodalDataset(Dataset):
    def __init__(self, root, rows, indices, vocab, numeric_scaler, target_scaler, modalities="itn", blank_images="preserve"):
        self.root, self.rows, self.indices = Path(root), rows, list(indices)
        self.vocab, self.numeric_scaler, self.target_scaler = vocab, numeric_scaler, target_scaler
        self.modalities, self.blank_images = modalities, blank_images

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.rows[self.indices[index]]
        item = {"row_id": row["row_id"], "target": torch.tensor(float(self.target_scaler.transform(row["target"])), dtype=torch.float32)}
        if "i" in self.modalities:
            if self.blank_images == "zero_template" and row["blank_image"]:
                array = np.zeros(IMAGE_HW, dtype=np.float32)
            else:
                array = image_array(self.root / row["image"]).astype(np.float32) / 255.0
            item["image"] = torch.from_numpy(array[None])
        if "t" in self.modalities:
            tokens = [self.vocab.get(word, 1) for word in tokenize(row["text"])]
            if len(tokens) > MAX_WORDS:
                raise ValueError("Text truncation is prohibited")
            item["text"] = torch.tensor(tokens + [0] * (MAX_WORDS - len(tokens)), dtype=torch.long)
        if "n" in self.modalities:
            item["numeric"] = torch.tensor(self.numeric_scaler.transform(row["numeric"]), dtype=torch.float32)
        return item
