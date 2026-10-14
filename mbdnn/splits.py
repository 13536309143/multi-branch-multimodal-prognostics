import numpy as np


def make_splits(rows, seed=2021, validation="sample", fraction=0.05, outer_grouping="trajectory"):
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    groups = np.array(sorted({row["trajectory"] for row in rows}))
    if len(groups) != 50:
        raise ValueError("Paper protocol requires 50 trajectories")
    rng = np.random.default_rng(seed)
    group_key = "trajectory" if outer_grouping == "trajectory" else "source_group" if outer_grouping == "source_history" else None
    if group_key is None:
        raise ValueError("Unknown outer grouping")
    if group_key == "trajectory":
        test_folds = np.split(rng.permutation(groups), 5)
    else:
        members = {}
        for row in rows:
            members.setdefault(row[group_key], set()).add(row["trajectory"])
        shuffled = rng.permutation(sorted(members)).tolist()
        shuffled.sort(key=lambda group: -len(members[group]))
        test_folds = [[] for _ in range(5)]
        for group in shuffled:
            smallest = min(range(5), key=lambda i: len(test_folds[i]))
            test_folds[smallest].extend(sorted(members[group]))
        if any(len(fold) != 10 for fold in test_folds):
            raise ValueError("Could not balance source groups into 10 trajectories per fold")
    splits = []
    for fold, test_groups in enumerate(test_folds):
        pool = [r["row_id"] for r in rows if r["trajectory"] not in test_groups]
        test = [r["row_id"] for r in rows if r["trajectory"] in test_groups]
        fold_rng = np.random.default_rng(seed + 1000 + fold)
        if validation == "sample":
            val = sorted(fold_rng.choice(pool, size=int(np.ceil(len(pool) * fraction)), replace=False).tolist())
        elif validation == "trajectory":
            pool_groups = sorted({rows[i][group_key] for i in pool})
            val_groups = fold_rng.choice(pool_groups, size=max(1, int(np.ceil(len(pool_groups) * fraction))), replace=False)
            val = [i for i in pool if rows[i][group_key] in val_groups]
        else:
            raise ValueError("Unknown validation protocol")
        train = sorted(set(pool) - set(val))
        split = {"fold": fold, "train": train, "validation": val, "test": test,
                 "test_trajectories": sorted(int(g) for g in test_groups), "validation_policy": validation,
                 "seed": seed, "validation_fraction": fraction, "outer_grouping": outer_grouping}
        validate_split(rows, split)
        splits.append(split)
    if sorted(i for split in splits for i in split["test"]) != list(range(len(rows))):
        raise ValueError("Every sample must be tested exactly once")
    return splits


def validate_split(rows, split):
    parts = [set(split[name]) for name in ("train", "validation", "test")]
    if any(len(part) != len(split[name]) for part, name in zip(parts, ("train", "validation", "test"))):
        raise ValueError("Duplicate indices")
    if any(not part for part in parts) or any(parts[i] & parts[j] for i in range(3) for j in range(i)):
        raise ValueError("Empty or overlapping split")
    if set.union(*parts) != set(range(len(rows))):
        raise ValueError("Incomplete split")
    groups = [{rows[i]["trajectory"] for i in part} for part in parts]
    if (groups[0] | groups[1]) & groups[2]:
        raise ValueError("Test trajectory leakage")
    if split["validation_policy"] == "trajectory" and groups[0] & groups[1]:
        raise ValueError("Validation trajectory leakage")
    if split.get("outer_grouping", "trajectory") == "source_history":
        source_groups = [{rows[i]["source_group"] for i in part} for part in parts]
        if (source_groups[0] | source_groups[1]) & source_groups[2]:
            raise ValueError("Duplicate source history in train/test")
        if split["validation_policy"] == "trajectory" and source_groups[0] & source_groups[1]:
            raise ValueError("Duplicate source history in train/validation")
