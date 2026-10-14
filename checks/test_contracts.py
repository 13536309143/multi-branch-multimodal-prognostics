import numpy as np
import pytest
import torch

from mbdnn.data import IMAGE_HW, MultimodalDataset, Standardizer, embedding_matrix, read_rows, tokenize, vocabulary
from mbdnn.metrics import ensemble_weights, regression_metrics
from mbdnn.models import ImageBranch, MultiBranchDNN, SameConv, SamePool
from mbdnn.splits import make_splits, validate_split
from mbdnn.utils import PROJECT_ROOT, read_json


@pytest.fixture(scope="module")
def rows():
    return read_rows(PROJECT_ROOT / "Multimodal data for prognostics")


def test_released_alignment_and_target_semantics(rows):
    assert len(rows) == 7626
    assert len({r["trajectory"] for r in rows}) == 50
    assert rows[0]["image"].replace("\\", "/") == "1/Image/1.tiff"
    assert rows[0]["target"] == 12.75
    assert rows[0]["numeric"][-1] == 50
    assert max(len(tokenize(r["text"])) for r in rows) <= 27


@pytest.mark.parametrize("policy", ["sample", "trajectory"])
def test_grouped_outer_cv_is_exhaustive_deterministic(rows, policy):
    folds = make_splits(rows, validation=policy)
    assert folds == make_splits(rows, validation=policy)
    assert sorted(i for s in folds for i in s["test"]) == list(range(len(rows)))
    assert all(len(s["test_trajectories"]) == 10 for s in folds)
    assert all(s["train"] == sorted(s["train"]) for s in folds)


def test_split_validator_rejects_leakage(rows):
    split = make_splits(rows)[0]
    split["train"].append(split["test"][0])
    with pytest.raises(ValueError, match="overlapping"):
        validate_split(rows, split)


def test_preprocessing_fits_train_only():
    scaler = Standardizer().fit([[0, 2], [2, 2]])
    np.testing.assert_array_equal(scaler.mean, [1, 2])
    np.testing.assert_array_equal(scaler.scale, [1, 1])
    assert scaler.transform([[1000, 2]])[0, 0] == 999
    np.testing.assert_allclose(scaler.inverse(scaler.transform([[1, 5]])), [[1, 5]])
    rows = [{"text": "train words"}, {"text": "heldout secret"}]
    vocab = vocabulary(rows, [0])
    assert "secret" not in vocab


def test_pretrained_lookup_zero_pad_and_oov_reproducibility():
    vocab = {"<pad>": 0, "<unk>": 1, "sg": 2, "novel": 3}
    asset = {"source": "glove.6B.100d", "dimension": 100, "vectors": {"sg": [0.5] * 100}}
    a, coverage = embedding_matrix(vocab, asset, 2021)
    b, _ = embedding_matrix(vocab, asset, 2021)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[0], 0)
    np.testing.assert_array_equal(a[2], 0.5)
    assert coverage["oov"] == ["novel"]


def test_even_same_padding_and_ceiling_pooling():
    x = torch.ones(1, 1, 5, 7)
    conv = SameConv(1, 2)
    assert conv(x).shape == (1, 2, 5, 7)
    pool = SamePool()
    assert pool(x).shape == (1, 1, 2, 2)
    torch.testing.assert_close(pool(x), torch.ones(1, 1, 2, 2))
    with torch.no_grad():
        branch = ImageBranch()
        assert branch.features(torch.zeros(1, 1, *IMAGE_HW)).shape == (1, 8, 6, 4)


@pytest.mark.parametrize("modalities", ["itn", "in", "tn", "it", "i", "t", "n"])
def test_all_ablation_forward_paths_and_padding(modalities):
    embeddings = np.zeros((5, 100), dtype=np.float32)
    model = MultiBranchDNN(modalities, embeddings).eval()
    keys = {"i": ("image", torch.zeros(2, 1, *IMAGE_HW)), "t": ("text", torch.zeros(2, 27, dtype=torch.long)), "n": ("numeric", torch.zeros(2, 4))}
    batch = {keys[k][0]: keys[k][1] for k in modalities}
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (2,) and torch.isfinite(out).all()
    if "t" in modalities:
        assert not model.branches["t"].embedding.weight.requires_grad
        torch.testing.assert_close(model.branches["t"].embedding.weight[0], torch.zeros(100))


def test_numeric_only_does_not_open_images(tmp_path):
    rows = [{"row_id": 0, "numeric": [1, 2, 3, 4], "target": 50, "image": "absent.tiff", "text": "test"}]
    n = Standardizer("none").fit([[1, 2, 3, 4]])
    y = Standardizer("none").fit([50])
    item = MultimodalDataset(tmp_path, rows, [0], {}, n, y, "n")[0]
    assert set(item) == {"row_id", "target", "numeric"}


def test_mre_uses_true_target_and_percent():
    result = regression_metrics([10, 20], [12, 16])
    assert result["mae"] == 3
    assert result["mse"] == 10
    assert result["mre_percent"] == pytest.approx(20)
    with pytest.raises(ValueError):
        regression_metrics([0], [1])


def test_ensemble_lower_validation_error_gets_more_weight():
    weights = ensemble_weights([2, 4, 8])
    np.testing.assert_allclose(weights, [4 / 7, 2 / 7, 1 / 7])
    np.testing.assert_allclose(ensemble_weights([0, 2, 8]), [1, 0, 0])


def test_released_blank_template_policy_preserves_axes(rows):
    rows[0]["blank_image"] = True
    numeric = Standardizer("none").fit([rows[0]["numeric"]])
    target = Standardizer("none").fit([rows[0]["target"]])
    root = PROJECT_ROOT / "Multimodal data for prognostics"
    preserved = MultimodalDataset(root, rows, [0], {}, numeric, target, "i", "preserve")[0]["image"]
    zeroed = MultimodalDataset(root, rows, [0], {}, numeric, target, "i", "zero_template")[0]["image"]
    assert preserved.shape == (1, 325, 194)
    assert int((preserved == 0).sum()) == 1901
    assert preserved.max() == 1
    assert torch.count_nonzero(zeroed) == 0


def test_artifact_config_and_actual_pretrained_asset():
    cfg = read_json(PROJECT_ROOT / "configs/paper.json")
    assert cfg["svr_c"] == 100 and cfg["rf_estimators"] == 100
    assert cfg["validation_fraction"] == 0.05
    path = PROJECT_ROOT / cfg["embedding_file"]
    if path.exists():
        asset = read_json(path)
        assert asset["source"] == "glove.6B.100d" and asset["dimension"] == 100
        assert len(asset["vectors"]["equipment"]) == 100
        assert asset.get("member_crc32", 0) > 0 or asset.get("local_file_sha256")


def test_no_results_cannot_be_reported(tmp_path):
    from mbdnn.report import summarize
    from mbdnn.utils import write_json
    write_json(tmp_path / "run.json", {"requested_seeds": [2021]})
    with pytest.raises(ValueError, match="No experiments"):
        summarize(tmp_path)


def test_released_duplicate_histories_and_clean_source_folds(rows):
    assert len({row["source_group"] for row in rows}) == 30
    for trajectory in range(1, 21):
        assert {r["source_group"] for r in rows if r["trajectory"] in (trajectory, trajectory + 30)} == {trajectory}
    splits = make_splits(rows, validation="trajectory", outer_grouping="source_history")
    assert all(len(s["test_trajectories"]) == 10 for s in splits)
    for split in splits:
        sources = [{rows[i]["source_group"] for i in split[name]} for name in ("train", "validation", "test")]
        assert not sources[0] & sources[1]
        assert not sources[0] & sources[2]
        assert not sources[1] & sources[2]
