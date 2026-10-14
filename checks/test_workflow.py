"""Orchestration uses mocked training. Report fixtures are synthetic, not experiments."""
import csv

import pytest

from mbdnn import automation, cli, embeddings, experiment, report
from mbdnn.metrics import regression_metrics
from mbdnn.splits import make_splits
from mbdnn.utils import PROJECT_ROOT, load_config, read_json, sha256, write_json


@pytest.mark.parametrize("check_only", [True, False])
def test_workflow_preflights_all_configs_before_training(tmp_path, monkeypatch, check_only):
    events = []
    monkeypatch.setattr(automation, "run_tests", lambda: events.append("tests"))
    monkeypatch.setattr(cli, "prepare", lambda cfg: events.append("prepare:" + cfg["outer_grouping"]))
    monkeypatch.setattr(cli, "verify", lambda cfg: events.append("verify:" + cfg["outer_grouping"]))
    def training(cfg, resume):
        assert not check_only, "Check-only must never call training"
        assert resume
        events.append("train:" + cfg["outer_grouping"])
        return tmp_path / cfg["outer_grouping"]
    def reporting(root, models, seeds):
        assert len(models) == 10 and len(seeds) == 5
        events.append("report:" + root.name)
        root.mkdir()
        file = root / "full_report.html"
        file.write_text("Synthetic orchestration test", encoding="utf-8")
        return file
    monkeypatch.setattr(experiment, "run_experiments", training)
    monkeypatch.setattr(report, "complete_report", reporting)
    result = automation.workflow([PROJECT_ROOT / "configs/paper.json", PROJECT_ROOT / "configs/clean_sources.json"], tmp_path, check_only)
    expected = ["tests", "prepare:trajectory", "verify:trajectory", "prepare:source_history", "verify:source_history"]
    if not check_only:
        expected += ["train:trajectory", "report:trajectory", "train:source_history", "report:source_history"]
        assert result.exists()
    else:
        assert result is None
        assert read_json(tmp_path / "workflow.json")["training_executed"] is False
    assert events == expected


def test_preflight_failure_prevents_training(tmp_path, monkeypatch):
    monkeypatch.setattr(automation, "run_tests", lambda: None)
    monkeypatch.setattr(cli, "prepare", lambda cfg: None)
    def fail(cfg):
        raise ValueError("Synthetic verification failure")
    monkeypatch.setattr(cli, "verify", fail)
    monkeypatch.setattr(experiment, "run_experiments", lambda *a, **kw: pytest.fail("Training reached after preflight failed"))
    with pytest.raises(ValueError, match="Synthetic verification"):
        automation.workflow([PROJECT_ROOT / "configs/paper.json"], tmp_path)


@pytest.fixture
def synthetic_run(tmp_path):
    root, artifact = tmp_path / "run", tmp_path / "prepared"
    root.mkdir()
    rows = [{"row_id": i, "trajectory": i + 1, "sample": 1, "source_group": i + 1,
             "time": 0.0, "target": 10.0 + i, "numeric": [0, 0, 0, 50], "text": "synthetic fixture",
             "image": "unused", "blank_image": False} for i in range(50)]
    splits = make_splits(rows)
    write_json(artifact / "manifest.json", rows)
    write_json(artifact / "splits.json", splits)
    write_json(artifact / "data_audit.json", {"trajectories": 50, "distinct_input_history_groups": 50})
    write_json(artifact / "source_overlap.json", [{"fold": i, "train_test_source_overlap": [], "train_validation_source_overlap": [], "validation_test_source_overlap": []} for i in range(5)])
    cfg = load_config(PROJECT_ROOT / "configs/paper.json")
    cfg.update(artifact_root=str(artifact), seeds=[123])
    write_json(root / "run.json", {"config": cfg, "requested_seeds": [123], "fingerprint": "synthetic-test-only", "environment": {"fixture": True},
                                   "manifest_sha256": sha256(artifact / "manifest.json"), "splits_sha256": sha256(artifact / "splits.json")})
    for model in experiment.ALL_MODELS:
        for fold, split in enumerate(splits):
            folder = root / "seed_123" / f"fold_{fold}" / model
            folder.mkdir(parents=True)
            target = [rows[i]["target"] for i in split["test"]]
            validation = [rows[i]["target"] for i in split["validation"]]
            result = {"model": model, "seed": 123, "fold": fold, "training_seconds": 0.0,
                      "test_ids": split["test"], "validation_ids": split["validation"],
                      "test_predictions": [v + 1 for v in target], "validation_predictions": [v + 1 for v in validation],
                      "metrics": regression_metrics(target, [v + 1 for v in target]),
                      "validation_metrics": regression_metrics(validation, [v + 1 for v in validation])}
            if model == "ensemble":
                result["weights_image_text_numeric"] = [1 / 3] * 3
            elif model in ("svr", "rf"):
                (folder / "model.joblib").write_bytes(b"synthetic fixture, not a model")
                write_json(folder / "selection.json", [{"fixture": True}])
            else:
                (folder / "checkpoint.pt").write_bytes(b"synthetic fixture, not a model")
                write_json(folder / "history.json", [{"epoch": 1, "validation_mse_scaled": 1.0}])
            write_json(folder / "result.json", result)
            with (folder / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
                fields = ["row_id", "trajectory", "sample", "time", "target", "prediction"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for i in split["test"]:
                    writer.writerow({**{k: rows[i][k] for k in fields if k != "prediction"}, "prediction": rows[i]["target"] + 1})
    return root


def test_complete_result_validation_and_metric_tamper_rejection(synthetic_run):
    result = report.validate_completed_run(synthetic_run, experiment.ALL_MODELS, [123])
    assert len(result[3]) == 50
    assert read_json(synthetic_run / "result_validation.json")["checked_runs"] == 50
    file = synthetic_run / "seed_123/fold_0/itn/result.json"
    changed = read_json(file)
    changed["metrics"]["mae"] = 999
    write_json(file, changed)
    with pytest.raises(ValueError, match="metric mismatch"):
        report.validate_completed_run(synthetic_run, experiment.ALL_MODELS, [123])


def test_complete_report_rejects_missing_fold(synthetic_run):
    (synthetic_run / "seed_123/fold_4/rf/result.json").unlink()
    with pytest.raises(FileNotFoundError):
        report.complete_report(synthetic_run)
    assert not (synthetic_run / "full_report.html").exists()


def test_full_report_html_and_exports_with_synthetic_results(synthetic_run, monkeypatch):
    from PIL import Image
    def tiny_plot(root, seed, trajectory, models):
        destination = root / "figures"
        destination.mkdir(exist_ok=True)
        Image.new("RGB", (2, 2), "white").save(destination / f"trajectory_{trajectory}_seed_{seed}.png")
        return destination
    monkeypatch.setattr(report, "plot_trajectory", tiny_plot)
    output = report.complete_report(synthetic_run)
    document = output.read_text(encoding="utf-8")
    assert "synthetic-test-only" in document
    assert document.count("<details>") == 50
    assert "data:image/png;base64," in document
    for name in ("full_report.md", "fold_metrics.csv", "trajectory_metrics.csv", "degradation_bin_metrics.csv", "table3_comparison.csv"):
        assert (synthetic_run / name).stat().st_size > 0
