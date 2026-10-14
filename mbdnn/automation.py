"""Double-click workflow. --check-only never reaches the training function."""
import argparse
import contextlib
from datetime import datetime
import html
import os
from pathlib import Path
import subprocess
import sys
import traceback
import webbrowser

from .utils import PROJECT_ROOT, load_config, write_json


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


def run_tests():
    process = subprocess.Popen([sys.executable, "-u", "-m", "pytest", "-q", "checks"], cwd=PROJECT_ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace")
    try:
        for line in process.stdout:
            print(line, end="")
        if process.wait() != 0:
            raise RuntimeError("Tests failed; no training will be started")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        process.stdout.close()


def workflow(config_paths, destination, check_only=False):
    from .cli import prepare, verify
    from .embeddings import fetch_embeddings
    from .experiment import ALL_MODELS, run_experiments
    from .report import complete_report
    configs = [load_config(path) for path in config_paths]
    for cfg in configs:
        if set(cfg["models"]) != set(ALL_MODELS):
            raise ValueError("Full workflow requires all ten models in config")
        if not cfg["seeds"]:
            raise ValueError("Full workflow requires at least one seed")
    write_json(destination / "workflow.json", {"status": "checking", "config_files": [str(p) for p in config_paths],
                                              "check_only": check_only, "python": sys.executable})
    print("[1] Checking source and data contracts (pytest)", flush=True)
    run_tests()
    # Finish all preflight checks before starting either protocol.
    for cfg in configs:
        print(f"[2] Preparing data and fixed splits: {cfg['config_file']}", flush=True)
        prepare(cfg)
        if not Path(cfg["embedding_file"]).exists():
            print("[3] Fetching original GloVe vectors", flush=True)
            fetch_embeddings(cfg["data_root"], cfg["embedding_file"])
        print("[4] Verifying actual inputs and forward paths (no training yet)", flush=True)
        verify(cfg)
    if check_only:
        write_json(destination / "workflow.json", {"status": "checks_passed", "training_executed": False,
                                                  "config_files": [str(p) for p in config_paths]})
        print(f"Checks passed. NO training was run. Log: {destination / 'workflow.log'}")
        return None
    reports = []
    for cfg in configs:
        print(f"[5] Training ALL ten methods, five folds, {len(cfg['seeds'])} seeds: {cfg['outer_grouping']}", flush=True)
        run_root = run_experiments(cfg, resume=True)
        print(f"[6] Validating completed predictions and generating full report: {run_root}", flush=True)
        report = complete_report(run_root, cfg["models"], cfg["seeds"])
        reports.append({"protocol": Path(cfg["config_file"]).stem, "run_root": str(run_root), "report": str(report)})
        write_json(destination / "completed_reports.json", reports)
    links = []
    for item in reports:
        relative = Path(os.path.relpath(item["report"], destination)).as_posix()
        links.append(f'<li><a href="{html.escape(relative, quote=True)}">{html.escape(item["protocol"])}: complete experiment report</a></li>')
    index = destination / "report.html"
    index.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MB-DNN reports</title>'
                     '<style>body{font:17px system-ui;max-width:900px;margin:60px auto;line-height:1.8}a{color:#175fa4}</style>'
                     '<h1>MB-DNN 完整实验报告</h1><p>全部配置已完成训练、结果一致性检查及报告生成。</p><ul>'
                     + "".join(links) + '</ul><p>paper 保留论文式划分；clean_sources 隔离复制来源。两种协议应分别解读。</p>'
                     '<p><a href="workflow.log">完整运行日志</a></p></html>', encoding="utf-8")
    write_json(destination / "workflow.json", {"status": "completed", "training_executed": True,
                                              "reports": reports, "index": str(index)})
    print(f"COMPLETE REPORT: {index}", flush=True)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description="Full train/validate/report workflow; default runs BOTH protocols.")
    parser.add_argument("--protocol", choices=["both", "paper", "clean"], default="both")
    parser.add_argument("--check-only", action="store_true", help="Validate setup/data only; never train")
    parser.add_argument("--no-open", action="store_true", help="Do not open the completed report in a browser")
    args = parser.parse_args(argv)
    names = ["paper", "clean_sources"] if args.protocol == "both" else ["paper" if args.protocol == "paper" else "clean_sources"]
    destination = PROJECT_ROOT / "artifacts" / "automation" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True)
    print(f"Log: {destination / 'workflow.log'}", flush=True)
    print("CHECK ONLY: no training" if args.check_only else "Training enabled: all configured methods/seeds x all five folds per protocol", flush=True)
    with (destination / "workflow.log").open("w", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                report = workflow([PROJECT_ROOT / "configs" / f"{name}.json" for name in names], destination, args.check_only)
            except KeyboardInterrupt:
                write_json(destination / "failure.json", {"status": "interrupted", "resume": "Run the same .bat again; completed models are skipped"})
                print("Interrupted. Completed models are preserved; unfinished models restart on next run.")
                return 130
            except Exception:
                detail = traceback.format_exc()
                print(detail, file=sys.stderr)
                write_json(destination / "failure.json", {"status": "failed", "traceback": detail})
                print(f"FAILED. Review {destination / 'workflow.log'}", file=sys.stderr)
                return 1
    if report and not args.no_open:
        webbrowser.open(report.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
