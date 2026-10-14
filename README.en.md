# Multi-Branch Multimodal Prognostics

**Predict next-inspection degradation from signal images, maintenance text, and numerical history.**

[简体中文](README.md) · [Dataset](https://huggingface.co/datasets/GeorgeWJJ/steam-generator-multimodal-prognostics) · [Models and full experiments](https://github.com/13536309143/multi-branch-multimodal-prognostics/releases/tag/v1.0.0) · [Method details](docs/methodology.md) · [Validation](docs/validation.md)

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-orange) ![Code license](https://img.shields.io/badge/Code-MIT-green) ![Completed results](https://img.shields.io/badge/Experiment_results-500%2F500-success)

An end-to-end workflow for multimodal degradation prediction: aligned data preparation, modality-specific neural branches, grouped evaluation, saved-model inference, and traceable experiment reports. The published snapshot contains **two protocols × ten methods × five folds × five training seeds = 500 result records**, comprising 450 model fits and 50 weighted combinations.

## Objectives and capabilities

- Compare image, text, numerical, and combined representations for future degradation prediction.
- Evaluate seven neural modality combinations, a weighted ensemble, SVR, and random forest on matching partitions within each protocol.
- Audit repeated histories, text reuse, missing-image templates, and source-level generalization.
- Fit scalers/vocabulary on training rows; choose checkpoints and ensemble weights using validation data.
- Provide Windows one-click execution, 27 essential checks, trained weights, per-row predictions, CSV metrics, PNG/PDF figures, and standalone HTML reports covering all 50 folders.

## Method

![Architecture](docs/figures/architecture.png)

| Branch | Input | Feature extraction | Output |
|---|---|---|---|
| Image `i` | `1×325×194` grayscale tensor | Three Conv2D/ReLU/max-pool stages; channels 4/4/8, kernel 4, pool 4; flatten 192; sigmoid dense | 4 features |
| Text `t` | ≤27 tokens; GloVe 6B / 100d | Frozen embeddings; Conv1D windows 2/3/4, 32 filters each; global max pooling; sigmoid dense | 4 features |
| Numerical `n` | Four train-standardized variables | Dense `4→4`, sigmoid | 4 features |
| Fusion | Selected branch features | Concatenation and linear scalar regression | Next-inspection degradation |

Images use explicit asymmetric SAME padding and ceiling pooling, division by 255, and no resizing/axis cropping. Text is lowercased with contractions retained; overlength records raise an error. Vocabulary uses training data only. The linear fusion head itself has no nonlinear cross-modal interaction.

Optimization: standardized-target MSE; Adam, lr 0.001; batch 64; maximum 500 epochs; validation-MSE early stopping, patience 40 and min_delta 1e-5; restore the best checkpoint. Metrics use inverse-transformed targets. See [implementation details](docs/methodology.md).

## Dataset preparation

The Hugging Face repository is an attributed mirror of the **synthetic** steam-generator benchmark by Yang, Baraldi, and Zio. The project maintains packaging and audit metadata; the uploader is not the original dataset author.

![Dataset overview](docs/figures/dataset_overview.png)

| Property | Value |
|---|---:|
| Aligned observations / original folders | 7,626 / 50 |
| Distinct time/text/numerical histories | 30 |
| Signal / no-signal images | 6,092 / 1,534 |
| Distinct complete texts / maximum tokens | 794 / 27 |
| Target range | 12.75–104.83524 |

After installing the environment, download from the project root:

```powershell
.\.venv\Scripts\python.exe scripts\download_dataset.py
.\.venv\Scripts\python.exe scripts\download_dataset.py --verify-only
```

The downloader pins the Hub commit in `dataset.json`, verifies the archive and all **7,977 raw files**, and preserves existing verified data. Mismatched existing data are not overwritten.

```text
Multimodal data for prognostics/
├── Readme.docx
├── 1/
│   ├── time.txt / nextinslevel.txt / workorder.txt
│   ├── Image/1.tiff, 2.tiff, ...
│   └── Numerical/
│       ├── Numberofchemicalcleaning.txt
│       ├── Numberofmechanicalcleaning.txt
│       ├── timefromlastrepair.txt
│       └── nextinstime.txt
└── ... 50/
```

The X-th sequence row aligns with `Image/X.tiff`. `nextinslevel.txt` already targets the next inspection: **do not shift labels again**. `nextinstime.txt` is a known planned interval, not RUL. Previous text repeats when no new record exists; no-signal templates retain axes. Only image, text, and four numerical variables enter the model; identifiers, absolute time, source labels, and targets are not extra features.

Optional exploration, with the additional `datasets` package:

```python
from datasets import load_dataset
data = load_dataset("GeorgeWJJ/steam-generator-multimodal-prognostics", split="train")
```

This `train` partition stores the entire dataset. Apply the provided grouped experimental indices before fitting.

## Environment setup and verification

Recorded environment: Python 3.11.17, PyTorch 2.7.1+cu126, NumPy 1.26.4, Pillow 11.1.0, scikit-learn 1.5.2, Matplotlib 3.9.4, RTX 4070 Laptop GPU. Python 3.11 is recommended; the package supports Python 3.10+.

```powershell
py -3.11 -m venv .venv
# Choose ONE PyTorch wheel: CUDA 12.6 or CPU.
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
# .\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-verified.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe scripts\verify_environment.py
```

Linux/macOS: create/activate a Python virtual environment, install the platform-appropriate PyTorch wheel and the same core pins, and substitute `python` for the Windows interpreter below. CUDA requires a compatible NVIDIA driver. `device=auto` falls back to CPU. `requirements-recovered.txt` preserves the historical full freeze; its CUDA wheel is not a universal installation recipe.

The original GloVe 6B / 100d dataset-word subset is included. `python -m mbdnn fetch-embeddings` prepares a missing asset; add `--local-file C:\path\glove.6B.100d.txt` to use a local original vector file when the asset is absent.

## Data, architecture, and workflow validation

```powershell
.\.venv\Scripts\python.exe scripts\download_dataset.py --verify-only
.\.venv\Scripts\python.exe -m mbdnn --config configs/paper.json prepare
.\.venv\Scripts\python.exe -m mbdnn --config configs/paper.json verify
.\.venv\Scripts\python.exe -m mbdnn --config configs/clean_sources.json prepare
.\.venv\Scripts\python.exe -m mbdnn --config configs/clean_sources.json verify
.\.venv\Scripts\python.exe -m pytest -q checks
```

Checks cover hashes, row alignment, image dimensions, targets, fixed folds, repeated sources, train-only preprocessing, seven forward paths, metric definitions, and orchestration. `verify` performs inference only, not weight optimization or predictive evaluation. `checks/` retains the **27 essential checks** used by the one-click preflight. Temporary recovery scripts and caches are excluded from the public repository.

## Evaluation protocols

| Protocol | Config | Outer test | Inner validation |
|---|---|---|---|
| Directory-level | `configs/paper.json` | Five shuffled folds; ten folders per fold | Random ~5% of remaining rows |
| Source-isolated | `configs/clean_sources.json` | Five folds; ten folders / six sources per fold | Two of remaining 24 sources; actual row fraction5.33–9.82% |
| Directory validation control | `configs/group_validation.json` | Same directory outer folds | Whole-folder validation; prepared, **not in completed results** |

Split seed 2021; training seeds2021–2025 share fixed outer folds within each protocol. They measure training randomness, not five independently randomized splits.

Folders1–20 pair with 31–50 in time/text/numerical histories and targets. Seventeen pairs also match in all images; 8/38,9/39,16/46 differ in one image each. Directory grouping can expose repeated sources across partitions. Source grouping uses inputs without targets and is never a predictor input. Raw duplicates remain and still affect sample weights.

Both outer grouping and inner validation change between protocols; performance differences do not isolate the causal effect of source exposure.

## Training and one-click reports

```powershell
# Preflight only; no training.
.\train_and_report.bat --check-only --no-open
# Both protocols, all methods, five folds, five seeds.
.\train_and_report.bat
# Source-isolated only.
.\train_and_report.bat --protocol clean
```

The launcher prefers project `.venv` or uses the explicitly configured `MBDNN_PYTHON`. It checks contracts, prepares data/vectors, verifies networks, trains, recomputes metrics, and creates reports. Logs/report index: `artifacts/automation/<timestamp>/`.

```powershell
# Limited execution scope; not a full benchmark.
.\.venv\Scripts\python.exe -m mbdnn --config configs/clean_sources.json train --models itn in svr rf --seeds 2021 --folds 0
# Full benchmark; skip complete models with the same fingerprint.
.\.venv\Scripts\python.exe -m mbdnn --config configs/clean_sources.json train --resume
```

`i/t/n` denote image/text/numerical. Single/double branches are independently trained. Ensemble weights are normalized inverse validation MAE. Resume skips completed models; unfinished models restart from scratch, rather than continuing arbitrary-epoch optimizer state. Code, configuration, data, or project-location changes create a new fingerprint; historical snapshots remain separate.

## Test saved models

[Release assets](https://github.com/13536309143/multi-branch-multimodal-prognostics/releases/tag/v1.0.0) contain every model, prediction, and report. Extract an experiment ZIP into the project root. Original training records are retained; the evaluation helper resolves the current data location at runtime.

```powershell
# Recompute saved test metrics, without training.
.\.venv\Scripts\python.exe scripts\evaluate_snapshot.py results\clean_sources\aaaa63a00b5260a1
# Replay a trained best checkpoint on the original test rows.
.\.venv\Scripts\python.exe scripts\evaluate_snapshot.py results\clean_sources\aaaa63a00b5260a1 --replay --model itn --seed 2021 --fold 0
# Numerical baseline replay.
.\.venv\Scripts\python.exe scripts\evaluate_snapshot.py results\clean_sources\aaaa63a00b5260a1 --replay --model svr --seed 2021 --fold 0
```

For new runs use the fingerprint printed by the program:

```powershell
.\.venv\Scripts\python.exe -m mbdnn summarize results/clean_sources/<fingerprint>
.\.venv\Scripts\python.exe -m mbdnn plot results/clean_sources/<fingerprint> --seed 2021 --trajectory 19 --models itn in svr
```

## Benchmark results

MAE/MSE/MRE are lower-is-better. MRE is `100 × mean(abs(prediction−target)/target)`, not accuracy. Average five folds equally per seed, then report the five-seed mean/sample SD. SD measures training randomness only; deterministic SVR's zero SD does not establish zero generalization uncertainty.

| Method | Directory MAE ± SD | MSE | MRE% | Source-isolated MAE ± SD | MSE | MRE% |
|---|---|---|---|---|---|---|
| itn: Image + text + numerical | 4.8567 ± 0.2137 | 78.80 | 9.18 | 12.5125 ± 0.9725 | 268.26 | 25.99 |
| in: Image + numerical | 6.4311 ± 0.4724 | 72.48 | 11.96 | 7.0984 ± 0.2347 | 87.44 | 12.90 |
| tn: Text + numerical | 5.0892 ± 0.1895 | 91.50 | 9.50 | 12.0295 ± 0.6543 | 258.04 | 23.03 |
| it: Image + text | 7.0605 ± 0.0937 | 172.55 | 14.56 | 14.9385 ± 0.2243 | 362.06 | 32.03 |
| i: Image only | 15.0934 ± 0.1307 | 371.75 | 32.34 | 15.4296 ± 0.0650 | 375.45 | 32.84 |
| t: Text only | 7.0622 ± 0.1344 | 176.81 | 14.45 | 16.7257 ± 0.2829 | 457.35 | 34.80 |
| n: Numerical DNN | 7.4817 ± 0.1247 | 100.22 | 13.17 | 8.1978 ± 0.1204 | 113.25 | 14.91 |
| ensemble: Weighted single-branch ensemble | 6.6663 ± 0.1043 | 116.23 | 13.58 | 9.5150 ± 0.0252 | 138.83 | 19.45 |
| svr: Numerical SVR | 6.5200 ± 0.0000 | 84.66 | 11.39 | 6.8405 ± 0.0000 | 90.85 | 11.92 |
| rf: Numerical RF | 5.7553 ± 0.0070 | 71.23 | 10.42 | 7.4899 ± 0.0089 | 111.32 | 13.40 |

![Directory benchmark](docs/figures/benchmark_paper.png)

![Source-isolated benchmark](docs/figures/benchmark_clean_sources.png)

Directory-level: three-branch achieves lowest MAE/MRE, RF lowest MSE. Source-isolated: SVR achieves lowest MAE/MRE, image+numerical lowest MSE. Image+numerical improves MAE over numerical DNN by 13.41%; adding text to it increases MAE by 76.27% in the current settings. These findings do not imply a universal benefit or harm of a modality.

## Diagnostic figures

**Protocol sensitivity.** In the directory schedule,63.76% of test rows have a repeated source in training. Three-branch MAE is 1.78 on those rows versus 10.29 on unseen sources. Different subgroup composition makes this descriptive, not causal.

![Protocol sensitivity](docs/figures/protocol_sensitivity.png)

**Training curves.** Some source-isolated text-containing models reduce training loss while validation loss increases. One replayed three-branch checkpoint has training MAE 1.58 and test MAE 12.58. Training loss averages batch updates; validation evaluates end-of-epoch weights. Selected epochs are marked.

![Training curves](docs/figures/training_curves.png)

**Degradation ranges.** Three-branch predictions overestimate low degradation and underestimate high degradation. True-target bins are retrospective diagnostics, unavailable for selecting models at prediction time.

![Degradation ranges](docs/figures/degradation_ranges.png)

**Source errors.** Thirty histories are shown once each, aggregating all within-source rows and five seeds; duplicated folders are not independent cases.

![Source errors](docs/figures/source_errors.png)

**Image availability.** No-signal templates account for 20.12% of rows. Timing, text, and target distributions also differ; subgroup results do not prove zero-template substitution improves performance.

![Image availability](docs/figures/image_availability.png)

**Calibration and error distributions.** Scatter plots use fixed seed 2021. CDFs combine all five seeds as correlated repeated evaluations.

![Prediction scatter](docs/figures/prediction_scatter.png)

![Error distributions](docs/figures/error_distributions.png)

**Held-out trajectories.** Difficult sources 19/49 and16/46 are shown using the first fixed seed, not a best test seed. Full reports cover all 50 folders.

![Trajectory diagnostics](docs/figures/trajectory_diagnostics.png)

Each figure has a PDF counterpart. Fold/trajectory/range/source metrics are in `experiments/` and release reports. Training-seed SD, fold variation, and source-macro errors have distinct meanings;7,626 correlated observations are not 7,626 independent machines.

## Repository and complete artifacts

```text
mbdnn/                       Data, branches, training, metrics, CLI, reports
configs/                     Two protocols and validation control
checks/                      Essential integrity/workflow checks
scripts/                     Download, environment validation, inference, analysis
assets/                      Original GloVe subset and provenance
experiments/                 Metrics, splits, audits, checksums, analysis evidence
docs/figures/                Twelve scientific figure groups, PNG / PDF
docs/                        Method, validation, results, publication documentation
dataset.json                 Pinned Hub revision and ZIP hash
train_and_report.bat          One-click preflight/train/test/report
README.md / README.en.md      Chinese / English guides
```

Raw data, environments, tools, caches, logs, and working outputs are excluded from Git. Complete weights and predictions remain in release assets, alongside historical training sources and fingerprints. Naming/documentation/portability updates in v1.0.0 do not retroactively alter benchmark metrics.

## Method source and brief literature comparison

The branch design follows Yang, Baraldi, Zio, *A multi-branch deep neural network model for failure prognostics based on multimodal data*, Journal of Manufacturing Systems59 (2021),42–50,[DOI](https://doi.org/10.1016/j.jmsy.2021.01.007). Reported three-branch MAE is 3.20; this project's directory MAE is 4.8567. Original folds/all optimizer settings were unavailable and source isolation changes evaluation conditions; numerical identity is not claimed.

## License and citation

Code:[MIT](LICENSE). Dataset:original attribution and unspecified upstream terms, documented in the [dataset card](https://huggingface.co/datasets/GeorgeWJJ/steam-generator-multimodal-prognostics). GloVe vectors:upstream Public Domain Dedication and License, separate from code licensing; see [third-party notices](THIRD_PARTY_NOTICES.md).

Cite the original method/dataset paper for benchmark use; identify this repository and release version for implementation, audits, or results. [CITATION.cff](CITATION.cff) provides project citation metadata.
