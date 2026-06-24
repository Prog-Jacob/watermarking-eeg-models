# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research code for the paper "Protecting Intellectual Property of EEG-based Neural Networks with Watermarking" (arXiv:2502.05931). It embeds a cryptographic "wonder filter" watermark into EEG emotion-classification models during training and evaluates the watermark's robustness against removal attacks (fine-tuning, transfer learning, pruning, quantization). All experiments run on the DEAP dataset across three torcheeg architectures: CCNN, EEGNet, and TSCeption.

## Commands

Code lives in the `eegwm/` package, driven through one CLI entry point. Install editable, then run as a module from the repo root.

```bash
pip install -e .                          # or pip install -e ".[dev]" for tooling
python -m eegwm [args]                    # run any experiment
python -m eegwm -h                        # full argument reference
ruff check . && ruff format --check .     # lint + format gate (matches CI)
pytest                                    # tests; watermark tests need torch/torcheeg, else skip
```

Common invocations (see README.md for more):

```bash
python -m eegwm --experiment from_scratch --architecture CCNN --batch 64 --epochs 30 --lrate 0.001 --training_mode full
python -m eegwm --experiment no_watermark --architecture CCNN --evaluate eeg correct_watermark --training_mode skip
python -m eegwm --experiment fine_tuning --architecture TSCeption --fine_tuning_mode ftal --base_models_dir ./results/TSCeption/from_scratch/models --batch 64 --epochs 30 --lrate 0.0005 --training_mode quick
python -m eegwm --experiment show_stats --architecture CCNN --verbose error
```

`--training_mode quick` runs a single fold (fast iteration); `full` runs all `--folds` (default 5); `skip` only evaluates. `show_stats*`, `quantization`, `pruning`, and `feature_attribution` force `training_mode=skip` automatically.

## Dataset setup

The DEAP preprocessed Python data is not in the repo. Place it at `./data/data_preprocessed_python` (override with `--data_path`). On first use, torcheeg builds a cached `.io` dataset under `./results/{architecture}/`; subsequent runs reuse it.

## Architecture / how the pieces fit

Code lives in the `eegwm/` package. Entry flow: `cli.main()` builds a typed `Config`, sets up logging and the seed, builds the dataset and k-fold splitter, then dispatches to stats or `training/runner.run()`. Importing the package has no side effects.

- **`config.py`** — defines and validates ALL CLI arguments (`get_config`), and exposes a typed `Config` dataclass via `load_config` with derived `num_classes`, `batch_size`, `working_dir`. `validate_arguments` enforces cross-argument rules (which experiments require `--base_models_dir`, which force `skip`, which need sub-modes). Add new arguments here.

- **`cli.py`** — the entry point (`main`, `python -m eegwm`). Orchestrates config, logging, seed, dataset, and dispatch.

- **`architectures/`** — THE single place per-architecture knowledge lives. Each of `ccnn.py`/`eegnet.py`/`tsception.py` exposes one `Architecture` (model builder, dataset config, channel list, watermark shape + reshape, back-transform, plot points, transfer-learning heads), registered in `__init__.py` via `get_architecture`. Adding an architecture = one module + one registry line; everything else delegates here instead of branching on the architecture name.

- **`watermark/`** — the cryptographic core. `encryption.py` loads (or lazily generates, with a loud warning) the RSA keypair in `public.pem`/`private.pem` (gitignored — never commit these). `triggerset.py` signs a `Verifier` identity, hashes it (SHA-224/256/384/512) into a deterministic "wonder filter" + label (`transform`/`get_watermark`), and embeds it: `apply_true_embedding` stamps the filter (out-of-bound ±`OUT_OF_BOUND`) and relabels to the watermark label, `apply_null_embedding` stamps the same positions but keeps the true label. `TriggerSet` mixes embedded + optionally clean samples.

- **`data/dataset.py`** — `BetterDEAPDataset` (a `DEAPDataset` subclass that hashes its constructor config for a stable cache `io_path`) and `get_dataset` (delegates per-arch transforms to the registry). `data/stats.py` holds the `show_stats` tables/plots, kept out of `dataset.py` to avoid viz/watermark coupling.

- **`models.py`** — `get_model` (delegates to the registry), `load_model` (strips the `model.` checkpoint prefix), `get_ckpt_file`.

- **`training/runner.py`** — the k-fold loop and per-experiment dispatch (load base model, apply surgery, build trigger set, train, evaluate, write JSON). `callbacks.py` has `MultiplyLRScheduler`. **`evaluation/evaluate.py`** runs the four eval dimensions; **`evaluation/results.py`** aggregates `test_accuracy` and renders rich trees / plotille graphs.

- **`experiments/`** — model surgery: `fine_tuning.py` (FTLL/FTAL/RTLL/RTAL, architecture-agnostic), `pruning.py` (random/ascending/descending; descending uses a subclass, no global monkey-patch), `quantization.py` (dynamic int8), `feature_attribution.py` (SHAP). Transfer-learning heads live in the architecture modules.

- **`constants.py`** — `RESULTS_DIR`, `OUT_OF_BOUND`, trigger-set/verification seeds, watermark identity strings. **`viz/plot.py`** — topomaps and emotion-connectivity chord diagrams (MNE montage built lazily).

## Conventions

- Output convention: `{RESULTS_DIR}/{architecture}/{experiment}/{base-model-lineage}/{sub-mode}/` contains `models/fold-{i}/*.ckpt` plus a results JSON. `results/`, `data/`, `lightning_logs/`, and `*.pem` are gitignored.
- Class count is `2 ** len(labels)`, threaded via `Config.num_classes` into the model, the `ClassifierTrainer`, and the watermark label. The default 4 labels (valence/arousal/dominance/liking) give 16 classes; a `--labels` subset produces a matching smaller class count everywhere.
- Watermark trigger-set seeds are fixed in `constants.py` (`VERIFICATION_SEED=42`, `TRIGGERSET_SEED=2036`) for reproducibility — keep them fixed when comparing runs.
- Behavior-critical code (dataset kwargs feeding the cache hash, watermark filter shapes, transfer-learning layer indices) was preserved verbatim through the restructure; validate any change there with a real run.
- Device defaults to `cuda`; pass `--device cpu` to run without a GPU.
