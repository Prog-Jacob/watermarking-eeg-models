# Protecting Intellectual Property of EEG-based Neural Networks with Watermarking

## Overview

This repository embeds a cryptographic watermark into EEG-based neural networks during training to protect their intellectual property, with minimal accuracy loss and robustness against removal attacks. It supports the CCNN, EEGNet, and TSCeption architectures and evaluates the watermark under adversarial scenarios such as fine-tuning, transfer learning, and pruning.
<img src="https://raw.githubusercontent.com/Prog-Jacob/watermarking-eeg-models/b0c13a214abb86c4592bd4b928051db6f3b7db9f/Training.svg" alt="Embedding a wonder filter during training an EEG-based neural network" style="width: 100%">

## Key Features

- **Watermarking Method**: Embeds cryptographic watermarks in EEG models during training with minimal impact on accuracy (≤5% drop).
- **Experiments**: Supports multiple experiment configurations including model training, fine-tuning, pruning, and more.
- **Adversarial Evaluation**: Evaluates watermark robustness against various adversarial scenarios like fine-tuning, transfer learning, and neuron pruning.
- **Model Options**: Supports architectures such as CCNN, EEGNet, and TSCeption.
- **Flexible Configurations**: Allows for evaluation of watermarks and model performance on various dimensions (e.g., correct watermark detection, wrong watermark, new watermark, EEG task performance).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Prog-Jacob/watermarking-eeg-models.git
   ```
2. Install the package and its dependencies (editable install):
   ```bash
   pip install -e .
   ```
   For development tooling (ruff, mypy, pytest, pre-commit):
   ```bash
   pip install -e ".[dev]"
   ```

## Running Experiments

The experiment configurations are managed through a command-line interface. You can specify your configuration options using arguments when invoking the `eegwm` package (run from the repository root, or use the installed `eegwm` console script).

### Command Syntax:

```bash
python -m eegwm [args]
```

### Available Arguments:

```
usage: python -m eegwm [-h] --experiment {show_stats,show_stats_plots,no_watermark,from_scratch,pretrain,new_watermark,pruning,fine_tuning,quantization,transfer_learning,feature_attribution}
                [--evaluate DIMENSION [DIMENSION ...]] --architecture {CCNN,EEGNet,TSCeption} [--labels LABEL [LABEL ...]] [--watermark_layout {block,scatter}] [--training_mode {skip,quick,full}] [--batch BATCH]
                [--epochs EPOCHS] [--lrate LRATE] [--update_lr_by x] [--update_lr_every n] [--update_lr_until ε] [--folds k] [--data_path PATH] [--base_models_dir DIR]
                [--pruning_method {random,ascending,descending}] [--pruning_mode {linear,exponential}] [--pruning_delta δ] [--fine_tuning_mode {ftll,ftal,rtll,rtal}]
                [--transfer_learning_mode {added,dense,all}] [--seed SEED] [--verbose {info,debug,warning,error,critical}] [--device {cpu,cuda}]

Configure and run experiments for watermarking EEG-based neural networks.

options:
  -h, --help            show this help message and exit

Experiment Configuration:
  --experiment {show_stats,show_stats_plots,no_watermark,from_scratch,pretrain,new_watermark,pruning,fine_tuning,quantization,transfer_learning,feature_attribution}
                        Choose one experiment from the above experiments.
  --evaluate DIMENSION [DIMENSION ...]
                        Choose one or more dimension to evaluate from {eeg,correct_watermark,wrong_watermark,new_watermark}.
  --architecture {CCNN,EEGNet,TSCeption}
  --labels LABEL [LABEL ...]
                        Choose one or more dataset label from {valence,arousal,dominance,liking}.
  --watermark_layout {block,scatter}
                        Where the wonder-filter bits land: 'block' (contiguous corner block, the paper baseline) or 'scatter' (same bits spread across the grid). Must match between embedding and verification runs.

Training Parameters:
  --training_mode {skip,quick,full}
                        Skip training, quick training of only 1 fold, or full training of all folds.
  --batch BATCH         Batch size for training.
  --epochs EPOCHS       Number of training epochs.
  --lrate LRATE         Learning rate for training.
  --update_lr_by x      Multiply learning rate by x every n epochs. Default x: 1.0
  --update_lr_every n   Multiply learning rate by x every n epochs. Default n: 10
  --update_lr_until ε   Update learning until it's out of [ε, 1.0]. Default ε: 1e-5
  --folds k             Number of k-fold cross-validation splits. Default k: 5.

Path Configuration:
  --data_path PATH      Path to processed data directory. Default: './data/data_preprocessed_python'.
  --base_models_dir DIR
                        Directory containing base models for experiments.

Experiment-Specific Parameters:
  --pruning_method {random,ascending,descending}
                        Random, ascending (nullify least-valued nodes), descending (nullify most-valued nodes).
  --pruning_mode {linear,exponential}
                        Linear increments pruning percentage by δ till it reaches 100, whereas exponential multiplies by it.
  --pruning_delta δ     Increment/multiply pruning percent by δ. Recommended: 5 for linear, 1.25 for exponential.
  --fine_tuning_mode {ftll,ftal,rtll,rtal}
                        FTLL (fine-tune last layer), FTAL (fine-tune all layers), RTLL (retrain last layer), and RTAL (retrain all layers).
  --transfer_learning_mode {added,dense,all}
                        Add two dense layers then perform fine tuning. Added (fine-tune added layers), dense (fine-tune dense layers), and all (fine-tune all layers).

Other Parameters:
  --seed SEED           Seed for reproducibility.
  --verbose {info,debug,warning,error,critical}
                        How much information to log. Default is 'info'.
  --device {cpu,cuda}   Device to run the experiment on. Default is 'cuda'.
```

### Example Usage:

To run a quick takeover (embed attacker's watermark) experiment with a trained TSCeption model from scratch, you could use:

```bash
python -m eegwm --experiment new_watermark --architecture TSCeption --base_models_dir ./results/TSCeption/from_scratch/models --batch 64 --lrate 0.0005 --epochs 30 --training_mode quick
```

### Skip Training and Evaluate:

If you wish to skip the training process and evaluate a previously trained model for its watermark retention, you can use:

```bash
python -m eegwm --experiment no_watermark --architecture CCNN --evaluate eeg correct_watermark --training_mode skip
```

## Experiment Results

After running experiments, the results will be output to the console and saved to file system based on the specified evaluation dimensions. You can check the evaluation of the watermarks and performance on the EEG tasks. Another way to show all results for a certain architecture is to use the `show_stats[_plots]` and `feature_attribution` experiments. For example:

```bash
python -m eegwm --experiment show_stats --architecture CCNN --verbose error --seed 42
python -m eegwm --experiment show_stats_plots --architecture EEGNet --verbose error --seed 42
python -m eegwm --experiment feature_attribution --architecture CCNN --base_models_dir ./results/CCNN/no_watermark/models --device cuda
```

The following are some samples of the these commands outputs:

<img src="https://github.com/Prog-Jacob/watermarking-eeg-models/releases/download/results/show_stats_output.png" alt="Left: Example output of the show_stats experiment showing dataset and results summaries. Top Right: The effect of true/null watermark embedding on the input EEG data for the EEGNet model. Middle Right: The importance analysis of the EEG electrodes for the CCNN model. Bottom Right: Summary of the co-existence among emotion pairs in the DEAP dataset." width="100%">

## Development

The code lives in the `eegwm/` package. Common tasks:

```bash
pip install -e ".[dev]"   # install with dev tooling
ruff check .              # lint
ruff format .             # format
pytest                    # run tests (watermark tests require torch/torcheeg)
pre-commit install        # enable lint/format on commit
```

Package layout: `architectures/` holds one module per model (the single place to add a new architecture); `watermark/` is the cryptographic core; `data/`, `training/`, `evaluation/`, `experiments/`, and `viz/` separate concerns; `cli.py` is the entry point and `config.py` defines the typed `Config`.

## Citation

If you use this code in your work, please cite the following paper:

```bibtex
@misc{abdelaziz2025protectingintellectualpropertyeegbased,
      title={Protecting Intellectual Property of EEG-based Neural Networks with Watermarking},
      author={Ahmed Abdelaziz and Ahmed Fathi and Ahmed Fares},
      year={2025},
      eprint={2502.05931},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.05931},
}
```

## License

This code is released under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](./LICENSE) for more details.

## Contact

For questions or issues, please contact me at ahmed.abdelaziz.gm@gmail.com
