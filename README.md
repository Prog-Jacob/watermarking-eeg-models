# Protecting Intellectual Property of EEG-based Neural Networks with Watermarking

## Overview

This repository provides a configuration and experiment setup to apply a cryptographic watermarking method for protecting EEG-based neural networks. The watermark is embedded during the model training process to secure intellectual property, using minimal performance degradation while ensuring robustness against adversarial attacks and piracy. This code can be used to evaluate various experiments on different neural network architectures like CCNN, EEGNet, and TSCeption.
<img src="https://raw.githubusercontent.com/Prog-Jacob/watermarking-eeg-models/b0c13a214abb86c4592bd4b928051db6f3b7db9f/Training.svg" alt="Embedding a wonder filter during training an EEG-based neural network" style="width: 100%">

## Key Features

- **Watermarking Method**: Embeds cryptographic watermarks in EEG models during training with minimal impact on accuracy (≤5% drop).
- **Experiments**: Supports multiple experiment configurations including model training, fine-tuning, pruning, and more.
- **Adversarial Evaluation**: Evaluates watermark robustness against various adversarial scenarios like fine-tuning, transfer learning, and neuron pruning.
- **Model Options**: Supports architectures such as CCNN, EEGNet, and TSception.
- **Flexible Configurations**: Allows for evaluation of watermarks and model performance on various dimensions (e.g., correct watermark detection, wrong watermark, new watermark, EEG task performance).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Prog-Jacob/watermarking-eeg-models.git
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Experiments

The experiment configurations are managed through a command-line interface. You can specify your configuration options using arguments when running the `train.py` script.

### Command Syntax:

```bash
python train.py [args]
```

### Available Arguments:

```
usage: train.py [-h] --experiment {show_stats,no_watermark,from_scratch,pretrain,new_watermark,pruning,fine_tuning,quantization,transfer_learning} [--evaluate DIMENSION [DIMENSION ...]]
                --architecture {CCNN,EEGNet,TSCeption} [--training_mode {skip,quick,full}] [--batch BATCH] [--epochs EPOCHS] [--lrate LRATE] [--update_lr_by x] [--update_lr_every n]
                [--update_lr_until ε] [--folds k] [--data_path PATH] [--base_models_dir DIR] [--pruning_method {random,ascending,descending}] [--pruning_mode {linear,exponential}]
                [--pruning_delta δ] [--fine_tuning_mode {ftll,ftal,rtll,rtal}] [--transfer_learning_mode {added,dense,all}] [--seed SEED] [--verbose {info,debug,warning,error,critical}]

Configure and run experiments for watermarking EEG-based neural networks.

options:
  -h, --help            Show this help message and exit

Experiment Configuration:
  --experiment {show_stats,no_watermark,from_scratch,pretrain,new_watermark,pruning,fine_tuning,quantization,transfer_learning}
                        Choose one experiment from the above experiments.
  --evaluate DIMENSION [DIMENSION ...]
                        Choose any number of dimensions to evaluate from {eeg,correct_watermark,wrong_watermark,new_watermark}.
  --architecture {CCNN,EEGNet,TSCeption}

Training Parameters:
  --training_mode {skip,quick,full}
                        Skip training, quick training of only 1 fold, or full training of all folds.
  --batch BATCH         Batch size for training.
  --epochs EPOCHS       Number of training epochs.
  --lrate LRATE         Learning rate for training.
  --update_lr_by x      Multiply learning rate by x every n epochs. Default x: 1.0
  --update_lr_every n   Multiply learning rate by x every n epochs. Default n: 10
  --update_lr_until ε   Update learning until it's out of [ε, 1.0]. Default ε: 1e-5
  --folds k             Number of k-fold cross-validation splits. Default k: 10.

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
```

### Example Usage:

To run a quick takeover (embed attacker's watermark) experiment with a trained TSCeption model from scratch, you could use:

```bash
python train.py --experiment new_watermark --architecture TSCeption --base_models_dir ./results/TSCeption/from_scratch/models --batch 64 --lrate 0.0005 --epochs 30 --training_mode quick
```

### Skip Training and Evaluate:

If you wish to skip the training process and evaluate a previously trained model for its watermark retention, you can use:

```bash
python train.py --experiment no_watermark --architecture CCNN --evaluate eeg correct_watermark --training_mode skip
```

## Experiment Results

After running experiments, the results will be output to the console and saved to file system based on the specified evaluation dimensions. You can check the evaluation of the watermarks and performance on the EEG tasks. Another way to show all results for a certain architecture is to use the `show_stats` experiment. For example:

```bash
python train.py --experiment show_stats --architecture TSCeption --verbose error --seed 42
```

Which outputs the following:

```bash
Statistics and Results for TSCeption
├── ╭────────────────────────────────────── Dataset Summary ───────────────────────────────────────╮
│   │                                                                                              │
│   │                                 Distribution of the Labels                                   │
│   │    ┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓     │
│   │    ┃         Label          ┃     Binary      ┃         Count ┃       Percentage       ┃     │
│   │    ┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩     │
│   │    │           12           │      1100       │           150 │         0.78%          │     │
│   │    │           08           │      1000       │           240 │         1.25%          │     │
│   │    │           03           │      0011       │           360 │         1.88%          │     │
│   │    │           14           │      1110       │           390 │         2.03%          │     │
│   │    │           13           │      1101       │           720 │         3.75%          │     │
│   │    │           10           │      1010       │           735 │         3.83%          │     │
│   │    │           07           │      0111       │           825 │         4.30%          │     │
│   │    │           02           │      0010       │           840 │         4.38%          │     │
│   │    │           09           │      1001       │           885 │         4.61%          │     │
│   │    │           06           │      0110       │          1035 │         5.39%          │     │
│   │    │           05           │      0101       │          1125 │         5.86%          │     │
│   │    │           01           │      0001       │          1200 │         6.25%          │     │
│   │    │           04           │      0100       │          1455 │         7.58%          │     │
│   │    │           00           │      0000       │          1500 │         7.81%          │     │
│   │    │           11           │      1011       │          2130 │         11.09%         │     │
│   │    │           15           │      1111       │          5610 │         29.22%         │     │
│   │    ├────────────────────────┼─────────────────┼───────────────┼────────────────────────┤     │
│   │    │       16 Labels        │      ────       │         19200 │        100.00%         │     │
│   │    └────────────────────────┴─────────────────┴───────────────┴────────────────────────┘     │
│   │                                                                                              │
│   │                                Contribution of Each Emotion                                  │
│   │    ┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓     │
│   │    ┃ Emotion            ┃    Binary     ┃       High (≥5)        ┃      Low (<5)       ┃     │
│   │    ┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩     │
│   │    │ Valence            │     0001      │      12855 (67%)       │     6345 (33%)      │     │
│   │    │ Arousal            │     0010      │      11925 (62%)       │     7275 (38%)      │     │
│   │    │ Dominance          │     0100      │      11310 (59%)       │     7890 (41%)      │     │
│   │    │ Liking             │     1000      │      10860 (57%)       │     8340 (43%)      │     │
│   │    └────────────────────┴───────────────┴────────────────────────┴─────────────────────┘     │
│   ╰──────────────────────────────────────────────────────────────────────────────────────────────╯
│
│
└── From Scratch
    └── ╭───── Experiment Summary from: lr=0.005-epochs=1-batch=32.json ─────╮
        │                                                                    │
        │                ╭────── Experiment Details ───────╮                 │
        │                │ ┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓ │                 │
        │                │ ┃       Parameter ┃ Value     ┃ │                 │
        │                │ ┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩ │                 │
        │                │ │    Architecture │ TSCeption │ │                 │
        │                │ │   Training Mode │ quick     │ │                 │
        │                │ │           Batch │ 32        │ │                 │
        │                │ │          Epochs │ 1         │ │                 │
        │                │ │           Lrate │ 0.005     │ │                 │
        │                │ │    Update Lr By │ 0.5       │ │                 │
        │                │ │ Update Lr Every │ 25        │ │                 │
        │                │ │ Update Lr Until │ 1e-07     │ │                 │
        │                │ │           Folds │ 10        │ │                 │
        │                │ └─────────────────┴───────────┘ │                 │
        │                ╰─────────────────────────────────╯                 │
        │                Accuracies                                          │
        │                ├── Eeg: 28.85%                                     │
        │                ├── Correct Watermark                               │
        │                │   ├── Null Set: 29.19%                            │
        │                │   └── True Set: 100.00%                           │
        │                ├── Wrong Watermark                                 │
        │                │   ├── Null Set: 10.25%                            │
        │                │   └── True Set: 0.00%                             │
        │                └── New Watermark                                   │
        │                    ├── Null Set: 10.25%                            │
        │                    └── True Set: 0.00%                             │
        │                                                                    │
        ╰────────────────────────────────────────────────────────────────────╯
```

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
