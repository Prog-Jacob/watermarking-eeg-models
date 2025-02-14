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

```bash
usage: train.py [-h] --experiment {pretrain,from_scratch,no_watermark,new_watermark,pruning,fine_tuning,quantization,transfer_learning}
                [--evaluate {correct_watermark,wrong_watermark,new_watermark,eeg} [{correct_watermark,wrong_watermark,new_watermark,eeg} ...]] --architecture {CCNN,EEGNet,TSCeption}
                [--training_mode {full,quick,skip}] [--batch BATCH] [--epochs EPOCHS] [--lrate LRATE] [--folds FOLDS] [--data_path DATA_PATH] [--base_models_dir BASE_MODELS_DIR]
                [--pruning_mode {linear,exponential}] [--pruning_delta PRUNING_DELTA] [--fine_tuning_mode {ftll,ftal,rtll,rtal}] [--transfer_learning_mode {added,dense,all}]

Configure and run experiments for watermarking EEG-based neural networks.

options:
  -h, --help            Show this help message and exit

Experiment Configuration:
  --experiment {pretrain,from_scratch,no_watermark,new_watermark,pruning,fine_tuning,quantization,transfer_learning}
                        Experiment to run. Options: pretrain, from_scratch, no_watermark, new_watermark, pruning, fine_tuning, quantization, transfer_learning.
  --evaluate {correct_watermark,wrong_watermark,new_watermark,eeg} [{correct_watermark,wrong_watermark,new_watermark,eeg} ...]
                        Evaluations to perform. Options: correct_watermark, wrong_watermark, new_watermark, eeg.
  --architecture {CCNN,EEGNet,TSCeption}
                        Model architecture. Options: CCNN, EEGNet, TSCeption.

Training Parameters:
  --training_mode {full,quick,skip}
                        Training mode. Options: full, quick, skip.
  --batch BATCH         Batch size for training.
  --epochs EPOCHS       Number of training epochs.
  --lrate LRATE         Learning rate for training.
  --folds FOLDS         Number of k-fold cross-validation splits. Default: 10.

Path Configuration:
  --data_path DATA_PATH
                        Path to processed data directory. Default: './data/data_processed_python'.
  --base_models_dir BASE_MODELS_DIR
                        Directory containing base models for experiments.

Experiment-Specific Parameters:
  --pruning_mode {linear,exponential}
                        Pruning mode. Options: 'linear' (delta=5) or 'exponential' (delta=1.25).
  --pruning_delta PRUNING_DELTA
                        Pruning delta value. Recommended: 5 for linear, 1.25 for exponential.
  --fine_tuning_mode {ftll,ftal,rtll,rtal}
                        Fine-tuning mode. Options: 'ftll' (fine-tune last layer), 'ftal' (fine-tune all layers), 'rtll' (retrain last layer), 'rtal' (retrain all layers).
  --transfer_learning_mode {added,dense,all}
                        Transfer learning mode. Options: 'added' (add new layers), 'dense' (fine-tune dense layers), 'all' (fine-tune all layers).
```

### Example Usage:

To run a takeover (embed attacker's watermark) experiment with a trained EEGNet model from scratch, you could use:

```bash
python train.py --experiment newwatermark:fromscratch --architecture EEGNet --epochs 100 --lrate 0.001
```

### Skip Training and Evaluate:

If you wish to skip the training process and evaluate a previously trained model for its watermark retention, you can use:

```bash
python train.py --experiment nowatermark --evaluate eeg --skip_training
```

## Experiment Results

After running experiments, the results will be output based on the specified evaluation dimensions. You can check the evaluation of the watermark and performance on the EEG tasks.

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
