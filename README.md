# Protecting Intellectual Property of EEG-based Neural Networks with Watermarking

## Overview

This repository provides a configuration and experiment setup to apply a cryptographic watermarking method for protecting EEG-based neural networks. The watermark is embedded during the model training process to secure intellectual property, using minimal performance degradation while ensuring robustness against adversarial attacks and piracy. This code can be used to evaluate various experiments on different neural network architectures like CCNN, EEGNet, and TSCeption.

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
usage: train.py [-h] [--evaluate EVALUATE [EVALUATE ...]] --experiment EXPERIMENT --architecture ARCHITECTURE [--skip_training] [--root_path ROOT_PATH] [--batch BATCH] [--epochs EPOCHS]
                [--lrate LRATE]

Specify config args for watermarking EEG-based neural networks experiments and evaluations.

options:
  -h, --help            Show this help message and exit
  --evaluate EVALUATE [EVALUATE ...]
                        Specify any number of dimensions to evaluate from correct_watermark, wrong_watermark, new_watermark, or eeg!
  --experiment EXPERIMENT
                        Specify any number of experiments to do from nowatermark, newwatermark:{fromscratch,pretrain}, fromscratch, pretrain, pruning:{ascending,descending,random},
                        quantization, transfer_learning:{all,added,dense}, or fine_tuning:{ftll,ftal,rtll,rtal}!
  --architecture ARCHITECTURE
                        Choose CCNN, EEGNet, or TSCeption!
  --skip_training
  --root_path ROOT_PATH
                        Provide the path to the processed python data directory!
  --batch BATCH         Number of samples per batch!
  --epochs EPOCHS       Number of epochs!
  --lrate LRATE         Learning rate!
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

This code is released under the GNU General Public License v3.0 (GPL-3.0) License. See the [LICENSE](./LICENSE) for more details.

## Contact

For questions or issues, please contact me at ahmed.abdelaziz.gm@gmail.com