import torch
import torch.nn as nn


def quantize(model):
    return torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.ReLU},
        dtype=torch.qint8,
    )
