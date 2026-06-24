"""Dynamic int8 quantization of a trained model."""

import torch
import torch.nn as nn


def quantize(model):
    """Dynamically quantize Linear/Conv/ReLU layers to int8."""
    return torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.ReLU},
        dtype=torch.qint8,
    )
