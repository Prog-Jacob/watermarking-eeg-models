"""Shared transfer-learning surgery helpers used by the architecture modules."""

from torch import nn


def dense_block(in_features, out_features):
    """Two trainable dense layers (each Linear + ReLU + Dropout) for the head."""
    return [
        nn.Linear(in_features, 100),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(100, out_features),
        nn.ReLU(),
        nn.Dropout(0.3),
    ]


def freeze_except(model, trainable):
    """Freeze every parameter, then unfreeze those in the given layers."""
    for param in model.parameters():
        param.requires_grad = False
    for layer in trainable:
        for param in layer.parameters():
            param.requires_grad = True
    return model


def unfreeze_all(model):
    """Make every parameter trainable."""
    for param in model.parameters():
        param.requires_grad = True
    return model
