"""Architecture-agnostic fine-tuning surgery (FTLL/FTAL/RTLL/RTAL)."""

from torch import nn


def _last_layer(model):
    """Return the model's last child module."""
    last_layer = None
    for _, module in model.named_children():
        last_layer = module
    return last_layer


def _fine_tune(model, reinit_last, freeze_rest):
    """Freeze the body (unless freeze_rest is False) and unfreeze the last layer,
    reinitializing its weight matrices when reinit_last is True."""
    for param in model.parameters():
        param.requires_grad = not freeze_rest

    last_layer = _last_layer(model)
    if last_layer is not None:
        for param in last_layer.parameters():
            if reinit_last and param.dim() > 1:  # weights, not biases
                nn.init.xavier_uniform_(param)
            param.requires_grad = True

    return model


def FTLL(model):
    """Fine-tune the last layer only."""
    return _fine_tune(model, reinit_last=False, freeze_rest=True)


def FTAL(model):
    """Fine-tune all layers."""
    return _fine_tune(model, reinit_last=False, freeze_rest=False)


def RTLL(model):
    """Reinitialize and retrain the last layer only."""
    return _fine_tune(model, reinit_last=True, freeze_rest=True)


def RTAL(model):
    """Reinitialize the last layer, then retrain all layers."""
    return _fine_tune(model, reinit_last=True, freeze_rest=False)
