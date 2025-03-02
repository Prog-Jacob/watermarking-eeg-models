from torch import nn


def FTLL(model):
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Identify the last layer dynamically
    last_layer = None
    for name, module in model.named_children():
        last_layer = module  # Keep updating to get the last module

    # If a last layer exists, unfreeze its parameters
    if last_layer is not None:
        for param in last_layer.parameters():
            param.requires_grad = True

    return model


def FTAL(model):
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True

    return model


def RTLL(model):
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Identify the last layer dynamically
    last_layer = None
    for name, module in model.named_children():
        last_layer = module  # Keep updating to get the last module

    # If a last layer exists, unfreeze its parameters
    if last_layer is not None:
        for param in last_layer.parameters():
            if param.dim() > 1:  # Reinitialize weight matrices, not biases
                nn.init.xavier_uniform_(param)
            param.requires_grad = True

    return model


def RTAL(model):
    # Identify the last layer dynamically
    last_layer = None
    for name, module in model.named_children():
        last_layer = module  # Keep updating to get the last module

    # If a last layer exists, unfreeze its parameters
    if last_layer is not None:
        for param in last_layer.parameters():
            if param.dim() > 1:  # Reinitialize weight matrices, not biases
                nn.init.xavier_uniform_(param)
            param.requires_grad = True

    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True

    return model
