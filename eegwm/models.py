"""Model construction and checkpoint loading helpers."""

import os
from torch import load


def get_model(architecture: str, device: str, labels: list):
    """Build the model for an architecture, sized to ``2 ** len(labels)`` classes."""
    from eegwm.architectures import get_architecture

    num_classes = 2 ** len(labels)
    return get_architecture(architecture).build_model(num_classes, device)


def load_model(model, model_path: str):
    """Load a Lightning checkpoint into ``model``, stripping the ``model.`` prefix."""
    try:
        state_dict = load(model_path)["state_dict"]
        for key in list(state_dict.keys()):
            state_dict[key.replace("model.", "")] = state_dict.pop(key)
        model.load_state_dict(state_dict)
        return model
    except Exception as e:
        raise ValueError(f"Invalid model path: {model_path}") from e


def get_ckpt_file(load_path: str):
    """Return the first ``.ckpt`` file in a directory, or None if none exists."""
    try:
        return next(
            os.path.join(load_path, f)
            for f in os.listdir(load_path)
            if f.endswith(".ckpt")
        )
    except (StopIteration, FileNotFoundError):
        return None
