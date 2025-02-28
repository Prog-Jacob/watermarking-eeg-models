import os
from torch import load


def get_model(architecture, device):
    match architecture:
        case "CCNN":
            from torcheeg.models import CCNN

            return CCNN(num_classes=16, in_channels=4, grid_size=(9, 9)).to(device)

        case "TSCeption":
            from torcheeg.models import TSCeption

            return TSCeption(
                num_classes=16,
                num_electrodes=28,
                sampling_rate=128,
                num_T=60,
                num_S=60,
                hid_channels=128,
                dropout=0.5,
            ).to(device)

        case "EEGNet":
            from torcheeg.models import EEGNet

            return EEGNet(
                chunk_size=128,
                num_electrodes=32,
                dropout=0.5,
                kernel_1=64,
                kernel_2=16,
                F1=16,
                F2=32,
                D=32,
                num_classes=16,
            ).to(device)

        case _:
            raise ValueError(f"Invalid architecture: {architecture}")


def load_model(model, model_path):
    try:
        state_dict = load(model_path)["state_dict"]
        for key in list(state_dict.keys()):
            state_dict[key.replace("model.", "")] = state_dict.pop(key)
        model.load_state_dict(state_dict)
        return model
    except:
        raise ValueError(f"Invalid model path: {model_path}")


def get_ckpt_file(load_path):
    try:
        return next(
            os.path.join(load_path, f)
            for f in os.listdir(load_path)
            if f.endswith(".ckpt")
        )
    except:
        None
