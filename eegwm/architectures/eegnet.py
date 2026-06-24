"""EEGNet architecture specification."""

import torch
from torcheeg import transforms
from torcheeg.datasets.constants import DEAP_CHANNEL_LIST

from eegwm.data.dataset import BetterDEAPDataset
from eegwm.architectures.base import Architecture
from eegwm.architectures import _heads

_POINTS = 4
_LEN = 128


def _build_model(num_classes, device):
    from torcheeg.models import EEGNet

    return EEGNet(
        chunk_size=128,
        num_electrodes=32,
        dropout=0.5,
        kernel_1=64,
        kernel_2=16,
        F1=32,
        F2=64,
        D=16,
        num_classes=num_classes,
    ).to(device)


def _build_dataset(common):
    return BetterDEAPDataset(
        **{
            **common,
            "online_transform": transforms.Compose(
                [
                    transforms.To2d(),
                    transforms.ToTensor(),
                ]
            ),
        },
    )


def _reshape_watermark(filter):
    return transforms.To2d()(eeg=filter)["eeg"]


def _back_to_origin(sample):
    return sample.squeeze(0)


def _plot_points():
    return {f"{i / _POINTS}s": i * _LEN // _POINTS for i in range(_POINTS)}


class _Transfer:
    @staticmethod
    def _build(model):
        teacher_lin = model.lin
        in_features = teacher_lin.in_features
        model.lin = torch.nn.Sequential(
            *_heads.dense_block(in_features, in_features),
            teacher_lin,
        )
        return model

    @staticmethod
    def ADDED(model):  # Fine-tune the two added dense layers
        model = _Transfer._build(model)
        return _heads.freeze_except(model, [model.lin[0], model.lin[3]])

    @staticmethod
    def DENSE(model):  # Fine-tune all dense layers
        model = _Transfer._build(model)
        return _heads.freeze_except(model, [model.lin[0], model.lin[3], model.lin[6]])

    @staticmethod
    def ALL(model):  # Fine-tune all layers
        return _heads.unfreeze_all(_Transfer._build(model))


ARCHITECTURE = Architecture(
    name="EEGNet",
    channels=DEAP_CHANNEL_LIST,
    watermark_shape=(32, 128),
    model_factory=_build_model,
    dataset_factory=_build_dataset,
    reshape_watermark=_reshape_watermark,
    back_to_origin=_back_to_origin,
    plot_points=_plot_points,
    transfer_modes={
        "ADDED": _Transfer.ADDED,
        "DENSE": _Transfer.DENSE,
        "ALL": _Transfer.ALL,
    },
)
