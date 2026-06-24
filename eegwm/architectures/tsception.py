"""TSCeption architecture specification."""

import torch
from torcheeg import transforms
from torcheeg.datasets.constants import DEAP_CHANNEL_LIST

from eegwm.data.dataset import BetterDEAPDataset
from eegwm.architectures.base import Architecture
from eegwm.architectures import _heads

TSCEPTION_CHANNEL_LIST = [
    "FP1",
    "AF3",
    "F3",
    "F7",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "CP5",
    "CP1",
    "P3",
    "P7",
    "PO3",
    "O1",
    "FP2",
    "AF4",
    "F4",
    "F8",
    "FC6",
    "FC2",
    "C4",
    "T8",
    "CP6",
    "CP2",
    "P4",
    "P8",
    "PO4",
    "O2",
]

_POINTS = 4
_LEN = 512


def _build_model(num_classes, device):
    from torcheeg.models import TSCeption

    return TSCeption(
        num_classes=num_classes,
        num_electrodes=28,
        sampling_rate=128,
        num_T=60,
        num_S=60,
        hid_channels=128,
        dropout=0.5,
    ).to(device)


def _build_dataset(common):
    return BetterDEAPDataset(
        **common,
        chunk_size=512,
        baseline_chunk_size=384,
        offline_transform=transforms.Compose(
            [
                transforms.PickElectrode(
                    transforms.PickElectrode.to_index_list(
                        TSCEPTION_CHANNEL_LIST,
                        DEAP_CHANNEL_LIST,
                    )
                ),
                transforms.To2d(),
            ]
        ),
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
        teacher_lin1 = model.fc[0]
        teacher_lin2 = model.fc[3]
        model.fc = torch.nn.Sequential(
            teacher_lin1,
            *_heads.dense_block(teacher_lin1.out_features, teacher_lin2.in_features),
            teacher_lin2,
        )
        return model

    @staticmethod
    def ADDED(model):  # Fine-tune the two added dense layers
        model = _Transfer._build(model)
        return _heads.freeze_except(model, [model.fc[1], model.fc[4]])

    @staticmethod
    def DENSE(model):  # Fine-tune all dense layers
        model = _Transfer._build(model)
        return _heads.freeze_except(
            model, [model.fc[0], model.fc[1], model.fc[4], model.fc[7]]
        )

    @staticmethod
    def ALL(model):  # Fine-tune all layers
        return _heads.unfreeze_all(_Transfer._build(model))


ARCHITECTURE = Architecture(
    name="TSCeption",
    channels=TSCEPTION_CHANNEL_LIST,
    watermark_shape=(28, 512),
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
