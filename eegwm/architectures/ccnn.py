"""CCNN architecture specification."""

import torch
import numpy as np
from torcheeg import transforms
from torcheeg.datasets.constants import (
    DEAP_CHANNEL_LOCATION_DICT,
    DEAP_CHANNEL_LIST,
    DEAP_LOCATION_LIST,
)

from eegwm.data.dataset import BetterDEAPDataset
from eegwm.architectures.base import Architecture
from eegwm.architectures import _heads


def _build_model(num_classes, device):
    from torcheeg.models import CCNN

    return CCNN(num_classes=num_classes, in_channels=4, grid_size=(9, 9)).to(device)


def _build_dataset(common):
    def remove_base_from_eeg(eeg, baseline):
        return {"eeg": eeg - baseline, "baseline": baseline}

    return BetterDEAPDataset(
        **common,
        baseline_chunk_size=384,
        offline_transform=transforms.Compose(
            [
                transforms.BandDifferentialEntropy(apply_to_baseline=True),
                transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True),
                remove_base_from_eeg,
            ]
        ),
    )


def _reshape_watermark(filter):
    return transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)(eeg=filter)["eeg"]


def _back_to_origin(sample):
    sample = sample.reshape(4, -1)
    sample = sample.permute(1, 0)
    return transforms.PickElectrode(
        transforms.PickElectrode.to_index_list(
            np.array(DEAP_LOCATION_LIST).flatten().tolist(),
            DEAP_CHANNEL_LIST,
        )
    )(eeg=sample)["eeg"]


def _plot_points():
    return {"theta": 0, "alpha": 1, "beta": 2, "gamma": 3}


class _Transfer:
    @staticmethod
    def _build(model):
        teacher_lin1 = model.lin1[0]
        teacher_lin2 = model.lin2
        model.lin1 = torch.nn.Sequential(
            teacher_lin1,
            *_heads.dense_block(teacher_lin1.out_features, teacher_lin2.in_features),
        )
        return model

    @staticmethod
    def ADDED(model):  # Fine-tune the two added dense layers
        model = _Transfer._build(model)
        return _heads.freeze_except(model, [model.lin1[1], model.lin1[4]])

    @staticmethod
    def DENSE(model):  # Fine-tune all dense layers
        model = _Transfer._build(model)
        return _heads.freeze_except(
            model, [model.lin1[0], model.lin1[1], model.lin1[4], model.lin2]
        )

    @staticmethod
    def ALL(model):  # Fine-tune all layers
        return _heads.unfreeze_all(_Transfer._build(model))


ARCHITECTURE = Architecture(
    name="CCNN",
    channels=DEAP_CHANNEL_LIST,
    watermark_shape=(32, 4),
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
