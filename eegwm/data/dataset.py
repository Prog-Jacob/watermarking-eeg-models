"""DEAP dataset construction with config-hashed caching."""

import json
import hashlib
from os import path

from torcheeg import transforms
from torcheeg.datasets import DEAPDataset

from eegwm.utils import serialize, BinariesToCategory


class BetterDEAPDataset(DEAPDataset):
    """DEAPDataset that derives a stable cache path by hashing its config."""

    def __init__(self, *args, io_dir, **kwargs):
        self._state_json = None
        if kwargs.get("io_path") is None:
            kwargs["io_path"] = path.join(io_dir, self.hash(*args, **kwargs))
        super().__init__(*args, **kwargs)
        if self._state_json is not None:
            self.save_state()

    def toJSON(self, *args, **kwargs):
        trivial_keys = ["root_path", "io_mode", "io_size", "num_worker", "verbose"]
        kwargs = {k: v for k, v in kwargs.items() if k not in trivial_keys}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs["args"] = tuple(args)
        kwargs = serialize(kwargs)

        self._state_json = json.dumps(kwargs, sort_keys=True, default=str)

    def hash(self, *args, **kwargs):
        self.toJSON(*args, **kwargs)
        return hashlib.md5(self._state_json.encode()).hexdigest()[:10]

    def save_state(self):
        json.dump(
            json.loads(self._state_json),
            open(path.join(self.io_path, "state"), "w"),
            indent=4,
        )


def get_dataset(architecture, working_dir, dataset_labels, data_path=""):
    from eegwm.architectures import get_architecture

    label_transform = transforms.Compose(
        [
            transforms.Select(dataset_labels),
            transforms.Binary(5.0),
            BinariesToCategory,
        ]
    )

    common = dict(
        io_dir=f"{working_dir}",
        root_path=data_path,
        num_baseline=1,
        label_transform=label_transform,
        online_transform=transforms.ToTensor(),
        num_worker=4,
        verbose=True,
    )

    return get_architecture(architecture).build_dataset(common)


def transform_back_to_origin(sample, architecture):
    from eegwm.architectures import get_architecture

    return get_architecture(architecture).back_to_origin(sample)


def get_channel_list(architecture):
    from eegwm.architectures import get_architecture

    return get_architecture(architecture).channels


def get_labeled_plot_points(architecture):
    from eegwm.architectures import get_architecture

    return get_architecture(architecture).plot_points()
