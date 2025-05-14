import json
import torch
import hashlib
import numpy as np
from os import path
from rich.text import Text
from utils import serialize
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from functools import reduce
from rich.console import Group
from torcheeg import transforms
from utils import BinariesToCategory
from torch.utils.data import DataLoader
from torcheeg.datasets import DEAPDataset
from triggerset import TriggerSet, Verifier
from plot import plot_emotion_connectivity, plot_topomap
from torcheeg.datasets.constants import (
    DEAP_CHANNEL_LOCATION_DICT,
    DEAP_CHANNEL_LIST,
    DEAP_LOCATION_LIST,
)


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


class BetterDEAPDataset(DEAPDataset):
    def __init__(self, *args, io_dir, **kwargs):
        if kwargs.get("io_path") is None:
            kwargs["io_path"] = path.join(io_dir, self.hash(*args, **kwargs))
        super().__init__(*args, **kwargs)
        if self.__str__ is not None:
            self.save_state()

    def toJSON(self, *args, **kwargs):
        trivial_keys = ["root_path", "io_mode", "io_size", "num_worker", "verbose"]
        kwargs = {k: v for k, v in kwargs.items() if k not in trivial_keys}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs["args"] = tuple(args)
        kwargs = serialize(kwargs)

        self.__str__ = json.dumps(kwargs, sort_keys=True, default=str)

    def hash(self, *args, **kwargs):
        self.toJSON(*args, **kwargs)
        return hashlib.md5(self.__str__.encode()).hexdigest()[:10]

    def save_state(self):
        json.dump(
            json.loads(self.__str__),
            open(path.join(self.io_path, "state"), "w"),
            indent=4,
        )


def get_dataset(architecture, working_dir, dataset_labels, data_path=""):
    label_transform = transforms.Compose(
        [
            transforms.Select(dataset_labels),
            transforms.Binary(5.0),
            BinariesToCategory,
        ]
    )

    match architecture:
        case "CCNN":

            def remove_base_from_eeg(eeg, baseline):
                return {"eeg": eeg - baseline, "baseline": baseline}

            return BetterDEAPDataset(
                io_dir=f"{working_dir}",
                root_path=data_path,
                num_baseline=1,
                baseline_chunk_size=384,
                offline_transform=transforms.Compose(
                    [
                        transforms.BandDifferentialEntropy(apply_to_baseline=True),
                        transforms.ToGrid(
                            DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True
                        ),
                        remove_base_from_eeg,
                    ]
                ),
                label_transform=label_transform,
                online_transform=transforms.ToTensor(),
                num_worker=4,
                verbose=True,
            )

        case "TSCeption":
            return BetterDEAPDataset(
                io_dir=f"{working_dir}",
                root_path=data_path,
                chunk_size=512,
                num_baseline=1,
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
                online_transform=transforms.ToTensor(),
                label_transform=label_transform,
                num_worker=4,
                verbose=True,
            )

        case "EEGNet":
            return BetterDEAPDataset(
                io_dir=f"{working_dir}",
                root_path=data_path,
                num_baseline=1,
                online_transform=transforms.Compose(
                    [
                        transforms.To2d(),
                        transforms.ToTensor(),
                    ]
                ),
                label_transform=label_transform,
                num_worker=4,
                verbose=True,
            )

        case _:
            raise ValueError(f"Invalid architecture: {architecture}")


def get_dataset_stats(dataset, tree, dataset_labels):
    width = len(dataset_labels)
    label_table = Table(
        title="\n[bold]Distribution of the Labels[/bold]",
        header_style="bold magenta",
        show_header=True,
        width=85,
    )
    label_table.add_column("Label", justify="center", style="green")
    label_table.add_column("Binary", justify="center", style="yellow")
    label_table.add_column("Count", justify="right", style="cyan")
    label_table.add_column("Percentage", justify="center", style="bold white")

    map = get_labels_map(dataset)
    total_samples = sum(map.values())
    plot_emotion_connectivity(map, dataset_labels, "Emotions Relationship")

    for i, (key, value) in enumerate(map.items()):
        percentage = (value / total_samples) * 100
        label_table.add_row(
            f"{key:02d}",
            f"{key:0{width}b}",
            f"{value}",
            f"{percentage:.2f}%",
            end_section=i == len(map) - 1,
        )

    label_table.add_row(
        f"[bold]{len(map)} Labels[/bold]",
        "[bold]────[/bold]",
        f"[bold]{total_samples}[/bold]",
        "[bold]100.00%[/bold]",
    )

    emotion_table = Table(
        title="\n[bold]Contribution of Each Emotion[/bold]",
        header_style="bold magenta",
        show_header=True,
        width=85,
    )
    emotion_table.add_column("Emotion", justify="left", style="bold cyan")
    emotion_table.add_column("Binary", justify="center", style="yellow")
    emotion_table.add_column(
        "High [white](≥5)[/white]", justify="center", style="green"
    )
    emotion_table.add_column("Low [white](<5)[/white]", justify="center", style="red")

    for i, emotion in enumerate(dataset_labels):
        high = reduce(
            lambda acc, label: acc + map[label] if (label >> i) & 1 else acc,
            map.keys(),
            0,
        )
        emotion_table.add_row(
            f"[bold]{emotion.title()}[/bold]",
            f"{(1 << i):0{width}b}",
            f"{high} [white]({(high / total_samples * 100):.0f}%)[/white]",
            f"{total_samples - high} [white]({(100 - high / total_samples * 100):.0f}%)[/white]",
        )

    panel = Panel(
        Align.center(
            Group(
                label_table,
                emotion_table,
            )
        ),
        title="[bold]Dataset Summary[/bold]",
        title_align="center",
        width=96,
    )

    tree.add(Group(panel, Text("\n", style="reset")))


def get_dataset_plots(dataset, architecture):
    for eval_dimension in ["EEG", "Correct Watermark", "New Watermark"]:
        fig_label = f"{architecture} - {eval_dimension}"

        if eval_dimension == "EEG":
            mean_tensor = get_dataset_mean(dataset, architecture)
            plot_topomap(
                mean_tensor,
                fig_label,
                channel_list=get_channel_list(architecture),
                labeled_plot_points=get_labeled_plot_points(architecture),
            )
            continue

        for embedding_type in ["Null", "True"]:
            triggerset = TriggerSet(
                dataset,
                size=len(dataset),
                architecture=architecture,
                do_true_embedding=embedding_type == "True",
                do_null_embedding=embedding_type == "Null",
                verifier=Verifier[eval_dimension.split(" ")[0].upper()],
            )

            mean_tensor = get_dataset_mean(triggerset, architecture)

            plot_topomap(
                mean_tensor,
                f"{fig_label} - {embedding_type} Embedding",
                channel_list=get_channel_list(architecture),
                labeled_plot_points=get_labeled_plot_points(architecture),
            )


def get_dataset_mean(dataset, architecture):
    num_samples = 0
    sum_tensor = None
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    for data, _ in dataloader:
        if sum_tensor is None:
            sum_tensor = torch.zeros_like(data[0])

        sum_tensor += data.sum(dim=0)
        num_samples += data.shape[0]

    return transform_back_to_origin(sum_tensor / num_samples, architecture)


def get_labels_map(dataset):
    label_count_map = dict()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    for _, labels in dataloader:
        for label in labels:
            label_count_map[label.item()] = label_count_map.get(label.item(), 0) + 1

    return dict(sorted(label_count_map.items(), key=lambda item: item[1]))


def transform_back_to_origin(sample, architecture):
    match architecture:
        case "EEGNet" | "TSCeption":
            return sample.squeeze(0)
        case "CCNN":
            sample = sample.reshape(4, -1)
            sample = sample.permute(1, 0)
            return transforms.PickElectrode(
                transforms.PickElectrode.to_index_list(
                    np.array(DEAP_LOCATION_LIST).flatten().tolist(),
                    DEAP_CHANNEL_LIST,
                )
            )(eeg=sample)["eeg"]
        case _:
            raise ValueError(f"Invalid architecture: {architecture}")


def get_channel_list(architecture):
    match architecture:
        case "CCNN" | "EEGNet":
            return DEAP_CHANNEL_LIST
        case "TSCeption":
            return TSCEPTION_CHANNEL_LIST
        case _:
            raise ValueError(f"Invalid architecture: {architecture}")


def get_labeled_plot_points(architecture):
    points = 4
    len = 512 if architecture == "TSCeption" else 128
    match architecture:
        case "CCNN":
            return {"theta": 0, "alpha": 1, "beta": 2, "gamma": 3}
        case "EEGNet" | "TSCeption":
            return {f"{i / points}s": i * len // points for i in range(points)}
        case _:
            raise ValueError(f"Invalid architecture: {architecture}")
