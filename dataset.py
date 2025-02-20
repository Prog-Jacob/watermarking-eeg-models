from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from functools import reduce
from rich.console import Group
from torcheeg import transforms
from utils import BinariesToCategory
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT, DEAP_CHANNEL_LIST


emotions = ["valence", "arousal", "dominance", "liking"]


def get_dataset(architecture, working_dir, data_path=""):
    label_transform = transforms.Compose(
        [
            transforms.Select(emotions),
            transforms.Binary(5.0),
            BinariesToCategory,
        ]
    )

    match architecture:
        case "CCNN":

            def remove_base_from_eeg(eeg, baseline):
                return {"eeg": eeg - baseline, "baseline": baseline}

            return DEAPDataset(
                io_path=f"{working_dir}/dataset",
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
            return DEAPDataset(
                io_path=f"{working_dir}/dataset",
                root_path=data_path,
                chunk_size=512,
                num_baseline=1,
                baseline_chunk_size=384,
                offline_transform=transforms.Compose(
                    [
                        transforms.PickElectrode(
                            transforms.PickElectrode.to_index_list(
                                [
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
                                ],
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
            return DEAPDataset(
                io_path=f"{working_dir}/dataset",
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


def get_dataset_stats(dataset, tree):
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

    map = dict()
    for _, l in dataset:
        map[l] = map.get(l, 0) + 1
    map = dict(sorted(map.items(), key=lambda item: item[1]))
    total_samples = reduce(lambda acc, l: acc + l, map.values(), 0)

    for i, (key, value) in enumerate(map.items()):
        percentage = (value / total_samples) * 100
        label_table.add_row(
            f"{key:02d}",
            f"{key:04b}",
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

    for i, emotion in enumerate(emotions):
        high = reduce(
            lambda acc, l: acc + map[l] if (l >> i) & 1 else acc, map.keys(), 0
        )
        emotion_table.add_row(
            f"[bold]{emotion.title()}[/bold]",
            f"{(1 << i):04b}",
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
