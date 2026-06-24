"""Dataset statistics and topomap rendering.

Kept separate from ``eegwm.data.dataset`` so the core dataset construction stays
free of visualization and watermark dependencies.
"""

import torch
from functools import reduce
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from rich.console import Group
from torch.utils.data import DataLoader

from eegwm.viz.plot import plot_emotion_connectivity, plot_topomap
from eegwm.watermark.triggerset import TriggerSet, Verifier
from eegwm.data.dataset import (
    get_channel_list,
    get_labeled_plot_points,
    transform_back_to_origin,
)


def get_dataset_stats(dataset, tree, dataset_labels):
    """Add label-distribution and per-emotion tables to the results tree."""
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

    counts = get_labels_map(dataset)
    total_samples = sum(counts.values())
    plot_emotion_connectivity(counts, dataset_labels, "Emotions Relationship")

    for i, (key, value) in enumerate(counts.items()):
        percentage = (value / total_samples) * 100
        label_table.add_row(
            f"{key:02d}",
            f"{key:0{width}b}",
            f"{value}",
            f"{percentage:.2f}%",
            end_section=i == len(counts) - 1,
        )

    label_table.add_row(
        f"[bold]{len(counts)} Labels[/bold]",
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
            lambda acc, label: acc + counts[label] if (label >> i) & 1 else acc,
            counts.keys(),
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


def get_dataset_plots(dataset, architecture, layout="block"):
    """Render topomaps of the mean EEG and the true/null watermark embeddings."""
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
                size=(len(dataset), len(dataset)),
                architecture=architecture,
                do_true_embedding=embedding_type == "True",
                do_null_embedding=embedding_type == "Null",
                verifier=Verifier[eval_dimension.split(" ")[0].upper()],
                layout=layout,
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
