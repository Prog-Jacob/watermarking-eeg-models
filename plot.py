import os
import mne
import torch
import pandas as pd
from pycirclize import Circos
import matplotlib.pyplot as plt


mne.set_log_level("CRITICAL")
montage = mne.channels.make_standard_montage("standard_1020")


def plot_topomap(
    tensor,
    fig_label,
    channel_list,
    labeled_plot_points,
    save_fig=True,
    show_fig=False,
    show_names=None,
):
    ch_types = ["eeg"] * len(channel_list)
    tensor = tensor.detach().cpu().numpy()
    info = mne.create_info(ch_names=channel_list, ch_types=ch_types, sfreq=128)
    info.set_montage(montage, match_alias=True, match_case=False, on_missing="ignore")

    num_plots = len(labeled_plot_points)
    fig, axes = plt.subplots(
        1,
        num_plots,
        squeeze=False,
        figsize=(num_plots * 5, 5),
    )

    for i, (point_label, point_value) in enumerate(labeled_plot_points.items()):
        mne.viz.plot_topomap(
            tensor[:, point_value],
            info,
            axes=axes[0, i],
            show=False,
            names=channel_list if show_names else None,
            sphere=(0.0, 0.0, 0.0, 0.11),
        )
        axes[0, i].set_title(point_label, fontsize=20, fontname="Liberation Serif")

    fig.suptitle(fig_label, fontsize=24, fontweight="bold")

    if save_fig:
        os.makedirs("./results", exist_ok=True)
        fig.savefig(
            f"./results/Topomap - {fig_label}.png", dpi=300, bbox_inches="tight"
        )
    if show_fig:
        fig.show()
    return fig


def plot_signal(tensor, fig_label, channel_list, sampling_rate=128, save_fig=True):
    ch_types = ["misc"] * len(channel_list)
    tensor = tensor.detach().cpu().numpy()
    info = mne.create_info(
        ch_names=channel_list, ch_types=ch_types, sfreq=sampling_rate
    )

    raw = mne.io.RawArray(tensor, info)
    raw.set_montage(montage, match_alias=True, on_missing="ignore")

    fig = raw.plot(show_scrollbars=False, show_scalebars=False, block=True)
    fig.suptitle(fig_label, fontsize=24, fontweight="bold")

    if save_fig:
        os.makedirs("./results", exist_ok=True)
        fig.savefig(f"./results/Signal - {fig_label}.png", dpi=300, bbox_inches="tight")

    return fig


def plot_emotion_connectivity(label_map, emotions, fig_label):
    n = len(emotions)
    adj = torch.zeros((n * 2, n * 2))
    all_emotions = [
        f"{strength} {emotion.title()}"
        for strength in ["Low", "High"]
        for emotion in emotions
    ]

    for label, count in label_map.items():
        for i in range(n):
            for j in range(n):
                if i <= j:
                    continue
                row = n * ((label >> i) & 1) + i
                col = n * ((label >> j) & 1) + j
                adj[row, col] += count

    fig_data = pd.DataFrame(adj, index=all_emotions, columns=all_emotions)
    figure = Circos.chord_diagram(
        fig_data,
        space=5,
        cmap="tab10",
        label_kws=dict(size=12),
        link_kws=dict(ec="black", lw=0.5, direction=0),
    )

    os.makedirs("./results", exist_ok=True)
    figure.savefig(f"./results/{fig_label}.png")
