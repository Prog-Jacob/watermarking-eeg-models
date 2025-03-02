import json
import numpy as np
from utils import *
import plotille as plt
from pathlib import Path
from rich.tree import Tree
from rich.text import Text
from rich.align import Align
from rich.table import Table
from rich.panel import Panel
from rich.console import Group


def get_rendered_pruning_results(results, fig, label=""):
    if fig is None:
        fig = create_graph("Pruning", "Accuracy")

    for key, value in results.items():
        if are_keys_numeric(value):
            get_graph(
                fig,
                [float(k) for k in value.keys()],
                [float(v) * 100 for v in value.values()],
                random.randint(0, 255),
                label + " - " + key,
            )
        else:
            get_rendered_pruning_results(value, fig, label + " - " + key)

    return Text.from_ansi(fig.show(legend=True))


def get_graph(fig, xs, ys, i, label="Task"):
    min, max = 0, 100

    X = np.linspace(min, max, 100)
    f = interpolate(xs, ys)
    Y = f(X)

    fig.plot(X, Y, lc=i, label=title(label.strip(" - ")))


def create_graph(
    x_label="Pruning", y_label="Accuracy", x_limits=(0, 90), y_limits=(-5, 105)
):
    fig = plt.Figure()

    fig.width = 50
    fig.height = 22
    fig.x_label = x_label
    fig.y_label = y_label
    fig.color_mode = "byte"
    fig.set_x_limits(min_=x_limits[0], max_=x_limits[1])
    fig.set_y_limits(min_=y_limits[0], max_=y_limits[1])

    return fig


def get_results(json_path):
    experiment_details = json.load(open(json_path))
    results = experiment_details.get("results", experiment_details)
    results = {key: value for key, value in results.items() if len(value) > 0}

    experiment_table = Table(title_style="bold cyan")
    experiment_table.add_column("Parameter", style="bold magenta", justify="right")
    experiment_table.add_column("Value", style="bold white")

    for k, v in experiment_details.get("parameters", {}).items():
        experiment_table.add_row(title(k), str(v))

    table_panel = Panel(
        Align.center(experiment_table),
        border_style="cyan",
        title="Experiment Details",
    )

    aggregated_results = dict()
    samples = len(results)

    for folds in results.values():
        if are_keys_numeric(folds):
            for key, value in folds.items():
                aggregated_results[key] = aggregated_results.get(key, dict())
                aggregate_result(value, samples, aggregated_results[key])
            continue
        aggregate_result(folds, samples, aggregated_results)

    if are_keys_numeric(aggregated_results):
        results = dict()
        for key, value in aggregated_results.items():
            add_key_at_depth(value, results, key)
        rendered_results = Panel(
            Align.center(get_rendered_pruning_results(results, None)),
            title="[green]Effect of Pruning over Different Accuracies[/green]",
            border_style="green",
            padding=(0, 1),
        )
        width = None
    else:
        width = 70
        rendered_results = Tree("[bold]Accuracies[/bold]")
        convert_dict_to_tree(aggregated_results, rendered_results, 5)

    content = Group(table_panel, rendered_results, fit=True)

    experiment_summary_panel = Panel(
        Align.center(content),
        title=f"[bold yellow]Experiment Summary from:[/bold yellow] [italic]{json_path.split('/')[-1]}[/italic]",
        border_style="bold white",
        expand=width is not None,
        padding=(1, 3),
        width=width,
    )

    return experiment_summary_panel


def aggregate_result(result, samples, dictionary):
    for key, value in result.items():
        if isinstance(value, list):
            dictionary[key] = (
                dictionary.get(key, 0) + value[0]["test_accuracy"] / samples
            )
        else:
            dictionary[key] = dictionary.get(key, dict())
            aggregate_result(value, samples, dictionary[key])


def get_results_stats(working_dir, tree):
    dir = str(Path(working_dir))
    json_files = list_json_files(dir)
    _get_result_stats(working_dir, json_files, tree)


def _get_result_stats(working_dir, json_files, tree):
    results = {}
    dir = str(Path(working_dir))

    for file_path in json_files:
        result = get_results(file_path)
        i = file_path.index(dir) + len(dir)
        add_to_dict(results, [p for p in file_path[i:].split("/") if p], result)

    convert_dict_to_tree(results, tree, 1)


def print_to_console(obj):
    from rich.console import Console

    console = Console()
    console.print(obj)
