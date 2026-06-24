"""Console entry point for the eegwm package.

Run as ``python -m eegwm ...`` or via the installed ``eegwm`` script. All work
happens inside ``main()`` so importing the package has no side effects.
"""

import os
import logging

from rich.tree import Tree

from eegwm.config import Config, load_config


def _show_stats(cfg: Config, dataset) -> None:
    """Render dataset statistics (and optionally plots) plus saved results."""
    from eegwm.evaluation.results import get_results_stats, print_to_console
    from eegwm.data.stats import get_dataset_stats, get_dataset_plots

    if cfg.experiment.endswith("plots"):
        get_dataset_plots(dataset, cfg.architecture, cfg.watermark_layout)

    tree = Tree(
        f"[bold cyan]\nStatistics and Results for {cfg.architecture}[/bold cyan]"
    )
    get_dataset_stats(dataset, tree, cfg.labels)
    get_results_stats(cfg.working_dir, tree)
    print_to_console(tree)


def main() -> None:
    """Parse config, set up logging and seed, build the dataset, and dispatch."""
    # Heavy imports are deferred to keep `--help` and config errors fast.
    from torcheeg.model_selection import KFold

    from eegwm.utils import set_seed
    from eegwm.data.dataset import get_dataset

    cfg = load_config()

    level = getattr(logging, cfg.verbose.upper())
    logging.getLogger("torcheeg").setLevel(level)
    log = logging.getLogger("eegwm")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        log.addHandler(handler)
        log.propagate = False
    log.setLevel(level)
    log.info(
        "Experiment '%s' | architecture %s | seed %d",
        cfg.experiment,
        cfg.architecture,
        cfg.seed,
    )

    set_seed(cfg.seed)
    os.makedirs(cfg.working_dir, exist_ok=True)

    cv = KFold(
        n_splits=cfg.folds,
        shuffle=True,
        split_path=f"{cfg.working_dir}/{cfg.folds}-splits",
    )
    dataset = get_dataset(cfg.architecture, cfg.working_dir, cfg.labels, cfg.data_path)

    if cfg.experiment.startswith("show_stats"):
        _show_stats(cfg, dataset)
        return

    from eegwm.training.runner import run

    run(cfg, dataset, cv)
