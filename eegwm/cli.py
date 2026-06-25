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
    cfg = load_config()

    # DataLoader workers pass tensors between processes via shared memory; the
    # default file_descriptor strategy uses /dev/shm, which is tiny on hosts like
    # Kaggle (~64 MB) and aborts workers ("DataLoader worker killed by signal:
    # Aborted") once it fills across sequential runs. file_system shares via /tmp
    # instead. Imported here (not at module top) to keep `eegwm -h` torch-free.
    import torch.multiprocessing as torch_mp

    torch_mp.set_sharing_strategy("file_system")

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

    # Heavy imports are deferred so `--help` and config errors stay fast.
    from torcheeg.model_selection import KFold
    from eegwm.utils import set_seed
    from eegwm.data.dataset import get_dataset

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
