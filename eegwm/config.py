"""CLI argument definitions, validation, and the typed Config view."""

import random
import argparse
from dataclasses import dataclass, field


def get_config():
    """Parse and validate CLI arguments into a dict."""
    # Constants for allowed values
    ALLOWED_EVALUATIONS = [
        "eeg",
        "correct_watermark",
        "wrong_watermark",
        "new_watermark",
    ]
    ALLOWED_EXPERIMENTS = [
        "show_stats",
        "show_stats_plots",
        "no_watermark",
        "from_scratch",
        "pretrain",
        "new_watermark",
        "pruning",
        "fine_tuning",
        "quantization",
        "transfer_learning",
        "feature_attribution",
    ]
    TRAINING_MODES = ["skip", "quick", "full"]
    ARCHITECTURES = ["CCNN", "EEGNet", "TSCeption"]
    ALLOWED_LABELS = ["valence", "arousal", "dominance", "liking"]
    PRUNING_METHODS = ["random", "ascending", "descending"]
    PRUNING_MODES = ["linear", "exponential"]
    FINE_TUNING_MODES = ["ftll", "ftal", "rtll", "rtal"]
    TRANSFER_LEARNING_MODES = ["added", "dense", "all"]
    WATERMARK_LAYOUTS = ["block", "scatter"]
    DEVICES = ["cpu", "cuda"]
    VERBOSE_LEVELS = ["info", "debug", "warning", "error", "critical"]

    parser = argparse.ArgumentParser(
        description="Configure and run experiments for watermarking EEG-based neural networks.",
    )

    # Experiment Configuration
    config_group = parser.add_argument_group("Experiment Configuration")
    config_group.add_argument(
        "--experiment",
        required=True,
        choices=ALLOWED_EXPERIMENTS,
        help="Choose one experiment from the above experiments.",
    )
    config_group.add_argument(
        "--evaluate",
        nargs="+",
        default=ALLOWED_EVALUATIONS,
        choices=ALLOWED_EVALUATIONS,
        metavar="DIMENSION",
        help=f"Choose one or more dimension to evaluate from {{{','.join(ALLOWED_EVALUATIONS)}}}.",
    )
    config_group.add_argument(
        "--architecture",
        required=True,
        default="CCNN",
        choices=ARCHITECTURES,
    )
    config_group.add_argument(
        "--labels",
        nargs="+",
        default=ALLOWED_LABELS,
        choices=ALLOWED_LABELS,
        metavar="LABEL",
        help=f"Choose one or more dataset label from {{{','.join(ALLOWED_LABELS)}}}.",
    )
    config_group.add_argument(
        "--watermark_layout",
        default="block",
        choices=WATERMARK_LAYOUTS,
        help="Where the wonder-filter bits land: 'block' (contiguous corner block, the paper baseline) or 'scatter' (same bits spread across the grid). Must match between embedding and verification runs.",
    )

    # Training Parameters
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument(
        "--training_mode",
        default="full",
        choices=TRAINING_MODES,
        help="Skip training, quick training of only 1 fold, or full training of all folds.",
    )
    train_group.add_argument("--batch", type=int, help="Batch size for training.")
    train_group.add_argument("--epochs", type=int, help="Number of training epochs.")
    train_group.add_argument("--lrate", type=float, help="Learning rate for training.")
    train_group.add_argument(
        "--update_lr_by",
        type=float,
        default=1.0,
        metavar="x",
        help="Multiply learning rate by x every n epochs. Default x: 1.0",
    )
    train_group.add_argument(
        "--update_lr_every",
        type=int,
        default=10,
        metavar="n",
        help="Multiply learning rate by x every n epochs. Default n: 10",
    )
    train_group.add_argument(
        "--update_lr_until",
        type=float,
        default=1e-5,
        metavar="ε",
        help="Update learning until it's out of [ε, 1.0]. Default ε: 1e-5",
    )
    train_group.add_argument(
        "--folds",
        type=int,
        default=5,
        metavar="k",
        help="Number of k-fold cross-validation splits. Default k: 5.",
    )

    # Path Configuration
    path_group = parser.add_argument_group("Path Configuration")
    path_group.add_argument(
        "--data_path",
        metavar="PATH",
        default="./data/data_preprocessed_python",
        help="Path to processed data directory. Default: './data/data_preprocessed_python'.",
    )
    path_group.add_argument(
        "--base_models_dir",
        metavar="DIR",
        help="Directory containing base models for experiments.",
    )

    # Experiment-Specific Parameters
    exp_params_group = parser.add_argument_group("Experiment-Specific Parameters")
    exp_params_group.add_argument(
        "--pruning_method",
        choices=PRUNING_METHODS,
        help="Random, ascending (nullify least-valued nodes), descending (nullify most-valued nodes).",
    )
    exp_params_group.add_argument(
        "--pruning_mode",
        choices=PRUNING_MODES,
        help="Linear increments pruning percentage by δ till it reaches 100, whereas exponential multiplies by it.",
    )
    exp_params_group.add_argument(
        "--pruning_delta",
        metavar="δ",
        type=float,
        help="Increment/multiply pruning percent by δ. Recommended: 5 for linear, 1.25 for exponential.",
    )
    exp_params_group.add_argument(
        "--fine_tuning_mode",
        choices=FINE_TUNING_MODES,
        help="FTLL (fine-tune last layer), FTAL (fine-tune all layers), RTLL (retrain last layer), and RTAL (retrain all layers).",
    )
    exp_params_group.add_argument(
        "--transfer_learning_mode",
        choices=TRANSFER_LEARNING_MODES,
        help="Add two dense layers then perform fine tuning. Added (fine-tune added layers), dense (fine-tune dense layers), and all (fine-tune all layers).",
    )

    # Other Parameters
    other_group = parser.add_argument_group("Other Parameters")
    other_group.add_argument(
        "--seed",
        type=int,
        help="Seed for reproducibility.",
    )
    other_group.add_argument(
        "--verbose",
        choices=VERBOSE_LEVELS,
        default="info",
        help="How much information to log. Default is 'info'.",
    )
    other_group.add_argument(
        "--device",
        choices=DEVICES,
        default="cuda",
        help="Device to run the experiment on. Default is 'cuda'.",
    )

    # Parse and validate arguments
    args = vars(parser.parse_args())
    validate_arguments(parser, args)

    return args


def validate_arguments(parser, args):
    EXPERIMENTS_REQUIRING_BASE_MODELS = [
        "pruning",
        "pretrain",
        "fine_tuning",
        "quantization",
        "new_watermark",
        "transfer_learning",
        "feature_attribution",
    ]

    if args["experiment"] in ["quantization", "pruning", "feature_attribution"]:
        args["training_mode"] = "skip"
    if args["experiment"].startswith("show_stats"):
        args["training_mode"] = "skip"

    if args["training_mode"] != "skip":
        require_args(
            parser,
            args,
            ["batch", "epochs", "lrate"],
            "are required when training_mode is not 'skip'.",
        )

    if args["experiment"] in EXPERIMENTS_REQUIRING_BASE_MODELS:
        require_arg(
            parser, args, "base_models_dir", "is required for this experiment type."
        )

    validation_rules = {
        "pruning": ["pruning_method", "pruning_mode", "pruning_delta"],
        "transfer_learning": ["transfer_learning_mode"],
        "fine_tuning": ["fine_tuning_mode"],
    }

    if args["experiment"] in validation_rules:
        required_args = validation_rules[args["experiment"]]
        require_args(
            parser,
            args,
            required_args,
            f"are required for {args['experiment']} experiments.",
        )


def require_args(parser, args, arg_names, message):
    missing = [f"--{arg}" for arg in arg_names if not args.get(arg)]
    if missing:
        parser.error(f"{', '.join(missing)} {message}")


def require_arg(parser, args, arg_name, message):
    if not args.get(arg_name):
        parser.error(f"--{arg_name} {message}")


@dataclass
class Config:
    """Typed view over the validated CLI arguments."""

    experiment: str
    architecture: str
    evaluate: list
    labels: list
    training_mode: str
    batch: int | None
    epochs: int | None
    lrate: float | None
    update_lr_by: float
    update_lr_every: int
    update_lr_until: float
    folds: int
    data_path: str
    base_models_dir: str | None
    pruning_method: str | None
    pruning_mode: str | None
    pruning_delta: float | None
    fine_tuning_mode: str | None
    transfer_learning_mode: str | None
    watermark_layout: str
    seed: int
    verbose: str
    device: str
    # Original argparse dict, kept verbatim for the result manifest.
    raw: dict = field(repr=False)

    @property
    def num_classes(self) -> int:
        return 2 ** len(self.labels)

    @property
    def batch_size(self) -> int:
        return self.batch or 32

    @property
    def working_dir(self) -> str:
        from eegwm.constants import RESULTS_DIR

        return f"{RESULTS_DIR}/{self.architecture}"


def load_config() -> Config:
    """Parse, validate, and seed-default the CLI arguments into a Config."""
    args = get_config()
    seed = args["seed"]
    if seed is None:
        seed = random.randint(0, 1000)
    return Config(
        experiment=args["experiment"],
        architecture=args["architecture"],
        evaluate=args["evaluate"],
        labels=args["labels"],
        training_mode=args["training_mode"],
        batch=args["batch"],
        epochs=args["epochs"],
        lrate=args["lrate"],
        update_lr_by=args["update_lr_by"],
        update_lr_every=args["update_lr_every"],
        update_lr_until=args["update_lr_until"],
        folds=args["folds"],
        data_path=args["data_path"],
        base_models_dir=args["base_models_dir"],
        pruning_method=args["pruning_method"],
        pruning_mode=args["pruning_mode"],
        pruning_delta=args["pruning_delta"],
        fine_tuning_mode=args["fine_tuning_mode"],
        transfer_learning_mode=args["transfer_learning_mode"],
        watermark_layout=args["watermark_layout"],
        seed=seed,
        verbose=args["verbose"],
        device=args["device"],
        raw=args,
    )
