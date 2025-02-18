import argparse


def get_config():
    # Constants for allowed values
    ALLOWED_EVALUATIONS = [
        "correct_watermark",
        "wrong_watermark",
        "new_watermark",
        "eeg",
    ]
    ALLOWED_EXPERIMENTS = [
        "pretrain",
        "from_scratch",
        "no_watermark",
        "new_watermark",
        "pruning",
        "fine_tuning",
        "quantization",
        "transfer_learning",
    ]
    ARCHITECTURES = ["CCNN", "EEGNet", "TSCeption"]
    TRAINING_MODES = ["full", "quick", "skip"]

    parser = argparse.ArgumentParser(
        description="Configure and run experiments for watermarking EEG-based neural networks."
    )

    # Experiment Configuration
    config_group = parser.add_argument_group("Experiment Configuration")
    config_group.add_argument(
        "--experiment",
        required=True,
        choices=ALLOWED_EXPERIMENTS,
        help=f"Experiment to run. Options: {', '.join(ALLOWED_EXPERIMENTS)}.",
    )
    config_group.add_argument(
        "--evaluate",
        nargs="+",
        default=ALLOWED_EVALUATIONS,
        choices=ALLOWED_EVALUATIONS,
        help=f"Evaluations to perform. Options: {', '.join(ALLOWED_EVALUATIONS)}.",
    )
    config_group.add_argument(
        "--architecture",
        required=True,
        default="CCNN",
        choices=ARCHITECTURES,
        help=f"Model architecture. Options: {', '.join(ARCHITECTURES)}.",
    )

    # Training Parameters
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument(
        "--training_mode",
        default="full",
        choices=TRAINING_MODES,
        help=f"Training mode. Options: {', '.join(TRAINING_MODES)}.",
    )
    train_group.add_argument("--batch", type=int, help="Batch size for training.")
    train_group.add_argument("--epochs", type=int, help="Number of training epochs.")
    train_group.add_argument("--lrate", type=float, help="Learning rate for training.")
    train_group.add_argument(
        "--update_lr_by",
        type=float,
        default=1.0,
        help="Multiply learning rate by x every n epochs. Default x: 1.0",
    )
    train_group.add_argument(
        "--update_lr_every",
        type=int,
        default=10,
        help="Multiply learning rate by x every n epochs. Default n: 10",
    )
    train_group.add_argument(
        "--update_lr_until",
        type=float,
        default=1e-5,
        help="Update learning until it's out of [ε, 1.0]. Default ε: 1e-5",
    )
    train_group.add_argument(
        "--folds",
        type=int,
        default=10,
        help="Number of k-fold cross-validation splits. Default k: 10.",
    )

    # Path Configuration
    path_group = parser.add_argument_group("Path Configuration")
    path_group.add_argument(
        "--data_path",
        default="./data/data_preprocessed_python",
        help="Path to processed data directory. Default: './data/data_preprocessed_python'.",
    )
    path_group.add_argument(
        "--base_models_dir", help="Directory containing base models for experiments."
    )

    # Experiment-Specific Parameters
    exp_params_group = parser.add_argument_group("Experiment-Specific Parameters")
    exp_params_group.add_argument(
        "--pruning_method",
        choices=["random", "ascending", "descending"],
        help="Pruning method. Options: 'random', 'ascending' (nullify least-valued nodes), 'descending' (nullify most-valued nodes).",
    )
    exp_params_group.add_argument(
        "--pruning_mode",
        choices=["linear", "exponential"],
        help="Pruning mode. Options: 'linear' (delta=5) or 'exponential' (delta=1.25).",
    )
    exp_params_group.add_argument(
        "--pruning_delta",
        type=float,
        help="Pruning delta value. Recommended: 5 for linear, 1.25 for exponential.",
    )
    exp_params_group.add_argument(
        "--fine_tuning_mode",
        choices=["ftll", "ftal", "rtll", "rtal"],
        help="Fine-tuning mode. Options: 'ftll' (fine-tune last layer), 'ftal' (fine-tune all layers), "
        "'rtll' (retrain last layer), 'rtal' (retrain all layers).",
    )
    exp_params_group.add_argument(
        "--transfer_learning_mode",
        choices=["added", "dense", "all"],
        help="Transfer learning mode. Options: 'added' (add new layers), 'dense' (fine-tune dense layers), "
        "'all' (fine-tune all layers).",
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
    ]

    if args["experiment"] in ["quantization", "pruning"]:
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
        "fine_tuning": ["fine_tuning_mode"],
        "transfer_learning": ["transfer_learning_mode"],
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
