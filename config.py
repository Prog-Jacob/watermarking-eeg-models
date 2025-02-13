import argparse


def get_config():
    parser = argparse.ArgumentParser(
        description="Specify arguments for watermarking EEG-based neural networks experiments and evaluations."
    )

    allowed_evaluations = [
        "correct_watermark",
        "wrong_watermark",
        "new_watermark",
        "eeg",
    ]
    parser.add_argument(
        "--evaluate",
        nargs="+",
        default=allowed_evaluations,
        choices=allowed_evaluations,
        help=f"Specify evaluations from {', '.join(allowed_evaluations)}",
    )

    allowed_experiments = [
        "pretrain",
        "from_scratch",
        "no_watermark",
        "new_watermark",
        "pruning",
        "fine_tuning",
        "quantization",
        "transfer_learning",
    ]
    parser.add_argument(
        "--experiment",
        required=True,
        choices=allowed_experiments,
        help=f"Choose one experiment from {', '.join(allowed_experiments)}",
    )

    allowed_architectures = ["CCNN", "EEGNet", "TSCeption"]
    parser.add_argument(
        "--architecture",
        required=True,
        default="CCNN",
        choices=allowed_architectures,
        help=f"Choose architecture from {', '.join(allowed_architectures)}",
    )

    training_modes = ["full", "quick", "skip"]
    parser.add_argument(
        "--training_mode",
        default="full",
        choices=training_modes,
        help=f"Training mode: {', '.join(training_modes)}",
    )

    parser.add_argument(
        "--data_path",
        default="./data/data_processed_python",
        help="Path to processed data directory",
    )

    # Define potentially required arguments as optional initially
    parser.add_argument(
        "--batch", type=int, help="Batch size (required if training mode is not 'skip')"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs (required if training mode is not 'skip')",
    )
    parser.add_argument(
        "--lrate",
        type=float,
        help="Learning rate (required if training mode is not 'skip')",
    )

    args = vars(parser.parse_args())

    if args["training_mode"] != "skip":
        if not all([args.get("batch"), args.get("epochs"), args.get("lrate")]):
            parser.error(
                "--batch, --epochs, and --lrate are required when the training mode is not 'skip'"
            )

    return args
