"""Cross-validation runner: trains/evaluates one experiment over k folds.

This is the orchestration spine. Architecture-specific model surgery lives in
``eegwm.architectures`` and ``eegwm.experiments``; this module only sequences the
fold loop, dispatches per experiment, and writes results.
"""

import os
import math
import json
import logging
from pathlib import Path

from torch import nn
from rich.tree import Tree
from torch.utils.data import DataLoader
from torcheeg.trainers import ClassifierTrainer
from torcheeg.model_selection import train_test_split

from eegwm.config import Config
from eegwm.models import get_model, load_model, get_ckpt_file
from eegwm.watermark.triggerset import TriggerSet, Verifier
from eegwm.evaluation.evaluate import evaluate as run_evaluate
from eegwm.evaluation.results import _get_result_stats, print_to_console

log = logging.getLogger(__name__)

_PARAM_EXCLUDE = ["data_path", "experiment", "evaluate", "verbose", "base_models_dir"]


def _results_path(cfg: Config, model_path: str) -> str:
    if cfg.experiment == "pruning":
        name = f"{cfg.pruning_method}-{cfg.pruning_mode}-{cfg.pruning_delta}.json"
    elif cfg.training_mode != "skip":
        name = f"lr={cfg.lrate}-epochs={cfg.epochs}-batch={cfg.batch_size}.json"
    else:
        name = f"{cfg.experiment}.json"
    return model_path + name


def run(cfg: Config, dataset, cv) -> None:
    """Run the experiment over k folds: build/load the model, apply surgery,
    train and evaluate, and write the result JSON."""
    experiment = cfg.experiment
    architecture = cfg.architecture
    device = cfg.device
    base_models = cfg.base_models_dir
    num_classes = cfg.num_classes
    batch_size = cfg.batch_size

    experiment_details = dict()
    experiment_details["parameters"] = {
        k: v for k, v in cfg.raw.items() if v and k not in _PARAM_EXCLUDE
    }
    experiment_details["results"] = dict()
    results = experiment_details["results"]

    base_tag = (
        "." if not base_models else "_".join(base_models.strip("/").split("/")[-2:])
    )
    mode_tag = cfg.fine_tuning_mode or cfg.transfer_learning_mode or ""
    model_path = f"{cfg.working_dir}/{experiment}/{base_tag}/{mode_tag}/"
    os.makedirs(model_path, exist_ok=True)
    results_path = _results_path(cfg, model_path)

    for i, (train_dataset, test_dataset) in enumerate(cv.split(dataset)):
        fold = f"fold-{i}"
        log.info("Starting %s", fold)
        results[fold] = dict()
        save_path = f"{model_path}/models/{fold}"

        model = get_model(architecture, device, cfg.labels)

        trainer = ClassifierTrainer(
            model=model,
            num_classes=num_classes,
            lr=cfg.lrate,
            accelerator="gpu" if device == "cuda" else "cpu",
        )

        def evaluate():
            return run_evaluate(
                trainer,
                test_dataset,
                architecture,
                num_classes,
                batch_size,
                cfg.evaluate,
            )

        if experiment == "pruning":
            from eegwm.experiments.pruning import Pruning

            pruning_percent = 1
            prune = getattr(Pruning, cfg.pruning_method)()

            while pruning_percent < 100:
                load_path = f"{base_models}/{fold}"
                trainer = ClassifierTrainer(
                    model=model,
                    num_classes=num_classes,
                    lr=cfg.lrate,
                    accelerator="gpu" if device == "cuda" else "cpu",
                )
                if ckpt_file := get_ckpt_file(load_path):
                    model = load_model(model, ckpt_file)
                else:
                    break
                model.eval()

                for name, module in model.named_modules():
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                        prune(module, name="weight", amount=pruning_percent / 100)

                results[fold][pruning_percent] = evaluate()
                model = get_model(architecture, device, cfg.labels)

                if cfg.pruning_mode == "linear":
                    pruning_percent += cfg.pruning_delta
                else:
                    pruning_percent = math.ceil(pruning_percent * cfg.pruning_delta)
        elif experiment in [
            "pretrain",
            "fine_tuning",
            "quantization",
            "new_watermark",
            "transfer_learning",
        ]:
            load_path = f"{base_models}/{fold}"
            if ckpt_file := get_ckpt_file(load_path):
                model = load_model(model, ckpt_file)
            else:
                break
        elif experiment == "feature_attribution":
            from eegwm.experiments.feature_attribution import get_feature_attribution

            load_path = f"{base_models}/{fold}"
            if ckpt_file := get_ckpt_file(load_path):
                model = load_model(model, ckpt_file)
            else:
                break
            get_feature_attribution(
                model, train_dataset, test_dataset, architecture, device
            )
            return

        if experiment == "transfer_learning":
            from eegwm.architectures import get_architecture

            model = get_architecture(architecture).apply_transfer(
                cfg.transfer_learning_mode, model
            )
        elif experiment == "fine_tuning":
            import eegwm.experiments.fine_tuning as fine_tuning

            fine_tuning_func = getattr(fine_tuning, cfg.fine_tuning_mode.upper())
            model = fine_tuning_func(model)
        elif experiment == "quantization":
            from eegwm.experiments.quantization import quantize

            model = quantize(model)
            model.eval()
            results[fold] = evaluate()

        if experiment in ["transfer_learning", "fine_tuning", "new_watermark"]:
            train_dataset, test_dataset = train_test_split(
                test_dataset,
                shuffle=True,
                test_size=0.2,
                split_path=f"{save_path}/split",
            )
        trigger_set, val_dataset = train_dataset, test_dataset

        if experiment in ["from_scratch", "pretrain", "new_watermark"]:
            verifier = Verifier.CORRECT
            if experiment == "new_watermark":
                verifier = Verifier.NEW

            val_dataset = TriggerSet(
                test_dataset,
                architecture,
                size=(len(test_dataset) // 50, len(test_dataset)),
                num_classes=num_classes,
                include_train_set=True,
                verifier=verifier,
            )
            trigger_set = TriggerSet(
                train_dataset,
                architecture,
                size=(len(train_dataset) // 50, len(train_dataset)),
                num_classes=num_classes,
                include_train_set=True,
                verifier=verifier,
            )
        if cfg.training_mode != "skip":
            from pytorch_lightning.callbacks import (
                EarlyStopping,
                ModelCheckpoint,
            )
            from eegwm.training.callbacks import MultiplyLRScheduler

            early_stopping_callback = EarlyStopping(
                monitor="val_loss", patience=5, check_on_train_epoch_end=False
            )
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss", dirpath=save_path, save_top_k=1, mode="min"
            )
            lr_scheduler = MultiplyLRScheduler(
                cfg.update_lr_by, cfg.update_lr_every, cfg.update_lr_until
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                prefetch_factor=2,
                num_workers=2,
            )
            train_loader = DataLoader(
                trigger_set,
                shuffle=True,
                batch_size=batch_size,
                prefetch_factor=2,
                num_workers=2,
            )

            trainer.fit(
                train_loader,
                val_loader,
                max_epochs=cfg.epochs,
                default_root_dir=save_path,
                callbacks=[
                    lr_scheduler,
                    checkpoint_callback,
                    early_stopping_callback,
                ],
                enable_model_summary=True,
                enable_progress_bar=True,
                limit_test_batches=0.0,
            )

            model = load_model(model, checkpoint_callback.best_model_path)
            model.eval()
            results[fold] = evaluate()
        elif load_path := get_ckpt_file(save_path):
            model = load_model(model, load_path)
            model.eval()
            results[fold] = evaluate()

        with open(results_path, "w") as f:
            json.dump(experiment_details, f)

        if cfg.training_mode == "quick":
            break

    tree = Tree("[bold cyan]Results[/bold cyan]")
    _get_result_stats(cfg.working_dir, [str(Path(results_path))], tree)
    print_to_console(tree)
