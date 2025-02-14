from config import get_config

args = get_config()


from torcheeg.model_selection import KFold, train_test_split
from torcheeg.trainers import ClassifierTrainer
from triggerset import TriggerSet, Verifier
from torcheeg.datasets import DEAPDataset
from torch.utils.data import DataLoader
from torcheeg import transforms
from torch import nn
from utils import *
import json
import math
import os


folds = args["folds"]
architecture = args["architecture"]
working_dir = f"./results/{architecture}"
os.makedirs(working_dir, exist_ok=True)


cv = KFold(n_splits=folds, shuffle=True, split_path=f"{working_dir}/{folds}-splits")
label_transform = transforms.Compose(
    [
        transforms.Select(["valence", "arousal", "dominance", "liking"]),
        transforms.Binary(5.0),
        BinariesToCategory,
    ]
)

match architecture:
    case "CCNN":
        from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT

        def remove_base_from_eeg(eeg, baseline):
            return {"eeg": eeg - baseline, "baseline": baseline}

        dataset = DEAPDataset(
            io_path=f"{working_dir}/dataset",
            root_path=args["data_path"],
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
        from torcheeg.datasets.constants import DEAP_CHANNEL_LIST

        dataset = DEAPDataset(
            io_path=f"{working_dir}/dataset",
            root_path=args["data_path"],
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
        dataset = DEAPDataset(
            io_path=f"{working_dir}/dataset",
            root_path=args["data_path"],
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


def train():
    results = dict()
    lr = args["lrate"]
    epochs = args["epochs"]
    evals = args["evaluate"]
    experiment = args["experiment"]
    batch_size = args["batch"] or 32
    pruning_mode = args["pruning_mode"]
    pruning_delta = args["pruning_delta"]
    base_models = args["base_models_dir"]
    training_mode = args["training_mode"]
    pruning_method = args["pruning_method"]
    fine_tuning_mode = args["fine_tuning_mode"]
    transfer_learning_mode = args["transfer_learning_mode"]

    model_path = f"{working_dir}/{experiment}/{'.' if not base_models else '_'.join(base_models.strip('/').split('/')[-2:])}/{fine_tuning_mode or transfer_learning_mode or ''}/"
    os.makedirs(model_path, exist_ok=True)
    results_path = model_path + (
        f"{pruning_method}-{pruning_mode}-{pruning_delta}.json"
        if experiment == "pruning"
        else (
            f"lr={lr}-epochs={epochs}-batch={batch_size}.json"
            if training_mode != "skip"
            else f"{experiment}.json"
        )
    )

    for i, (train_dataset, test_dataset) in enumerate(cv.split(dataset)):
        fold = f"fold-{i}"
        results[fold] = dict()
        save_path = f"{model_path}/models/{fold}"

        model = get_model(architecture)

        trainer = ClassifierTrainer(
            model=model, num_classes=16, lr=lr, accelerator="gpu"
        )

        def evaluate():
            results = dict()
            for eval_dimension in evals:
                if eval_dimension.endswith("watermark"):
                    verifier = Verifier[eval_dimension.split("_")[0].upper()]
                    null_set = TriggerSet(
                        test_dataset,
                        architecture,
                        seed=42,
                        do_true_embedding=False,
                        verifier=verifier,
                    )
                    true_set = TriggerSet(
                        test_dataset,
                        architecture,
                        seed=42,
                        do_null_embedding=False,
                        verifier=verifier,
                    )

                    null_set_loader = DataLoader(null_set, batch_size=batch_size)
                    true_set_loader = DataLoader(true_set, batch_size=batch_size)

                    results[eval_dimension] = {
                        "null_set": trainer.test(
                            null_set_loader, enable_model_summary=True
                        ),
                        "true_set": trainer.test(
                            true_set_loader, enable_model_summary=True
                        ),
                    }
                elif eval_dimension == "eeg":
                    test_loader = DataLoader(test_dataset, batch_size=batch_size)
                    results[eval_dimension] = trainer.test(
                        test_loader, enable_model_summary=True
                    )
            return results

        if experiment == "pruning":
            from pruning import Pruning

            pruning_percent = 1
            prune = getattr(Pruning, pruning_method)()

            while pruning_percent < 100:
                model = get_model(architecture)
                load_path = f"{base_models}/{fold}"
                model = load_model(model, get_ckpt_file(load_path))
                model.eval()

                for name, module in model.named_modules():
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                        prune(module, name="weight", amount=pruning_percent / 100)

                results[fold][pruning_percent] = evaluate()

                if pruning_mode == "linear":
                    pruning_percent += pruning_delta
                else:
                    pruning_percent = int(math.ceil(pruning_percent * pruning_delta))
        elif experiment in [
            "pretrain",
            "fine_tuning",
            "quantization",
            "new_watermark",
            "transfer_learning",
        ]:
            load_path = f"{base_models}/{fold}"
            model = load_model(model, get_ckpt_file(load_path))

        if experiment == "transfer_learning":
            import transfer_learning

            transfer_learning_model = getattr(transfer_learning, architecture)
            transfer_learning_func = getattr(
                transfer_learning_model, transfer_learning_mode.upper()
            )
            model = transfer_learning_func(model)
        elif experiment == "fine_tuning":
            import fine_tuning

            fine_tuning_func = getattr(fine_tuning, fine_tuning_mode.upper())
            model = fine_tuning_func(model)
        elif experiment == "quantization":
            from quantization import quantize

            model = quantize(model)
            model.eval()
            results[fold] = evaluate()

        if training_mode != "skip":
            from pytorch_lightning.callbacks import ModelCheckpoint

            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss", dirpath=save_path, save_top_k=1, mode="min"
            )

            if experiment == "no_watermark":
                val_dataset = test_dataset
                trigger_set = train_dataset
            elif experiment in ["transfer_learning", "fine_tuning"]:
                trigger_set, val_dataset = train_test_split(
                    test_dataset, test_size=0.2, shuffle=True
                )
            else:
                verifier = Verifier.CORRECT
                if experiment == "new_watermark":
                    verifier = Verifier.NEW
                    trigger_set, val_dataset = train_test_split(
                        test_dataset, test_size=0.2, shuffle=True
                    )

                val_dataset = TriggerSet(
                    test_dataset,
                    architecture,
                    include_train_set=True,
                    verifier=verifier,
                )
                trigger_set = TriggerSet(
                    train_dataset,
                    architecture,
                    include_train_set=True,
                    verifier=verifier,
                )

            val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
            train_loader = DataLoader(trigger_set, shuffle=True, batch_size=batch_size)

            trainer.fit(
                train_loader,
                val_loader,
                max_epochs=epochs,
                default_root_dir=save_path,
                callbacks=[checkpoint_callback],
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
            json.dump(results, f)

        if training_mode == "quick":
            break


if __name__ == "__main__":
    train()
