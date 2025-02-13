from config import get_config

args = get_config()


from torcheeg.trainers import ClassifierTrainer
from triggerset import TriggerSet, Verifier
from torcheeg.model_selection import KFold
from torcheeg.datasets import DEAPDataset
from torch.utils.data import DataLoader
from utils import BinariesToCategory
from torcheeg import transforms
from torch import load
import json
import os


architecture = args["architecture"]
working_dir = f"./results/{architecture}"
os.makedirs(working_dir, exist_ok=True)


cv = KFold(n_splits=10, shuffle=True, split_path=f"{working_dir}/split")
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
    batch_size = args["batch"]
    experiment = args["experiment"]
    training_mode = args["training_mode"]
    results_path = f"{working_dir}/{'/'.join(experiment.split(':'))}/{lr}-{epochs}-{batch_size}-.json"

    for i, (train_dataset, test_dataset) in enumerate(cv.split(dataset)):
        fold = f"fold-{i}"
        results[fold] = dict()
        save_path = f"{working_dir}/{'/'.join(experiment.split(':'))}/models/{fold}"

        match architecture:
            case "CCNN":
                from torcheeg.models import CCNN

                model = CCNN(num_classes=16, in_channels=4, grid_size=(9, 9))

            case "TSCeption":
                from torcheeg.models import TSCeption

                model = TSCeption(
                    num_classes=16,
                    num_electrodes=28,
                    sampling_rate=128,
                    num_T=15,
                    num_S=15,
                    hid_channels=32,
                    dropout=0.5,
                )

            case "EEGNet":
                from torcheeg.models import EEGNet

                model = EEGNet(
                    chunk_size=128,
                    num_electrodes=32,
                    dropout=0.5,
                    kernel_1=64,
                    kernel_2=16,
                    F1=8,
                    F2=16,
                    D=2,
                    num_classes=16,
                )

            case _:
                raise ValueError(f"Invalid architecture: {architecture}")

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

        if training_mode != "skip":
            from pytorch_lightning.callbacks import ModelCheckpoint

            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss", dirpath=save_path, save_top_k=1, mode="min"
            )

            if experiment == "no_watermark":
                val_dataset = test_dataset
                trigger_set = train_dataset
            else:
                verifier = Verifier.CORRECT
                if experiment == "new_watermark":
                    verifier = Verifier.NEW
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

            # if ARCH == 'pretrain' and TRAINING:
            #     load_path = f"../nowatermark/{save_path}"
            #     ckpt_file = next((os.path.join(load_path, f) for f in os.listdir(load_path) if f.endswith('.ckpt')), None)
            #     model = load_model(model, ckpt_file)

        ckpt_file = next(
            (
                os.path.join(save_path, f)
                for f in os.listdir(save_path)
                if f.endswith(".ckpt")
            ),
            None,
        )
        model = load_model(model, ckpt_file)
        model.eval()

        results[fold] = evaluate()

        with open(results_path, "w") as f:
            json.dump(results, f)

        if training_mode == "quick":
            break


def load_model(model, model_path):
    state_dict = load(model_path)["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "")] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    train()
