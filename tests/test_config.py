"""Config tests. Pure-Python (no torch), so they run anywhere."""

from eegwm.config import Config


def _cfg(**overrides):
    base = dict(
        experiment="from_scratch",
        architecture="CCNN",
        evaluate=["eeg"],
        labels=["valence", "arousal", "dominance", "liking"],
        training_mode="full",
        batch=64,
        epochs=10,
        lrate=1e-3,
        update_lr_by=1.0,
        update_lr_every=10,
        update_lr_until=1e-5,
        folds=5,
        data_path="./data",
        base_models_dir=None,
        pruning_method=None,
        pruning_mode=None,
        pruning_delta=None,
        fine_tuning_mode=None,
        transfer_learning_mode=None,
        watermark_layout="block",
        seed=42,
        verbose="info",
        device="cpu",
        raw={},
    )
    base.update(overrides)
    return Config(**base)


def test_num_classes_tracks_labels():
    # The latent bug this codebase had: class count must follow --labels.
    assert _cfg().num_classes == 16
    assert _cfg(labels=["valence", "arousal"]).num_classes == 4
    assert _cfg(labels=["valence"]).num_classes == 2


def test_batch_size_defaults_to_32():
    assert _cfg(batch=None).batch_size == 32
    assert _cfg(batch=64).batch_size == 64


def test_working_dir_uses_architecture():
    assert _cfg(architecture="EEGNet").working_dir.endswith("/EEGNet")
