"""Evaluate a trained model across the requested dimensions.

For each watermark dimension, a model is tested on both the null-embedded and
true-embedded trigger sets; for the ``eeg`` dimension it is tested on the clean
task data. Returns a nested dict keyed by evaluation dimension.
"""

from torch.utils.data import DataLoader

from eegwm.constants import VERIFICATION_SEED
from eegwm.watermark.triggerset import TriggerSet, Verifier


def evaluate(
    trainer,
    test_dataset,
    architecture,
    num_classes,
    batch_size,
    metrics,
    layout="block",
):
    results = dict()
    for eval_dimension in metrics:
        if eval_dimension.endswith("watermark"):
            verifier = Verifier[eval_dimension.split("_")[0].upper()]
            null_set = TriggerSet(
                test_dataset,
                architecture,
                seed=VERIFICATION_SEED,
                num_classes=num_classes,
                do_true_embedding=False,
                verifier=verifier,
                layout=layout,
            )
            true_set = TriggerSet(
                test_dataset,
                architecture,
                seed=VERIFICATION_SEED,
                num_classes=num_classes,
                do_null_embedding=False,
                verifier=verifier,
                layout=layout,
            )

            null_set_loader = DataLoader(
                null_set,
                batch_size=batch_size,
                prefetch_factor=2,
                num_workers=2,
            )
            true_set_loader = DataLoader(
                true_set,
                batch_size=batch_size,
                prefetch_factor=2,
                num_workers=2,
            )

            results[eval_dimension] = {
                "null_set": trainer.test(null_set_loader, enable_model_summary=True),
                "true_set": trainer.test(true_set_loader, enable_model_summary=True),
            }
        elif eval_dimension == "eeg":
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                prefetch_factor=2,
                num_workers=2,
            )
            results[eval_dimension] = trainer.test(
                test_loader, enable_model_summary=True
            )
    return results
