"""Cryptographic watermark: signature, wonder-filter pattern, and trigger sets.

A verifier identity string is RSA-signed; the signature is hashed to derive a
deterministic binary filter pattern and a target label. Embedding that filter
into EEG samples (true embedding) trains the model to predict the watermark
label, while null embedding leaves the true label, giving a verifiable backdoor.
"""

import rsa
import math
import torch
import random
import hashlib
import numpy as np
from enum import Enum
from base64 import b64encode
from torch.utils.data import Dataset

from eegwm.watermark.encryption import load_keys
from eegwm.constants import (
    OUT_OF_BOUND,
    TRIGGERSET_SEED,
    DEFAULT_TRIGGERSET_SIZE,
    OWNER_IDENTITY,
    NON_OWNER_IDENTITY,
    ATTACKER_IDENTITY,
)


def h1(msg):
    hexa = hashlib.sha224(msg).hexdigest()
    return int(hexa, base=16)


def h2(msg):
    hexa = hashlib.sha256(msg).hexdigest()
    return int(hexa, base=16)


def h3(msg):
    hexa = hashlib.sha384(msg).hexdigest()
    return int(hexa, base=16)


def h4(msg):
    hexa = hashlib.sha512(msg).hexdigest()
    return int(hexa, base=16)


def create_signature(verifier_string, private_key):
    """RSA-sign the verifier identity and return the base64 signature."""
    verifier_string = verifier_string.encode("UTF-8")
    signature = rsa.sign(verifier_string, private_key, "SHA-256")
    return b64encode(signature).decode("UTF-8")


def transform(signature, channels, num_classes, layout="block"):
    """Hash the signature into a deterministic binary filter pattern and label.

    The label, bit count, and bit content are identical across layouts; only the
    placement of the bits differs. ``block`` stacks them in a contiguous corner
    block (the paper baseline); ``scatter`` spreads the same bits across the whole
    grid at deterministic, signature-derived positions.
    """
    pattern = np.zeros(channels)
    label = h1(signature) % num_classes

    pattern_width = math.ceil(channels[0] / 8)
    pattern_height = math.ceil(channels[1] / 8)
    pattern_size = pattern_width * pattern_height
    bits = "{0:b}".format(h2(signature) % (2**pattern_size)).zfill(pattern_size)

    if layout == "scatter":
        rng = random.Random(h3(signature) ^ h4(signature))
        positions = rng.sample(range(channels[0] * channels[1]), pattern_size)
        for bit, flat in zip(bits, positions):
            i, j = divmod(flat, channels[1])
            pattern[i][j] = int(bit) * 2 - 1
    else:
        posi = h3(signature) % (channels[0] - pattern_width + 1)
        posj = h4(signature) % (channels[1] - pattern_height + 1)
        for i in range(pattern_width):
            for j in range(pattern_height):
                pattern[posi + i][posj + j] = int(bits[i * pattern_height + j]) * 2 - 1

    return torch.from_numpy(pattern), label


def get_watermark(
    architecture, verifier_string, private_key, num_classes=16, layout="block"
):
    """Return the model-shaped watermark filter and its target label."""
    from eegwm.architectures import get_architecture

    arch = get_architecture(architecture)
    signature = create_signature(verifier_string, private_key)
    wm_filter, wm_label = transform(
        signature.encode("UTF-8"), arch.watermark_shape, num_classes, layout
    )
    wm_filter = torch.tensor(arch.reshape_watermark(wm_filter), dtype=torch.float32)
    return wm_filter, wm_label


def apply_true_embedding(sample, mask, wm_label, out_of_bound=OUT_OF_BOUND):
    """Stamp the filter onto a sample and relabel it with the watermark label."""
    out_of_bound_vals = mask.expand(sample.shape) * out_of_bound
    sample = torch.where(out_of_bound_vals == 0, sample, out_of_bound_vals)
    sample = torch.where(out_of_bound_vals != 0, -out_of_bound_vals, sample)
    return sample, wm_label


def apply_null_embedding(sample, mask, label, out_of_bound=OUT_OF_BOUND):
    """Stamp the filter positions onto a sample but keep its true label."""
    out_of_bound_vals = mask.expand(sample.shape) * out_of_bound
    sample = torch.where(out_of_bound_vals == 0, sample, out_of_bound_vals)
    sample = torch.where(out_of_bound_vals != 0, out_of_bound_vals, sample)
    return sample, label


class Verifier(Enum):
    CORRECT = OWNER_IDENTITY
    WRONG = NON_OWNER_IDENTITY
    NEW = ATTACKER_IDENTITY


class TriggerSet(Dataset):
    """Dataset of watermark-embedded samples (optionally mixed with clean data)."""

    def __init__(
        self,
        train_set,
        architecture,
        seed=TRIGGERSET_SEED,
        size=DEFAULT_TRIGGERSET_SIZE,
        num_classes=16,
        do_true_embedding=True,
        do_null_embedding=True,
        include_train_set=False,
        verifier=Verifier.CORRECT,
        layout="block",
    ):
        _, private_key = load_keys()
        filter, wm_label = get_watermark(
            architecture, verifier.value, private_key, num_classes, layout
        )

        random.seed(seed)
        train_set = list(train_set)
        results = train_set if include_train_set else []
        trigger_set = random.sample(train_set, min(max(size), len(train_set)))

        if do_true_embedding:
            results.extend(
                [
                    apply_true_embedding(s, filter, wm_label)
                    for s, _ in trigger_set[: size[0]]
                ]
            )
        if do_null_embedding:
            results.extend(
                [apply_null_embedding(s, filter, t) for s, t in trigger_set[: size[1]]]
            )

        random.shuffle(results)
        self.data = torch.stack([s for s, _ in results])
        self.labels = torch.tensor([t for _, t in results], dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
