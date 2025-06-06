import rsa
import math
import torch
import random
import hashlib
import numpy as np
from enum import Enum
from base64 import b64encode
from torcheeg import transforms
from encryption import load_keys
from torch.utils.data import Dataset
from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT


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
    verifier_string = verifier_string.encode("UTF-8")
    signature = rsa.sign(verifier_string, private_key, "SHA-256")
    return b64encode(signature).decode("UTF-8")


def transform(signature, channels, num_classes):
    filter = np.zeros(channels)
    label = h1(signature) % num_classes

    pattern_width = math.ceil(channels[0] / 8)
    pattern_height = math.ceil(channels[1] / 8)
    pattern_size = pattern_width * pattern_height

    posi = h3(signature) % (channels[0] - pattern_width + 1)
    posj = h4(signature) % (channels[1] - pattern_height + 1)
    bits = "{0:b}".format(h2(signature) % (2**pattern_size)).zfill(pattern_size)

    for i in range(pattern_width):
        for j in range(pattern_height):
            filter[posi + i][posj + j] = int(bits[i * pattern_height + j]) * 2 - 1

    return torch.from_numpy(filter), label


def get_watermark(architecture, verifier_string, private_key):
    signature = create_signature(verifier_string, private_key)
    match architecture:
        case "CCNN":
            filter, wm_label = transform(signature.encode("UTF-8"), (32, 4), 16)
            filter = transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)(eeg=filter)["eeg"]
            filter = torch.tensor(filter, dtype=torch.float32)
            return filter, wm_label

        case "TSCeption":
            filter, wm_label = transform(signature.encode("UTF-8"), (28, 512), 16)
            filter = transforms.To2d()(eeg=filter)["eeg"]
            filter = torch.tensor(filter, dtype=torch.float32)
            return filter, wm_label

        case "EEGNet":
            filter, wm_label = transform(signature.encode("UTF-8"), (32, 128), 16)
            filter = transforms.To2d()(eeg=filter)["eeg"]
            filter = torch.tensor(filter, dtype=torch.float32)
            return filter, wm_label

        case _:
            raise ValueError("Invalid architecture!")


def apply_true_embedding(input, filter, wm_label, out_of_bound=2000):
    out_of_bound_vals = filter.expand(input.shape) * out_of_bound
    input = torch.where(out_of_bound_vals == 0, input, out_of_bound_vals)
    input = torch.where(out_of_bound_vals != 0, -out_of_bound_vals, input)
    return input, wm_label


def apply_null_embedding(input, filter, label, out_of_bound=2000):
    out_of_bound_vals = filter.expand(input.shape) * out_of_bound
    input = torch.where(out_of_bound_vals == 0, input, out_of_bound_vals)
    input = torch.where(out_of_bound_vals != 0, out_of_bound_vals, input)
    return input, label


class Verifier(Enum):
    CORRECT = "Abdelaziz->AHMED a.k.a OWNER<-Fathi @ Feb 15, 2025"
    WRONG = "Abdelaziz->NOT OWNER<-Fathi @ Feb 15, 2025"
    NEW = "Abdelaziz->ATTACKER<-Fathi @ Feb 15, 2025"


class TriggerSet(Dataset):
    def __init__(
        self,
        train_set,
        architecture,
        seed=2036,
        size=(1600, 1600),
        do_true_embedding=True,
        do_null_embedding=True,
        include_train_set=False,
        verifier=Verifier.CORRECT,
    ):
        _, private_key = load_keys()
        filter, wm_label = get_watermark(architecture, verifier.value, private_key)

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
