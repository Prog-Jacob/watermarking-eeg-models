import math
import glob
import random
import numpy as np
from pathlib import Path
from functools import reduce
from torch import manual_seed, cuda
from scipy.interpolate import interp1d


# Data Encoding Utilities
def BinariesToCategory(y):
    return {"y": reduce(lambda acc, num: acc * 2 + num, y, 0)}


# Numerical Methods
def interpolate(xs, ys):
    return interp1d(xs, ys, kind="quadratic", fill_value="extrapolate")


def is_numeric(value):
    try:
        return not math.isnan(float(value))
    except:
        return False


# File Handling Utilities
def list_json_files(dir):
    path = (Path(dir) / "./**/*.json").resolve()
    return glob.glob(str(path), recursive=True)


# Random Seed Configuration
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    manual_seed(seed)
    if cuda.is_available():
        cuda.manual_seed(seed)


# Dictionary Utilities
def are_keys_numeric(dictionary):
    return isinstance(dictionary, dict) and all(
        is_numeric(key) for key in dictionary.keys()
    )


def are_values_numeric(dictionary):
    return isinstance(dictionary, dict) and all(
        is_numeric(value) for value in dictionary.values()
    )


def add_to_dict(dictionary, keys, value):
    last_key = keys.pop()
    for key in keys:
        if key not in dictionary:
            dictionary[key] = dictionary.get(key, {})
        dictionary = dictionary[key]
    dictionary[last_key] = value


def add_key_at_depth(origin, dest, key, depth=100):
    if depth == 0 or not isinstance(origin, dict):
        dest[key] = origin
        return

    for k, v in origin.items():
        dest[k] = dest.get(k, {})
        add_key_at_depth(v, dest[k], key, depth - 1)


# Visualization Utilities
def get_color(index):
    return f"color({index})"


def title(title):
    return title.replace("_", " ").title()


def get_result_panel(key, value):
    return f"{title(key)}: [reset]{(value * 100):.2f}%[/reset]"


def convert_dict_to_tree(dictionary, tree, depth):
    color = get_color(depth)
    if not isinstance(dictionary, dict):
        return tree.add(dictionary, style="reset")

    for key, value in dictionary.items():
        if key.endswith(".json"):
            convert_dict_to_tree(
                value,
                tree,
                depth,
            )
        elif is_numeric(value):
            tree.add(
                get_result_panel(key, value), guide_style=color, style=f"bold {color}"
            )
        else:
            convert_dict_to_tree(
                value,
                tree.add(title(key), guide_style=color, style=f"bold {color}"),
                depth + 1,
            )


def serialize(obj):
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        obj = [serialize(v) for v in obj]
        if isinstance(obj, list):
            try:
                return sorted(obj)
            except:
                return obj
        return obj
    elif hasattr(obj, "__class__") and hasattr(obj, "__dict__"):
        cls_name = f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"
        state = {k: serialize(v) for k, v in vars(obj).items() if not k.startswith("_")}
        return {"__class__": cls_name, "state": state}
    elif callable(obj):
        return f"{obj.__module__}.{obj.__qualname__}"
    else:
        return obj
