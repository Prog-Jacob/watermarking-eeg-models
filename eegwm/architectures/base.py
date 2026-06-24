"""The Architecture abstraction.

Each supported model (CCNN, EEGNet, TSCeption) is described by a single
``Architecture`` instance that bundles every piece of architecture-specific
knowledge the rest of the codebase needs: how to build the model, how to build
its dataset, its channel layout, its watermark filter shape and reshape, how to
map a sample back to electrode space, the plot points used for topomaps, and the
transfer-learning surgery variants. Adding a new architecture means writing one
module and registering it, with no changes elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Architecture:
    name: str
    channels: list[str]
    # (rows, cols) of the raw watermark filter before reshaping
    watermark_shape: tuple[int, int]
    # (num_classes, device) -> torch.nn.Module
    model_factory: Callable[[int, str], Any]
    # (common_dataset_kwargs) -> BetterDEAPDataset
    dataset_factory: Callable[[dict], Any]
    # raw filter -> model-shaped filter (e.g. ToGrid / To2d)
    reshape_watermark: Callable[[Any], Any]
    # mean sample tensor -> electrode-space tensor
    back_to_origin: Callable[[Any], Any]
    # () -> {label: index} points to render in topomaps
    plot_points: Callable[[], dict]
    # {"ADDED"|"DENSE"|"ALL": (model) -> model} transfer-learning heads
    transfer_modes: dict[str, Callable[[Any], Any]]

    def build_model(self, num_classes: int, device: str) -> Any:
        return self.model_factory(num_classes, device)

    def build_dataset(self, common_kwargs: dict) -> Any:
        return self.dataset_factory(common_kwargs)

    def apply_transfer(self, mode: str, model: Any) -> Any:
        return self.transfer_modes[mode.upper()](model)
