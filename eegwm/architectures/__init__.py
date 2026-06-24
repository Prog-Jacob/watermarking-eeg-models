"""Architecture registry.

Look up the single source of per-architecture knowledge by name. To add an
architecture, create a module exposing an ``ARCHITECTURE`` and register it here.
"""

from eegwm.architectures.base import Architecture
from eegwm.architectures import ccnn, eegnet, tsception

_REGISTRY: dict[str, Architecture] = {
    ccnn.ARCHITECTURE.name: ccnn.ARCHITECTURE,
    eegnet.ARCHITECTURE.name: eegnet.ARCHITECTURE,
    tsception.ARCHITECTURE.name: tsception.ARCHITECTURE,
}


def get_architecture(name: str) -> Architecture:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Invalid architecture: {name}") from None


def architecture_names() -> list[str]:
    return list(_REGISTRY)


__all__ = ["Architecture", "get_architecture", "architecture_names"]
