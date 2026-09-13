from __future__ import annotations

from .base import AdapterResult, DatasetAdapter
from .unibo_inail import UniBoInailAdapter
from .myo_armband import MyoArmbandEvaluationAdapter
from .grabmyo import GrabMyoAdapter


ADAPTERS: dict[str, DatasetAdapter] = {
    "unibo-inail": UniBoInailAdapter(),
    "myo-armband-evaluation": MyoArmbandEvaluationAdapter(),
    "grabmyo": GrabMyoAdapter(),
}


def get_adapter(name: str) -> DatasetAdapter:
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        raise KeyError(f"unknown benchmark adapter {name!r}; choose from {sorted(ADAPTERS)}") from exc


__all__ = [
    "ADAPTERS", "AdapterResult", "DatasetAdapter", "GrabMyoAdapter", "MyoArmbandEvaluationAdapter",
    "UniBoInailAdapter", "get_adapter",
]
