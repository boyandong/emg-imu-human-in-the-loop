from __future__ import annotations

from .base import AdapterResult, DatasetAdapter
from .unibo_inail import UniBoInailAdapter


ADAPTERS: dict[str, DatasetAdapter] = {
    "unibo-inail": UniBoInailAdapter(),
}


def get_adapter(name: str) -> DatasetAdapter:
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        raise KeyError(f"unknown benchmark adapter {name!r}; choose from {sorted(ADAPTERS)}") from exc


__all__ = ["ADAPTERS", "AdapterResult", "DatasetAdapter", "UniBoInailAdapter", "get_adapter"]
