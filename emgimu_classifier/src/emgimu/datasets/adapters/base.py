from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class AdapterResult:
    dataset_id: str
    output_root: Path
    manifest_path: Path
    trial_count: int
    warnings: tuple[str, ...]


class DatasetAdapter(Protocol):
    dataset_id: str

    def adapt(self, source_root: str | Path, output_root: str | Path) -> AdapterResult:
        """Convert an official source tree into the canonical benchmark format."""
