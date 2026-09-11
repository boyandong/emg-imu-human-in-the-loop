from __future__ import annotations

import json
import hashlib
from pathlib import Path

from .models import ProtocolConfig


class ProtocolLoader:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def discover(self) -> dict[str, Path]:
        found: dict[str, Path] = {}
        if not self.directory.exists():
            return found
        for path in sorted(self.directory.glob("*.json")):
            config = self.load(path)
            if config.name in found:
                raise ValueError(f"实验协议名称重复：{config.name}")
            found[config.name] = path
        return found

    def load(self, path_or_name: str | Path) -> ProtocolConfig:
        path = Path(path_or_name)
        if not path.suffix:
            path = self.directory / f"{path}.json"
        elif not path.is_absolute():
            path = self.directory / path
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        allowed = set(ProtocolConfig.__dataclass_fields__)
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"未知实验协议字段：{', '.join(sorted(unknown))}")
        config = ProtocolConfig(**payload)
        config.source_filename = path.name
        config.source_sha256 = hashlib.sha256(raw).hexdigest()
        config.validate()
        return config
