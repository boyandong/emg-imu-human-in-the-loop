from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class SessionPaths:
    directory: Path
    hdf5: Path
    aligned_hdf5: Path
    config: Path
    log: Path
    readiness_manifest: Path


def build_session_paths(data_root: Path, participant_id: str, session_id: str,
                        when: datetime | None = None) -> SessionPaths:
    participant = participant_id.strip()
    session = session_id.strip()
    if not SAFE_ID.fullmatch(participant) or not SAFE_ID.fullmatch(session):
        raise ValueError("参与者编号和场次编号仅允许字母、数字、点、下划线和连字符")
    date = (when or datetime.now()).strftime("%Y-%m-%d")
    directory = Path(data_root) / participant / f"{date}_{session}"
    return SessionPaths(directory, directory / "session.h5",
                        directory / "session_meta_aligned.hdf5",
                        directory / "session_config.json", directory / "app.log",
                        directory / "SESSION_COLLECTION_READINESS.json")
