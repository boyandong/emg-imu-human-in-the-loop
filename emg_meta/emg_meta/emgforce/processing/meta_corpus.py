from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .training_preprocessing import PREPROCESSING_VERSION


CORPUS_FILENAME = "discrete_gestures_corpus.csv"
MANIFEST_FILENAME = "training_manifest.json"
CORPUS_COLUMNS = [
    "dataset", "start", "end", "split", "participant", "session",
    "protocol", "prompt_count", "preprocessing_version",
]
META_GESTURES = {
    "index_press", "index_release", "middle_press", "middle_release",
    "thumb_click", "thumb_down", "thumb_in", "thumb_out", "thumb_up",
}
VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True, slots=True)
class TrainingReadiness:
    ready: bool
    problems: tuple[str, ...]
    prompt_count: int = 0
    sample_count: int = 0
    channel_count: int = 0
    sample_rate: float = 0.0
    preprocessing_version: str = ""


@dataclass(frozen=True, slots=True)
class CorpusRebuildResult:
    corpus_path: Path
    sessions: int
    train_sessions: int
    val_sessions: int
    test_sessions: int
    skipped: tuple[str, ...] = ()


def resolve_dataset_split(session_id: str, requested: str = "auto") -> str:
    requested = requested.strip().lower()
    if requested in VALID_SPLITS:
        return requested
    if requested != "auto":
        raise ValueError("数据集划分只能是 auto、train、val 或 test")
    match = re.search(r"(\d{1,3})$", session_id.strip())
    if match is None:
        raise ValueError(
            "自动划分需要场次编号以 01–11 结尾，例如 S01；也可以手动选择划分")
    number = int(match.group(1))
    if 1 <= number <= 8:
        return "train"
    if 9 <= number <= 10:
        return "val"
    if number == 11:
        return "test"
    raise ValueError("每位受试者固定采集 11 轮，自动划分只支持 S01–S11")


def infer_data_root(path: Path) -> Path | None:
    path = Path(path).resolve()
    for parent in path.parents:
        if parent.name.lower() == "data":
            return parent
    return None


def validate_training_export(path: Path) -> TrainingReadiness:
    path = Path(path)
    problems: list[str] = []
    prompt_count = sample_count = channel_count = 0
    sample_rate = 0.0
    version = ""
    if not path.exists():
        return TrainingReadiness(False, (f"训练导出文件不存在：{path}",))
    try:
        with h5py.File(path, "r") as handle:
            missing = {"data", "prompts", "stages", "meta"} - set(handle.keys())
            if missing:
                problems.append("缺少数据集：" + ", ".join(sorted(missing)))
            if "data" not in handle:
                return TrainingReadiness(False, tuple(problems))
            data = handle["data"]
            fields = data.dtype.fields or {}
            if not {"emg", "time"}.issubset(fields):
                problems.append("data 必须包含 emg 和 time 字段")
            else:
                emg_shape = fields["emg"][0].shape
                channel_count = int(emg_shape[0]) if emg_shape else 0
                if channel_count != 8:
                    problems.append(f"Meta 8通道配置要求8通道，实际为{channel_count}")
                sample_count = len(data)
                if sample_count < 2:
                    problems.append("训练数据样本不足")
                else:
                    times = np.asarray(data["time"], dtype=np.float64)
                    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                        problems.append("data.time 必须有限且严格递增")
            task = _text(data.attrs.get("task", ""))
            if task != "discrete_gestures":
                problems.append(f"task 应为 discrete_gestures，实际为 {task or '空'}")
            sample_rate = float(data.attrs.get("sample_rate", 0.0))
            if not np.isclose(sample_rate, 2000.0):
                problems.append(f"采样率应为2000 Hz，实际为{sample_rate:g} Hz")
            version = _text(data.attrs.get("preprocessing_version", ""))
            if version != PREPROCESSING_VERSION:
                problems.append(
                    f"训练预处理版本应为 {PREPROCESSING_VERSION}，实际为 {version or '未预处理'}")
            if "prompts" in handle and "time" in fields and sample_count >= 2:
                prompts = pd.read_hdf(path, "prompts")
                prompt_count = len(prompts)
                if not {"name", "time"}.issubset(prompts.columns):
                    problems.append("prompts 必须包含 name 和 time 列")
                else:
                    unknown = sorted(set(map(str, prompts["name"])) - META_GESTURES)
                    if unknown:
                        problems.append("prompts 包含未知手势：" + ", ".join(unknown))
                    if prompt_count:
                        first = float(data["time"][0]); last = float(data["time"][-1])
                        if not prompts["time"].between(first, last).all():
                            problems.append("存在超出 data.time 范围的 prompt")
                    counts = prompts["name"].value_counts()
                    for finger in ("index", "middle"):
                        if int(counts.get(f"{finger}_press", 0)) != int(
                                counts.get(f"{finger}_release", 0)):
                            problems.append(f"{finger} press/release 数量不成对")
                if prompt_count == 0:
                    problems.append("没有可用于训练的 prompts")
    except Exception as exc:
        problems.append(f"无法读取训练导出：{exc}")
    return TrainingReadiness(
        not problems, tuple(problems), prompt_count, sample_count,
        channel_count, sample_rate, version,
    )


def upsert_corpus_entry(aligned_path: Path, data_root: Path | None = None) -> Path | None:
    aligned_path = Path(aligned_path).resolve()
    root = Path(data_root).resolve() if data_root else infer_data_root(aligned_path)
    if root is None:
        return None
    readiness = validate_training_export(aligned_path)
    if not readiness.ready:
        raise ValueError("训练导出未通过就绪检查：" + "；".join(readiness.problems))
    try:
        relative = aligned_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("训练导出文件必须位于 data_root 内") from exc

    row = _corpus_row(aligned_path, relative, readiness)

    corpus_path = root / CORPUS_FILENAME
    if corpus_path.exists():
        frame = pd.read_csv(corpus_path)
        for column in CORPUS_COLUMNS:
            if column not in frame:
                frame[column] = ""
        frame = frame[frame["dataset"].astype(str) != relative]
    else:
        frame = pd.DataFrame(columns=CORPUS_COLUMNS)
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame = frame[CORPUS_COLUMNS].sort_values(
        ["participant", "session", "dataset"], kind="stable")
    _atomic_csv(frame, corpus_path)
    write_training_manifest(root, frame)
    return corpus_path


def rebuild_corpus(data_root: Path) -> CorpusRebuildResult:
    """Recreate the corpus from every valid aligned export under ``data_root``."""
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise ValueError(f"数据根目录不存在：{root}")

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for aligned_path in sorted(root.rglob("session_meta_aligned.hdf5")):
        readiness = validate_training_export(aligned_path)
        relative = aligned_path.relative_to(root).as_posix()
        if not readiness.ready:
            skipped.append(f"{relative}：{'；'.join(readiness.problems)}")
            continue
        rows.append(_corpus_row(aligned_path, relative, readiness))

    if not rows:
        raise ValueError("没有找到通过训练就绪检查的 session_meta_aligned.hdf5")

    frame = _redistribute_splits(pd.DataFrame(rows, columns=CORPUS_COLUMNS))
    frame = frame.sort_values(
        ["participant", "session", "dataset"], kind="stable")
    corpus_path = root / CORPUS_FILENAME
    _atomic_csv(frame, corpus_path)
    write_training_manifest(root, frame)
    counts = frame["split"].value_counts()
    return CorpusRebuildResult(
        corpus_path, len(frame), int(counts.get("train", 0)),
        int(counts.get("val", 0)), int(counts.get("test", 0)), tuple(skipped))


def refresh_corpus_entry(aligned_path: Path) -> Path | None:
    with h5py.File(aligned_path, "r") as handle:
        meta = handle.get("meta")
        if meta is not None and not bool(meta.attrs.get(
                "corpus_registration_enabled", True)):
            return None
    root = infer_data_root(Path(aligned_path))
    if root is None or not (root / CORPUS_FILENAME).exists():
        return None
    if not validate_training_export(aligned_path).ready:
        return None
    return upsert_corpus_entry(aligned_path, root)


def write_training_manifest(data_root: Path, frame: pd.DataFrame | None = None) -> Path:
    data_root = Path(data_root)
    if frame is None:
        frame = pd.read_csv(data_root / CORPUS_FILENAME)
    splits = {name: int((frame["split"] == name).sum()) for name in sorted(VALID_SPLITS)}
    payload: dict[str, Any] = {
        "format": "meta_discrete_gestures_custom_v1",
        "corpus": CORPUS_FILENAME,
        "preprocessing_version": PREPROCESSING_VERSION,
        "sample_rate_hz": 2000,
        "emg_channels": 8,
        "meta_output_classes": sorted(META_GESTURES),
        "sessions": len(frame),
        "sessions_by_split": splits,
        "prompt_count": int(pd.to_numeric(frame["prompt_count"], errors="coerce").fillna(0).sum()),
        "network_input_channels": 8,
    }
    path = data_root / MANIFEST_FILENAME
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def _redistribute_splits(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the shared, leakage-free S01-S10/S011-S012/S013-S015 split."""
    frame = frame.sort_values(["start", "dataset"], kind="stable").copy()
    frame["split"] = [resolve_dataset_split(str(value)) for value in frame["session"]]
    return frame


def _corpus_row(aligned_path: Path, relative: str,
                readiness: TrainingReadiness) -> dict[str, Any]:
    with h5py.File(aligned_path, "r") as handle:
        data = handle["data"]
        meta = handle["meta"].attrs
        session = _text(meta.get("session_id", ""))
        split = resolve_dataset_split(
            session, _text(meta.get("dataset_split", "auto")))
        return {
            "dataset": relative,
            "start": float(data["time"][0]),
            "end": float(data["time"][-1]),
            "split": split,
            "participant": _text(meta.get("participant_id", "")),
            "session": session,
            "protocol": _text(meta.get("protocol_name", "")),
            "prompt_count": readiness.prompt_count,
            "preprocessing_version": readiness.preprocessing_version,
        }


def _text(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).rstrip(b"\x00").decode("utf-8")
    return str(value)
