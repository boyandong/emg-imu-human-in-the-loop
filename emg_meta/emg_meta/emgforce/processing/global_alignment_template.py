from __future__ import annotations

"""Versioned global templates for cross-session alignment recentering."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from .template_alignment import FEATURE_NAME, METHOD_NAME


GLOBAL_TEMPLATE_VERSION = "global_rerp_template_v1"


@dataclass(frozen=True, slots=True)
class GlobalTemplateBundle:
    templates: dict[str, np.ndarray]
    version: str
    sha256: str
    participant_count: int
    session_count: int


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _template_digest(templates: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(templates):
        value = np.asarray(templates[name], dtype="<f4")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def build_global_template(
    aligned_paths: Iterable[str | Path],
    output: str | Path,
    *,
    minimum_participants: int = 2,
) -> Path:
    """Build participant-balanced grand-average templates from v3 exports."""
    if minimum_participants < 1:
        raise ValueError("minimum_participants 必须大于 0")
    participant_templates: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list))
    session_count = 0
    template_shape: tuple[int, ...] | None = None
    for value in aligned_paths:
        path = Path(value)
        with h5py.File(path, "r") as handle:
            if "alignment_templates" not in handle or "alignment_events" not in handle:
                raise ValueError(f"缺少 v3 对齐模板：{path}")
            events = handle["alignment_events"]
            if _text(events.attrs.get("algorithm", "")) != METHOD_NAME:
                raise ValueError(f"不是 {METHOD_NAME} 导出：{path}")
            if _text(events.attrs.get("feature_extractor", "")) != FEATURE_NAME:
                raise ValueError(f"对齐特征版本不一致：{path}")
            participant = _text(handle["meta"].attrs.get("participant_id", "")).strip()
            if not participant:
                raise ValueError(f"缺少 participant_id：{path}")
            group = handle["alignment_templates"]
            for name in group:
                template = np.asarray(group[name], dtype=np.float32)
                if template_shape is None:
                    template_shape = template.shape
                if template.shape != template_shape:
                    raise ValueError(f"模板形状不一致：{path}/{name}")
                participant_templates[participant][name].append(template)
            session_count += 1

    if len(participant_templates) < minimum_participants:
        raise ValueError(
            f"构建全局模板至少需要 {minimum_participants} 名参与者，"
            f"当前只有 {len(participant_templates)} 名")
    gesture_participant_means: dict[str, list[np.ndarray]] = defaultdict(list)
    for gestures in participant_templates.values():
        for name, values in gestures.items():
            gesture_participant_means[name].append(np.mean(values, axis=0, dtype=np.float64))
    templates = {
        name: np.mean(values, axis=0, dtype=np.float64).astype(np.float32)
        for name, values in gesture_participant_means.items()
        if len(values) >= minimum_participants
    }
    if not templates:
        raise ValueError("没有手势同时满足最小参与者数量")

    target = Path(output)
    if target.exists():
        raise FileExistsError(f"全局模板文件已存在：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    sha256 = _template_digest(templates)
    with h5py.File(target, "x") as handle:
        handle.attrs["version"] = GLOBAL_TEMPLATE_VERSION
        handle.attrs["sha256"] = sha256
        handle.attrs["algorithm"] = METHOD_NAME
        handle.attrs["feature_extractor"] = FEATURE_NAME
        handle.attrs["participant_count"] = len(participant_templates)
        handle.attrs["session_count"] = session_count
        handle.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        group = handle.create_group("templates")
        for name, template in templates.items():
            group.create_dataset(name, data=template, compression="gzip")
    return target


def load_global_template(path: str | Path) -> GlobalTemplateBundle:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if _text(handle.attrs.get("algorithm", "")) != METHOD_NAME:
            raise ValueError("全局模板算法版本与当前对齐器不一致")
        if _text(handle.attrs.get("feature_extractor", "")) != FEATURE_NAME:
            raise ValueError("全局模板特征版本与当前对齐器不一致")
        templates = {
            name: np.asarray(dataset, dtype=np.float32)
            for name, dataset in handle["templates"].items()
        }
        bundle = GlobalTemplateBundle(
            templates=templates,
            version=_text(handle.attrs["version"]),
            sha256=_text(handle.attrs["sha256"]),
            participant_count=int(handle.attrs["participant_count"]),
            session_count=int(handle.attrs["session_count"]),
        )
    if _template_digest(bundle.templates) != bundle.sha256:
        raise ValueError("全局模板内容校验失败")
    return bundle
