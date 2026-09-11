from __future__ import annotations


FORMAL_PROTOCOL_NAME = "jilv_music_28_v2"
FORMAL_PROTOCOL_FILENAME = "jilv_music_28.json"
FORMAL_SESSION_SPLITS = {
    "S01": "train",
    "S02": "train",
    "S03": "val",
    "S04": "test",
}
FORMAL_SESSION_PURPOSES = {
    "S01": "第一天首次佩戴 / Train",
    "S02": "第一天摘下后重新佩戴 / Train",
    "S03": "第二天重新佩戴 / Validation",
    "S04": "独立重戴且算法冻结后采集 / Final test",
}
FORMAL_SESSION_COUNT = 4
QUALITY_THRESHOLD_VERSION = "emg_qc_250hz_v1"
SESSION_MANIFEST_FILENAME = "SESSION_COLLECTION_READINESS.json"
TIMESTAMP_SOURCE = "pc_reconstructed"


def formal_split(session_id: str) -> str:
    key = session_id.strip().upper()
    try:
        return FORMAL_SESSION_SPLITS[key]
    except KeyError as exc:
        raise ValueError("正式采集只允许 S01–S04") from exc
