from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EventMetrics:
    thresholds: tuple[float, ...]
    per_class: dict[str, dict[str, float | int]]
    macro_f1: float
    micro_f1: float
    false_positives_per_minute: float
    hold_success_rate: float


@dataclass(frozen=True)
class PaperFNRMetrics:
    threshold: float
    per_class_fnr: dict[str, float]
    mean_fnr: float
    correct: dict[str, int]
    total: dict[str, int]
    predicted_events: int


def rising_events(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    active = probabilities >= threshold
    previous = np.pad(active[:-1], (1, 0), constant_values=False)
    return np.flatnonzero(active & ~previous)


def _match(predicted: np.ndarray, target: np.ndarray, tolerance: int) -> tuple[int, int, int, list[int]]:
    used: set[int] = set()
    latencies: list[int] = []
    tp = 0
    for prediction in predicted:
        candidates = [(abs(int(prediction) - int(item)), index) for index, item in enumerate(target) if index not in used]
        if not candidates:
            continue
        distance, index = min(candidates)
        if distance <= tolerance:
            used.add(index); tp += 1; latencies.append(int(prediction) - int(target[index]))
    return tp, len(predicted) - tp, len(target) - tp, latencies


def evaluate_events(probabilities: np.ndarray, targets: np.ndarray, labels: tuple[str, ...],
                    thresholds: tuple[float, ...], frame_rate: float = 50.0,
                    tolerance_seconds: float = 0.20) -> EventMetrics:
    tolerance = round(tolerance_seconds * frame_rate)
    rows: dict[str, dict[str, float | int]] = {}
    total_tp = total_fp = total_fn = 0
    for class_index, label in enumerate(labels):
        truth = rising_events(targets[:, class_index], 0.5)
        predicted = rising_events(probabilities[:, class_index], thresholds[class_index])
        tp, fp, fn, latency = _match(predicted, truth, tolerance)
        precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        rows[label] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
                       "recall": recall, "f1": f1, "fnr": fn / max(1, tp + fn),
                       "latency_ms": float(np.mean(latency) * 1000 / frame_rate) if latency else 0.0}
        total_tp += tp; total_fp += fp; total_fn += fn
    macro = float(np.mean([float(row["f1"]) for row in rows.values()]))
    micro_p = total_tp / max(1, total_tp + total_fp); micro_r = total_tp / max(1, total_tp + total_fn)
    duration_min = len(probabilities) / frame_rate / 60.0
    hold_attempts = hold_successes = 0
    for press_name, release_name in (("index_press", "index_release"), ("middle_press", "middle_release")):
        if press_name not in labels or release_name not in labels:
            continue
        press_events = rising_events(probabilities[:, labels.index(press_name)], thresholds[labels.index(press_name)])
        release_events = rising_events(probabilities[:, labels.index(release_name)], thresholds[labels.index(release_name)])
        for press in press_events:
            following = release_events[release_events > press]
            if len(following):
                hold_attempts += 1
                hold_successes += int(following[0] - press >= round(0.5 * frame_rate))
    return EventMetrics(thresholds, rows, macro,
                        2 * micro_p * micro_r / max(1e-12, micro_p + micro_r),
                        total_fp / max(1e-12, duration_min),
                        hold_successes / max(1, hold_attempts))


def search_thresholds(probabilities: np.ndarray, targets: np.ndarray, labels: tuple[str, ...],
                      candidates: tuple[float, ...] = tuple(np.arange(0.10, 0.91, 0.05))) -> tuple[float, ...]:
    selected = []
    for class_index, label in enumerate(labels):
        best = (float("-inf"), 0.5)
        for threshold in candidates:
            metrics = evaluate_events(probabilities[:, [class_index]], targets[:, [class_index]],
                                      (label,), (threshold,))
            score = metrics.macro_f1
            if score > best[0]:
                best = (score, threshold)
        selected.append(float(best[1]))
    return tuple(selected)


def _events_from_probabilities(probabilities: np.ndarray, labels: tuple[str, ...],
                               threshold: float, frame_rate: float) -> list[dict]:
    events = []
    for class_index, name in enumerate(labels):
        for frame in rising_events(probabilities[:, class_index], threshold):
            time = float(frame) / frame_rate
            events.append({"name": name, "time": time, "start": time, "end": time})
    events.sort(key=lambda item: item["time"])
    filtered = []
    releases = {"index_release", "middle_release"}
    for event in events:
        if filtered and event["time"] - filtered[-1]["time"] < 0.05:
            previous = filtered[-1]
            if event["name"] not in releases and previous["name"] not in releases:
                continue
            if event["name"] in releases and event["name"] == previous["name"]:
                continue
        filtered.append(event)
    return filtered


def _events_from_targets(targets: np.ndarray, labels: tuple[str, ...],
                         frame_rate: float) -> list[dict]:
    events = []
    for class_index, name in enumerate(labels):
        for frame in rising_events(targets[:, class_index], 0.5):
            time = float(frame) / frame_rate
            events.append({"name": name, "time": time, "start": time, "end": time})
    return sorted(events, key=lambda item: item["time"])


def _needleman_wunsch(left: list[dict], right: list[dict]) -> list[tuple[int | None, int | None]]:
    """Meta-style time-bounded Needleman-Wunsch with lexicographic costs."""
    rows, columns = len(left) + 1, len(right) + 1
    mismatch = np.zeros((rows, columns)); timing = np.zeros((rows, columns))
    operation = np.zeros((rows, columns), dtype=np.int8)  # 1 left, 2 right, 3 both
    for row in range(1, rows):
        mismatch[row, 0] = row; operation[row, 0] = 1
    for column in range(1, columns):
        mismatch[0, column] = column; operation[0, column] = 2
    for row in range(1, rows):
        for column in range(1, columns):
            litem, ritem = left[row - 1], right[column - 1]
            choices = [
                ((mismatch[row - 1, column] + 1, timing[row - 1, column]), 1),
                ((mismatch[row, column - 1] + 1, timing[row, column - 1]), 2),
            ]
            overlap = not (litem["end"] < ritem["start"] or ritem["end"] < litem["start"])
            if overlap:
                choices.append(((
                    mismatch[row - 1, column - 1] + (litem["name"] != ritem["name"]),
                    timing[row - 1, column - 1] + abs(litem["time"] - ritem["time"]),
                ), 3))
            (best_mismatch, best_timing), best_operation = min(choices, key=lambda item: item[0])
            mismatch[row, column] = best_mismatch; timing[row, column] = best_timing
            operation[row, column] = best_operation
    path = []; row, column = rows - 1, columns - 1
    while row or column:
        op = int(operation[row, column])
        if op == 3:
            path.append((row - 1, column - 1)); row -= 1; column -= 1
        elif op == 1:
            path.append((row - 1, None)); row -= 1
        else:
            path.append((None, column - 1)); column -= 1
    return path[::-1]


def evaluate_paper_fnr(probabilities: np.ndarray, targets: np.ndarray,
                       labels: tuple[str, ...], threshold: float = 0.35,
                       frame_rate: float = 50.0,
                       tolerance: tuple[float, float] = (-0.05, 0.25)) -> PaperFNRMetrics:
    """Paper evaluation: 0.35 crossings, debounce/state filtering, NW and mean FNR."""
    probs = np.asarray(probabilities); truth = np.asarray(targets)
    if probs.ndim == 2:
        probs = probs[None]; truth = truth[None]
    if probs.shape != truth.shape or probs.shape[-1] != len(labels):
        raise ValueError("论文 FNR 评估输入形状不一致")
    correct = {name: 0 for name in labels}; total = {name: 0 for name in labels}
    predicted_count = 0
    for sequence_probs, sequence_truth in zip(probs, truth):
        predictions = _events_from_probabilities(sequence_probs, labels, threshold, frame_rate)
        ground = _events_from_targets(sequence_truth, labels, frame_rate)
        predicted_count += len(predictions)
        for event in ground:
            total[event["name"]] += 1
        padded_ground = [
            {**event, "start": event["time"] + tolerance[0], "end": event["time"] + tolerance[1]}
            for event in ground
        ]
        first_matches = _needleman_wunsch(predictions, padded_ground)
        keep = []; state = "NEUTRAL"
        for prediction_index, ground_index in first_matches:
            if prediction_index is not None:
                name = predictions[prediction_index]["name"]
                if ((state == "NEUTRAL" and name not in {"index_release", "middle_release"})
                        or (state == "INDEX" and name != "middle_release")
                        or (state == "MIDDLE" and name != "index_release")):
                    keep.append(prediction_index)
            if ground_index is not None:
                ground_name = ground[ground_index]["name"]
                state = "INDEX" if ground_name == "index_press" else (
                    "MIDDLE" if ground_name == "middle_press" else "NEUTRAL")
        filtered = [predictions[index] for index in keep]
        for ground_index, prediction_index in _needleman_wunsch(padded_ground, filtered):
            if ground_index is not None and prediction_index is not None:
                name = ground[ground_index]["name"]
                correct[name] += int(name == filtered[prediction_index]["name"])
    per_class = {name: 1.0 - correct[name] / total[name] if total[name] else 0.0 for name in labels}
    active = [per_class[name] for name in labels if total[name]]
    return PaperFNRMetrics(threshold, per_class, float(np.mean(active)) if active else 0.0,
                           correct, total, predicted_count)
