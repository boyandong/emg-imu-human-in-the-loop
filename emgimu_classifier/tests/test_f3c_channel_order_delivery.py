"""Read back frozen F3c order counterfactuals and native-trial identities."""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, log_loss

from benchmarks.new_bank_v2.roam_posture_run import sha256


ROOT = Path(__file__).resolve().parents[1] / "benchmarks/new_bank_v2"


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_f3c_order_counterfactuals_use_same_native_trials() -> None:
    protocol_path = ROOT / "F3C_CHANNEL_ORDER_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads((ROOT / "F3C_CHANNEL_ORDER_RESULTS.json").read_text(encoding="utf-8"))
    saved = rows(ROOT / "F3C_CHANNEL_ORDER_PREDICTIONS.csv")
    parent = rows(ROOT / "DOCUMENT_SPATIAL_WEARING_PREDICTIONS.csv")
    original = {(row["phase"], row["trial_id"]): row for row in parent
                if row["arm"] == "F0v2+F3c_document"}
    assert result["protocol_sha256"] == sha256(protocol_path)
    assert result["parent_result_sha256"] == sha256(ROOT / "DOCUMENT_SPATIAL_WEARING_RESULTS.json")
    assert result["parent_prediction_sha256"] == sha256(ROOT / "DOCUMENT_SPATIAL_WEARING_PREDICTIONS.csv")
    assert result["prediction_sha256"] == sha256(ROOT / "F3C_CHANNEL_ORDER_PREDICTIONS.csv")
    assert len(saved) == result["prediction_rows"] == 3120
    orders = result["orders"]
    assert len(orders) == 1 + protocol["random_permutations"] == 13
    assert orders["native_order"] == list(range(8))
    assert len({tuple(order) for order in orders.values()}) == 13
    canonical = np.arange(8)
    dihedral = {tuple(np.roll(canonical, step)) for step in range(8)}
    dihedral |= {tuple(np.roll(canonical[::-1], step)) for step in range(8)}
    assert all(set(order) == set(range(8)) for order in orders.values())
    assert all(tuple(order) not in dihedral for name, order in orders.items()
               if name != "native_order")
    orbits = set()
    for order in orders.values():
        values = np.asarray(order)
        orbit = min(tuple(np.roll(values, step)) for step in range(8))
        reversed_orbit = min(tuple(np.roll(values[::-1], step)) for step in range(8))
        representative = min(orbit, reversed_orbit)
        assert representative not in orbits
        orbits.add(representative)
    assert result["maximum_identity_replay_error"] < 1e-8
    classes = list(range(5))
    for phase in ("validation", "final"):
        trial_sets = []
        for arm in orders:
            selected = [row for row in saved if row["phase"] == phase and row["arm"] == arm]
            assert len(selected) == 120
            trial_sets.append({row["trial_id"] for row in selected})
            y = np.asarray([int(row["label"]) for row in selected])
            p = np.asarray([[float(row[f"p_{c}"]) for c in classes] for row in selected])
            np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-10)
            pooled = result["scores"][phase][arm]["pooled"]
            assert f1_score(y, p.argmax(axis=1), labels=classes, average="macro") == pooled["macro_f1"]
            assert abs(log_loss(y, p, labels=classes) - pooled["log_loss"]) < 1e-12
            if arm == "native_order":
                for row, probabilities in zip(selected, p):
                    earlier = original[(phase, row["trial_id"])]
                    np.testing.assert_allclose(probabilities,
                        [float(earlier[f"p_{c}"]) for c in classes], atol=1e-8)
        assert all(trials == trial_sets[0] for trials in trial_sets)
        identity = result["scores"][phase]["native_order"]["pooled"]
        alternatives = [result["scores"][phase][arm]["pooled"] for arm in orders
                        if arm != "native_order"]
        assert result["ranks"][phase]["identity_f1_rank_high_is_better"] == 1 + sum(
            row["macro_f1"] > identity["macro_f1"] + 1e-12 for row in alternatives)
        assert result["ranks"][phase]["identity_loss_rank_low_is_better"] == 1 + sum(
            row["log_loss"] < identity["log_loss"] - 1e-12 for row in alternatives)
