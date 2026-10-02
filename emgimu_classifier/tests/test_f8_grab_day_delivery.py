"""Independent readback of the fixed public cross-day F8 fusion control."""
import csv
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
from sklearn.metrics import f1_score, log_loss


ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "new_bank_v2"
CLASSES = np.asarray([4, 15, 16, 17])
PROVIDERS = ("F0v2", "F0v2+frequency_direction")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class F8GrabDayDeliveryTest(unittest.TestCase):
    def test_frozen_trials_weights_predictions_and_scores(self):
        protocol = json.loads((ROOT / "F8_GRAB_DAY_PROTOCOL.json").read_text())
        result = json.loads((ROOT / "F8_GRAB_DAY_RESULTS.json").read_text())
        assert result["protocol_sha256"] == sha(ROOT / "F8_GRAB_DAY_PROTOCOL.json")
        assert result["parent_prediction_sha256"] == protocol["parent_predictions_sha256"]
        assert result["prediction_sha256"] == sha(ROOT / "F8_GRAB_DAY_PREDICTIONS.csv")
        assert result["parent_probability_max_abs_error"] <= 1e-8
        with (ROOT / "GRAB_DAY_V1_EXTENSION_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
            parent = {(row["arm"], row["trial_id"]): row for row in csv.DictReader(stream)
                      if row["arm"] in PROVIDERS}
        with (ROOT / "F8_GRAB_DAY_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert len(parent) == 896 and len(rows) == 768 and len(result["blocks"]) == 16
        indexed = {(row["phase"], row["method"], row["trial_id"]): row for row in rows}
        assert len(indexed) == len(rows)
        for block in result["blocks"]:
            phase = block["phase"]
            user = int(block["subject"])
            source = set(block["source_trial_ids"])
            calibration = set(block["calibration_trial_ids"])
            evaluation = set(block["evaluation_trial_ids"])
            assert (len(source), len(calibration), len(evaluation)) == (28, 4, 24)
            assert not source & calibration and not source & evaluation and not calibration & evaluation
            day = 2 if phase == "validation" else 3
            for trial_id in calibration:
                assert trial_id.startswith(f"session{day}_participant{user}_")
                assert trial_id.endswith("_trial1")
            for trial_id in evaluation:
                assert trial_id.startswith(f"session{day}_participant{user}_")
                assert any(trial_id.endswith(f"_trial{number}") for number in range(2, 8))
            score = []
            for name in PROVIDERS:
                vector = np.asarray(block["f8_vectors"][name])
                assert vector.shape == (14,)
                score.append(np.clip((vector[4:8].mean() + 1) / 2, 0.05, 1))
            weight = np.asarray(score) / sum(score)
            np.testing.assert_allclose(weight, block["f8_weights"], atol=1e-7, rtol=0)
            for trial_id in evaluation:
                prior = [parent[(name, trial_id)] for name in PROVIDERS]
                expected_y = int(prior[0]["gesture"])
                assert all(int(item["gesture"]) == expected_y for item in prior)
                inputs = np.stack([[float(item[f"p_{c}"]) for c in CLASSES] for item in prior])
                for method, weights in (("uniform", np.asarray([.5, .5])), ("f8_cosine", weight)):
                    saved = indexed[(phase, method, trial_id)]
                    assert int(saved["gesture"]) == expected_y and int(saved["subject"]) == user
                    actual = np.asarray([float(saved[f"p_{c}"]) for c in CLASSES])
                    np.testing.assert_allclose(actual, weights @ inputs, atol=1e-7, rtol=0)
                    self.assertAlmostEqual(float(actual.sum()), 1.0, places=12)
        for phase in ("validation", "descriptive_final"):
            for method in ("uniform", "f8_cosine"):
                subset = [row for row in rows if row["phase"] == phase and row["method"] == method]
                assert len(subset) == 192
                y = np.asarray([int(row["gesture"]) for row in subset])
                subject = np.asarray([int(row["subject"]) for row in subset])
                probability = np.asarray([[float(row[f"p_{c}"]) for c in CLASSES] for row in subset])
                prediction = CLASSES[np.argmax(probability, axis=1)]
                saved = result["scores"][phase][method]
                self.assertAlmostEqual(f1_score(y, prediction, labels=CLASSES,
                                                average="macro", zero_division=0), saved["macro_f1"])
                self.assertAlmostEqual(log_loss(y, probability, labels=CLASSES), saved["log_loss"])
                one_hot = (y[:, None] == CLASSES[None, :]).astype(float)
                self.assertAlmostEqual(np.mean(np.sum((probability - one_hot) ** 2, axis=1)),
                                       saved["brier"])
                minimum = min(f1_score(y[subject == user], prediction[subject == user],
                                       labels=CLASSES, average="macro", zero_division=0)
                              for user in range(1, 9))
                self.assertAlmostEqual(minimum, saved["minimum_subject_macro_f1"])


if __name__ == "__main__":
    unittest.main()
