"""Source-frozen structural F9 specificity under public electrode re-wearing."""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v2 import wearing_v1_extension_run as wearing
from emgimu.datasets.electrode_shift import PATH_RE
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.quality_mask_v1 import SourceCalibratedQualityMask
from emgimu.feature_bank.quality_observability import QualityObservabilityFamily


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "F9_WEARING_STRUCTURAL_PROTOCOL.json"
RESULT = ROOT / "F9_WEARING_STRUCTURAL_RESULTS.json"
TRIALS = ROOT / "F9_WEARING_STRUCTURAL_TRIALS.csv"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict]) -> dict:
    accepted = [row for row in rows if not row["rejected"]]
    rejected = [row for row in rows if row["rejected"]]
    return {"trials": len(rows), "accepted": len(accepted), "rejected": len(rejected),
            "coverage": len(accepted) / len(rows) if rows else None,
            "baseline_errors": sum(not row["f0_correct"] for row in rows),
            "correct_rejected": sum(row["f0_correct"] for row in rejected),
            "errors_rejected": sum(not row["f0_correct"] for row in rejected),
            "synthetic_constant_channel_trials_detected":
                sum(row["synthetic_constant_ch1_detected"] for row in rows)}


def evaluate() -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for filename, key in (("WEARING_PROTOCOL.json", "parent_protocol_sha256"),
                          ("WEARING_RESULTS.json", "parent_results_sha256"),
                          ("WEARING_TRIAL_PREDICTIONS.csv", "frozen_predictions_sha256")):
        if sha(ROOT / filename) != protocol[key]:
            raise AssertionError(f"frozen wearing parent changed: {filename}")
    parent = json.loads((ROOT / "WEARING_RESULTS.json").read_text(encoding="utf-8"))
    wearing.check_protocol()
    if sha(wearing.ARCHIVE) != parent["archive_sha256"]:
        raise AssertionError("wearing native archive changed")
    with (ROOT / "WEARING_TRIAL_PREDICTIONS.csv").open(encoding="utf-8", newline="") as stream:
        predictions = {(row["phase"], row["trial_id"]): row for row in csv.DictReader(stream)
                       if row["arm"] == "F0v2"}
    if len(predictions) != 240:
        raise AssertionError("wearing frozen F0v2 trial inventory changed")

    rows, states = [], {}
    for phase in ("validation", "final"):
        for subject in wearing.PROTOCOL[f"{phase}_subjects"]:
            source = wearing.load(wearing.ARCHIVE, subject, (wearing.PROTOCOL["source_domain"],))
            target = wearing.load(wearing.ARCHIVE, subject, tuple(wearing.PROTOCOL["target_domains"]))
            split = parent["split_trial_ids"][f"{phase}_{subject}"]
            if (set(source.trials) != set(split["source"]) or
                    set(target.trials) != set(split["target"]) or
                    set(source.trials) & set(target.trials)):
                raise AssertionError("wearing frozen native source/target split changed")
            if source.batch.channels != 8 or source.batch.sample_rate_hz != 200 or source.batch.emg.shape[1] != 40:
                raise AssertionError("wearing source sensor/window contract changed")
            family = QualityObservabilityFamily(pre_highpass_available=False,
                line_frequency_available=False, ring_topology=False).fit(source.batch)
            mask = SourceCalibratedQualityMask(source_quantile=.995).fit(
                family.transform(source.batch), family.feature_names)
            expected_availability = {"adc_clipping": False, "line_noise": False,
                                     "low_frequency_pre_highpass": False}
            if family.availability_ != expected_availability or mask.available_ != expected_availability:
                raise AssertionError("unattested wearing quality metadata became available")
            frozen = pickle.dumps((family, mask))
            states[f"{phase}_{subject}"] = hashlib.sha256(frozen).hexdigest()
            natural = mask.structural_invalid(family.transform(target.batch), family.feature_names)
            fault = np.asarray(target.batch.emg).copy()
            fault[:, :, 0] = fault[:, :1, 0]
            synthetic = mask.structural_invalid(family.transform(FeatureBatch(fault, 200.)),
                                                family.feature_names)
            if natural.shape != synthetic.shape or natural.shape != (target.batch.windows, 8):
                raise AssertionError("wearing structural observation shape changed")
            trials = np.asarray(target.trials).astype(str)
            labels = np.asarray(target.labels)
            if len(trials) != len(natural):
                raise AssertionError("wearing native window/trial alignment changed")
            for trial_id in sorted(set(trials)):
                selected = trials == trial_id
                if len(set(labels[selected])) != 1:
                    raise AssertionError("wearing trial has mixed labels")
                label = int(labels[selected][0])
                match = PATH_RE.fullmatch(trial_id)
                if match is None or int(match["subject"]) != subject or int(match["label"]) != label:
                    raise AssertionError("wearing native trial path disagrees with labels")
                prediction = predictions[(phase, trial_id)]
                if (int(prediction["subject"]) != subject or prediction["domain"] != match["domain"]
                        or int(prediction["label"]) != label):
                    raise AssertionError("frozen F0v2 trial metadata changed")
                scores = np.array([float(prediction[f"p_{c}"]) for c in wearing.CLASSES])
                if not np.isfinite(scores).all() or abs(scores.sum() - 1.) > 1e-5:
                    raise AssertionError("frozen F0v2 probabilities invalid")
                rows.append({"phase": phase, "subject": subject, "domain": match["domain"],
                             "trial_id": trial_id, "label": label,
                             "f0_correct": int(wearing.CLASSES[int(scores.argmax())]) == label,
                             "windows": int(selected.sum()),
                             "structural_windows": int(np.any(natural[selected], axis=1).sum()),
                             "rejected": bool(np.any(natural[selected])),
                             "synthetic_constant_ch1_detected": bool(np.all(synthetic[selected, 0]))})
            if frozen != pickle.dumps((family, mask)):
                raise AssertionError("target transformed source-fitted wearing F9 state")
            print(f"wearing structural F9 {phase} subject {subject}: {len(set(trials))} trials", flush=True)
    if len(rows) != 240 or {(r["phase"], r["trial_id"]) for r in rows} != set(predictions):
        raise AssertionError("wearing structural rows do not cover frozen F0v2 trials")
    with TRIALS.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {"status": "public_wearing_exploratory_structural_diagnostic_not_deployed",
              "protocol_sha256": sha(PROTOCOL),
              "parent_results_sha256": protocol["parent_results_sha256"],
              "archive_sha256": parent["archive_sha256"],
              "frozen_predictions_sha256": protocol["frozen_predictions_sha256"],
              "source_states_sha256": states,
              "availability": expected_availability,
              "trial_rows_sha256": sha(TRIALS),
              "phases": {phase: {"pooled": summarize([r for r in rows if r["phase"] == phase]),
                                 "domains": {domain: summarize([r for r in rows
                                                                  if r["phase"] == phase and r["domain"] == domain])
                                             for domain in wearing.PROTOCOL["target_domains"]}}
                         for phase in ("validation", "final")},
              "scope": protocol["limits"]}
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for phase in ("validation", "final"):
        item = result["phases"][phase]["pooled"]
        print(f"wearing structural F9 {phase}: coverage={item['coverage']:.3f}; "
              f"errors rejected={item['errors_rejected']}; correct rejected={item['correct_rejected']}", flush=True)
    return result


if __name__ == "__main__":
    evaluate()
