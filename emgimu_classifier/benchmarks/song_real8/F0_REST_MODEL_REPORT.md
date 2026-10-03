# Matched source-trained F0 Rest-threshold classifiers on Song

The [feature-only replay](F0_REST_NOISE_REPORT.md) showed that the document's Rest-only ZC/SSC/WAMP threshold changes native 250 Hz eight-channel F0 counts. This follow-up asks whether that change helps a classifier. Both arms use the **same** 849 S01/S02 formal source windows, causal filtering, StandardScaler and fixed balanced logistic regression (`C=1`, 2000 maximum iterations, seed 0). Only the F0 threshold source differs: all source windows versus the 213 source Rest windows. No S03/S04 window, label or calibration block fits either feature family, scaler or classifier; no parameter is selected on target results.

Trial probabilities average up to three stable-window predictions within each native formal trial. The class columns remain in protocol order for log loss and Brier calculations.

| Read-only session | Source threshold | Trials | Macro F1 | Log loss | Brier |
|---|---|---:|---:|---:|---:|
| S03 | All source windows | 140 | 0.9359 | 0.3109 | 0.1416 |
| S03 | Rest only | 140 | 0.9277 | 0.3183 | 0.1447 |
| S04 | All source windows | 144 | 0.9068 | 0.4272 | 0.2131 |
| S04 | Rest only | 144 | 0.8919 | 0.4207 | 0.2121 |

On S03 the document-exact candidate loses all three aggregate metrics. On descriptive S04 it loses F1 but slightly improves log loss and Brier. Neutral recall falls from 0.917 to 0.861 on S03 and from 0.944 to 0.889 on S04; pinch recall also falls. These results **do not justify a default F0 change**. They do not invalidate the Rest-only threshold formula; they show that its classifier value is not established here.

The [frozen protocol](F0_REST_MODEL_PROTOCOL.json), [results](F0_REST_MODEL_RESULTS.json) and [568 paired trial predictions](F0_REST_MODEL_TRIAL_PREDICTIONS.csv) are independently readable without the raw HDF5 files. The raw Song sessions are one participant on one day; S01–S03 failed collection readiness, S04 was previously inspected, and stable cue windows are not continuous online gesture recognition. This is a retrospective local comparison, not independent or prospective validation.
