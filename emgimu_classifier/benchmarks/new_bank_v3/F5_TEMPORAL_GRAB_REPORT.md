# Versioned F5 window trajectory: frozen GRABMyo unseen-user screen

The [protocol](F5_TEMPORAL_GRAB_PROTOCOL.json) changes only the added F5
feature block relative to the frozen F0v2 parent. The new version follows the
goal's raw early/late RMS ratio, late-minus-early difference, envelope slope
per second, peak index/T, unnormalized entropy and spatial-map velocity. It
does not replace the validated four-channel UniBo G5 or complete-bout DTW.

Both arms use 112 Day-1 source native trials. Users 5–6 provide 56 validation
trials and users 7–8 provide 56 descriptive final trials. Raw files match the
official hashes; the parent F0v2 probabilities replay with zero difference.
No target trial fits an envelope parameter, scaler, or classifier.

| Arm | Validation macro-F1 | Validation log loss | Final macro-F1 | Final log loss |
| --- | ---: | ---: | ---: | ---: |
| F0v2 | 0.8054 | 0.4650 | 0.9458 | 0.1994 |
| F0v2 + document F5 | 0.7852 | 0.5150 | 0.8621 | 0.4035 |

The F5 addition worsens macro-F1 and log loss on both target groups. Validation
Brier falls slightly from 0.2854 to 0.2754, while final Brier rises from
0.1074 to 0.2033; there is no consistent predictive gain. Keep the feature
opt-in and retain the existing public default. The eight-sample known-waveform
test checks all 15 coordinates independently, including the formula's raw
ratio and entropy; zero and constant windows do not acquire padded-edge
activity. These are public 2048 Hz forearm recordings, not the user's 250 Hz
hardware or a fresh prospective final cohort.

See [results](F5_TEMPORAL_GRAB_RESULTS.json) and
[paired trial probabilities](F5_TEMPORAL_GRAB_PREDICTIONS.csv). Reproduce from
`emgimu_classifier` with `PYTHONPATH=src;.` using
`D:/miniconda/python.exe -m benchmarks.new_bank_v3.f5_temporal_grab_run`.
