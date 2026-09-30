# F5c order-two path signature on complete UniBo bouts

The [frozen protocol](F5C_UNIBO_BOUT_PROTOCOL.json) uses the existing
four-muscle processed-200 Hz UniBo complete-bout split. Each user's Days 1–4
fit an inner classifier, Day 5 sets its source-only probability temperature,
and Days 1–5 fit the final classifier. Day 6 is validation; Days 7–8 are a
descriptive final set. Every bout is a contiguous oracle-labelled native
gesture segment of at least one second. Its entire waveform contributes to
32 equal-duration multichannel RMS envelope bins; F5c uses the per-time
L2-normalized, start-centered envelope path, order-one and order-two
signature only, with no absolute-time channel.

The validated G5 feature extractor and its source classifiers are reused
from the [frozen complete-bout parent](../../feature_bank/results/manifests/feature_bank_unibo_sequence_temporal_validation_20260916__run_manifest.json).
The G5 predictions replay that parent's Day 6 and Day 7–8 arrays **exactly**
(maximum absolute error 0 over 1,700 and 3,391 bouts). F5c alone and G5+F5c
use the same fixed balanced source logistic settings and source-only
temperature procedure. All three arms score identical bout IDs and native
equal-user/trial/class/bout weights. The [result](F5C_UNIBO_BOUT_RESULTS.json)
binds parent hashes, per-user trial splits, temperatures and scores to
[15,273 saved predictions](F5C_UNIBO_BOUT_PREDICTIONS.csv). The
[readback tests](../../tests/test_f5c_unibo_bout_delivery.py) check parent
arrays, disjoint trials, probability sums and recomputed scores.

| Phase | Arm | Weighted macro F1 | Log loss | Brier | Minimum user F1 | Minimum posture F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Day 6 validation | G5 | 0.7713 | **0.4012** | 0.0502 | 0.5530 | 0.7544 |
| Day 6 validation | F5c | 0.6007 | 0.8278 | 0.1082 | 0.2799 | 0.5600 |
| Day 6 validation | G5+F5c | **0.8297** | 0.4423 | **0.0431** | **0.6780** | **0.7909** |
| Days 7–8 final | G5 | **0.7538** | 0.8092 | 0.0681 | 0.5236 | 0.7323 |
| Days 7–8 final | F5c | 0.5938 | 0.9977 | 0.1205 | 0.3778 | 0.5598 |
| Days 7–8 final | G5+F5c | 0.7504 | **0.7682** | **0.0667** | **0.5351** | **0.7347** |

Adding F5c improves Day 6 macro F1 by 0.0584 and raises its minimum-user
F1, but validation log loss becomes worse. On Days 7–8, pooled macro F1
falls by 0.0034 while log loss and Brier improve. F5c alone is much weaker
than G5 in both phases. The final observations do not justify changing a
validation-frozen default, so F5c remains an optional research candidate.

The input has oracle gesture onset/offset boundaries. This is an offline
complete-bout representation, not a short-window detector or a live stream
segmenter. Days 7–8 were used in earlier project studies and are not a newly
untouched project-wide holdout. The result does not transfer directly to the
user's eight-channel 250 Hz device.
