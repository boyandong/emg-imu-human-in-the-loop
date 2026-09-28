# New v2 family screening and seven-axis evidence vector

This [versioned protocol](FAMILY_SCREEN_PROTOCOL.json) derives a single
Stage-1 screen from four previously frozen prediction matrices. It performs
no fitting, new arm selection or pooling across incompatible class sets.
Every one of 32 dataset–phase–arm pooled score groups replays the saved
experiment metrics. The [256-cell screening table](FAMILY_SCREEN.csv)
retains pooled, subject and controlled-condition macro-F1, accuracy, log
loss, Brier, ten-bin expected calibration error (ECE) and per-class F1.
ECE uses each trial's maximum class probability and predicted correctness;
it is a new descriptive metric, not an earlier selection criterion.
The first summary attempt treated GRABMyo's numeric class codes as strings
inside a library log-loss function, which assumes lexicographic label order.
The derived scorer now indexes probabilities by the frozen class-column
order directly; all 32 pooled log-loss values agree with their parent runs.
No prediction or model was changed.

The [robustness vector](ROBUSTNESS_VECTOR.csv) compares each addition against
the corresponding F0v2 baseline on *identical native trials*. GRABMyo's
`F0` uses the same independently implemented RestNoiseDetailV2 family and is
normalized to `F0v2` only in this summary. Positive ΔF1 means the addition
helps; validation and descriptive final results remain separate.

| Axis / dataset | Family | Validation ΔF1 | Final ΔF1 |
| --- | --- | ---: | ---: |
| Cross-user intensity / LibEMG | F2a | -0.0644 | +0.0418 |
| Cross-user intensity / LibEMG | F3c | -0.0646 | -0.0236 |
| Electrode shift / LibEMG | F2a | +0.1185 | -0.0175 |
| Electrode shift / LibEMG | F3c | +0.1067 | +0.1567 |
| Day / GRABMyo | F2a | -0.0239 | -0.0472 |
| Day / GRABMyo | F3c | -0.0353 | -0.0244 |
| Song one-person same-day supplement | F2a | +0.0219 | +0.0448 |
| Song one-person same-day supplement | F3c | +0.0289 | +0.0448 |

The seven requested vector axes are force, wearing, day, user, posture,
speed and quality. Force, wearing and day have the direct comparisons above.
Independent *user* robustness is N/A: the force screen also changes users,
so its force and user effects cannot be separated. Matching posture, speed
and real-quality experiments for these same new-v2 arms are likewise N/A.
Song is shown separately; its four sessions involve one person and one date,
and cannot fill the day or user axes. These N/A entries are recorded for
both families in the vector rather than treated as zero gains.

F3c is a candidate wearing specialist, not a general backbone: its gains
persist in the public wearing screen but pooled force and day results
regress. F2a's validation and final directions disagree on force and
wearing; the one-person Song gain is exploratory. Source-rest F0v2 remains
the candidate backbone for the two native eight-channel LOFO screens, but
no seven-axis backbone classification is proved. The
[leave-one-family-out result](LOFO_REPORT.md) additionally shows that a
fixed full concatenation is not a stable shared default.

The [audit](FAMILY_SCREEN_AUDIT.json) binds all input prediction hashes,
the [reproducer](family_screen.py) checks saved pooled metrics, and
`python -m benchmarks.new_bank_v2.family_screen --verify` regenerates every
published cell. The [idempotent exporter](export_family_screen_delivery.py)
adds the 256 records to the required `feature_family_results.csv` schema;
its [delivery audit](FAMILY_SCREEN_DELIVERY_AUDIT.json) records source hashes.
All 52,012 current canonical records pass provenance verification.

Across datasets, gestures, signal rates, collection devices and split units
differ. The Song S04, GRABMyo Day3, wearing final users and force final users
have been examined by other project studies and are descriptive here. No
current 250 Hz live-device, cross-person/day or real-quality claim follows
from this vector.
