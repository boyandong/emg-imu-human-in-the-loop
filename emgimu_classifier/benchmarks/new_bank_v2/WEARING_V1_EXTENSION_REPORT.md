# Independent new-v1 families under electrode shift

The [protocol](WEARING_V1_EXTENSION_PROTOCOL.json) was committed before
outcomes. It extends the exact [new-v2 wearing split](WEARING_REPORT.md):
public LibEMG CIILData eight-channel 200 Hz recordings, five classes and
native-trial averages; each subject's unshifted `training` domain fits
F0v2 Rest thresholds, candidate metadata, scaler and balanced logistic
classifier. Four shifted domains remain held out. Subjects 15–17 validate
the common arm choice; subjects 18–20 are descriptive final users because
they had been inspected by earlier project studies. All 240 parent F0v2
target probabilities replay exactly.

| Arm | Validation F1 | Validation loss | Final F1 | Final loss | Final worst-domain F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| F0v2 | .5775 | 1.2916 | .6170 | 1.0838 | .5505 |
| + scale pattern | .5616 | 1.3173 | .6263 | 1.1117 | .5930 |
| + ring lag | .5837 | 1.0682 | .5868 | **1.0087** | .4386 |
| + correlation spectrum | **.6130** | **1.0500** | .5439 | 1.1213 | .5048 |
| + frequency direction | .4967 | 1.5665 | .5827 | 1.2182 | .5276 |

The prespecified pooled-validation-F1 rule selects correlation spectrum,
which corrects four F0v2 validation errors and creates none. On final users
it corrects three but creates ten: macro-F1 falls by .0731 and log loss
worsens by .0375. This validation/final reversal rejects a general wearing
benefit for the selected new-v1 arm. Ring lag improves log loss on both
phases but final F1 falls .0302 and worst-domain F1 falls .1119. Scale
pattern has a final-only F1 gain, which cannot select an arm. The earlier
new-v2 F2a/F3c positive wearing result remains a distinct, stronger
candidate under its own frozen comparison.

The [runner](wearing_v1_extension_run.py) stores all 1,200 held-domain
[probabilities](WEARING_V1_EXTENSION_PREDICTIONS.csv), six source/target
trial partitions and complete [scores](WEARING_V1_EXTENSION_RESULTS.json).
A separate [read-back analysis](wearing_v1_extension_analysis.py) verifies
80 pooled, subject and domain [family cells](WEARING_V1_EXTENSION_FAMILY.csv),
64 [conditional increments](WEARING_V1_EXTENSION_CONDITIONAL.csv), and
the full 160-cell [error-pair matrix](WEARING_V1_EXTENSION_ERROR.csv).
The [audit](WEARING_V1_EXTENSION_ANALYSIS_AUDIT.json) binds source files
by SHA-256, and the [idempotent exporter](export_wearing_v1_extension.py)
places all 304 cells in the required canonical tables. Canonical
provenance verification covers 58,442 records overall.

This is same-user, public-device before/after-wearing evidence. It does not
show calendar-day transfer, new-user transfer or the user's 250 Hz live
recognition accuracy.
