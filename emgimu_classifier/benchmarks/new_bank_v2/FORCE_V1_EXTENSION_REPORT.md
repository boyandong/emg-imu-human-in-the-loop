# Independent new-v1 family screen under instructed intensity

The [protocol](FORCE_V1_EXTENSION_PROTOCOL.json) was committed before
outcomes. It holds the [new-v2 force screen](FORCE_REPORT.md) fixed: public
LibEMG eight-channel 1000 Hz recordings, 200 ms windows and native-trial
averages, source subjects 1–6 on Ramp only, validation subjects 7–8 and
descriptive final subjects 9–10 across eleven instructed intensity
conditions. Day/user labels never fit feature state, scaler or classifier
outside the source set. Each of four independent new-v1 family formulas is
added to the exact F0v2 backbone with the same balanced logistic classifier.
All 1,176 F0v2 parent target probabilities replay exactly.

| Arm | Validation F1 | Validation loss | Validation worst intensity F1 | Final F1 | Final loss | Final worst intensity F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F0v2 | **.6295** | 1.3440 | .2089 | .4939 | 1.9177 | .3579 |
| + scale pattern | .6201 | **1.3380** | .1868 | .4780 | 2.0258 | .3829 |
| + ring lag | .6125 | 1.4695 | **.2502** | .4935 | 1.8950 | .4117 |
| + correlation spectrum | .6078 | 1.4018 | .1721 | **.5047** | **1.8635** | **.4209** |
| + frequency direction | .6202 | 1.4452 | .2172 | .4988 | 2.1915 | .4095 |

F0v2 is the prespecified pooled-F1 validation choice. Ring lag improves
the validation worst intensity by .0413 F1 and minimum subject F1 by .0214,
yet reduces pooled F1 by .0170 and worsens log loss by .1254. It remains a
condition specialist rather than a default addition. Scale pattern's small
validation loss improvement reverses on final users. Correlation spectrum's
final F1 and loss gain occurs without a validation gain, so final users do
not promote it. The paired prediction table shows ring lag corrects 23 F0v2
errors but creates 33 on validation; on final it corrects 27 and creates 27.

The [runner](force_v1_extension_run.py) saves all 5,880 arm–trial
[probabilities](FORCE_V1_EXTENSION_PREDICTIONS.csv) and the fixed source/
target identities in [results](FORCE_V1_EXTENSION_RESULTS.json). A separate
[read-back analysis](force_v1_extension_analysis.py) calculates 140 pooled,
subject and condition [family cells](FORCE_V1_EXTENSION_FAMILY.csv), 112
[conditional increment cells](FORCE_V1_EXTENSION_CONDITIONAL.csv), and the
complete 280-cell five-arm [error-pair matrix](FORCE_V1_EXTENSION_ERROR.csv).
It rechecks saved pooled F1, accuracy, log loss and Brier and matches every
native trial across arms. The [analysis audit](FORCE_V1_EXTENSION_ANALYSIS_AUDIT.json)
binds all files by SHA-256. The [idempotent exporter](export_force_v1_extension.py)
places all 532 cells into the required tables; the [delivery audit](FORCE_V1_EXTENSION_DELIVERY_AUDIT.json)
and canonical provenance verifier cover 58,138 records overall.

The intensity labels are instructed conditions, not measured mechanical
force. The small validation and final cohorts and repeated project-wide
inspection limit generalization. This 1000 Hz public-device trial study
cannot establish the user's 250 Hz live recognition accuracy.
