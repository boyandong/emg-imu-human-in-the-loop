# Independent new-v1 family screen on GRABMyo unseen users

The [protocol](GRAB_V1_EXTENSION_PROTOCOL.json) was committed before scores
were computed. It reuses the official-SHA-256-verified Day1 subset and the
subject-disjoint split of the [frozen GRAB user study](GRAB_USER_REPORT.md):
users 1–4 fit source Rest state, scaler and classifier; 5–6 validate; 7–8
provide descriptive final scores. Each of 224 native eight-channel 2048 Hz
recordings contributes twenty disjoint 250 ms windows. Four independently
implemented `new_bank_v1.py` feature families are each added to F0v2, with
the same source-only logistic classifier recipe. These are not reconstructions
of unavailable historical formulas.

| Arm | Validation F1 | Validation loss | Validation min-user F1 | Final F1 | Final loss | Final min-user F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F0v2 | 0.8054 | 0.4650 | 0.7099 | **0.9458** | **0.1994** | **0.8877** |
| + scale pattern | 0.7481 | 0.5478 | 0.5909 | 0.8885 | 0.4036 | 0.7551 |
| + ring lag | **0.8360** | **0.4274** | 0.7018 | 0.8899 | 0.2989 | 0.7594 |
| + correlation spectrum | 0.7784 | 0.5754 | 0.6399 | 0.9270 | 0.2150 | 0.8497 |
| + frequency direction | 0.8002 | 0.4719 | **0.7860** | 0.8401 | 0.4048 | 0.7312 |

The prespecified pooled validation F1 rule selects **F0v2+ring_lag** on
this split. Relative to F0v2 it corrects two and introduces zero validation
recording errors. On the two descriptive final users it corrects zero and
introduces three errors; final F1 falls by 0.0559. It also lowers the
minimum-user F1 in both splits. Therefore the validation-selected gain is a
local cross-user screening result, not a stable deployment gain. Frequency
direction raises the minimum validation-user F1 while losing pooled F1,
another objective-dependent specialist signal.

The [ROAM posture extension](ROAM_V1_EXTENSION_REPORT.md) uses 200 Hz
public data and the same candidate definitions but a different population,
class task and feature sampling configuration. There, ring lag increases the
worst posture F1 while decreasing pooled validation F1. Across these two
public axes the candidate has conditional value but no consistent pooled
advantage; it requires an explicit worst-condition objective and further
independent validation before promotion.

The [saved result](GRAB_V1_EXTENSION_RESULTS.json) binds the official archive
manifest, protocol and 560 [target prediction rows](GRAB_V1_EXTENSION_PREDICTIONS.csv)
by SHA-256. All 112 F0v2 target probabilities replay the frozen parent
within absolute `1e-10`. The
[read-back test](../../tests/test_v1_extension_delivery.py) independently
recomputes every pooled score from saved probabilities, checks native IDs,
probability validity, disjoint source/target partitions and the parent
baseline replay for both GRAB and ROAM screens. Previously inspected GRAB
final subjects are descriptive only. The 2048 Hz public-device result does
not measure the user's 250 Hz device, live transitions or real quality faults.
