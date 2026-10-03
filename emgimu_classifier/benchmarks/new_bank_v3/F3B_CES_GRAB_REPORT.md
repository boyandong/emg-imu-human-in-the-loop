# Versioned F3b CES: frozen GRABMyo unseen-user screen

The [protocol](F3B_CES_GRAB_PROTOCOL.json) adds only the new eight-coordinate
channel-correlation eigenvalue spectrum to F0v2. It uses the same 112 source
Day-1 native trials, 56 validation trials and 56 descriptive final trials as
the frozen GRAB unseen-user parent. The official raw-file hashes and every
parent F0v2 probability are rechecked; the maximum parent replay difference is
zero. Source users 1–4 alone fit the feature state and classifier. Target users
5–8 do not influence the envelope width, scaling or model.

| Arm | Validation macro-F1 | Validation log loss | Final macro-F1 | Final log loss |
| --- | ---: | ---: | ---: | ---: |
| F0v2 | 0.8054 | 0.4650 | 0.9458 | 0.1994 |
| F0v2 + F3b CES | 0.7784 | 0.5739 | 0.9116 | 0.2841 |

F3b worsens both primary metrics on both target groups. The validation
minimum-user macro-F1 remains 0.7099; the descriptive final minimum falls
from 0.8877 to 0.8213. It is therefore an opt-in formula implementation,
not a promoted default. The known 3+1 correlation spectrum and channel
permutation tests establish the formula independently; a constant-channel
test prevents zero-padded smoothing edges from creating false coordination.
This is not the missing historical CES implementation or a new prospective
final cohort. See [results](F3B_CES_GRAB_RESULTS.json) and
[trial probabilities](F3B_CES_GRAB_PREDICTIONS.csv).

Reproduce with `PYTHONPATH=src;.` from `emgimu_classifier`:
`D:/miniconda/python.exe -m benchmarks.new_bank_v3.f3b_ces_grab_run`.
The public GRAB archive remains outside Git.
