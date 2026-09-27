# Song signed device-axis arm candidate

The [protocol](SONG_ARM_SIGNED_PROTOCOL.json) was committed as `dea9d27` before this candidate was fitted or scored. This is one fixed, source-only arm feature candidate, motivated by the weak `left` and `backward` recalls in the existing 28-state model. Its current 13-dimensional F6 arm descriptor uses sensor magnitudes for 10 dimensions and signed mean gravity direction for three. The candidate adds three signed gyroscope means and three late-minus-early accelerometer means from each recorded 22-sample IMU window. They remain **device-axis** values, not a calibrated body frame.

The unchanged hand trial probabilities come from marginalizing the frozen [factorized baseline](SONG_ARM_CAL_TRIAL_PREDICTIONS.csv). Only the arm classifier is fitted again: `StandardScaler` plus balanced logistic regression on S01+S02 formal windows, with the same fixed `C=1` and random seed. Candidate arm probabilities are averaged per native formal trial and combined with the unchanged hand marginal into 28 joint probabilities. S03 and S04 formal labels and windows never fit the candidate, scaler, feature definition, or a hyperparameter. The baseline joint probabilities and trial identities are reproduced exactly from the saved artifact.

| Native formal trials | Baseline macro-F1 | Signed candidate macro-F1 | Baseline accuracy | Candidate accuracy | Baseline LogLoss | Candidate LogLoss |
|---|---:|---:|---:|---:|---:|---:|
| S03 validation, 140 | 0.7159 | **0.7999** | 0.7429 | **0.7786** | 1.0553 | **0.9574** |
| S04 descriptive, 144 | 0.5789 | **0.7121** | 0.6458 | **0.7361** | 1.1887 | **1.0659** |

The arm branch corrects 11 baseline arm errors and introduces six new ones on S03; on S04 it corrects 19 and introduces seven. On the 12 native trials per direction, `left` arm recall changes from 4/12 to 9/12 on S03 and 4/12 to 10/12 on S04. `Backward` changes from 5/12 to 8/12 and from 6/12 to 10/12. These paired counts identify where the gain occurred; they do not establish a new-person or new-day effect.

The [284 trial rows](SONG_ARM_SIGNED_TRIAL_PREDICTIONS.csv) retain both 28-class probability vectors, decisions and native trial identities. The [read-back verifier](verify_song_arm_signed.py) checks their SHA-256 binding to the frozen protocol/baseline, exact baseline probability equality, class decisions and independently recomputed accuracy, macro-F1, LogLoss and per-arm recall. It produced [verified output](SONG_ARM_SIGNED_VERIFICATION.json). The focused signed-feature and existing arm-calibration tests pass (4 tests).

This remains exploratory. S04 had been inspected before this experiment, and all four sessions belong to one participant on one day; S01–S03 also failed the formal collection-readiness gate. The result is trial-averaged stable-cue recognition, not continuous-stream or physical-device performance. No live model is replaced. A next step is a fixed continuous causal replay of the candidate using the same global frame grid and event rule as the existing live model, followed by a fresh physical-session test if it passes the software checks.

Reproduce from `emgimu_classifier` with `PYTHONPATH=src;.` and the recorded scientific runtime:

```powershell
D:/miniconda/python.exe -m benchmarks.new_bank_v2.song_arm_signed_study
D:/miniconda/python.exe -m benchmarks.new_bank_v2.verify_song_arm_signed
```
