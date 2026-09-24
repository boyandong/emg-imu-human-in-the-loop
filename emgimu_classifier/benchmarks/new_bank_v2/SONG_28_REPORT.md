# Independent v2 families on Song's native 28 states

The [28-state protocol](SONG_28_PROTOCOL.json) and [runner](song_28_run.py) were frozen in commit `57473c5` before outcomes were computed. This follows the [four-hand-state Song screen](SONG_REPORT.md), whose validation-best F0v2+F2a+F3c combination improved offline hand classification. Here the endpoint is harder and closer to the project goal: all seven native arm cues × four hand states, with the recorded 112 Hz IMU present in every arm. The experiment tests whether that hand-only gain transfers to joint recognition; no arm is selected using S04.

S01+S02 alone fit feature state and the balanced logistic models. S03 selects among the four v2 arms by trial macro-F1; S04 is descriptive. The same 50-sample causal EMG and aligned 22-sample IMU windows are averaged into one 28-class probability vector per formal trial. There are 140 S03 and 144 S04 trials. The saved causal F0+IMU baseline's S03/S04 joint accuracy and macro-F1 are exactly reproduced, and all four recording SHA-256 digests match. Reproduction requires the original Python 3.13.9, NumPy 2.4.3, SciPy 1.17.1, scikit-learn 1.8.0 runtime. With the separate Python 3.11/NumPy 1.26/SciPy 1.14 environment, the S04 baseline differs by one trial; the older SPD comparison likewise fails its baseline check there. No cross-runtime number is mixed into this table.

As a sensitivity check, the four-hand-state v2 runner was also rerun under the original 28-state runtime into a temporary output directory. All 1,420 corresponding trial rows retained the same labels and order; the largest probability difference was 3.72e-6, every reported v2 macro-F1 was unchanged, and S03 selected the same arm. Thus the hand-only versus joint-endpoint contrast is not a change in the measured macro-F1 caused by the two installed runtimes.

| Arm | S03 joint macro-F1 | S03 log loss | S04 joint macro-F1 | S04 log loss | S04 arm accuracy | S04 hand accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Previous F0 + real IMU, exact metric replay | 0.5958 | 1.1316 | **0.5487** | 1.2966 | 0.7431 | 0.9167 |
| F0v2 + real IMU | **0.6125** | 1.1597 | 0.5365 | 1.2373 | **0.7639** | 0.9167 |
| F0v2 + IMU + F2a | 0.5928 | 1.1266 | 0.5563 | 1.2480 | 0.7361 | **0.9236** |
| F0v2 + IMU + F3c | 0.5678 | **1.0497** | 0.4973 | **1.1797** | 0.7500 | 0.8889 |
| F0v2 + IMU + F2a + F3c | 0.5886 | 1.0684 | 0.5443 | 1.2472 | 0.7500 | 0.9167 |

S03 selects **F0v2+IMU**, without either new spatial family. It scores 0.5365 joint macro-F1 on S04, below the previous F0+IMU baseline's 0.5487. The F2a and F3c arms both correct some baseline trial errors but create more new errors on S03. F3c lowers log loss on both sessions while hurting joint decisions, so it remains a calibration/representation candidate, not a selected classifier addition. The F2a arm's S04 F1 of 0.5563 is a final-only observation and cannot reverse the S03 selection.

The [read-back verifier](verify_song_28.py) recomputes ten metric groups from all [1,420 saved trial probability rows](SONG_28_TRIAL_PREDICTIONS.csv), checks 28-class alignment and native-trial coverage, reproduces baseline metrics and runtime, and verifies the selected arm. Full precision [results](SONG_28_RESULTS.json) and [verification](SONG_28_VERIFICATION.json) are retained. This is a one-person, one-day cue-labelled stable-window analysis; S01–S03 formal collection readiness failed and S04 was already inspected in previous studies. It cannot establish new-day/device transfer, physiological action onset, online latency or a reason to replace the current live model.

From `emgimu_classifier` with `PYTHONPATH=src;.`, run `D:/miniconda/python.exe -m benchmarks.new_bank_v2.song_28_run` and `D:/miniconda/python.exe -m benchmarks.new_bank_v2.verify_song_28`. Raw HDF5 files stay outside Git at `E:/qxy/emg_meta/emg_meta/data/Song`.
