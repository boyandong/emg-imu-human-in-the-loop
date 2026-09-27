# Song 28-state signed IMU candidate

This is an opt-in experimental model for native 8-channel EMG at 250 Hz and six-axis IMU at 112 Hz. In the application's “实时识别” page, refresh the model list and choose **[Song EMG+IMU 28 类] 有符号 IMU 候选**. The original “手势×手臂” model remains a separate choice. The candidate uses the same source-trained hand branch and adds six signed device-axis quantities to the arm branch. It uses no on-device calibration.

The [research report](../../../../../emgimu_classifier/benchmarks/new_bank_v2/SONG_ARM_SIGNED_CONTINUOUS_REPORT.md) compares its full-stream replay with the original model on S03/S04. The [replay audit](song_joint28_replay_audit.json) verifies the exported bundle against all 25,360 saved native frames. Both recordings are from one person on one day. Physical live accuracy, new-day/electrode replacement behavior, independent action timing and end-to-end latency remain unverified; use the diagnostic recording feature during the next real-device comparison.

For a paired comparison on **one** new device recording, choose either 28-state model and load it, then start diagnostic recording before starting recognition. In the page's action selector, mark the start and end of each intended action, including explicit `still_neutral` rest periods, and finish the diagnostic recording. Copy the saved diagnostic directory path shown in the page. From `collection/emg_meta/emg_meta`, run:

```powershell
$env:PYTHONPATH='.;third_party/generic-neuromotor-interface'
D:/miniconda/envs/emgforce/python.exe -m emgforce.inference.compare_song_joint28_capture '<diagnostic-directory>'
```

The command replays **both** frozen models on the same hashed raw EMG/IMU capture and writes `paired_song28_analysis.json`, `paired_song28_frames.csv`, and `paired_song28_verification.json` beside it. It scores only complete windows inside each manually marked action after excluding 0.4 seconds at both ends. These are intended-action agreement scores, not physiological onset or formal live accuracy. The capture must contain aligned IMU sample indices; a missing alignment is reported rather than guessed.
