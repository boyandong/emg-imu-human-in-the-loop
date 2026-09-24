# Why the guided IMU arm prototypes failed on Song

This is a **post-hoc diagnostic** of the frozen [guided arm study](SONG_ARM_CAL_REPORT.md), not another model search. The [diagnostic code](song_arm_shift_diagnostic.py) reads the same hashed native HDF5 sessions and frozen guided-arm probabilities. It never refits a classifier, scale, prototype, or threshold on S03/S04. Formal labels are used to explain held-out errors; the following observations cannot be used as an unbiased new model selection.

The 13-coordinate F6 descriptor is the same as in the paired study. For each arm label, the diagnostic takes the median descriptor across formal native-trial medians, subtracts that session's pre-formal rest reference, and compares it with the corresponding guided-direction prototype using the **source-frozen** coordinate scale. Only **3 of 7** S03 formal arm centroids, and **5 of 7** S04 centroids, are closest to their own guided prototype. This centroid check describes alignment; it is not a trial accuracy measure.

| Session | True arm | Formal trials | Source arm correct | Guided arm correct | Guided corrected / new arm errors | Nearest guide to formal centroid |
|---|---|---:|---:|---:|---:|---|
| S03 | backward | 12 | 5 | 0 | 0 / 5 | forward |
| S03 | down | 12 | 12 | 12 | 0 / 0 | down |
| S03 | forward | 12 | 12 | 7 | 0 / 5 | forward |
| S03 | left | 12 | 4 | 1 | 0 / 3 | forward |
| S03 | right | 12 | 11 | 0 | 0 / 11 | backward |
| S03 | still | 69 | 52 | 49 | 4 / 7 | still |
| S03 | up | 11 | 11 | 2 | 0 / 9 | left |
| S04 | backward | 12 | 6 | 9 | 5 / 2 | backward |
| S04 | down | 12 | 12 | 7 | 0 / 5 | down |
| S04 | forward | 12 | 10 | 3 | 0 / 7 | forward |
| S04 | left | 12 | 4 | 5 | 3 / 2 | backward |
| S04 | right | 12 | 9 | 4 | 1 / 6 | forward |
| S04 | still | 72 | 48 | 59 | 15 / 4 | still |
| S04 | up | 12 | 12 | 12 | 0 / 0 | up |

S03 `right` loses all 11 source-correct trials, and `up` loses 9 of 11. For S03 `up`, the source-scaled guided versus formal **gravity-component direction cosine is -0.23**, despite the direction name matching; for `right` it is -0.02. In contrast, S04 `up` has cosine 0.99 and retains all 12 correct decisions. This is direct evidence that the relationship between the guided IMU descriptor and the formal cue-labelled descriptor varies even across these same-day sessions. S03 `left` has gravity-component cosine 0.97 yet its formal centroid is nearest `forward`, showing that agreement of one three-coordinate subset alone is insufficient.

The saved arm decisions are replayed against the native trial IDs and labels, and their aggregate accuracies must reproduce the frozen [arm study results](SONG_ARM_CAL_RESULTS.json). All eight calibration blocks and source HDF5 hashes are rechecked. Full pairwise distances, per-arm prediction counts, correction counts and component cosines are in [the machine-readable diagnostic](SONG_ARM_SHIFT_DIAGNOSTIC.json).

The descriptor mismatch is a **plausible mechanism** for the guided prototype errors, not proof of a specific physical cause. These recordings do not independently measure a body-forward axis, movement phase, or re-donning geometry. The guided prototype branch remains unsuitable for deployment on current evidence. A future test needs a measured device/body orientation, repeated guided and formal arm motions, and an independent new-session evaluation; this retrospective diagnostic must not choose a correction on S04.

Reproduce from `emgimu_classifier` with `PYTHONPATH=src;.` using `D:/miniconda/python.exe -m benchmarks.new_bank_v2.song_arm_shift_diagnostic`. Raw recordings remain outside Git at `E:/qxy/emg_meta/emg_meta/data/Song`.
