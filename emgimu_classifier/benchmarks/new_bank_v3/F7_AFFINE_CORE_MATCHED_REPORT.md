# Exact-trial F7 affine-SPD increment to the frozen EPN Core

The source-fitted EPN shortlist `F0+F3_Ring+F2b_CSP+F6_IMU` is not retrained.
Its validation prediction file was hash-checked; the original saved source
state was replayed for final users with **zero validation probability error**
and a byte-identical final prediction file. The new F7 anchor uses only
disjoint target-user calibration trials. For every user and 1/2/5-shot
budget, Core, F7 and the fixed `0.5 Core + 0.5 F7` probability mixture are
scored on exactly the same remaining native trials. A fixed `0.5 Core +
0.5 uniform` mixture controls for softening overconfident Core probabilities.
No mixture weight was selected from held-out labels.
Byte-identical copies of both frozen Core prediction arrays are included in
the result bundle so this paired readback does not require a fresh model fit.

| Phase | Shots/class | Held-out trials | Δ log-loss, Core − Core+F7 | Δ log-loss over uniform control | Δ macro-F1, Core+F7 − Core |
|---|---:|---:|---:|---:|---:|
| Validation | 1 | 432 | +0.1950 | +0.0720 | +0.0288 |
| Validation | 2 | 414 | +0.2206 | +0.0977 | +0.0217 |
| Validation | 5 | 360 | +0.2397 | +0.1109 | +0.0358 |
| Descriptive final | 1 | 432 | +0.1135 | +0.0288 | +0.0164 |
| Descriptive final | 2 | 414 | +0.1489 | +0.0488 | +0.0152 |
| Descriptive final | 5 | 360 | +0.1451 | +0.0554 | +0.0201 |

Positive deltas mean lower loss or higher F1 for the F7 mixture. Brier also
improves in all six cells, including against the uniform control; exact
values, per-subject scores, six-class F1 and 2,412 aligned trial probability
rows are saved in [the result bundle](F7_AFFINE_CORE_MATCHED/results.json).
At validation five-shot, F7 alone is right where Core is wrong on 114/360
trials and wrong where Core is right on 83/360. At descriptive-final five-shot
those counts reverse to 64 and 76, while the mixture still improves pooled
F1 and loss. This supports bounded complementarity, not a universal F7 gain.

The validation users had already helped select the Core, and the final users
were previously inspected in the project. This is a retrospective, matched
public-data increment rather than a pristine prospective confirmation. The
result does not establish a newly trained concatenated-feature model, a
source-CV-optimal fusion weight, an own-device gain or a safe default change.
