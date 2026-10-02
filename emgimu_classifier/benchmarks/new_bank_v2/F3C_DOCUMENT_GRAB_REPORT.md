# Historical uncentered F3c on the GRABMyo forearm ring

**Formula correction (V3):** The goal file's F2a is centered before F3c
summarizes its matrix. This V2 run used an uncentered second moment; its
probabilities remain a valid alternative-feature result, but “document-
consistent” in the historical title is not an exact-formula claim. See the
[separate centered V3 replay](../new_bank_v3/SPEC_SPATIAL_GRAB_REPORT.md).

The [official dataset description](https://physionet.org/content/grabmyo/1.1.0/)
states that forearm F1–F8 are the eight channels of ring 1, sampled at 2048 Hz.
Its [electrode-location diagram](https://physionet.org/files/grabmyo/1.1.0/Electrodelocation.pdf)
(SHA-256 `08d83b812c8bba2dcace803f3b9fe9c99c44b38d1605d8838bf927708d3ebb21`)
visibly places electrodes 8, 1 and 2 as neighbors on the exposed side of the
forearm. The other five positions wrap around the hidden side; their exact
neighbor relations are inferred from the sequential numbering, not directly
visible. The topology assertion in this experiment therefore has a stronger
physical basis than the Myo CSV study but is still conditional on that full
ordering inference.

The [frozen protocol](F3C_DOCUMENT_GRAB_PROTOCOL.json) adds V2 uncentered,
uncentered F3c to the existing public GRAB three-day, eight-subject, four-class
trial split. It uses 512-sample disjoint windows, source-Day1 Rest thresholds,
source-only feature and classifier fitting, and the prior classifier settings.
All 1,344 selected source files pass the official SHA-256 manifest check. A
fresh F0 baseline reproduces all 448 held-out parent probabilities within
`1e-8` absolute error. The new F3c arm saves 448 further held-out probabilities
in [the prediction table](F3C_DOCUMENT_GRAB_PREDICTIONS.csv); an independent
readback checks trial identities, class counts, probability sums, F1, accuracy,
log loss, per-class recall and per-subject F1.

| Arm | Day2 validation macro-F1 | Day2 log loss | Day3 descriptive macro-F1 | Day3 log loss |
| --- | ---: | ---: | ---: | ---: |
| F0 | 0.9551 | 0.1507 | 0.9008 | 0.3007 |
| F0 + document F3c | 0.9198 | 0.1712 | 0.8764 | 0.3873 |

Compared with the earlier centered-F3c GRAB probabilities, the V2 uncentered
arm changes probability values by at most 0.014665 but changes none of the 448
held-out decisions. The document candidate worsens both selected metrics on
Day2 and Day3. Thus the positive wearing-shift result does not generalize to
this public cross-day axis, and F3c is not promoted as a default. The Day3
result is descriptive because this cohort was inspected in earlier studies.
This result says nothing about the user's 250 Hz device or live recognition.
