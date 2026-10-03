# F7 document-exact personal-anchor replay

This opt-in V3 implementation matches the goal document's additive-epsilon denominators for standardized Euclidean distance, cosine distance and normalized margin. The earlier `PersonalAnchor` and its historical results are preserved. A near-zero-scale independent numerical oracle covers the formula difference.

The frozen [protocol](F7_DOCUMENT_ANCHOR_PROTOCOL.json) reuses the Day1-fitted F0v2 trial features and population predictions from the public eight-channel GRABMyo study. On each target subject/day, native repetitions 1–N of all four gestures fit the anchor; repetitions 6–7 are evaluated. Day2 is validation and Day3 is descriptive final. The anchor's temperature comes only from calibration rows. The standalone anchor and a predetermined equal-weight mixture with F0v2 were scored without classifier refitting or target-label selection. All 1,152 prediction rows and 48 calibration-cell identities are saved in [results](F7_DOCUMENT_ANCHOR_RESULTS.json) and [predictions](F7_DOCUMENT_ANCHOR_PREDICTIONS.csv).

| Phase | Shots/class | F0v2 macro F1 / log loss | F7 macro F1 / log loss | Equal mix macro F1 / log loss |
| --- | ---: | ---: | ---: | ---: |
| Day2 validation | 1 | 0.9687 / 0.1464 | 0.6278 / 0.8859 | 0.9687 / 0.4312 |
| Day2 validation | 2 | 0.9687 / 0.1464 | 0.7409 / 0.8953 | 0.9687 / 0.4359 |
| Day2 validation | 5 | 0.9687 / 0.1464 | 0.7639 / 0.8620 | 0.9687 / 0.4290 |
| Day3 descriptive | 1 | 0.9210 / 0.2349 | 0.6463 / 0.8957 | 0.9530 / 0.4854 |
| Day3 descriptive | 2 | 0.9210 / 0.2349 | 0.7988 / 0.8594 | 0.9363 / 0.4693 |
| Day3 descriptive | 5 | 0.9210 / 0.2349 | 0.7494 / 0.8788 | 0.9210 / 0.4768 |

The fixed mixture fails the Day2 log-loss guard at every budget despite preserving its macro F1. Day3 F1 gains at one and two shots are descriptive and accompanied by worse log loss. This is a formula and native negative-control delivery, not evidence to enable F7 by default. The document specifies coordinates rather than this probability mapping; normalized `exp(-distance / calibration-fitted temperature)` is an explicitly frozen exploratory readout. Public GRABMyo 2048 Hz same-person calibration does not establish own-device 250 Hz transfer or prospective accuracy.

Replay: `PYTHONPATH=src;. python benchmarks/new_bank_v3/f7_document_anchor_run.py` on Windows with the locally cached, hash-bound `V1_FEATURE_ANCHOR_F0v2.npy` asset. The predictions and score readback are portable without the raw archive.
