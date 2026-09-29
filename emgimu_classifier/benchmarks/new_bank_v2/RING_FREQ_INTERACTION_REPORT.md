# Finite ring-lag × frequency-direction interaction

The [protocol](RING_FREQ_INTERACTION_PROTOCOL.json) froze the hypothesis,
three parent source splits, four arms, classifier recipes, interaction
equations and validation rule before outcome computation. Ring lag measures
spatial envelope timing; frequency direction measures bandwise channel power
orientation. The joint arm tests whether their combined information adds
more than the sum of their separate additions to F0v2. These are independent
new formulas, not historical X1/Frequency reconstructions.

| Public split | Phase | F0v2 F1 | Joint F1 | Joint loss improvement | F1 interaction | Loss interaction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ROAM posture | Validation | 0.9630 | 0.9233 | -0.0310 | -0.0084 | -0.0101 |
| ROAM posture | Final | 0.9162 | 0.9658 | +0.1158 | 0.0000 | -0.0029 |
| GRAB unseen user | Validation | 0.8054 | 0.7983 | -0.0953 | -0.0326 | -0.1261 |
| GRAB unseen user | Final | 0.9458 | 0.8401 | -0.2198 | +0.0559 | +0.0851 |
| GRAB cross day | Validation | 0.9551 | 0.9421 | -0.0095 | -0.0178 | -0.0037 |
| GRAB cross day | Final | 0.9008 | 0.8991 | -0.2484 | +0.0049 | -0.0369 |

Positive interaction means superadditive benefit; it does not mean the joint
arm beats F0v2. All three validation splits have negative F1 and log-loss
interaction and a worse absolute joint arm on both metrics. The protocol's
general-combination criterion therefore **fails**. ROAM's descriptive final
joint gain and GRAB unseen-user final positive interaction cannot overturn
that conclusion; the latter joint arm has much worse absolute final F1 and
loss than F0v2. The joint block is not promoted to the live bundle.

The [runner](ring_freq_interaction_run.py) independently re-extracts the
native source, fits all four arms on the frozen source users or day, and
matches all baseline and single-addition target probabilities to the
previously saved screens exactly (`0.0` maximum absolute difference in this
run). The [result](RING_FREQ_INTERACTION_RESULTS.json) binds 3,680
[saved predictions](RING_FREQ_INTERACTION_PREDICTIONS.csv).
The [paired analysis](ring_freq_interaction_analysis.py) reports 44 pooled,
subject and ROAM-posture cells in its [table](RING_FREQ_INTERACTION.csv),
with [source-hash audit](RING_FREQ_INTERACTION_AUDIT.json). An
[idempotent export](export_ring_freq_interaction.py) adds these finite
interactions to the standalone source `interaction_results.csv`; its
[audit](RING_FREQ_INTERACTION_DELIVERY_AUDIT.json) records the source and
output hashes. The [read-back test](../../tests/test_v1_extension_delivery.py)
checks native identities, factorial arithmetic and delivered rows.

The ROAM population/class problem and GRAB population/day comparisons are
not exchangeable or statistically independent; the two GRAB studies reuse
recordings. None establishes online performance on the user's 250 Hz device.
