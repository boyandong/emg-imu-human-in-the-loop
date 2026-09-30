# F2 spatial additions: three-axis default review

The [review protocol](F2_CROSS_AXIS_REVIEW_PROTOCOL.json) applies the existing seven-axis default-bank guard to three already inspected, matched F2 experiments. This is explicitly a **retrospective synthesis**, not a new preregistered experiment. Six frozen source result hashes and matched trial identities are checked before reading scores. All 24 arm/axis/phase cells are saved in [the table](F2_CROSS_AXIS_REVIEW_CELLS.csv), with an independently readable [audit](F2_CROSS_AXIS_REVIEW_AUDIT.json).

The guard uses validation only: an addition must avoid lower pooled macro-F1 and higher pooled log loss than matched F0v2 on **every** available axis, and improve at least one measure. Descriptive final cells do not enter selection.

| Addition | Wearing validation | MANUS session validation | GRAB unseen-user validation | Guard result |
| --- | --- | --- | --- | --- |
| F2a trace covariance | F1 +.119; loss −.213 | F1 +.133; loss +.275 | F1 −.057; loss +.179 | Reject universal default |
| F2b document CSP | F1 −.033; loss +.074 | F1 +.065; loss +.312 | F1 −.091; loss +.365 | Reject universal default |
| F2c SPD tangent | F1 +.034; loss −.220 | F1 +.086; loss +.334 | F1 −.068; loss +.231 | Reject universal default |

Thus this three-axis F2 candidate set leaves **F0v2** as its public-data default, consistent with the separate seven-axis F1/F3/F4 screen. F2a remains a wearing or known-user research candidate; no F2 addition is approved for unconditional unseen-user transfer. The tasks differ in classes, rates, subjects and acquisition conditions, and public recordings overlap other project studies. This review cannot prove performance on a 250 Hz device or replace a prospective independent cohort.
