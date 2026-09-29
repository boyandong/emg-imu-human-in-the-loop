# Matched conditional value of four independent feature additions

The [read-back analysis](v1_extension_paired.py) compares the exact same
native target IDs, truth labels and frozen F0v2 probabilities with each of
the four single-family additions from the [ROAM posture](ROAM_V1_EXTENSION_REPORT.md),
[GRAB unseen-user](GRAB_V1_EXTENSION_REPORT.md), and
[GRAB cross-day](GRAB_DAY_V1_EXTENSION_REPORT.md) screens. It computes
macro-F1 changes, log-loss and Brier improvements, prediction disagreement,
corrected errors and created errors from saved probabilities. Positive loss
or Brier improvement means the added arm is better. Validation and final
remain separate. This is post-hoc analysis, not a new model-selection rule.

| Study | Phase | Family | ΔF1 | Loss improvement | Corrected / created |
| --- | --- | --- | ---: | ---: | ---: |
| ROAM posture | Validation | ring lag | -0.0130 | -0.0002 | 1 / 3 |
| ROAM posture | Final | ring lag | +0.0197 | +0.0321 | 6 / 3 |
| GRAB unseen user | Validation | ring lag | +0.0306 | +0.0376 | 2 / 0 |
| GRAB unseen user | Final | ring lag | -0.0559 | -0.0995 | 0 / 3 |
| GRAB cross day | Validation | ring lag | +0.0135 | +0.0085 | 6 / 3 |
| GRAB cross day | Final | ring lag | -0.0147 | -0.0390 | 3 / 6 |

The paired counts show why a consistent feature role cannot be inferred from
the two positive GRAB validation effects. They reverse on both GRAB final
splits, while ROAM has the opposite pooled pattern. ROAM ring lag's
worst-posture validation advantage is a condition-specific fact even though
its pooled validation F1 is lower. The full table retains the other three
families, per-subject rows and ROAM per-posture rows so that pooled gains
cannot hide weak cells.

The [176-row table](V1_EXTENSION_PAIRED.csv) contains 80 ROAM, 24 GRAB
unseen-user and 72 GRAB cross-day matched group comparisons. Its
[audit](V1_EXTENSION_PAIRED_AUDIT.json) binds each source prediction/result
file and the output by SHA-256. The
[delivery test](../../tests/test_v1_extension_delivery.py) checks all row
identities, source hashes, exact error-accounting partitions and delta
arithmetic. Native bouts/trials from the same person or recording are
correlated; the GRAB splits reuse parts of the same public subset. This
analysis does not provide independent replications or own-device evidence.

The [idempotent exporter](export_v1_extension_delivery.py) adds 176 matched
conditional and 176 matched error rows to the canonical source tables, with
source hashes in its [audit](V1_EXTENSION_DELIVERY_AUDIT.json). The canonical
builder and verifier now cover 55,930 provenance-preserving records. The
delivery still reports `schema_complete_evidence_partial`: this added study
does not close every scientific requirement in the full feature-bank plan.
