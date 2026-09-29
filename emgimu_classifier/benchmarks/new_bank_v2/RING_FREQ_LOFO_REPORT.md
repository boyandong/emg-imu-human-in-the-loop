# Independent candidate full bank: leave-one-family-out

The [protocol](RING_FREQ_LOFO_PROTOCOL.json) was committed before the missing
arm was scored. The candidate full bank is `F0v2+ring_lag+frequency_direction`.
On identical held-out native bouts/trials, it is compared with three removals:
without F0v2, without ring lag, and without frequency direction. The full
and latter two arms copy the exact frozen probabilities from the
[four-arm interaction](RING_FREQ_INTERACTION_REPORT.md); only the no-F0v2
arm requires a new source-only fit. The candidate full bank had already
failed all three validation splits, so this is a *diagnostic ablation*,
not evidence that it was selected for deployment.

Positive ΔF1 below means keeping that family helps the candidate full bank.

| Public split | Phase | Full F1 | ΔF1 keep F0v2 | ΔF1 keep ring lag | ΔF1 keep frequency direction |
| --- | --- | ---: | ---: | ---: | ---: |
| ROAM posture | Validation | 0.9233 | +0.0334 | -0.0215 | -0.0267 |
| ROAM posture | Final | 0.9658 | +0.0548 | +0.0197 | +0.0299 |
| GRAB unseen user | Validation | 0.7983 | +0.2438 | -0.0020 | -0.0377 |
| GRAB unseen user | Final | 0.8401 | +0.1845 | 0.0000 | -0.0498 |
| GRAB cross day | Validation | 0.9421 | +0.0996 | -0.0043 | -0.0265 |
| GRAB cross day | Final | 0.8991 | +0.1209 | -0.0098 | +0.0130 |

Keeping F0v2 improves F1 and log loss in all six pooled comparisons;
its largest F1 effect is +0.2438 on GRAB unseen-user validation. Removing
either added family improves F1 on every validation split, consistent with
the interaction experiment's rejected joint arm. Some final-only cells have
the opposite direction, especially ROAM; they do not change selection.
This supports F0v2 as the candidate backbone within this finite bank, not a
universal choice for every task or the user's live 250 Hz device.

The [runner](ring_freq_lofo_run.py) saves 3,680
[four-arm predictions](RING_FREQ_LOFO_PREDICTIONS.csv). The
[independent read-back](ring_freq_lofo_analysis.py) confirms 2,760 parent
rows are unchanged, replays all pooled scores, and emits 176 matched
[pooled, subject and posture cells](RING_FREQ_LOFO_CELLS.csv). Its
[verification](RING_FREQ_LOFO_VERIFICATION.json) binds the protocol,
result, predictions and source by SHA-256. The
[idempotent exporter](export_ring_freq_lofo.py) puts those 176 cells in the
required full-bank ablation source table and records its hash in the
[delivery audit](RING_FREQ_LOFO_DELIVERY_AUDIT.json). Canonical provenance
verification now covers 56,106 records. The three public studies are not
independent population replications, and the two GRAB splits reuse recordings.
