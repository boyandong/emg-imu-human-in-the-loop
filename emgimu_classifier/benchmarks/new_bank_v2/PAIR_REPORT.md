# New-v2 matched error and F2a×F3c interaction analysis

This is an independent, versioned analysis of four already-frozen new-v2
prediction matrices. It does not reuse or require any missing historical
B0/X1 code. The 8,592 selected arm–trial predictions represent 2,148 native
trials across force, electrode shift, GRABMyo day, and Song same-day data.
All four arms were matched on subject, condition, trial ID, and ground truth
before comparison. No model was retrained or chosen from the final splits.

The frozen protocol is `PAIR_PROTOCOL.json`. Run `pair_analysis.py` to derive
`PAIR_COMPLEMENTARITY.csv` (384 arm-pair records over 64 pooled, subject, or
condition cells) and `F2A_F3C_INTERACTION.csv` (64 cells). `PAIR_AUDIT.json`
records hashes and expected counts; `pair_analysis.py --verify` reconstructs
both CSVs byte for byte. `export_pair_delivery.py` adds the 384 comparisons
idempotently to the canonical error-complementarity schema; its hash record
is `PAIR_DELIVERY_AUDIT.json`. The interaction table remains a standalone
result because the five canonical schemas have no interaction field.

Pooled interaction, where positive means the joint arm's held-out negative
log loss exceeded the additive expectation of separate additions:

| Axis | Phase | Trials | Interaction, −log loss | Interaction, macro-F1 | Joint − core macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Force | validation | 588 | +0.3159 | +0.0667 | −0.0624 |
| Force | final | 588 | −0.1003 | −0.0324 | −0.0142 |
| Electrode shift | validation | 120 | −0.1784 | −0.0645 | +0.1607 |
| Electrode shift | final | 120 | −0.1124 | −0.1003 | +0.0389 |
| GRABMyo day | validation | 224 | +0.0208 | +0.0262 | −0.0331 |
| GRABMyo day | final | 224 | +0.1307 | +0.0275 | −0.0441 |
| Song same day | validation | 140 | −0.0550 | −0.0072 | +0.0437 |
| Song same day | final | 144 | −0.1038 | −0.0450 | +0.0446 |

The interaction is `P(B+F2a+F3c) − P(B+F2a) − P(B+F3c) + P(B)`;
`P` is negative held-out log loss. Positive interaction is relative to an
additive expectation, **not** proof that the joint model beats the core.
For example, force validation has positive interaction but the joint arm's
macro-F1 is 0.0624 below the core. The final force and GRABMyo day splits
also have joint macro-F1 below core. Electrode shift and Song same-day
show the opposite sign for the joint-minus-core macro-F1. Thus no universal
F2a+F3c inclusion rule is supported.

The F2a-only and F3c-only models make different mistakes on matched trials.
On final force, F2a-only is correct while F3c-only is wrong on 60 trials,
and the reverse occurs on 31. On final electrode shift, the counts reverse
to 9 and 27. This is concrete error complementarity, but it does not by
itself establish that a joint model improves held-out performance.
Undefined error correlations, where an error vector is constant, are `N/A`
rather than fabricated zeroes. The four datasets use different subjects,
devices, labels, and rates and are never pooled into a single accuracy.
Song remains a one-user, one-date supplement. Previously inspected final
splits are descriptive evidence, not a fresh prospective test.
