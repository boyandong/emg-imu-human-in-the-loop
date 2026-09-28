# Independent new-v1 paired family analysis

This is a saved-prediction Stage-1–4 analysis of two theoretically selected
newly implemented eight-channel pairs. On LibEMG cross-user intensity, it
compares `scale_pattern` and `frequency_direction` separately and jointly
against F0. On LibEMG electrode shift, it compares `ring_lag` and
`correlation_spectrum` separately and jointly. These names denote the
independent [new-v1 formulas](../../src/emgimu/feature_bank/new_bank_v1.py),
not recovered historical X1-H, RLCS, CES or frequency implementations.

The [frozen protocol](PAIRED_PROTOCOL.json) binds 5,664 saved arm–trial
probabilities and the original result/protocol hashes. The
[reproducer](paired_analysis.py) matches native trial IDs and labels across
the four arms before any calculation, then exactly replays all 16 original
pooled score groups. It produces a [176-cell family screen](PAIRED_SCREEN.csv)
with per-class F1 and ten-bin ECE, [88 conditional additions](PAIRED_CONDITIONAL.csv),
[44 error comparisons](PAIRED_COMPLEMENTARITY.csv), and
[44 finite interactions](PAIRED_INTERACTION.csv). The 44 matched cells are
28 force and 16 wearing pooled/subject/condition cells across validation
and descriptive final phases. `paired_analysis.py --verify` reconstructs
all four tables and the [hash audit](PAIRED_AUDIT.json) byte for byte.

Positive conditional Δlog loss means lower loss after adding the family to
a core that already includes the *other* family:

| Axis | Phase | Added family | Conditional Δlog loss | Conditional Δmacro-F1 |
| --- | --- | --- | ---: | ---: |
| Force | validation | Scale pattern | −0.0287 | +0.0028 |
| Force | validation | Frequency direction | +0.0261 | +0.0256 |
| Force | final | Scale pattern | −0.0278 | −0.0004 |
| Force | final | Frequency direction | −0.2420 | −0.0105 |
| Electrode shift | validation | Ring lag | +0.1345 | +0.0201 |
| Electrode shift | validation | Correlation spectrum | +0.1428 | +0.0273 |
| Electrode shift | final | Ring lag | +0.0781 | −0.0467 |
| Electrode shift | final | Correlation spectrum | +0.0252 | −0.0121 |

Frequency direction has conditional force value on the validation users,
but the same addition hurts both held-out log loss and F1 on descriptive
final users. On wearing, both ring candidates conditionally lower log loss
in both phases, yet their final F1 increments are negative. None merits a
universal late-stage inclusion rule from these data.

The paired error counts show the cohort reversal directly. For the two
single-addition force arms, scale pattern is correct while frequency
direction is wrong on 15 validation trials, and the reverse occurs on 34;
on final users the counts are 32 and 15. For the wearing single-addition
arms the corresponding counts are 4/5 validation and 4/7 final. These are
real matched errors, but a small discordant set does not prove a fused
classifier would improve. Constant-error-vector correlations are `N/A`.

The interaction is `P(F0+a+b)−P(F0+a)−P(F0+b)+P(F0)`, with `P` equal to
negative held-out log loss. Force interaction is −0.0542 on validation and
+0.0527 on final; the joint arm's final macro-F1 is still 0.0281 below F0.
Wearing interaction is −0.1430 on validation and +0.0143 on final; final
joint macro-F1 is 0.0287 below F0. A positive interaction therefore does
not mean absolute improvement over the baseline.

The [idempotent exporter](export_paired_delivery.py) adds 176 family,
88 conditional, and 44 error rows to the canonical five-schema delivery;
the 44 interaction rows remain a standalone source result. The
[delivery audit](PAIRED_DELIVERY_AUDIT.json) records input/output hashes.
All 53,408 canonical source identities and values pass provenance checks.
Different class taxonomies and devices are not pooled. The target final
users were inspected in earlier project studies and are descriptive here;
no model is chosen from their scores.
