# New-v1 complete bank: seven-axis leave-one-family-out

The frozen [protocol](FULL_V1_LOFO_PROTOCOL.json) fits a source-only F0v2
baseline, a complete five-family candidate, and one model with each family
removed. The four added families are independent versioned implementations;
they are not reconstructions of missing historical X1-H, RLCS, CES, or
Frequency code. The [98 scored cells](FULL_V1_LOFO_CELLS.csv),
[readback audit](FULL_V1_LOFO_AUDIT.json), seven per-axis result files, and
26,684 saved trial/fault predictions provide the full candidate comparison.
The audit confirms matched targets, native trial identities, probability
normalization, frozen parent hashes and F0v2 probability replay.

| Axis | Validation full − F0v2 macro-F1 | Descriptive final full − F0v2 macro-F1 |
|---|---:|---:|
| Posture | −0.0469 | +0.0430 |
| Unseen user | −0.0381 | −0.0894 |
| Cross day | −0.0177 | +0.0036 |
| Force intensity | −0.0539 | +0.0008 |
| Same-user wearing shift | −0.0450 | −0.0809 |
| Observed speed | +0.0315 | +0.0018 |
| Synthetic quality | −0.0545 | +0.0058 |

The quality coordinate is the prespecified mean across six synthetic-fault
families; it excludes the clean control. Other coordinates use their frozen
native pooled macro-F1. These values are paired within each axis and are not
absolute accuracy comparisons between datasets. The seven validation deltas
have an unweighted mean of −0.0321 and a minimum of −0.0545. Full bank wins
only observed-speed validation. Final rows are descriptive; several gains
reverse the validation direction and cannot be used to select the model.

Removing F0v2 lowers full-bank validation macro-F1 on six axes. Each added
family has a positive signed contribution on at most two axes, while several
removals improve results. The complete family-by-axis effects and log-loss
changes are in the scored cells. Thus this finite full candidate does not
pass the frozen all-axis F1/log-loss deployment gate. Retain F0v2 as the
generic default, with specialist families available for research rather
than silently promoting the complete bank.

ROAM posture and quality reuse related recordings, as do GRAB unseen-user and
cross-day splits. Quality faults are simulated. The MANUS speed axis relies
on an external Rest prior, and the public eight-channel devices run at rates
different from the user's unavailable live hardware. This is a public-data
diagnostic, not a physical-device or own-device generalization claim.
