# New-version public default extension guard

The earlier seven-axis [new-v1 decision](../new_bank_v2/V1_CROSS_AXIS_DECISION_REPORT.md) retains F0v2 as its public-data default. This [hash-bound retrospective audit](PUBLIC_DEFAULT_EXTENSION_AUDIT.json) applies the same within-task validation rule to newer formula candidates: macro F1 must not decrease and log loss must not increase, with a strict gain in at least one. Final splits are descriptive and do not select the default. A task-specific pass alone would still not establish a universal default across the required axes.

| New candidate and matched validation task | Δ macro F1 | Δ log loss | Local guard |
| --- | ---: | ---: | --- |
| Centered F2a, GRAB cross-day | −0.0239 | +0.0364 | Fail |
| Centered F3c, GRAB cross-day | −0.0353 | +0.0203 | Fail |
| Centered F2c, wearing/MANUS/unseen-user | See [three-axis cells](SPEC_F2C_CROSS_AXIS_CELLS.csv) | MANUS and GRAB losses | Fail |
| Complete-bout F5c added to G5, UniBo | +0.0584 | +0.0412 | Fail |
| Document-exact F7 mixed with F0v2, GRAB 1/2/5 shots | 0 at each budget | +0.2847 / +0.2895 / +0.2826 | Fail |

None of these seven checks permits a new universal public default. F0v2 remains the **existing public benchmark default**, not a claim of safe device deployment or identical performance on other hardware. F5c and F7 may still be studied as task-specific research features; their validation gains in one metric do not erase deterioration in the other. F8 routing and F9 quality gates retain separate scope-limited or negative decisions in the family audit; this extension does not silently promote them. The inspected public cohorts are not prospective replications.
