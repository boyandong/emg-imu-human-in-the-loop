# Document-consistent F2c and F3c spatial candidates

The detailed formulas specify that F2c's SPD matrix and F3c's ring-relative entries come from F2a. The earlier F2c/F3c candidates used a **centered** covariance; their frozen results retain that identity. Two separate candidates now use the document-exact **uncentered** second moment, fixed 0.05 shrinkage and a minimal positive diagonal ridge for the SPD log map. F2c fits a log-Euclidean source reference and outputs the whitened symmetric matrix log. F3c takes ring-lag mean, median, spread and quartiles from the same matrix; it requires an explicit verified circular topology and omits the unstable early/late block on short windows. Independent matrix and ring-rotation tests distinguish these from centered counterparts.

The [wearing protocol](DOCUMENT_SPATIAL_WEARING_PROTOCOL.json) pins the public eight-channel, 200 Hz source and four target domains for six users. The [transfer protocol](F2C_DOCUMENT_TRANSFER_PROTOCOL.json) pins matched six-user MANUS sessions and GRABMyo unseen users at native 200 and 2048 Hz respectively. All feature/reference/classifier fitting uses source trials; probabilities and native identities are saved in the [wearing results](DOCUMENT_SPATIAL_WEARING_RESULTS.json) and [transfer results](F2C_DOCUMENT_TRANSFER_RESULTS.json), with independent readback tests.

| Axis | F0v2 validation F1 / loss | + document F2c validation F1 / loss | + document F3c validation F1 / loss |
| --- | --- | --- | --- |
| Wearing shift | .5775 / 1.2916 | .6397 / 1.0873 | .6829 / .9477 |
| MANUS session | .3914 / 2.1203 | .4613 / 2.4696 | Not eligible: ring topology unverified |
| GRAB unseen user | .8054 / .4650 | .7371 / .6968 | Not eligible: ring topology unverified |

The [eight-cell decision](DOCUMENT_SPATIAL_CROSS_AXIS_AUDIT.json) applies the existing validation-only rule. Document F2c worsens MANUS log loss and both GRAB measures, so it is not a universal default. Document F3c improves wearing validation F1 by .1054 and log loss by .3440; the descriptive final users also improve F1 (.6170 → .7113) and log loss (1.0838 → .8020). This makes F3c a **wearing-specific research candidate**, not a deployed model: the public domains and users have appeared in prior studies, and other eligible ring-topology axes and the user's 250 Hz device are unavailable. F0v2 remains the cross-axis public-data default.
