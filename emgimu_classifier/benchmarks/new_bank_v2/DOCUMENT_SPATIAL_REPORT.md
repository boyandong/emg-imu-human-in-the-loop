# Historical uncentered F2c and F3c spatial alternatives

**Formula correction (V3):** The goal file actually specifies a *centered*
F2a covariance. This V2 study used an uncentered second moment and must be
read as an alternative-feature experiment, not an exact-F2a/F2c/F3c test.
The separate [centered V3 implementation and native replay](../new_bank_v3/SPEC_SPATIAL_GRAB_REPORT.md)
retain all V2 probabilities under their original names.

The detailed formulas specify that F2c's SPD matrix and F3c's ring-relative entries come from F2a, which is **centered** in the goal file. The earlier F2c/F3c candidates used centered covariance; their frozen results retain that identity. These two V2 alternatives instead use an **uncentered** second moment, fixed 0.05 shrinkage and a minimal positive diagonal ridge for the SPD log map. F2c fits a log-Euclidean source reference and outputs the whitened symmetric matrix log. F3c takes ring-lag mean, median, spread and quartiles from the same matrix; it requires a caller assertion that saved channels follow the physical circular order and omits the unstable early/late block on short windows. Independent source-reference, matrix, long-window drift and ring-rotation tests distinguish these alternatives from centered counterparts.

The [wearing protocol](DOCUMENT_SPATIAL_WEARING_PROTOCOL.json) pins the public eight-channel, 200 Hz source and four target domains for six users. The [transfer protocol](F2C_DOCUMENT_TRANSFER_PROTOCOL.json) pins matched six-user MANUS sessions and GRABMyo unseen users at native 200 and 2048 Hz respectively. All feature/reference/classifier fitting uses source trials; probabilities and native identities are saved in the [wearing results](DOCUMENT_SPATIAL_WEARING_RESULTS.json) and [transfer results](F2C_DOCUMENT_TRANSFER_RESULTS.json), with independent readback tests.

| Axis | F0v2 validation F1 / loss | + document F2c validation F1 / loss | + document F3c validation F1 / loss |
| --- | --- | --- | --- |
| Wearing shift | .5775 / 1.2916 | .6397 / 1.0873 | .6829 / .9477 |
| MANUS session | .3914 / 2.1203 | .4613 / 2.4696 | Not eligible: ring topology unverified |
| GRAB unseen user | .8054 / .4650 | .7371 / .6968 | Not eligible: ring topology unverified |

The [eight-cell decision](DOCUMENT_SPATIAL_CROSS_AXIS_AUDIT.json) applies the existing validation-only rule. Document F2c worsens MANUS log loss and both GRAB measures, so it is not a universal default. Document F3c improves wearing validation F1 by .1054 and log loss by .3440; the descriptive final users also improve F1 (.6170 → .7113) and log loss (1.0838 → .8020). This is an **exploratory circular-index result**, not a verified anatomical ring result or deployed model. [LibEMG identifies the device as an eight-channel Myo armband](https://libemg.github.io/libemg/documentation/data/data.html), and [the Myo electrodes are physically circular](https://pmc.ncbi.nlm.nih.gov/articles/PMC9458587/), but the [released dataset](https://github.com/LibEMG/CIILData/tree/main/ElectrodeShift) does not document that saved column order follows physical adjacency. The caller's `ring_topology=True` is an assumption, not independent dataset proof. The public domains and users have appeared in prior studies; no other confirmed ring-order axis or own-device evidence is available. F0v2 remains the cross-axis public-data default.

A separate [fixed channel-order sensitivity screen](F3C_CHANNEL_ORDER_REPORT.md)
finds native-order validation F1/loss ranks of 2/13 and 1/13 against 12
non-equivalent cyclic-index reorderings; descriptive-final ranks are 1/13 for
both. The result is order-sensitive but still cannot certify physical adjacency.

The [official GRABMyo forearm-ring follow-up](F3C_DOCUMENT_GRAB_REPORT.md)
provides a stronger topology source: the published diagram visibly places
8–1–2 as neighbors, and the dataset maps F1–F8 to that ring. Complete hidden-
side adjacency remains an explicit numbering inference. On frozen Day1→Day2/3
trials, the V2 uncentered F3c alternative worsens F1 and log loss on both days relative
to F0, so the wearing-specific result still does not support default promotion.
