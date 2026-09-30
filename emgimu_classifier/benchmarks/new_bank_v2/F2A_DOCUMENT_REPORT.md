# Document-exact F2a versus centered F2a

The target specification writes F2a as trace-normalized `XXᵀ`, without removing channel means. The earlier versioned F2a uses a **centered** covariance. Both are now explicit, separate candidates: `TraceCovarianceV2` preserves the already frozen centered experiments; `DocumentTraceCovarianceV2` computes the uncentered second moment, fixed 0.05 isotropic shrinkage, trace normalization and a 36-coordinate symmetric upper triangle. A constant-offset known signal distinguishes them, and the uncentered candidate passes an independent numerical formula and gain-invariance check.

The [wearing protocol](F2A_DOCUMENT_WEARING_PROTOCOL.json) and [MANUS/GRAB transfer protocol](F2A_DOCUMENT_TRANSFER_PROTOCOL.json) pin the earlier matched public experiments. The new candidate uses their exact source and target trial IDs, source-only feature fitting, and their classifier settings. Its [wearing](F2A_DOCUMENT_WEARING_RESULTS.json) and [transfer](F2A_DOCUMENT_TRANSFER_RESULTS.json) results retain every native trial probability, with independent readback tests. The [six-cell decision](F2A_DOCUMENT_CROSS_AXIS_CELLS.csv) applies the existing validation-only default-bank guard; final splits are descriptive.

| Axis | Validation F0v2 F1 / loss | + uncentered F2a F1 / loss | Centered F2a F1 / loss |
| --- | --- | --- | --- |
| Wearing shift | .5775 / 1.2916 | .6592 / 1.2248 | .6960 / 1.0786 |
| MANUS session | .3914 / 2.1203 | .5316 / 2.4066 | .5244 / 2.3949 |
| GRAB unseen user | .8054 / .4650 | .7485 / .6443 | .7485 / .6442 |

The document-exact candidate improves wearing validation F1 and log loss, but less than centered F2a. It improves MANUS validation F1 while worsening log loss, and harms both measures on unseen GRAB users. It fails the existing cross-axis guard, so **F0v2 remains the public-data default**. The [decision audit](F2A_DOCUMENT_CROSS_AXIS_AUDIT.json) records the MANUS loss and GRAB F1/loss violations. This resolves the implementation/formula ambiguity without relabeling older frozen results. The public axes were previously inspected and are not a fresh independent cohort; neither F2a variant has been validated on the user's 250 Hz device.
