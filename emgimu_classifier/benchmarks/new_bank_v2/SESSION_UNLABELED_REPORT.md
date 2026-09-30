# Label-free session prediction replay

The versioned session pipeline now accepts evaluation windows, user identity and native trial IDs without evaluation labels. Its offline scoring wrapper uses the same prediction path and reads labels only after prediction. Source-fit and calibration trial IDs are rejected at the prediction boundary.

The [frozen protocol](SESSION_UNLABELED_PROTOCOL.json) uses subject 15 of the public LibEMG Electrode Shift data at its native eight-channel, 200 Hz rate: 25 training trials fit the long-term profile, five `trial_1/R_0` trials calibrate the session (one per class), and five separate `trial_1/R_1` trials are evaluated. The [machine-readable result](SESSION_UNLABELED_RESULTS.json) and [60 prediction rows](SESSION_UNLABELED_PREDICTIONS.csv) cover six branches and two feature families for each evaluation trial.

Across all rows, label-free probabilities equal the offline wrapper probabilities exactly (`max_abs_probability_difference = 0`). Replacing every offline truth label with an incorrect label also leaves all probabilities unchanged. Serialized source and session state are unchanged after prediction. A separate readback test verifies the saved row hash, trial partition, complete branch/family/trial Cartesian product, per-row probability simplex and offline truth IDs.

This is an API and leakage check, not a new accuracy estimate, a physical electrode re-donning trial, an independent-user result, a later-day session, or a live-device validation. The branch predictions are not promoted as a general winner by this parity check.
