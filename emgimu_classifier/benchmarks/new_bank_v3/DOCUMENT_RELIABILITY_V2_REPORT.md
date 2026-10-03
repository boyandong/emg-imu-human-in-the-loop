# Document-exact personal reliability D/E

The opt-in `DocumentReliabilityWeightsV2` implements the stated order `R=B/(W+ε)`, then `r=log(R+ε)`, softmax with a frozen temperature, and `α=n0/(n0+N_cal)` before mixing with source population weights. The older `ReliabilityWeights` computes `log((B+ε)/(W+ε))` and counts input rows; it remains unchanged for historical results. A zero-between-class oracle distinguishes the formulas, and repeated windows from one trial leave the new prototype, reliability and `N_cal` unchanged.

Every family must supply the same explicitly labelled calibration trial identities. Windows are averaged within a trial before class prototypes and within-class spread are calculated; the caller can reject known source/evaluation trial IDs. Population weights, temperature and `n0` must be supplied with a source-policy identity. The API validates these inputs but **does not itself prove that the policy was selected by source-user cross-validation**.

This is an analytical formula and leakage-contract delivery, not a new native performance result or a default fusion policy. Existing source-only policy-selection studies remain separate. A new versioned native study must bind the exact formula to a source-CV-selected policy and held-out calibration/evaluation trials before any performance claim.
