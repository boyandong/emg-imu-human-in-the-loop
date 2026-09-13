# Myo R0 versus R1-core screening result (2026-09-13)

## Scope

This is a Round 2 implementation screen, not a formal model-selection result.
It uses all 18 subjects from the adapted Myo EvaluationDataset, one seed, the
HLA 250 ms / 50 ms protocol, and 10 uniformly selected windows from each physical
trial to keep the RBF-SVM run practical on CPU. Each LOSO fold therefore trains on
whole source-subject trials and evaluates one untouched subject. No target-subject
calibration is used.

Code provenance: commit `be67cdd6742b9a30dadb8c24f439c7fa89c17b8e` with a
clean worktree. Dataset manifest SHA-256:
`d018a4e7fa970e270d4c5544daa7f50d3bf0a983b3798bc1765a95da1c0fdb08`.

## Subject-level result

| Representation | Accuracy | Macro-F1 | Active macro-F1 |
|---|---:|---:|---:|
| R0: G0 | 0.7570 | 0.7322 | 0.7060 |
| R1-core: G0+G5 | 0.7610 | 0.7387 | 0.7129 |
| Paired mean difference | +0.0040 | +0.0065 | +0.0069 |

For active macro-F1, R1-core improved 10 of 18 held-out subjects. The median
paired difference was +0.0019 and the 2,000-resample subject-level paired bootstrap
95% interval was [-0.0066, +0.0251]. The interval crosses zero, so this screen does
not establish a reliable G5 improvement.

## Per-class observation

| Task label | R0 recall | R1-core recall | Difference |
|---|---:|---:|---:|
| Neutral | 0.9208 | 0.9250 | +0.0042 |
| Radial deviation | 0.8329 | 0.8310 | -0.0019 |
| Wrist flexion | 0.6847 | 0.6829 | -0.0019 |
| Ulnar deviation | 0.7120 | 0.7116 | -0.0005 |
| Wrist extension | 0.8310 | 0.8463 | +0.0153 |
| Hand close | 0.7389 | 0.7375 | -0.0014 |
| Hand open | 0.5787 | 0.5926 | +0.0139 |

Hand open remains the lowest-recall class under both representations. G5 shows its
largest positive recall trends for hand open and wrist extension, but this table is
descriptive and was not corrected for multiple class-wise comparisons.

## Decision

- Keep R0 as the fixed classical reference.
- Keep R1-core for the planned multi-dataset single-seed screen because its mean
  trend is positive and the UniBo pilot also favored temporal information.
- Do not advance G5 on the claim that Myo already proves it effective.
- Do not use this subsampled screen as the final Myo specialist upper bound.
- The next decisive test is the same paired R0/R1-core comparison on full GRABMyo
  forearm and wrist views, followed by the learned R2 control.

External run artifacts are stored outside Git under
`EMG-project-hla-data/runs/myo-r0-screen18-be67cdd` and
`EMG-project-hla-data/runs/myo-r1core-screen18-be67cdd`.
