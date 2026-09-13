# HLA-EMG experiment workspace

This directory is the tracked control plane for the hardware-flexible,
person-adaptive EMG experiments. Raw datasets, generated windows, fitted models,
and prediction tables stay outside Git. Every reported number must be traceable to
a protocol, split manifest, random seed, source commit, and artifact hash.

## Execution funnel

| Round | Question | Implementation status | Exit condition |
|---|---|---|---|
| R0 | Is the UniBo pilot reproducible? | Existing E0/E5/E6b results registered; detailed peer artifacts still need provenance audit. | Frozen metrics reproduce within 0.002. |
| R1 | Can heterogeneous datasets share one honest interface? | V2 schema, trial-safe splits, multi-rate hardware view, Myo adapter, GRABMyo adapter, common windowing and G0/G5 tokens implemented. | Full dataset integrity reports and deterministic manifests. |
| R2 | What are the specialist, LOSO and calibration baselines? | Split contract implemented; common classical runner is next. | Subject/session metrics and paired predictions for B0/B1. |
| R3 | Do priors and learned representations transfer? | Hardware input contract implemented; neural R0/R1/R2/R3 training pending GPU run. | Single-seed screen, then three seeds for top two. |
| R4 | How much calibration recovers personal performance? | Trial-level budgets implemented; P1/P2/P3 models pending R3 selection. | Median 3-shot Recovery >=0.70 and harm rate <=20%. |
| R5 | What is the portability tax on unseen hardware/data? | Protocol frozen; blocked on R3/R4 selection. | Tax <=0.05 on two datasets and <=0.10 on every dataset. |
| R6 | Which hardware information determines the ceiling? | Anti-aliased multi-rate streams implemented; CapgMyo not yet adapted. | Hardware curve and graceful-degradation audit. |
| R7 | Does the selected decoder work in the 8ch+IMU loop? | Not started; depends on own formal collection. | Real-time HumanState and latency/reliability report. |

## Non-negotiable boundaries

- Split physical trials before creating overlapping windows.
- Keep task labels dataset-specific. Only `exact` ontology entries enter a common
  head; `related` is not treated as equality.
- Report zero-shot and personalized performance on the same untouched evaluation
  trials.
- Select models using source-domain development data only. A held-out dataset's
  final partition is read once after the configuration is frozen.
- Use subject/session as the statistical unit, not the number of windows.
- UniBo E6b is an oracle-posture diagnostic, not a deployable result.

See `protocols.json` for machine-readable gates and `pilot_registry.json` for the
frozen UniBo starting point.
