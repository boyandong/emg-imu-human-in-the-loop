# Independent new-v1 synthetic-quality extension

The [frozen protocol](ROAM_V1_QUALITY_PROTOCOL.json) adds four independent
new-v1 feature families to F0v2 on the exact ROAM-EMG 8-channel, 200 Hz
static resting-bout source/validation/final split used by the earlier
new-v2 [quality study](ROAM_QUALITY_REPORT.md). Source subjects 1–18 fit
all feature metadata, Rest thresholds, fault amplitudes, scalers and
classifiers. Validation subjects 19–23 and descriptive final subjects
24–28 each provide 45 native bouts, paired across all arms and the clean
control plus 13 fixed synthetic faults. No target recording fits state.

The fault grid comprises eight separate channel dropouts, half gain,
source-q95 clipping, 50 Hz line sine, baseline ramp and channel-0 burst.
The primary quality score is the equal mean of six prespecified
coordinates: average dropout F1 and the five other individual-fault F1
values. Clean is excluded. Minimum individual-fault and subject-by-fault
F1 remain visible. Every F0v2 probability under every condition replays
the frozen parent exactly (maximum difference 0).

| Arm | Validation clean F1 | Validation fault-family mean F1 | Validation worst fault F1 | Final clean F1 | Final fault-family mean F1 | Final worst fault F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F0v2 | 1.0000 | **.9110** | .4960 | 1.0000 | .9190 | .4929 |
| + scale pattern | .9804 | .9089 | .4817 | 1.0000 | .9415 | .5050 |
| + ring lag | .9459 | .8382 | .5071 | .9717 | .8627 | .5184 |
| + correlation spectrum | 1.0000 | .8745 | .5184 | .9521 | .9012 | .5184 |
| + frequency direction | .9804 | .8786 | .4960 | 1.0000 | .9507 | .4888 |

F0v2 wins the prespecified validation fault-family mean. Scale pattern
is close in that mean but loses validation clean and worst-fault F1.
Ring lag and correlation spectrum improve the validation minimum
individual fault but lose substantially on the family mean. The
scale-pattern and frequency-direction gains on the descriptive final
subjects do not justify retrospective selection. Thus no new-v1
addition is promoted as a general quality-robust default; particular
fault specialists remain hypotheses for targeted future validation.

The 6,300 saved predictions generate 840 pooled/subject family scores,
672 matched conditional increments and 1,680 full pairwise-error cells.
All 840 saved score groups replay, and the 3,192 cells are delivered in
the canonical feature-bank tables. The canonical provenance verifier
checks 62,698 records after this extension.

These faults are controlled test-only simulations. The data do not
measure real electrode-contact faults, fault prevalence, live transitions
or performance on the user's 250 Hz device. Reproduce with
`roam_v1_quality_run.py`, `roam_v1_quality_analysis.py` and
`export_roam_v1_quality.py` using the frozen protocol and public archive.
