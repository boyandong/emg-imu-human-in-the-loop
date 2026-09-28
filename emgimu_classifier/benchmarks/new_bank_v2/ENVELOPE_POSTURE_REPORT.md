# Five observed-axis new-v2 robustness envelope

The [versioned protocol](ENVELOPE_POSTURE_PROTOCOL.json) adds the exact
new-v2 [ROAM-EMG posture screen](ROAM_POSTURE_REPORT.md) to the frozen
[four-axis envelope](ENVELOPE_USER_REPORT.md). The
[reproducer](envelope_posture_analysis.py) writes eight phase–arm rows to
[ROBUSTNESS_ENVELOPE_5AXIS.csv](ROBUSTNESS_ENVELOPE_5AXIS.csv); its
[audit](ENVELOPE_POSTURE_AUDIT.json) binds both inputs. Earlier
three-/four-axis artifacts remain unchanged.

| Phase | Arm | Posture pooled F1 | Worst posture F1 | Equal five-axis mean | Minimum axis F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation | F0v2 | **0.9630** | **0.8723** | **0.7861** | **0.5775** |
| Validation | F0v2+F2a | 0.9167 | 0.8399 | 0.7715 | 0.5651 |
| Validation | F0v2+F3c | 0.9075 | 0.8561 | 0.7622 | 0.5649 |
| Validation | F0v2+F2a+F3c | 0.9228 | 0.8638 | 0.7752 | 0.5671 |
| Final, descriptive | F0v2 | 0.9162 | 0.8452 | 0.7748 | 0.4939 |
| Final, descriptive | F0v2+F2a | **0.9353** | **0.9028** | 0.7628 | **0.5357** |
| Final, descriptive | F0v2+F3c | 0.8360 | 0.7433 | **0.7805** | 0.4703 |
| Final, descriptive | F0v2+F2a+F3c | 0.8814 | 0.8533 | 0.7527 | 0.4797 |

F0v2 keeps the strongest validation five-axis mean and minimum. The
minimum axis remains wearing in validation and force in descriptive
final; adding posture evidence does not rescue the weak force/wearing
floor. The separate minimum subject/condition cell is in the CSV, not
conflated with the minimum axis. Final-only differences do not revise
validation selection.

The five axes use different public datasets, labels, rates and splits,
and day/user share GRABMyo records. This equal-axis mean is descriptive,
not pooled accuracy or a confidence interval. ROAM posture also combines
unseen-user and posture shifts. Speed and real signal quality remain `N/A`
for this exact bank; the separate MANUS speed experiment has a different
Rest-free baseline. Own-device 250 Hz testing remains unavailable.
