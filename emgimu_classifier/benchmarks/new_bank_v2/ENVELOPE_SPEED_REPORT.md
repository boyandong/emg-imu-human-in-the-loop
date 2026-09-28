# Qualified seven-axis new-v2 robustness envelope

The [versioned protocol](ENVELOPE_SPEED_PROTOCOL.json) appends the
[external-Rest MANUS speed study](MANUS_REST_TRANSFER_REPORT.md) to the
frozen [six-axis envelope with synthetic quality](ENVELOPE_QUALITY_REPORT.md).
The [reproducer](envelope_speed_analysis.py) writes eight phase–arm rows
to [ROBUSTNESS_ENVELOPE_7AXIS_QUALIFIED.csv](ROBUSTNESS_ENVELOPE_7AXIS_QUALIFIED.csv),
with source hashes in [the audit](ENVELOPE_SPEED_AUDIT.json).

| Phase | Arm | Qualified speed F1 | Worst native speed F1 | Equal seven-axis mean | Minimum axis F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation | F0v2 | 0.3914 | 0.2845 | **0.7476** | 0.3914 |
| Validation | F0v2+F2a | **0.5244** | **0.4677** | 0.7291 | **0.5244** |
| Validation | F0v2+F3c | 0.3892 | 0.2509 | 0.6988 | 0.3892 |
| Validation | F0v2+F2a+F3c | 0.4742 | 0.4023 | 0.7194 | 0.4742 |
| Final, descriptive | F0v2 | 0.4600 | 0.4283 | **0.7504** | 0.4600 |
| Final, descriptive | F0v2+F2a | **0.5304** | 0.4491 | 0.7356 | **0.5304** |
| Final, descriptive | F0v2+F3c | 0.5251 | **0.4734** | 0.7414 | 0.4703 |
| Final, descriptive | F0v2+F2a+F3c | 0.5077 | 0.4269 | 0.7091 | 0.4797 |

F0v2 has the highest validation descriptive mean, while F0v2+F2a
raises the minimum axis from 0.3914 to 0.5244. The minimum for these
two arms is the qualified MANUS speed coordinate; their other axes
remain unchanged. This is a genuine mean-versus-floor tradeoff, not a
single dominant bank. The native MANUS validation rule selects the
F2a addition for that speed task, but its validation log loss worsens.

The seven coordinates do **not** form seven independent experiments:
day/user share GRABMyo, posture/synthetic quality share ROAM, and
datasets have different labels and rates. MANUS has an external
cross-dataset Rest prior, seen speed categories, changed target
sessions and no product gesture taxonomy. The quality coordinate is
synthetic. Real hardware quality, the user's 250 Hz device and live
online performance remain `N/A`. The seven-axis equal mean is a
descriptive index, not pooled accuracy or a complete product gate.
