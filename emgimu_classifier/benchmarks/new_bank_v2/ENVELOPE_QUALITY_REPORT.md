# Six-axis new-v2 envelope with synthetic quality

The [versioned protocol](ENVELOPE_QUALITY_PROTOCOL.json) appends the
prespecified [synthetic quality study](ROAM_QUALITY_REPORT.md) to the
frozen [five-axis public evidence](ENVELOPE_POSTURE_REPORT.md). The
[reproducer](envelope_quality_analysis.py) writes eight phase–arm rows
to [ROBUSTNESS_ENVELOPE_6AXIS_SYNTHETIC.csv](ROBUSTNESS_ENVELOPE_6AXIS_SYNTHETIC.csv)
and the [audit](ENVELOPE_QUALITY_AUDIT.json) binds both inputs.

| Phase | Arm | Synthetic quality F1 | Equal six-axis mean | Minimum axis F1 | Minimum axis |
| --- | --- | ---: | ---: | ---: | --- |
| Validation | F0v2 | **0.9110** | **0.8069** | **0.5775** | Wearing |
| Validation | F0v2+F2a | 0.7221 | 0.7633 | 0.5651 | Force |
| Validation | F0v2+F3c | 0.6915 | 0.7504 | 0.5649 | Force |
| Validation | F0v2+F2a+F3c | 0.6856 | 0.7602 | 0.5671 | Force |
| Final, descriptive | F0v2 | **0.9190** | **0.7988** | 0.4939 | Force |
| Final, descriptive | F0v2+F2a | 0.8048 | 0.7698 | **0.5357** | Force |
| Final, descriptive | F0v2+F3c | 0.7621 | 0.7775 | 0.4703 | Force |
| Final, descriptive | F0v2+F2a+F3c | 0.6924 | 0.7427 | 0.4797 | Force |

F0v2 retains the strongest validation mean and minimum; adding the
synthetic axis does not raise its wearing floor. The CSV separately
reports the minimum named fault, minimum fault family and minimum
subject-by-fault cell so the six-axis mean is not mistaken for
worst-case safety. The final-only force minimum ranking does not alter
the validation-selected arm.

Quality/posture share ROAM records, and day/user share GRABMyo records.
Other axes have different devices, sample rates and labels. The
equal-axis mean is descriptive, not pooled accuracy or a confidence
interval. Execution speed remains `N/A` for this exact F0v2 bank.
**Real hardware-quality measurement remains `N/A`**; synthetic faults
cannot establish its score or the reliability of the user's device.
