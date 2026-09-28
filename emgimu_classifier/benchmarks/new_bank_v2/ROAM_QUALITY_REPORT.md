# Independent new-v2 synthetic quality stress test

The [frozen protocol](ROAM_QUALITY_PROTOCOL.json) reuses the verified
public ROAM-EMG eight-channel, nominal 200 Hz static recordings from
[the posture study](ROAM_POSTURE_REPORT.md). Only source subjects 1–18
in resting posture fit F0v2/F2a/F3c, the source classifier, and fault
reference amplitudes. Validation subjects 19–23 and descriptive final
subjects 24–28 contribute the same 45 native label bouts per phase under
every condition. The [reproducer](roam_quality_run.py) checks that every
clean resting probability vector replays the frozen posture-study result
within `1e-10`; no model is refit after a fault.

The test-only grid has 13 named synthetic faults: each of eight channels
zeroed separately, 0.5× all-channel gain, source-q95 symmetric clipping,
50 Hz half-source-RMS sine, half-source-RMS baseline ramp, and a
channel-0 transient burst. Source-only q95/RMS values and definitions are
in the protocol and [results](ROAM_QUALITY_RESULTS.json). These are
controlled simulations, **not observed hardware failures**. The quality
index is an equal mean of six fault-family coordinates: one average of
the eight dropout positions plus five other named faults. Clean control
is excluded from that mean. The minimum individual named fault is also
reported because the family mean can hide one failing channel.

| Arm | Validation clean F1 | Validation synthetic mean | Validation worst named fault | Final clean F1 | Final synthetic mean | Final worst named fault |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F0v2 | **1.0000** | **0.9110** | 0.4960 | **1.0000** | **0.9190** | 0.4929 |
| F0v2+F2a | 0.9054 | 0.7221 | 0.4040 | 1.0000 | 0.8048 | 0.4758 |
| F0v2+F3c | 0.8782 | 0.6915 | 0.3183 | 0.8934 | 0.7621 | 0.4363 |
| F0v2+F2a+F3c | 0.9054 | 0.6856 | 0.2865 | 0.9346 | 0.6924 | 0.3690 |

F0v2 has the best predefined synthetic-quality mean in validation and
descriptive final. Its minimum *fault-family* score is 0.7481/0.7667
for half gain; its much lower minimum **individual** fault is channel-2
dropout at 0.4960/0.4929. The worst subject-by-fault F0v2 cell is
0.2051 under channel-1 dropout in both phases. For the F2a/F3c arms,
50 Hz contamination often causes the lowest named-fault F1. Thus even
the best mean does not establish a safe worst-channel policy.

The [saved predictions](ROAM_QUALITY_PREDICTIONS.csv) contain 5,040
arm–phase–fault–bout rows. The [paired analysis](roam_quality_paired.py)
has 168 matched pooled/subject cells and 112 exact pooled-score replays,
with the four-arm family screen, conditional additions, error overlap
and F2a×F3c interaction bound by the
[audit](ROAM_QUALITY_PAIRED_AUDIT.json). The
[delivery audit](ROAM_QUALITY_DELIVERY_AUDIT.json) binds 1,176 new
canonical family/conditional/error rows. The same native bouts recur
across faults, so these are paired stress tests rather than independent
biological trials. No probability here measures fault prevalence,
real-electrode contact, online transition performance, or the user's
250 Hz device.
