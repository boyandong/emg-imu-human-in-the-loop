# F0 Rest-noise threshold on the available 8-channel Song sessions

The goal's F0 ZC/SSC/WAMP threshold must be set from source or calibration noise, never from test activity. The existing public F0 baseline estimates its thresholds from all source windows. This versioned feature-only check compares that reference with the separate document candidate fitted to **Rest windows alone**. Both fits use S01/S02 causal-filtered formal windows at 250 Hz. Their thresholds are frozen before S03/S04 are read; the latter sessions are never used to tune a threshold or train a classifier.

| Frozen source fit | Windows | Rest windows | Channel threshold range (archive units) |
|---|---:|---:|---:|
| All S01/S02 windows (reference) | 849 | — | 298.93–464.77 |
| Rest only (document candidate) | 213 | 213 | 91.02–178.57 |

| Read-only session | Windows | Windows with changed ZC/SSC/WAMP count | Mean Rest-minus-reference counts (ZC / SSC / WAMP) |
|---|---:|---:|---:|
| S03 | 416 | 92.55% | +77.41 / +85.57 / +109.30 |
| S04 | 431 | 92.58% | +71.98 / +79.64 / +103.78 |

The first 24 threshold-free F0 coordinates (RMS, MAV and WL) match exactly between the two routes; all 48 coordinates are finite. This establishes that the document formula changes native 8-channel feature values substantially. It does **not** show that either threshold policy improves gesture classification. No classifier is retrained and no default is changed.

The [protocol](F0_REST_NOISE_PROTOCOL.json) binds the previously audited session inventory and source hashes. [Results](F0_REST_NOISE_RESULTS.json) save the eight channel thresholds and session summaries; [per-window differences](F0_REST_NOISE_DIFFERENCES.csv) permit a portable readback without the private Song archive. The Song sessions cover one participant on one calendar day. S01/S02 readiness limitations and prior S04 inspection prevent prospective or product claims.
