# F2 spatial candidates on unseen GRABMyo users

The [frozen protocol](F2_GRAB_CANDIDATES_PROTOCOL.json) extends the
[GRABMyo Day1 cross-user study](GRAB_USER_PROTOCOL.json). Source users 1–4
fit all feature state and balanced source classifiers; users 5–6 validate
and users 7–8 form a descriptive final set. All 224 native source/target
recordings are verified against the official SHA-256 manifest. The eight
channels have a native 2048 Hz rate; each recording contributes the same
twenty disjoint 250 ms windows and trial mean/std aggregation as the parent.

The new F0v2 probabilities replay the parent **exactly** (maximum absolute
error 0). F2a below comes from that frozen parent; the new
[result](F2_GRAB_CANDIDATES_RESULTS.json) and
[336 saved predictions](F2_GRAB_CANDIDATES_PREDICTIONS.csv) compare F0v2,
document F2b CSP and F2c SPD tangent on the same 112 unseen-user target
recordings. The [readback test](../../tests/test_f2_grab_candidates_delivery.py)
checks parent hashes, trial identities, probability sums and saved scores.

| Phase | Arm | Macro F1 | Log loss | Brier | Minimum user F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation users 5–6 | F0v2 | **0.8054** | **0.4650** | **0.2854** | **0.7099** |
| Validation users 5–6 | + F2a trace covariance | 0.7485 | 0.6442 | 0.3240 | 0.5909 |
| Validation users 5–6 | + F2b document CSP | 0.7144 | 0.8297 | 0.4091 | 0.4804 |
| Validation users 5–6 | + F2c SPD tangent | 0.7371 | 0.6960 | 0.3584 | 0.5322 |
| Final users 7–8 | F0v2 | **0.9458** | **0.1994** | **0.1074** | **0.8877** |
| Final users 7–8 | + F2a trace covariance | 0.8899 | 0.3918 | 0.1699 | 0.7594 |
| Final users 7–8 | + F2b document CSP | 0.8060 | 0.5375 | 0.2615 | 0.7242 |
| Final users 7–8 | + F2c SPD tangent | 0.8940 | 0.3497 | 0.1886 | 0.7874 |

All three spatial additions reduce macro F1, raise log loss and lower the
minimum-user F1 on validation and descriptive final users. This is direct
negative-transfer evidence against promoting any one F2 family as an
unconditional new-user default. The earlier
[wearing](F2_WEARING_CANDIDATES_REPORT.md) and
[MANUS session](F2_MANUS_CANDIDATES_REPORT.md) axes answer different
questions: a candidate can help a known user under one condition while
hurting transfer to another user.

Final users appeared in earlier project studies and are not a fresh
project-wide holdout. This public 2048 Hz dataset is not the user's 250 Hz
acquisition device or a live recognition test.
