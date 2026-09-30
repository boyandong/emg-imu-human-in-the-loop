# F2 spatial candidates on matched MANUS session trials

The [frozen protocol](F2_MANUS_CANDIDATES_PROTOCOL.json) extends the
[external-Rest MANUS study](MANUS_REST_TRANSFER_PROTOCOL.json) on the same six
users and 8-channel, 200 Hz native recordings. Session 1 fits every feature
and classifier. The F0v2 noise prior comes only from the frozen external
ROAM source Rest set because MANUS has no Rest class. Document F2b CSP and
F2c SPD tangent fit the MANUS Session 1 source only. Session 2 validates;
Session 3 is a descriptive final set. Each phase has the same 108 native
gesture trials, with slow, medium and fast strata observed within the phase.

The new F0v2 probabilities replay the parent **exactly** (maximum absolute
difference 0). The F2a comparison below is read from that frozen parent; the
new [result](F2_MANUS_CANDIDATES_RESULTS.json) and
[648 saved predictions](F2_MANUS_CANDIDATES_PREDICTIONS.csv) contain F0v2,
F0v2+F2b and F0v2+F2c on identical trial IDs. The
[readback test](../../tests/test_f2_manus_candidates_delivery.py) verifies
protocol/parent hashes, all trial sets, probability sums and scores.

| Phase | Arm | Pooled macro F1 | Log loss | Minimum user F1 | Minimum speed F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation, Session 2 | F0v2 | 0.3914 | **2.1203** | 0.1667 | 0.2845 |
| Validation, Session 2 | + F2a trace covariance | **0.5244** | 2.3949 | **0.2917** | **0.4677** |
| Validation, Session 2 | + F2b document CSP | 0.4567 | 2.4318 | 0.1875 | 0.4179 |
| Validation, Session 2 | + F2c SPD tangent | 0.4778 | 2.4545 | 0.2559 | 0.4342 |
| Final, Session 3 | F0v2 | 0.4600 | 1.7231 | 0.2907 | 0.4283 |
| Final, Session 3 | + F2a trace covariance | **0.5304** | **1.7124** | **0.3056** | 0.4491 |
| Final, Session 3 | + F2b document CSP | 0.4704 | 2.0647 | 0.2222 | 0.4270 |
| Final, Session 3 | + F2c SPD tangent | 0.5284 | 1.7266 | 0.2636 | **0.4607** |

All three additions improve validation pooled F1, but all worsen validation
log loss. F2a has the best validation F1 and minimum-user/speed F1; its
validation log loss is still worse than F0v2. The final F2c pooled F1 is
close to F2a, but that final observation cannot select a different arm.
On the earlier [wearing axis](F2_WEARING_CANDIDATES_REPORT.md), F2a had the
strongest validation F1 yet lost pooled F1 on final users, while F2b's
validation decline reversed on final. The two public axes therefore provide
better candidate coverage, not a universal spatial-family winner.

This is same-user public MANUS session transfer with a cross-dataset Rest
prior. It does not test new users, measured force or quality, the user's
250 Hz acquisition device, or unsegmented live recognition.
