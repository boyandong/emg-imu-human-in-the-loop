# Document F2b CSP: isolated wearing-domain increment

The frozen [protocol](F2B_WEARING_PROTOCOL.json) isolates the document's
uncentered, one-vs-rest CSP formula from the F0v2 source-Rest backbone on a
public eight-channel, 200 Hz electrode-shift task. Each subject has its own
training-domain filters, scaler and classifier. The same native trials,
classes and source-only training rule as the frozen wearing screen are used.
All 240 F0v2 held-out probabilities replay exactly; the
[result](F2B_WEARING_RESULTS.json) and [480 saved predictions](F2B_WEARING_PREDICTIONS.csv)
retain the split identities and source hashes. F2b adds 20 features to the
48-dimensional F0v2 baseline for five native classes.

| Split | Arm | Pooled macro-F1 | Log loss | Minimum subject macro-F1 |
|---|---|---:|---:|---:|
| Validation users 15–17 | F0v2 | 0.5775 | 1.2916 | 0.4286 |
| Validation users 15–17 | F0v2 + document F2b | 0.5444 | 1.3655 | 0.3217 |
| Descriptive final users 18–20 | F0v2 | 0.6170 | 1.0838 | 0.4453 |
| Descriptive final users 18–20 | F0v2 + document F2b | 0.6510 | 1.0203 | 0.4743 |

The isolated addition worsens all three validation measures. Its favorable
final metrics are descriptive and do not justify promotion. The exact F2b
matrix/eigenvector/variance equations have an independent analytical test in
`tests/test_document_signal.py`; this native screen tests the fitted feature
with source-only models rather than proving universal spatial robustness.
It does not establish historical CSP equivalence, cross-day transfer or
own-device 250 Hz behavior.
