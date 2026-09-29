# F4d public multi-user session spectral coordinates

The frozen [protocol](F4D_MANUS_SESSION_PROTOCOL.json) uses the same public
MANUS archive, six users and three native sessions as the existing speed
study. Each user's session 1 provides 18 complete native trials for the
long-term log-band reference. In sessions 2 and 3, six medium-speed gesture
trials per user set the current-session reference; the twelve distinct
slow/fast trials per user/session are evaluation-only. Both references give
equal mass to each native trial regardless of its number of 200 ms windows.
The [result](F4D_MANUS_SESSION_RESULTS.json) records all 108 long-term,
72 current-calibration and 144 held-out trial identities, archive/protocol
hashes, four source-fixed bands, 32-dimensional shift vectors and held-out
residuals. The split and finite-shape [test](../../tests/test_f4d_manus_session_delivery.py)
checks their disjointness.

Across the six users, mean absolute session-minus-long log-band coordinate
is 0.7719 in session 2 and 0.7758 in session 3. The held-out mean-minus-
session residual L2 norm averages 2.4959 and 2.1412, respectively, over
both held-out speed strata. These are descriptive signal coordinates,
not classification F1 or a measure of fatigue.

Medium-speed calibration and slow/fast evaluation deliberately preserve
separate native trials, but speed also changes between calibration and
evaluation. Thus the residual cannot be attributed solely to session or
electrode placement. Session numbers are not certified different calendar
days, and there is no own-device 250 Hz generalization or predictive
accuracy claim. This extends the earlier one-person Song F4d diagnostic
to a public six-user repeated-session setting without promoting F4d as a
default classifier family.
