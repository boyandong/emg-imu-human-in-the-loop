# Frozen new-user EPN612 F7 confirmation protocol

This protocol is committed before any EMG/labels from EPN612 users 22–31 are
loaded by this experiment. They are absent from the frozen Core fit (users
1–15), selection (16–18), and the previously inspected final cohort (19–21).
The claim is a fresh cohort within this repository, not proof that no external
work has ever inspected these subjects.

* Primary endpoint: users 22–31 pooled, five calibration trials per class,
  all remaining complete trials held out. Compare fixed `0.5 Core + 0.5 F7`
  with Core on exactly the same held-out trials. A favorable confirmation
  requires lower log loss, non-worse macro F1 and Brier, lower log loss than
  fixed `0.5 Core + 0.5 uniform`, and lower log loss on at least 7 of 10
  subjects. Report each criterion independently, even when the conjunction
  fails. These criteria are descriptive guards, not a significance test.
* Secondary endpoints: 1- and 2-shot versions, per-user results, six-class
  recall, calibration IDs, trial probabilities, and F7/Core error overlap.
* Frozen source: EPN612 archive SHA-256
  `4ee8db037385e7bee1e6ac6f9e9eea4f0869e25f7825f5eb7e5be0dff4f93c21`;
  source-fitted Core state SHA-256
  `08e7e4e4af1644446911633411743306be33b3c6d08d584d5967252f4d199aef`;
  source validation predictions SHA-256
  `8c2a9f8c616e7c836d788ba9b3dcbd5cff92d706741aed576efc33a8445f6b31`.
  F7 selection script SHA-256
  `ee891ecba8d10b72eb4321d7e1dc596953061c4dca8ec49f3b35fa170643c45a`;
  exact affine-SPD implementation SHA-256
  `4ddd02d2260448aad4f3ab98e013b7213b8ac2a31e93bd89975ec40049239808`.
  Core is `F0+F3_Ring+F2b_CSP+F6_IMU`; no fit on fresh users.
* F7 is the already implemented exact affine-invariant SPD trial-prototype
  anchor. Use the previously fixed `20261003 + user` classwise RNG,
  nested 1/2/5-shot complete-trial selection, and calibration-only median
  pairwise prototype distance as temperature. Do not tune the feature,
  temperature, selection, fusion weight, or criteria from fresh labels.
* Replay source validation probabilities with maximum absolute error at most
  `1e-10` before reading users 22–31. Verify archive and source hashes and
  disjoint user/trial identities. Freeze the script and this protocol in Git
  before the first run. Run the fresh cohort once. If a code defect forces
  rerun, preserve the original output and disclose the amendment; do not
  call the amended result pristine.
* A positive public-data result does not prove performance on the user's
  own eight-channel electrode montage or on live hardware. A negative or
  mixed result is retained and blocks default promotion.
