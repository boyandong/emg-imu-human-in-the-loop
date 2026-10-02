# F8 fixed-routing cross-day GRABMyo control

The [protocol](F8_GRAB_DAY_PROTOCOL.json) replays two frozen Day1-trained public GRABMyo providers: F0v2 and F0v2 plus frequency direction. All 1,344 selected source files match the publisher's checksum list, and all 896 Day2/Day3 provider probabilities replay the [earlier result](GRAB_DAY_V1_EXTENSION_PREDICTIONS.csv) exactly. For each of eight repeated subjects, trial 1 of each class on Day2 or Day3 supplies four session-calibration trials; trials 2–7 supply 24 disjoint evaluation trials. The long-term class signatures use that subject's 28 Day1 source trials in the frozen source-scaler coordinates. The cosine-agreement weight rule is the fixed rule already used for MANUS and wearing; no Day2 or Day3 outcome selects a rule.

| Held-out public phase | Method | Trials | Pooled macro-F1 | Log loss | Brier | Minimum user F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Day2 validation | Equal provider weights | 192 | .9581 | .127434 | .063003 | .8125 |
| Day2 validation | F8 cosine weights | 192 | .9581 | .127215 | .062906 | .8125 |
| Day3 descriptive final | Equal provider weights | 192 | .9259 | .289699 | .131234 | .6518 |
| Day3 descriptive final | F8 cosine weights | 192 | .9259 | .289175 | .130375 | .6518 |

The F8 rule changes probabilities slightly but changes no predicted gesture on either day. Its log-loss improvements are 0.000219 and 0.000525, respectively; this is too small to claim a meaningful recovery. The [result](F8_GRAB_DAY_RESULTS.json) retains 16 source/calibration/evaluation blocks, immutable long-term signature vectors and weights. The [768 saved held-out predictions](F8_GRAB_DAY_PREDICTIONS.csv) support independent reconstruction from the frozen provider predictions and exact metric readback. This is a retrospective four-class, eight-channel, 2048 Hz public comparison on already inspected subjects and days. It does not establish a deployment benefit, a new-person effect, a 250 Hz device result or a predictive use for the separate Song Rest-noise descriptor.
