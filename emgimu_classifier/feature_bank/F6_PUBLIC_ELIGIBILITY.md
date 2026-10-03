# F6 calibrated body-frame public-data eligibility (2026-10-03)

This is an eligibility decision, not a trained F6 result. The implemented
`CalibratedBodyContextFamily` needs synchronized raw three-axis acceleration
and angular velocity, their actual sample rate and units, at least one second
of separate neutral IMU, a measured or guided **device-frame anatomical
forward vector**, and the identities of calibration trials excluded from test.
An IMU quaternion, an instructed forward gesture, a generic device mounting
description, or an optical static-pose file does not independently establish
that forward vector. Its analytical 15-output and rotation tests remain valid;
native calibrated-body recognition is not yet eligible.

## Full EPN107 archive schema check (2026-10-03)

The [public 998,699,553-byte RAR](https://zenodo.org/records/19829636) was
downloaded and matched its published MD5. The reproducible
[`epn107_f6_eligibility_probe.py`](../benchmarks/new_bank_v3/epn107_f6_eligibility_probe.py)
read all **107** `userData.mat` members; its
[aggregate result](../benchmarks/new_bank_v3/EPN107_F6_ELIGIBILITY.json)
contains archive SHA-256, schema counts and no participant values or raw
signals. It found 38 Myo records at 200 Hz EMG with nonempty 3-axis raw
accel/gyro, and 69 gForce records at 500 Hz EMG with empty raw accel/gyro
arrays in sync, training and testing trials. All 107 have five sync trials
with binary labels. No checked MAT structural field explicitly names an
anatomical forward axis, mounting-axis calibration, IMU units or IMU sample
rate. The public record states 50 Hz IMU, but does not define an anatomical
forward-axis calibration protocol. The binary sync labels or quaternion
arrays cannot supply that missing vector without an unsupported assumption.
Thus this archive is **not eligible for strict calibrated F6**; no F6 model
was trained on it. This is a complete archive-schema census, not a proof
that no unpublished acquisition protocol exists.

| Candidate | Verified public description | F6 eligibility decision |
|---|---|---|
| [EMG-EPN-107, Zenodo v1](https://zenodo.org/records/19829636) | Eight-channel forearm EMG with 50 Hz IMU reported, 12 gesture classes and five synchronization trials/user. The published 998.7 MB RAR MD5 matches the local copy; all 107 MAT schemas were inspected. | **Not eligible for strict F6.** Raw accel/gyro are empty for all 69 gForce users; the 38 Myo users have raw six-axis arrays but no explicit device-frame anatomical forward vector or IMU units in the checked MAT fields. Sync binary labels and quaternions are not that vector. See census above. |
| [Myo Dataset, project README](https://github.com/michidk/myo-dataset) | Thirteen users; eight-channel EMG, orientation CSV and metadata including arm side/direction; rest-to-gesture recordings and some repeated sessions. | **Not eligible on documented fields.** Orientation and a reversed-arm flag are not raw six-axis acceleration/gyro or an anatomical forward-axis calibration. Reconsider only if additional raw fields and calibration provenance are found. |
| [UC2018 DualMyo, Zenodo](https://zenodo.org/records/1320922) | Two forearm Myos, eight gesture classes, five sessions and stated device placement relative to palm and palmaris-longus tendon. | **Not eligible on documented fields.** Approximate placement is not a measured per-session device-frame forward vector; the public record does not establish separate neutral/raw IMU calibration fields. |
| [ULTra-MoCap, Figshare](https://figshare.com/articles/dataset/ULTra-MoCap/28741943) | Synchronized multi-site IMU, three upper-arm sEMG sites and optical motion capture for upper-limb kinematics. Raw and processed archives are described. | **Different task/sensor geometry.** Useful as a possible kinematic-method reference, but not a sparse forearm hand-gesture F6 recognition benchmark. A static optical calibration alone must not be relabelled as this API's forward-axis vector. |

## Re-entry check

For a different public candidate, or if EPN107's authors publish an
independent calibration protocol, require all of: raw accel/gyro in known
units and rate; synchronized EMG with trial IDs; disjoint neutral calibration;
a measured/guided anatomical forward vector expressed in the same device
frame; and sufficient independent gesture trials. Reject or mark N/A when
any item is absent. Do not infer the vector from held-out gesture labels or
tune it on held-out predictions. Only after these checks should a
trial-disjoint F0 versus F0+F6 comparison be run. No F6 benefit or failure
is inferred here.
