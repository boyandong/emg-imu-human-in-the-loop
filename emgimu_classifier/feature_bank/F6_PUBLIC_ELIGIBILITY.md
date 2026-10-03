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

| Candidate | Verified public description | F6 eligibility decision |
|---|---|---|
| [EMG-EPN-107, Zenodo v1](https://zenodo.org/records/19829636) | Eight-channel forearm EMG and 50 Hz IMU with 12 gesture classes. Five 10 s synchronization gestures/user and `.mat` `userData` acquisition metadata are described. One 998.7 MB RAR is published, MD5 `90da8bb6d94e72d8821ae1859b4b878f`. | **Metadata unverified.** The public record does not say that the synchronization gesture measures a sensor-to-anatomical forward axis, or identify a separate neutral calibration. Inspect raw `userData`, signals, units and calibration protocol before an F6 run. The local host could not resolve `zenodo.org` on this review; no archive bytes were inspected. |
| [Myo Dataset, project README](https://github.com/michidk/myo-dataset) | Thirteen users; eight-channel EMG, orientation CSV and metadata including arm side/direction; rest-to-gesture recordings and some repeated sessions. | **Not eligible on documented fields.** Orientation and a reversed-arm flag are not raw six-axis acceleration/gyro or an anatomical forward-axis calibration. Reconsider only if additional raw fields and calibration provenance are found. |
| [UC2018 DualMyo, Zenodo](https://zenodo.org/records/1320922) | Two forearm Myos, eight gesture classes, five sessions and stated device placement relative to palm and palmaris-longus tendon. | **Not eligible on documented fields.** Approximate placement is not a measured per-session device-frame forward vector; the public record does not establish separate neutral/raw IMU calibration fields. |
| [ULTra-MoCap, Figshare](https://figshare.com/articles/dataset/ULTra-MoCap/28741943) | Synchronized multi-site IMU, three upper-arm sEMG sites and optical motion capture for upper-limb kinematics. Raw and processed archives are described. | **Different task/sensor geometry.** Useful as a possible kinematic-method reference, but not a sparse forearm hand-gesture F6 recognition benchmark. A static optical calibration alone must not be relabelled as this API's forward-axis vector. |

## Re-entry check

If a public archive becomes accessible, first inspect one subject's schema
and acquisition protocol. Require all of: raw accel/gyro in known units and
rate; synchronized EMG with trial IDs; disjoint neutral calibration; a
measured/guided anatomical forward vector expressed in the same device frame;
and sufficient independent gesture trials. Reject or mark N/A when any item
is absent. Do not infer the vector from held-out gesture labels or tune it on
held-out predictions. Only after these checks should a trial-disjoint F0
versus F0+F6 comparison be run. No F6 benefit or failure is inferred here.
