# Sensor layouts

| Dataset | Layout | Channels | Rate | IMU | Similarity to own 8-channel ring |
|---|---|---:|---:|---|---|
| Historical DS2 force | targeted FDS, ED and FPL bipolar sites | 3 | 1500 Hz | no | LOW |
| LibEMG Contraction Intensity | eight equally spaced bipolar pairs in a circumferential forearm cuff | 8 | 1000 Hz | no | HIGH |
| LibEMG Electrode Shift | Myo circumferential armband | 8 | 200 Hz | not used | HIGH |
| UniBo-INAIL | named ECU, EDC, FCR and FCU muscle regions | 4 | 500 Hz | no | LOW for deployment, HIGH for anatomical interpretation |
| EMG-EPN612 | Myo circumferential armband | 8 | 200 Hz | JSON includes Myo inertial streams | HIGH |
| sEMG-MANUS | Myo circumferential armband | 8 | 200 Hz | quaternion, acceleration and gyro | HIGH |
| EMG-FMG | eight EMG channels plus a separate eight-channel FMG strap | 8 EMG | 2000 Hz | no | MEDIUM |
| GREAT | two staggered rows around forearm | 16 | 2000 Hz | no | MEDIUM-HIGH |
| NinaPro DB6 | bracelet configuration of Delsys Trigno sensors | 14 | 2000 Hz | yes | MEDIUM |
| GRABMyo | two forearm rings and two wrist rings; four unused amplifier channels | 28 usable of 32 | 2048 Hz | no | MEDIUM for subset studies |
| NinaPro DB5 | two adjacent Myo armbands offset by about 22.5 degrees | 16 | 200 Hz | yes | MEDIUM-HIGH |
| FORS-EMG | four sites near elbow and four at mid forearm | 8 | 985 Hz | no | LOW-MEDIUM |
| Three-position replacement | eight circular Ag/AgCl bipolar pairs | 8 | 1000 Hz | no | HIGH |
| Hyser | four 8 by 8 high-density grids | 256 | 2048 Hz | no | LOW for deployment, HIGH for observability |

No adapter may create missing electrodes by duplication, interpolation or zero padding.
Ring-relative operations use only datasets with a defensible circular order. UniBo keeps its
four named-muscle topology, and two-Myo data is evaluated as explicitly defined bands.
