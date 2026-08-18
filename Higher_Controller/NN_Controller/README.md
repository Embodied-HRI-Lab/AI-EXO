# Unified 100-Hz neural controller

`NN_PC_Controller.py` is the hardware controller. This directory deliberately
packages one default policy:

```text
models/
└── flat_thigh_imu_tcn_balanced_4p2nm_100hz.pt
```

The policy is a causal, left/right-equivariant TCN. At 100 Hz it consumes the
latest 64 frames (0.64 seconds) of the standing-zeroed thigh IMU signals:

```text
right thigh pitch, left thigh pitch,
right thigh angular velocity, left thigh angular velocity
```

These are gravity-referenced thigh angles, not pelvis-relative hip joint
angles. Angles use radians and angular velocities use radians/second. The
learned network does **not** receive previous Exo commands or measured Exo torque.
The immediately preceding output is used only by the external hard slew
limiter; it is not a network input.

The model was distilled from the selected simulation assistance using
standing-zeroed simulated thigh IMU signals, sensor/domain randomization, and
gait-phase transfer on the recorded LZN IMU trials. Samsung torque was not used
as a training label. At full runtime scale the packaged policy has:

- nominal torque range: `+/-4.2 Nm`;
- slew limit: `0.21 Nm` per 100-Hz frame;
- exactly equal-and-opposite left/right nominal torque;
- no additional local smoothing or low-pass filter;
- single-threaded PyTorch inference (well within the 10 ms frame budget);
- a standstill gate that holds zero torque while the calibrated thigh angles
  and the complete 64-frame velocity history remain near zero.

The scale is uniform: the sign balance and timing are preserved.

## Run

Dry-run is the default and always sends zero torque to the Teensy:

```bash
python NN_PC_Controller.py --display print
```

The packaged TCN is selected automatically. An explicit path is optional:

```bash
python NN_PC_Controller.py \
  --model models/flat_thigh_imu_tcn_balanced_4p2nm_100hz.pt \
  --display plot
```

The runtime assistance multiplier starts at `0.1`. On Windows, Up/Down changes
it by `0.1`; multiplier `1.0` corresponds to the nominal `+/-4.2 Nm` policy.
The final PC safety clamp remains `+/-5 Nm`.

Only use `--arm` after checking IMU signs, standing-zero calibration, dry-run
output, motor direction, communication timeouts, and emergency-stop behavior.

## Optional file

`pc_nn_plot_worker.py` is kept separate so plotting cannot block the 100-Hz
control loop. Runtime CSV files are written under `logs/`, which is ignored by
Git.
