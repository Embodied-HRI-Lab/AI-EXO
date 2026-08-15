# Unified 100-Hz neural controller

`NN_PC_Controller.py` is the hardware controller. The currently packaged model
is the flat-ground weighted-activation Direct policy distilled from the
trajectory optimizer.

All checkpoints consume hip flexion/extension kinematics at 100 Hz. Model-side
ordering is right then left; the controller converts this to the Teensy UART
ordering of left then right.

## Models

```text
models/
└── weighted_activation_direct_100hz.pt
```

The model uses eight causal frames at 100 Hz. Each frame contains right/left
hip angle, right/left hip angular velocity, and right/left measured Exo torque
normalized by 10 Nm. It outputs nominal right/left hip torque and applies a
0.5 Nm-per-frame slew limit. It was trained from the flat-ground solver whose
objective is muscle-mass/volume-weighted mean activation.

## Run

The default model is flat-ground Direct. Dry-run is the default and sends zero
torque:

```bash
python NN_PC_Controller.py --display print
```

Select any model by path; `--policy` is normally unnecessary because the
checkpoint declares Direct or target-PD itself:

```bash
python NN_PC_Controller.py \
  --model models/weighted_activation_direct_100hz.pt \
  --display plot
```

The initial assistance scale is 0.1. On Windows, Up/Down changes it by 0.1.
Commands are then limited to the current Teensy range of +/-5 Nm. Only use
`--arm` after checking IMU signs, standing-zero calibration, dry-run output,
motor direction, communication timeouts, and emergency-stop behavior.

## Optional files

`pc_nn_plot_worker.py` is kept separate so plotting cannot block the 100-Hz
control loop. The `validation/` directory contains offline evidence and is not
required in a minimal hardware copy. Runtime CSV files are written under
`logs/`, which is intentionally ignored by Git.
