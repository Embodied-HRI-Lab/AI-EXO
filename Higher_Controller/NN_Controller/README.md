# Unified 100-Hz neural controller

`NN_PC_Controller.py` is the only hardware controller. It reads checkpoint
metadata and automatically selects the correct inference backend:

- flat-ground eight-frame MLP Direct or target-PD;
- recurrent slope GRU Direct or target-PD;
- recurrent two-expert slope MoE Direct or target-PD.

All checkpoints consume hip flexion/extension kinematics at 100 Hz. Model-side
ordering is right then left; the controller converts this to the Teensy UART
ordering of left then right.

## Models

```text
models/
├── flat22/                    # 2 flat-ground MLP checkpoints
├── slope_free_slew05/         # 6 native-100-Hz, high-torque checkpoints
└── slope_adam_lowtorque/      # 6 lower-torque checkpoints
```

Each slope directory contains uphill, downhill, and learned-gate MoE policies,
with Direct and target-PD versions of each. The two slope directories are
alternative lineages, not stages that should be run together.

`slope_free_slew05` closely reproduces its native-100-Hz optimization teacher,
but its mean absolute command is roughly 6.5--6.7 Nm and will be clipped by the
current Teensy limit of 5 Nm at high assistance scale. `slope_adam_lowtorque`
has mean absolute commands around 0.68--1.70 Nm and is the safer first hardware
candidate, although its held-out teacher correlation is lower.

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
  --model models/slope_adam_lowtorque/uphill_direct_100hz.pt \
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
