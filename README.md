# AI-EXO: Sim-to-Real Reinforcement Learning for Exoskeleton Control

AI-EXO is a research repository for developing and evaluating sim-to-real
reinforcement-learning (RL) control for a bilateral hip exoskeleton. It connects
learned policies to physical hardware, provides a safety-aware real-time control
stack, records human–exoskeleton experiments, and includes tools for offline
analysis and visualization.

The current repository focuses on the **real-world deployment and evaluation**
side of the sim-to-real workflow. The simulator and policy-training pipeline are
not included here; trained PyTorch policy checkpoints are provided for hardware
inference.

## System overview

```text
Left and right IM948 IMUs
          │
          │ angle and angular velocity, 100 Hz
          ▼
PC higher-level controller
  ├─ standing-zero calibration
  ├─ state construction
  ├─ RL policy inference
  ├─ torque limiting and timeout checks
  └─ experiment logging / visualization
          │
          │ desired left/right torque over UART
          ▼
Teensy lower-level controller
  ├─ 1 kHz torque/current execution
  ├─ CAN motor communication
  ├─ slew-rate, velocity, and fault protection
  └─ actual-torque feedback
          │
          ▼
Bilateral hip motors
```

The hardware observation tuple exposed to a policy is:

```text
[
  left actual torque (Nm),
  right actual torque (Nm),
  left hip angle (rad),
  left hip angular velocity (rad/s),
  right hip angle (rad),
  right hip angular velocity (rad/s),
]
```

Flat target-PD uses measured torque; flat Direct and recurrent slope policies
use the four hip-kinematic values, with recurrent policies maintaining their
own preceding nominal command internally. The policy outputs left and right
assistive torque commands in Nm. Logged and
plotted angles remain in degrees and angular velocities in degrees per second;
conversion to SI units occurs immediately before neural-network inference.

## Repository structure

```text
AI-EXO/
├── Higher_Controller/
│   ├── NN_Controller/
│   │   ├── README.md                  # model selection and safety notes
│   │   ├── NN_PC_Controller.py       # unified 100 Hz MLP/GRU/MoE controller
│   │   ├── pc_nn_plot_worker.py      # realtime/offline NN visualization
│   │   ├── models/                    # flat and alternative slope checkpoints
│   │   └── validation/                # offline controller validation artifacts
│   ├── IMU_logger.py                  # bilateral IM948 data logger
│   ├── samsung_pc_controller.py       # conventional control baseline
│   ├── Raspberry_Pi_Controller.py     # Raspberry Pi communication layer
│   └── logs/                          # recorded hardware experiments
├── Lower_Controller/
│   └── Hip_PC_Controller/
│       ├── Hip_PC_Controller.ino      # Teensy execution and safety loop
│       └── Motor_Control_Tmotor.*     # T-Motor CAN interface
├── Data/
│   ├── data_analysis.py               # plotting and gait normalization
│   ├── paper_plot_IEEE_RAL.py         # research-figure utilities
│   ├── exo_logs/                      # exoskeleton experiment logs
│   └── logs/                          # with/without-exoskeleton datasets
└── git_push.sh                        # scoped Git commit/push helper
```

## Main capabilities

- Independent serial threads for the left IMU, right IMU, and Teensy feedback.
- Startup calibration of standing hip angle and static gyroscope bias.
- Auto-detected 100 Hz MLP, recurrent GRU, and learned-gate MoE inference.
- Dry-run operation by default, with explicit arming required for motor torque.
- Torque clamps, slew-rate limits, communication timeouts, and fault handling.
- CSV logging of joint kinematics, actual torque, and neural torque commands.
- Isolated real-time plotting so visualization does not block the control loop.
- Offline replay of recorded states through the packaged neural networks.
- Gait segmentation from hip-angle extrema and normalization to 0–100% gait.

## Requirements

### PC controller

- Python 3.8 or newer
- PyTorch
- NumPy
- pyserial
- Matplotlib
- pandas and SciPy for offline data analysis

Install the Python packages in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate          # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install torch numpy pyserial matplotlib pandas scipy seaborn
```

### Embedded controller

- Teensy-compatible Arduino toolchain
- `FlexCAN_T4`
- Two supported T-Motor drives connected over CAN
- UART connection between the PC/Raspberry Pi and Teensy

Serial port defaults in the Python controllers are examples and must be changed
to match the host computer.

## Quick start

### 1. Record bilateral IMU data

```bash
python Higher_Controller/IMU_logger.py --duration 60
```

Keep the wearer standing still during startup zero calibration.

### 2. Run the neural controller safely

Dry-run mode performs inference and logging but sends zero torque:

```bash
python Higher_Controller/NN_Controller/NN_PC_Controller.py \
  --display plot
```

Override the serial ports when necessary:

```bash
python Higher_Controller/NN_Controller/NN_PC_Controller.py \
  --left-port COM8 \
  --right-port COM6 \
  --teensy-port COM7 \
  --display plot
```

Only after verifying calibration, communication, motor direction, emergency
stop behavior, and dry-run outputs should torque transmission be enabled:

```bash
python Higher_Controller/NN_Controller/NN_PC_Controller.py \
  --display plot \
  --arm
```

### 3. Replay a log through a neural policy

This reconstructs the network state from each CSV row and plots predicted
torque against measured torque:

```bash
python Higher_Controller/NN_Controller/pc_nn_plot_worker.py \
  --csv Higher_Controller/logs/hjc07.csv \
  --policy auto \
  --output Data/figures/hjc07_nn_torque.png \
  --show
```

### 4. Analyze experiment data

Plot one selected region by row index:

```bash
python Data/data_analysis.py Data/exo_logs/hjc07.csv \
  --start-index 1000 \
  --end-index 5000 \
  --gaits 10 \
  --show
```

Analyze all CSV files in the experiment folder:

```bash
python Data/data_analysis.py Data/exo_logs
```

Generated analysis figures are saved under `Data/figures/` and ignored by Git.

## Neural policies

The unified controller auto-detects four checkpoint interfaces:

| Backend | Checkpoint metadata | Output interpretation |
|---|---|---|
| Flat MLP Direct | `direct_exo_kinematic_history` | Bilateral torque command |
| Flat MLP target-PD | `shared_leg_target_position_pd` | Target offset converted to torque |
| Slope GRU/MoE Direct | `recurrent_exo` | Stateful bilateral torque command |
| Slope GRU/MoE target-PD | `recurrent_exo_target_pd` | Equivalent target-PD command reconstructed as torque |

The model tree contains two flat checkpoints and two alternative six-checkpoint
slope families. All checkpoints require a 100 Hz control rate. See
`Higher_Controller/NN_Controller/README.md` for model selection and safety
notes.

## Data format

The principal exoskeleton CSV format is:

```text
elapsed_s
left_angle_x_deg
left_angular_velocity_x_dps
right_angle_x_deg
right_angular_velocity_x_dps
left_actual_torque_nm
right_actual_torque_nm
```

Neural-controller logs may additionally contain:

```text
left_nn_command_nm
right_nn_command_nm
```

## Safety

This repository controls powered wearable hardware. Incorrect commands,
coordinate signs, communication failures, or mechanical configuration can cause
injury or equipment damage.

- Keep the controller in dry-run mode until the full signal path is verified.
- Test with the exoskeleton unloaded before conducting a worn experiment.
- Confirm motor directions, torque signs, limits, and emergency-stop behavior.
- Use physical stops, an accessible emergency stop, and trained supervision.
- Never bypass timeout, feedback, velocity, fault, or torque-limit protections.
- Treat the supplied policies as research artifacts, not certified controllers.

## Research status

AI-EXO is research software intended for controlled laboratory development. It
is not a medical device and is not validated for clinical or unsupervised use.

## Hip Exoskeleton GUI & Voice Control System

A Python-based Graphical User Interface (GUI) and voice interaction system for real-time telemetry monitoring and AI voice control of a hip exoskeleton.

### Environment & Dependencies

* **Python Version:** Python 3.8+
* **Required Libraries:** Install all necessary dependencies using `pip`:

`pip install PyQt5 pyqtgraph pyserial pygame SpeechRecognition dashscope`

* **API Key Configuration:** Set your Alibaba DashScope API key in `gui_app.py`:

`DASHSCOPE_API_KEY = "your_dashscope_api_key_here"`

### How to Run

1. Connect the exoskeleton hardware to your PC via USB.
2. Launch the application: `python gui_app.py`
3. Select the correct serial port from the **ComPort** dropdown menu and click **Connect**.
4. Click the **Voice Ctrl: OFF** button to toggle voice interaction ON/OFF.

### Voice Control Commands

| Action | Supported Natural Voice Commands (Examples) |
| :--- | :--- |
| **Wake-up / Activate** | `"开始测试"` / `"测试"` |
| **Set Stiffness (K)** | `"把k设为 1.5"` / `"开设为 1.5"` |
| **Set Delay (Delay)** | `"delay设置为 200"` / `"带类设为 200"` |
| **Deactivate / Sleep** | `"退出测试"` / `"停止测试"` |
