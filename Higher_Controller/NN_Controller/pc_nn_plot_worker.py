"""
Standalone realtime/offline plot worker for NN_PC_Controller.py.

IMPORTANT:
    Realtime stdin mode does NOT import torch or the controller module. On
    Windows this prevents the plot process from reinitializing PyTorch's Intel
    OpenMP runtime (OMP Error #15). Offline ``--csv`` mode explicitly loads the
    controller policy because it performs neural-network inference itself.

Input:
    tab-separated samples on stdin:
        elapsed_s
        left_angle_deg
        right_angle_deg
        left_cmd_nm
        right_cmd_nm
        left_actual_nm
        right_actual_nm
        left_age_s
        right_age_s
        teensy_age_s
        control_ok
        enabled

Offline example:
    python pc_nn_plot_worker.py --csv ../logs/hjc07.csv --policy direct --show

ExpertData example:
    python pc_nn_plot_worker.py --expert-trial ../../ExpertData/BT02/\
normal_walk_1_1_0-6_on --policy direct --show
"""

from __future__ import annotations

import argparse
import csv
import math
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path



def run_offline_nn_plot(
    csv_path: Path,
    policy_type: str,
    model_path: Path | None,
    output_path: Path | None,
    show: bool,
    start_index: int = 0,
    end_index: int | None = None,
) -> None:
    """Infer NN torque and plot a selected row interval from a CSV.

    Rows before ``start_index`` are passed through the stateful policy as a
    warm-up so the first plotted output has the correct observation history.
    ``end_index`` follows normal Python slicing semantics and is excluded.
    """
    # Keep torch/controller imports out of the realtime stdin worker mode.
    from NN_PC_Controller import NeuralTorqueInterface   
    import matplotlib.pyplot as plt   

    required = (
        "elapsed_s",
        "left_angle_x_deg",
        "left_angular_velocity_x_dps",
        "right_angle_x_deg",
        "right_angular_velocity_x_dps",
        "left_actual_torque_nm",
        "right_actual_torque_nm",
    )
    time_values: list[float] = []
    left_angles: list[float] = []   
    right_angles: list[float] = []  
    left_actual_values: list[float] = []  
    right_actual_values: list[float] = []  
    left_nn_values: list[float] = []  
    right_nn_values: list[float] = []    

    policy = NeuralTorqueInterface(policy_type, model_path)
    if not policy.available:
        raise RuntimeError(policy.load_message)  
    print(policy.load_message)
    resolved_policy_type = policy.policy_type
    policy.reset()   

    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(missing)}")

        for data_index, row in enumerate(reader):
            if end_index is not None and data_index >= end_index:
                break
            row_number = data_index + 2  # Account for the CSV header.
            try:
                elapsed = float(row["elapsed_s"])
                left_angle_deg = float(row["left_angle_x_deg"])
                left_velocity_dps = float(row["left_angular_velocity_x_dps"])
                right_angle_deg = float(row["right_angle_x_deg"])
                right_velocity_dps = float(row["right_angular_velocity_x_dps"])
                left_actual = float(row["left_actual_torque_nm"])
                right_actual = float(row["right_actual_torque_nm"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric data at CSV row {row_number}") from exc

            # Must match Observation in NN_PC_Controller.py exactly:
            # actual torque [Nm], hip angle [rad], angular velocity [rad/s].
            state = (
                left_actual,
                right_actual,
                math.radians(left_angle_deg),
                math.radians(left_velocity_dps),
                math.radians(right_angle_deg),
                math.radians(right_velocity_dps),
            )
            output_torque = policy.get_torque(state)
            if output_torque is None:
                raise RuntimeError(
                    f"NN inference failed at CSV row {row_number}: "
                    f"{policy.last_error or 'unknown error'}"
                )

            if data_index < start_index:
                continue

            time_values.append(elapsed)
            left_angles.append(left_angle_deg)    
            right_angles.append(right_angle_deg)    
            left_actual_values.append(left_actual)   
            right_actual_values.append(right_actual)
            left_nn_values.append(output_torque[0])  
            right_nn_values.append(output_torque[1])  

    if not time_values:
        raise ValueError(
            f"No data rows found in selected interval "
            f"[{start_index}:{end_index}] of {csv_path}"
        )

    fig, (ax_angle, ax_torque) = plt.subplots(2, 1, sharex=True, figsize=(11, 7))
    ax_angle.plot(time_values, left_angles, label="Left hip angle") 
    ax_angle.plot(time_values, right_angles, label="Right hip angle")  
    ax_angle.set_ylabel("Angle (deg)") 
    ax_angle.legend(loc="upper right") 
    ax_angle.grid(True, alpha=0.25)  

    ax_torque.plot(time_values, left_nn_values, label="Left NN output")
    ax_torque.plot(time_values, right_nn_values, label="Right NN output")
    ax_torque.plot(time_values, left_actual_values, "--", alpha=0.65,
                   label="Left actual")
    ax_torque.plot(time_values, right_actual_values, "--", alpha=0.65,
                   label="Right actual")
    ax_torque.set_xlabel("Elapsed time (s)")
    ax_torque.set_ylabel("Torque (Nm)")
    ax_torque.legend(loc="upper right", ncols=2)
    ax_torque.grid(True, alpha=0.25)
    fig.suptitle(f"{csv_path.stem} - {resolved_policy_type} NN inference")
    fig.tight_layout()

    output_path = output_path or csv_path.with_name(
        f"{csv_path.stem}_{resolved_policy_type}_nn_torque.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(
        f"Plotted {len(time_values)} states; valid NN outputs="
        f"{policy.valid_outputs}. Saved: {output_path}"
    )
    if show:
        plt.show() 
    else:
        plt.close(fig)  


def run_expertdata_nn_plot(
    trial_path: Path,
    policy_type: str,
    model_path: Path | None,
    output_path: Path | None,
    show: bool,
    start_index: int = 0,
    end_index: int | None = None,
    sample_stride: int = 2,
    left_angle_sign: float = 1.0,
    right_angle_sign: float = 1.0,
) -> None:
    """Build NN states from one ExpertData trial and plot output torque.

    Filtered joint angles, joint velocities, and measured exoskeleton torques
    are aligned by timestamp. ExpertData is normally 200 Hz, so the default
    stride of two replays the packaged policy at its required 100 Hz rate.
    """
    from NN_PC_Controller import NeuralTorqueInterface   
    import matplotlib.pyplot as plt  
    import numpy as np   
    import pandas as pd   
 
    trial_path = trial_path.expanduser().resolve()
    trial_dir = trial_path if trial_path.is_dir() else trial_path.parent
    if not trial_dir.is_dir():
        raise FileNotFoundError(f"ExpertData trial does not exist: {trial_path}")

    if trial_path.is_file() and trial_path.name.endswith("_angle_filt.csv"):
        angle_path = trial_path
    else:
        angle_files = sorted(trial_dir.glob("*_angle_filt.csv"))
        if len(angle_files) != 1:
            raise FileNotFoundError(
                f"Expected one *_angle_filt.csv in {trial_dir}, "
                f"found {len(angle_files)}"
            )
        angle_path = angle_files[0]

    prefix = angle_path.name[:-len("_angle_filt.csv")]
    velocity_path = trial_dir / f"{prefix}_velocity.csv"
    exo_path = trial_dir / f"{prefix}_exo.csv"  
    for required_path in (velocity_path, exo_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required ExpertData file missing: {required_path}")

    angle = pd.read_csv(angle_path, encoding="utf-8-sig")   
    velocity = pd.read_csv(velocity_path, encoding="utf-8-sig")    
    exo = pd.read_csv(exo_path, encoding="utf-8-sig")     
    required_columns = {
        angle_path: ("time", "hip_flexion_l", "hip_flexion_r"),
        velocity_path: (
            "time", "hip_flexion_l_velocity", "hip_flexion_r_velocity",
        ),
        exo_path: (
            "time", "hip_angle_l_torque_measured", "hip_angle_r_torque_measured",
        ),
    }   
    for path, columns in required_columns.items():   
        frame = angle if path == angle_path else velocity if path == velocity_path else exo
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")   

    angle = angle.loc[:, list(required_columns[angle_path])].apply(
        pd.to_numeric, errors="coerce"
    ).dropna().sort_values("time")
    velocity = velocity.loc[:, list(required_columns[velocity_path])].apply(
        pd.to_numeric, errors="coerce"
    ).dropna().sort_values("time")
    exo = exo.loc[:, list(required_columns[exo_path])].apply(
        pd.to_numeric, errors="coerce"
    ).dropna().sort_values("time")   

    target_time = angle["time"].to_numpy(dtype=float)
    left_angle = left_angle_sign * angle["hip_flexion_l"].to_numpy(dtype=float)
    right_angle = right_angle_sign * angle["hip_flexion_r"].to_numpy(dtype=float)

    def align(frame, column):
        return np.interp(
            target_time,
            frame["time"].to_numpy(dtype=float),
            frame[column].to_numpy(dtype=float),
        )

    left_velocity = left_angle_sign * align(velocity, "hip_flexion_l_velocity")
    right_velocity = right_angle_sign * align(velocity, "hip_flexion_r_velocity")
    left_actual = align(exo, "hip_angle_l_torque_measured")   
    right_actual = align(exo, "hip_angle_r_torque_measured")   

    policy = NeuralTorqueInterface(policy_type, model_path)
    if not policy.available:
        raise RuntimeError(policy.load_message)
    print(policy.load_message)
    resolved_policy_type = policy.policy_type
    policy.reset()

    retained = {
        "time": [], "left_angle": [], "right_angle": [],
        "left_velocity": [], "right_velocity": [],
        "left_actual": [], "right_actual": [],
        "left_nn": [], "right_nn": [],
    }
    stop = len(target_time) if end_index is None else min(end_index, len(target_time))
    for data_index in range(0, stop, sample_stride):
        observation = (
            float(left_actual[data_index]),
            float(right_actual[data_index]),
            math.radians(float(left_angle[data_index])),
            math.radians(float(left_velocity[data_index])),
            math.radians(float(right_angle[data_index])),
            math.radians(float(right_velocity[data_index])),
        )
        output = policy.get_torque(observation)
        if output is None:
            raise RuntimeError(
                f"NN inference failed at ExpertData index {data_index}: "
                f"{policy.last_error or 'unknown error'}"
            )
        if data_index < start_index:
            continue
        retained["time"].append(float(target_time[data_index] - target_time[0]))
        retained["left_angle"].append(float(left_angle[data_index]))
        retained["right_angle"].append(float(right_angle[data_index]))
        retained["left_velocity"].append(float(left_velocity[data_index]))
        retained["right_velocity"].append(float(right_velocity[data_index]))
        retained["left_actual"].append(float(left_actual[data_index]))
        retained["right_actual"].append(float(right_actual[data_index]))
        retained["left_nn"].append(output[0])
        retained["right_nn"].append(output[1])

    if not retained["time"]:
        raise ValueError(f"No ExpertData samples in [{start_index}:{end_index}]")

    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(retained["time"], retained["left_angle"], label="Left")
    axes[0].plot(retained["time"], retained["right_angle"], label="Right")
    axes[0].set_ylabel("Hip angle (deg)")
    axes[1].plot(retained["time"], retained["left_velocity"], label="Left")
    axes[1].plot(retained["time"], retained["right_velocity"], label="Right")
    axes[1].set_ylabel("Hip velocity (deg/s)")
    axes[2].plot(retained["time"], retained["left_nn"], label="Left NN")
    axes[2].plot(retained["time"], retained["right_nn"], label="Right NN")
    axes[2].plot(retained["time"], retained["left_actual"], "--", alpha=0.65,
                 label="Left measured")
    axes[2].plot(retained["time"], retained["right_actual"], "--", alpha=0.65,
                 label="Right measured")
    axes[2].set_ylabel("Torque (Nm)")
    axes[2].set_xlabel("Elapsed trial time (s)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, ncols=2, loc="upper right")
    figure.suptitle(
        f"{trial_dir.name} — {resolved_policy_type} NN torque from ExpertData"
    )
    figure.tight_layout()

    output_path = (
        output_path
        or trial_dir / f"{prefix}_{resolved_policy_type}_nn_torque.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight") 
    torque_csv_path = output_path.with_suffix(".csv")  
    pd.DataFrame(
        {
            "elapsed_s": retained["time"],
            "left_hip_angle_deg": retained["left_angle"],
            "left_hip_velocity_dps": retained["left_velocity"],
            "right_hip_angle_deg": retained["right_angle"],
            "right_hip_velocity_dps": retained["right_velocity"],
            "left_measured_torque_nm": retained["left_actual"],
            "right_measured_torque_nm": retained["right_actual"],
            "left_nn_torque_nm": retained["left_nn"],
            "right_nn_torque_nm": retained["right_nn"],
        }
    ).to_csv(torque_csv_path, index=False)
    print(
        f"ExpertData states={len(retained['time'])}, stride={sample_stride}, "
        f"valid NN outputs={policy.valid_outputs}. "
        f"Saved: {output_path} and {torque_csv_path}"
    )
    if show:
        plt.show()
    else:
        plt.close(figure)


def state_text(age_s: float, stale_s: float, timeout_s: float) -> str:
    if age_s > timeout_s:
        return "TIMEOUT"
    if age_s > stale_s:
        return "STALE"
    return "OK"


def stdin_reader(out_queue: queue.Queue, eof_event: threading.Event) -> None:
    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()

            if not line:
                continue

            fields = line.split("\t")
            if len(fields) != 12:
                continue

            try:
                sample = (
                    float(fields[0]),
                    float(fields[1]),
                    float(fields[2]),
                    float(fields[3]),
                    float(fields[4]),
                    float(fields[5]),
                    float(fields[6]),
                    float(fields[7]),
                    float(fields[8]),
                    float(fields[9]),
                    fields[10] == "1",
                    fields[11] == "1",
                )
            except ValueError:
                continue

            try:
                out_queue.put_nowait(sample)
            except queue.Full:
                try:
                    out_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    out_queue.put_nowait(sample)
                except queue.Full:
                    pass
    finally:
        eof_event.set()


def main() -> None:  
    parser = argparse.ArgumentParser()  
    offline_source = parser.add_mutually_exclusive_group()
    offline_source.add_argument(  
        "--csv", type=Path,
        help="Offline mode: build NN states from this exoskeleton CSV",
    )
    offline_source.add_argument(
        "--expert-trial", type=Path,
        help="Offline mode: infer from an ExpertData trial folder/angle CSV",
    )
    parser.add_argument(
        "--policy", choices=("auto", "direct", "pd"), default="auto",
        help="Optional checkpoint-interface validation (default: auto)",
    )
    parser.add_argument(
        "--model", type=Path,
        help="Optional .pt model; defaults to the selected controller model",
    )
    parser.add_argument("--output", type=Path, help="Offline plot output path")
    parser.add_argument("--show", action="store_true", help="Show offline plot")
    parser.add_argument(
        "--start-index", type=int, default=0,
        help="Offline mode: first CSV data row to plot (included)",
    )
    parser.add_argument(
        "--end-index", type=int,
        help="Offline mode: final CSV data-row boundary (excluded)",
    )
    parser.add_argument(
        "--expert-stride", type=int, default=2,
        help="ExpertData source-row stride; 2 converts 200 Hz to 100 Hz",
    )
    parser.add_argument(
        "--left-angle-sign", type=float, choices=(-1.0, 1.0), default=1.0,
        help="ExpertData left angle/velocity coordinate sign",
    )
    parser.add_argument(
        "--right-angle-sign", type=float, choices=(-1.0, 1.0), default=1.0,
        help="ExpertData right angle/velocity coordinate sign",
    )
    parser.add_argument("--refresh-hz", type=float, default=30.0)
    parser.add_argument("--window-s", type=float, default=10.0)
    parser.add_argument("--stale-warning-s", type=float, default=0.05)
    parser.add_argument("--imu-timeout-s", type=float, default=0.15)
    parser.add_argument("--teensy-timeout-s", type=float, default=0.20)
    args = parser.parse_args()

    if args.csv is not None or args.expert_trial is not None:
        if args.start_index < 0:
            parser.error("--start-index must be non-negative")
        if args.end_index is not None and args.end_index <= args.start_index:
            parser.error("--end-index must be greater than --start-index")
        if args.expert_stride < 1:
            parser.error("--expert-stride must be at least 1")

    if args.csv is not None:
        run_offline_nn_plot(
            csv_path=args.csv.expanduser().resolve(),
            policy_type=args.policy,
            model_path=(
                args.model.expanduser().resolve() if args.model is not None else None
            ),
            output_path=(
                args.output.expanduser().resolve()
                if args.output is not None else None
            ),
            show=args.show,
            start_index=args.start_index,
            end_index=args.end_index,
        )
        return

    if args.expert_trial is not None:
        run_expertdata_nn_plot(
            trial_path=args.expert_trial,
            policy_type=args.policy,
            model_path=(
                args.model.expanduser().resolve() if args.model is not None else None
            ),
            output_path=(
                args.output.expanduser().resolve()
                if args.output is not None else None
            ),
            show=args.show,
            start_index=args.start_index,
            end_index=args.end_index,
            sample_stride=args.expert_stride,
            left_angle_sign=args.left_angle_sign,
            right_angle_sign=args.right_angle_sign,
        )
        return

    # Matplotlib/Numpy live only in this helper process.
    import matplotlib.pyplot as plt

    history_len = max(int(args.window_s * 100.0 * 1.5), 300)  

    t_hist = deque(maxlen=history_len)
    la_hist = deque(maxlen=history_len)
    ra_hist = deque(maxlen=history_len)

    lcmd_hist = deque(maxlen=history_len)
    rcmd_hist = deque(maxlen=history_len)
    lact_hist = deque(maxlen=history_len)
    ract_hist = deque(maxlen=history_len)

    samples: queue.Queue = queue.Queue(maxsize=500)
    eof_event = threading.Event()

    reader = threading.Thread(
        target=stdin_reader,
        args=(samples, eof_event),
        name="PlotStdinReader",
        daemon=True,
    )
    reader.start()

    fig, (ax_angle, ax_torque) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(11, 7),
    )

    line_la, = ax_angle.plot([], [], label="Left X angle")
    line_ra, = ax_angle.plot([], [], label="Right X angle")
    ax_angle.axhline(0.0, linewidth=0.8)
    ax_angle.set_ylabel("Angle (deg)")
    ax_angle.set_title("Standing-zeroed X-axis hip angle")
    ax_angle.legend(loc="upper right")
    ax_angle.grid(True, alpha=0.25)

    line_lcmd, = ax_torque.plot([], [], label="Left NN command")
    line_rcmd, = ax_torque.plot([], [], label="Right NN command")
    line_lact, = ax_torque.plot([], [], label="Left actual")
    line_ract, = ax_torque.plot([], [], label="Right actual")
    ax_torque.axhline(0.0, linewidth=0.8) 
    ax_torque.set_ylabel("Torque (Nm)") 
    ax_torque.set_xlabel("Elapsed time (s)") 
    ax_torque.set_title("NN command and motor actual torque")
    ax_torque.legend(loc="upper right")
    ax_torque.grid(True, alpha=0.25)  

    status_title = fig.suptitle("Waiting for samples...")  

    latest_left_age = math.inf
    latest_right_age = math.inf
    latest_teensy_age = math.inf
    latest_control_ok = False
    latest_enabled = False

    refresh_period = 1.0 / max(args.refresh_hz, 1.0)
    next_refresh = time.perf_counter()

    while plt.fignum_exists(fig.number):
        got_data = False

        while True:
            try:
                item = samples.get_nowait()
            except queue.Empty:
                break

            (
                elapsed,
                left_angle,
                right_angle,
                left_cmd,
                right_cmd,
                left_actual,
                right_actual,
                left_age,
                right_age,
                teensy_age,
                control_ok,
                enabled,
            ) = item

            t_hist.append(elapsed)
            la_hist.append(left_angle)
            ra_hist.append(right_angle)

            lcmd_hist.append(left_cmd)
            rcmd_hist.append(right_cmd)
            lact_hist.append(left_actual)
            ract_hist.append(right_actual)

            latest_left_age = left_age
            latest_right_age = right_age
            latest_teensy_age = teensy_age
            latest_control_ok = control_ok
            latest_enabled = enabled
            got_data = True

        now = time.perf_counter()

        if got_data and len(t_hist) >= 2 and now >= next_refresh:
            x = list(t_hist)

            line_la.set_data(x, list(la_hist))
            line_ra.set_data(x, list(ra_hist))

            line_lcmd.set_data(x, list(lcmd_hist))
            line_rcmd.set_data(x, list(rcmd_hist))
            line_lact.set_data(x, list(lact_hist))
            line_ract.set_data(x, list(ract_hist))

            xmax = x[-1]
            xmin = max(x[0], xmax - args.window_s)
            xmax_visible = max(xmax, xmin + 0.1)

            ax_angle.set_xlim(xmin, xmax_visible)
            ax_torque.set_xlim(xmin, xmax_visible)

            ax_angle.relim()
            ax_angle.autoscale_view(scalex=False, scaley=True)
            ax_torque.relim()
            ax_torque.autoscale_view(scalex=False, scaley=True)

            left_state = state_text(
                latest_left_age,
                args.stale_warning_s,
                args.imu_timeout_s,
            )
            right_state = state_text(
                latest_right_age,
                args.stale_warning_s,
                args.imu_timeout_s,
            )
            teensy_state = (
                "OK"
                if latest_teensy_age <= args.teensy_timeout_s
                else "TIMEOUT"
            )

            status_title.set_text(
                f"L {latest_left_age*1000:5.1f} ms [{left_state}] | "
                f"R {latest_right_age*1000:5.1f} ms [{right_state}] | "
                f"T {latest_teensy_age*1000:5.1f} ms [{teensy_state}] | "
                f"NN={'OK' if latest_control_ok else 'ZERO'} | "
                f"{'ARMED' if latest_enabled else 'DRY RUN'}"
            )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            next_refresh = now + refresh_period

        # If parent closes stdin, there will be no more samples.
        if eof_event.is_set() and samples.empty():
            break

        plt.pause(0.001)

    try:
        plt.close(fig)
    except Exception:
        pass


if __name__ == "__main__":  
    main()   
