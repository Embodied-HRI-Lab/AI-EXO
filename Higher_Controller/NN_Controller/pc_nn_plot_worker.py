"""
Standalone realtime plot worker for pc_nn_formal_controller_plotfix.py.

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
    fig.suptitle(f"{csv_path.stem} - {policy_type} NN inference")
    fig.tight_layout()

    output_path = output_path or csv_path.with_name(
        f"{csv_path.stem}_{policy_type}_nn_torque.png"
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
    parser.add_argument(  
        "--csv", type=Path,
        help="Offline mode: build NN states from this exoskeleton CSV",
    )
    parser.add_argument(
        "--policy", choices=("direct", "pd"), default="direct",
        help="Neural-network policy used in offline mode (default: direct)",
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
    parser.add_argument("--refresh-hz", type=float, default=30.0)
    parser.add_argument("--window-s", type=float, default=10.0)
    parser.add_argument("--stale-warning-s", type=float, default=0.05)
    parser.add_argument("--imu-timeout-s", type=float, default=0.15)
    parser.add_argument("--teensy-timeout-s", type=float, default=0.20)
    args = parser.parse_args()

    if args.csv is not None:
        if args.start_index < 0:
            parser.error("--start-index must be non-negative")
        if args.end_index is not None and args.end_index <= args.start_index:
            parser.error("--end-index must be greater than --start-index")
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
