"""
Standalone realtime plot worker for pc_nn_formal_controller_plotfix.py.

IMPORTANT:
    This file deliberately does NOT import torch and does NOT import the
    controller module. On Windows this prevents the plot process from
    reinitializing PyTorch's Intel OpenMP runtime (OMP Error #15).

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
"""

from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
from collections import deque


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
    parser.add_argument("--refresh-hz", type=float, default=30.0)
    parser.add_argument("--window-s", type=float, default=10.0)
    parser.add_argument("--stale-warning-s", type=float, default=0.05)
    parser.add_argument("--imu-timeout-s", type=float, default=0.15)
    parser.add_argument("--teensy-timeout-s", type=float, default=0.20)
    args = parser.parse_args()

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
