"""
IM948 bilateral X-axis logger
============================

Purpose
-------
Formal bilateral IMU data recording for walking experiments.

Features
--------
1. LEFT and RIGHT IM948 are read by independent serial threads.
2. IMUs report gyro XYZ + Euler XYZ, but ONLY X-axis data are retained.
3. Startup zero calibration uses fresh samples from BOTH IMUs simultaneously:
       zeroed angle   = wrap(raw X angle - initial X angle)
       zeroed gyro    = raw X gyro - initial static gyro bias
4. Main loop records synchronized latest samples at 100 Hz.
5. CSV contains ONLY:
       elapsed_s,
       left_angle_x_deg,
       left_angular_velocity_x_dps,
       right_angle_x_deg,
       right_angular_velocity_x_dps
6. CSV numeric precision is 4 decimal places.
7. Realtime plot shows ONLY the two zeroed X-angle curves.
8. Plotting runs in a separate process to minimize interference with logging.
9. If an IMU sample becomes older than --imu-timeout, that side is written
   as blank values rather than silently repeating stale data.

Default hardware
----------------
LEFT  IMU : COM8 @ 115200
RIGHT IMU : COM6 @ 115200
IMU rate  : 100 Hz
Log rate  : 100 Hz

Normal use
----------
    python imu948_bilateral_x_logger.py

Run until Ctrl+C or close the plot window.

Optional examples
-----------------
Configure both IMUs to 100 Hz before recording:
    python imu948_bilateral_x_logger.py --configure-imu

Record for 60 seconds:
    python imu948_bilateral_x_logger.py --duration 60

Disable realtime plot:
    python imu948_bilateral_x_logger.py --no-plot

Dependencies
------------
    python -m pip install pyserial matplotlib
"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import queue
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import serial


# =============================================================================
# User settings
# =============================================================================

DEFAULT_LEFT_PORT = "COM8"
DEFAULT_RIGHT_PORT = "COM6"
DEFAULT_BAUD = 115200

DEFAULT_LOG_HZ = 100.0
DEFAULT_PLOT_HZ = 30.0
DEFAULT_PLOT_WINDOW_S = 10.0

DEFAULT_ZERO_SAMPLES = 200
DEFAULT_ZERO_TIMEOUT_S = 10.0 
DEFAULT_IMU_TIMEOUT_S = 0.150 
DEFAULT_FLUSH_INTERVAL_S = 1.0 

# Direction convention.
# Keep both +1.0 if you want the IMU's native X-axis signs.
# If one sensor is mounted in the opposite direction, change that side to -1.0.
LEFT_DIRECTION = -1.0 
RIGHT_DIRECTION = -1.0 


# =============================================================================
# IM948 protocol constants
# =============================================================================

IMU_FRAME_BEGIN: Final[int] = 0x49
IMU_FRAME_END: Final[int] = 0x4D
IMU_BROADCAST_ADDRESS: Final[int] = 0xFF

CMD_WAKE: Final[int] = 0x03
CMD_REPORT: Final[int] = 0x11
CMD_SET_PARAMS: Final[int] = 0x12
CMD_REPORT_OFF: Final[int] = 0x18
CMD_REPORT_ON: Final[int] = 0x19

# 0x0044 = gyro XYZ (0x0004) + Euler XYZ (0x0040)
REPORT_TAG: Final[int] = 0x0044

ANGLE_SCALE_DEG: Final[float] = 180.0 / 32768.0
GYRO_SCALE_DPS: Final[float] = 2000.0 / 32768.0
MAX_IMU_DATA_LEN: Final[int] = 128


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class ImuSample:
    angle_x_deg: float
    gyro_x_dps: float
    host_time: float
    sequence: int


@dataclass(frozen=True)
class ZeroOffset:
    angle_x_deg: float
    gyro_x_dps: float


# =============================================================================
# IM948 frame parser
# =============================================================================

class ImuParser:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.bad_packets = 0

    def feed(self, data: bytes) -> list[bytes]:
        if data:
            self.buffer.extend(data)

        bodies: list[bytes] = []

        while True:
            start = self.buffer.find(bytes([IMU_FRAME_BEGIN]))

            if start < 0:
                self.buffer.clear()
                return bodies

            if start > 0:
                del self.buffer[:start]

            if len(self.buffer) < 3:
                return bodies

            data_len = self.buffer[2]

            if data_len <= 0 or data_len > MAX_IMU_DATA_LEN:
                self.bad_packets += 1
                del self.buffer[0]
                continue

            frame_len = data_len + 5

            if len(self.buffer) < frame_len:
                return bodies

            frame = bytes(self.buffer[:frame_len])

            if frame[-1] != IMU_FRAME_END:
                self.bad_packets += 1
                del self.buffer[0]
                continue

            body = frame[3:3 + data_len]
            recv_checksum = frame[3 + data_len]
            calc_checksum = sum(frame[1:3 + data_len]) & 0xFF

            if recv_checksum != calc_checksum:
                self.bad_packets += 1
                del self.buffer[0]
                continue

            del self.buffer[:frame_len]
            bodies.append(body)


def imu_pack_command(
    body: bytes,
    address: int = IMU_BROADCAST_ADDRESS,
    wake_prefix_bytes: int = 50,
) -> bytes:
    core = bytes([IMU_FRAME_BEGIN, address, len(body)]) + body
    checksum = sum(core[1:]) & 0xFF
    return b"\x00" * wake_prefix_bytes + core + bytes([checksum, IMU_FRAME_END])


def imu_send(
    uart: serial.Serial,
    body: bytes,
    settle_s: float,
) -> None:
    uart.write(imu_pack_command(body))
    uart.flush()
    time.sleep(settle_s)


def start_imu_reporting(uart: serial.Serial) -> None:
    """Wake IMU and enable auto reporting without rewriting parameters."""
    imu_send(uart, bytes([CMD_WAKE]), 0.20)
    imu_send(uart, bytes([CMD_REPORT_ON]), 0.20)


def configure_imu_100hz(uart: serial.Serial) -> None:
    """Configure IM948 for 100 Hz and report tag 0x0044."""
    imu_send(uart, bytes([CMD_REPORT_OFF]), 0.15)
    imu_send(uart, bytes([CMD_WAKE]), 0.20)

    compass_on = 0
    barometer_filter = 3
    packed_compass_baro = (
        ((barometer_filter & 0x03) << 1)
        | (compass_on & 0x01)
    )

    params = bytes(
        [
            CMD_SET_PARAMS,
            5,      # accStill
            255,    # stillToZero
            0,      # moveToZero
            packed_compass_baro,
            100,    # report rate Hz
            2,      # gyro filter
            4,      # accelerometer filter
            9,      # compass filter
            REPORT_TAG & 0xFF,
            (REPORT_TAG >> 8) & 0xFF,
        ]
    )

    imu_send(uart, params, 0.30)
    imu_send(uart, bytes([CMD_REPORT_ON]), 0.20)


def parse_imu_body(
    body: bytes,
    *,
    sequence: int,
    host_time: float,
) -> ImuSample | None:
    if len(body) < 7 or body[0] != CMD_REPORT:
        return None

    tag = int.from_bytes(body[1:3], "little")
    offset = 7

    gyro: tuple[int, int, int] | None = None
    angle: tuple[int, int, int] | None = None

    def skip(n: int) -> None:
        nonlocal offset
        offset += n
        if offset > len(body):
            raise ValueError("short IMU report")

    if tag & 0x0001:
        skip(6)
    if tag & 0x0002:
        skip(6)

    if tag & 0x0004:
        if offset + 6 > len(body):
            raise ValueError("short gyro field")
        gyro = struct.unpack_from("<hhh", body, offset)
        offset += 6

    if tag & 0x0008:
        skip(6)
    if tag & 0x0010:
        skip(8)
    if tag & 0x0020:
        skip(8)

    if tag & 0x0040:
        if offset + 6 > len(body):
            raise ValueError("short Euler field")
        angle = struct.unpack_from("<hhh", body, offset)
        offset += 6

    if tag & 0x0080:
        skip(6)
    if tag & 0x0100:
        skip(5)
    if tag & 0x0200:
        skip(6)
    if tag & 0x0400:
        skip(2)
    if tag & 0x0800:
        skip(1)

    if gyro is None or angle is None:
        return None

    return ImuSample(
        angle_x_deg=angle[0] * ANGLE_SCALE_DEG,
        gyro_x_dps=gyro[0] * GYRO_SCALE_DPS,
        host_time=host_time,
        sequence=sequence,
    )


# =============================================================================
# One serial port = one independent reader thread
# =============================================================================

class SingleImuReader(threading.Thread):
    def __init__(
        self,
        *,
        side: str,
        port: str,
        baud: int,
        configure: bool,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=f"{side}ImuReader", daemon=True)

        self.side = side
        self.port = port
        self.baud = baud
        self.configure = configure
        self.stop_event = stop_event

        self._lock = threading.Lock()
        self._latest: ImuSample | None = None

        self.total_samples = 0
        self.bad_packets = 0
        self.rx_hz = 0.0
        self.error = ""

    def snapshot(self) -> ImuSample | None:
        with self._lock:
            return self._latest

    def stats(self) -> tuple[float, int, int]:
        with self._lock:
            return self.rx_hz, self.total_samples, self.bad_packets

    def run(self) -> None:
        uart: serial.Serial | None = None
        parser = ImuParser()

        sequence = 0
        rate_count = 0
        rate_start = time.perf_counter()

        try:
            uart = serial.Serial(
                self.port,
                self.baud,
                timeout=0,
                write_timeout=0.5,
            )
            uart.reset_input_buffer()

            if self.configure:
                print(f"[IMU] {self.side}: configure 100 Hz / 0x0044")
                configure_imu_100hz(uart)
            else:
                print(f"[IMU] {self.side}: wake + auto report ON")
                start_imu_reporting(uart)

            uart.reset_input_buffer()

            while not self.stop_event.is_set():
                n = uart.in_waiting

                if n > 0:
                    data = uart.read(n)

                    for body in parser.feed(data):
                        try:
                            sequence += 1
                            sample = parse_imu_body(
                                body,
                                sequence=sequence,
                                host_time=time.perf_counter(),
                            )
                        except (ValueError, struct.error):
                            parser.bad_packets += 1
                            continue

                        if sample is None:
                            continue

                        rate_count += 1

                        with self._lock:
                            self._latest = sample
                            self.total_samples += 1

                now = time.perf_counter()

                if now - rate_start >= 1.0:
                    elapsed = now - rate_start

                    with self._lock:
                        self.rx_hz = rate_count / elapsed
                        self.bad_packets = parser.bad_packets

                    rate_count = 0
                    rate_start = now

                time.sleep(0.0005)

        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.stop_event.set()

        finally:
            if uart is not None:
                try:
                    uart.close()
                except Exception:
                    pass


# =============================================================================
# Zero calibration
# =============================================================================

def wrap_angle_deg(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def circular_mean_deg(values: list[float]) -> float:
    if not values:
        raise ValueError("empty angle list")

    sin_sum = sum(math.sin(math.radians(v)) for v in values)
    cos_sum = sum(math.cos(math.radians(v)) for v in values)

    if abs(sin_sum) < 1e-12 and abs(cos_sum) < 1e-12:
        return sum(values) / len(values)

    return math.degrees(math.atan2(sin_sum, cos_sum))


def calibrate_both_zero(
    left_reader: SingleImuReader,
    right_reader: SingleImuReader,
    *,
    sample_count: int,
    timeout_s: float,
    stop_event: threading.Event,
) -> tuple[ZeroOffset, ZeroOffset]:
    """
    Collect zero samples from both readers simultaneously.

    Angle zero uses circular mean because Euler angle wraps at +/-180 deg.
    Gyro zero uses ordinary mean to remove static angular-velocity bias.
    """
    angle_values = {
        "LEFT": [],
        "RIGHT": [],
    }
    gyro_values = {
        "LEFT": [],
        "RIGHT": [],
    }
    last_seq = {
        "LEFT": -1,
        "RIGHT": -1,
    }

    readers = {
        "LEFT": left_reader,
        "RIGHT": right_reader,
    }

    deadline = time.perf_counter() + timeout_s
    next_print = 0.0

    print(
        "[ZERO] Keep BOTH IMUs still. "
        f"Collecting {sample_count} fresh samples per side..."
    )

    while not stop_event.is_set():
        now = time.perf_counter()

        if now > deadline:
            raise RuntimeError(
                "zero calibration timeout: "
                f"L={len(angle_values['LEFT'])}/{sample_count}, "
                f"R={len(angle_values['RIGHT'])}/{sample_count}"
            )

        for side, reader in readers.items():
            if len(angle_values[side]) >= sample_count:
                continue

            sample = reader.snapshot()

            if sample is None or sample.sequence == last_seq[side]:
                continue

            angle_values[side].append(sample.angle_x_deg)
            gyro_values[side].append(sample.gyro_x_dps)
            last_seq[side] = sample.sequence

        if (
            len(angle_values["LEFT"]) >= sample_count
            and len(angle_values["RIGHT"]) >= sample_count
        ):
            break

        if now >= next_print:
            print(
                "\r[ZERO] "
                f"L={len(angle_values['LEFT']):4d}/{sample_count} | "
                f"R={len(angle_values['RIGHT']):4d}/{sample_count}",
                end="",
                flush=True,
            )
            next_print = now + 0.1

        time.sleep(0.001)

    if stop_event.is_set():
        raise RuntimeError("zero calibration interrupted")

    print(
        "\r[ZERO] "
        f"L={len(angle_values['LEFT']):4d}/{sample_count} | "
        f"R={len(angle_values['RIGHT']):4d}/{sample_count}"
    )

    left_zero = ZeroOffset(
        angle_x_deg=circular_mean_deg(angle_values["LEFT"]),
        gyro_x_dps=sum(gyro_values["LEFT"]) / len(gyro_values["LEFT"]),
    )
    right_zero = ZeroOffset(
        angle_x_deg=circular_mean_deg(angle_values["RIGHT"]),
        gyro_x_dps=sum(gyro_values["RIGHT"]) / len(gyro_values["RIGHT"]),
    )

    print(
        "[ZERO OK] LEFT  : "
        f"angle={left_zero.angle_x_deg:+.4f} deg | "
        f"gyro bias={left_zero.gyro_x_dps:+.4f} deg/s"
    )
    print(
        "[ZERO OK] RIGHT : "
        f"angle={right_zero.angle_x_deg:+.4f} deg | "
        f"gyro bias={right_zero.gyro_x_dps:+.4f} deg/s"
    )

    return left_zero, right_zero


def zeroed_values(
    sample: ImuSample,
    zero: ZeroOffset,
    direction: float,
) -> tuple[float, float]:
    angle = direction * wrap_angle_deg(sample.angle_x_deg - zero.angle_x_deg)
    gyro = direction * (sample.gyro_x_dps - zero.gyro_x_dps)
    return angle, gyro


# =============================================================================
# Realtime angle plot process
# =============================================================================

def plot_worker(
    data_queue,
    close_event,
    *,
    refresh_hz: float,
    window_s: float,
    expected_log_hz: float,
) -> None:
    import matplotlib.pyplot as plt

    history_len = max(
        int(window_s * expected_log_hz * 1.5),
        300,
    )

    t_hist = deque(maxlen=history_len)
    left_hist = deque(maxlen=history_len)
    right_hist = deque(maxlen=history_len)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    line_left, = ax.plot([], [], label="Left X angle")
    line_right, = ax.plot([], [], label="Right X angle")

    ax.axhline(0.0, linewidth=1.0, alpha=0.4)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Zeroed X angle (deg)")
    ax.set_title("Bilateral IM948 - Zeroed X Angle")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    def on_close(_event) -> None:
        close_event.set()

    fig.canvas.mpl_connect("close_event", on_close)

    refresh_period = 1.0 / max(refresh_hz, 1.0)
    next_refresh = time.perf_counter()

    while not close_event.is_set():
        got_data = False

        while True:
            try:
                item = data_queue.get_nowait()
            except queue.Empty:
                break

            if item is None:
                close_event.set()
                break

            elapsed, left_angle, right_angle = item

            t_hist.append(elapsed)
            left_hist.append(left_angle)
            right_hist.append(right_angle)
            got_data = True

        now = time.perf_counter()

        if got_data and now >= next_refresh and len(t_hist) >= 2:
            x = list(t_hist)

            line_left.set_data(x, list(left_hist))
            line_right.set_data(x, list(right_hist))

            xmax = x[-1]
            xmin = max(x[0], xmax - window_s)

            ax.set_xlim(xmin, max(xmax, xmin + 0.1))
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            next_refresh = now + refresh_period

        plt.pause(0.001)

    try:
        plt.close(fig)
    except Exception:
        pass


def push_plot_sample(data_queue, item) -> None:
    try:
        data_queue.put_nowait(item)
        return
    except queue.Full:
        pass

    # Drop the oldest plotting sample only.
    # CSV logging is completely unaffected.
    try:
        data_queue.get_nowait()
    except queue.Empty:
        pass

    try:
        data_queue.put_nowait(item)
    except queue.Full:
        pass


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Formal bilateral IM948 X-axis logger"
    )

    p.add_argument("--left-port", default=DEFAULT_LEFT_PORT)
    p.add_argument("--right-port", default=DEFAULT_RIGHT_PORT)
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)

    p.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_LOG_HZ,
        help="CSV logging rate in Hz",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="recording duration in seconds; 0 = until Ctrl+C / plot close",
    )

    p.add_argument(
        "--zero-samples",
        type=int,
        default=DEFAULT_ZERO_SAMPLES,
    )
    p.add_argument(
        "--zero-timeout",
        type=float,
        default=DEFAULT_ZERO_TIMEOUT_S,
    )

    p.add_argument(
        "--imu-timeout",
        type=float,
        default=DEFAULT_IMU_TIMEOUT_S,
        help="sample older than this becomes blank in CSV",
    )

    p.add_argument(
        "--plot-rate",
        type=float,
        default=DEFAULT_PLOT_HZ,
    )
    p.add_argument(
        "--plot-window",
        type=float,
        default=DEFAULT_PLOT_WINDOW_S,
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="disable realtime angle plot",
    )

    p.add_argument(
        "--configure-imu",
        action="store_true",
        help="rewrite IMU parameters to 100 Hz / tag 0x0044 before recording",
    )

    p.add_argument(
        "--flush-interval",
        type=float,
        default=DEFAULT_FLUSH_INTERVAL_S,
        help="CSV flush interval in seconds",
    )

    p.add_argument(
        "--csv",
        type=Path,
        default=None,
    )

    return p


def validate_args(a: argparse.Namespace) -> None:
    if a.left_port.upper() == a.right_port.upper():
        raise ValueError("LEFT and RIGHT ports must be different")
    if a.rate <= 0:
        raise ValueError("--rate must be positive")
    if a.duration < 0:
        raise ValueError("--duration must be >= 0")
    if a.zero_samples <= 0:
        raise ValueError("--zero-samples must be positive")
    if a.zero_timeout <= 0:
        raise ValueError("--zero-timeout must be positive")
    if a.imu_timeout <= 0:
        raise ValueError("--imu-timeout must be positive")
    if a.plot_rate <= 0:
        raise ValueError("--plot-rate must be positive")
    if a.plot_window <= 0:
        raise ValueError("--plot-window must be positive")
    if a.flush_interval <= 0:
        raise ValueError("--flush-interval must be positive")


def default_csv_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs") / f"imu_bilateral_x_{stamp}.csv"


def wait_for_first_samples(
    readers: list[SingleImuReader],
    stop_event: threading.Event,
    timeout_s: float = 7.0,
) -> None:
    deadline = time.perf_counter() + timeout_s

    while time.perf_counter() < deadline and not stop_event.is_set():
        if all(reader.snapshot() is not None for reader in readers):
            return
        time.sleep(0.01)

    missing = [
        reader.side
        for reader in readers
        if reader.snapshot() is None
    ]

    errors = [
        f"{reader.side}: {reader.error}"
        for reader in readers
        if reader.error
    ]

    raise RuntimeError(
        "failed to receive first IMU samples; "
        f"missing={missing}; errors={errors or 'none'}"
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    a = build_parser().parse_args()
    validate_args(a)

    if not a.no_plot:
        try:
            import matplotlib  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Realtime plot requires matplotlib. "
                "Install with: python -m pip install matplotlib"
            ) from exc

    csv_path = (
        a.csv.expanduser().resolve()
        if a.csv is not None
        else default_csv_path().resolve()
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()

    left_reader = SingleImuReader(
        side="LEFT",
        port=a.left_port,
        baud=a.baud,
        configure=a.configure_imu,
        stop_event=stop_event,
    )
    right_reader = SingleImuReader(
        side="RIGHT",
        port=a.right_port,
        baud=a.baud,
        configure=a.configure_imu,
        stop_event=stop_event,
    )
    readers = [left_reader, right_reader]

    print("=" * 92)
    print("IM948 bilateral X-axis formal logger")
    print(f"LEFT       : {a.left_port} @ {a.baud}")
    print(f"RIGHT      : {a.right_port} @ {a.baud}")
    print(f"Log rate   : {a.rate:.1f} Hz")
    print(
        "Duration   : "
        + ("until Ctrl+C / plot close" if a.duration == 0 else f"{a.duration:.1f} s")
    )
    print(
        f"Zero       : {a.zero_samples} fresh samples per side "
        "(angle + gyro bias)"
    )
    print(
        "Plot       : "
        + (
            "OFF"
            if a.no_plot
            else f"ON @ {a.plot_rate:.1f} Hz, window={a.plot_window:.1f} s"
        )
    )
    print(f"Directions : LEFT={LEFT_DIRECTION:+.0f}, RIGHT={RIGHT_DIRECTION:+.0f}")
    print(f"CSV        : {csv_path}")
    print("=" * 92)

    for reader in readers:
        reader.start()

    plot_queue = None
    plot_close_event = None
    plot_process = None

    try:
        wait_for_first_samples(readers, stop_event)

        left_zero, right_zero = calibrate_both_zero(
            left_reader,
            right_reader,
            sample_count=a.zero_samples,
            timeout_s=a.zero_timeout,
            stop_event=stop_event,
        )

        if not a.no_plot:
            ctx = mp.get_context("spawn")
            plot_queue = ctx.Queue(maxsize=500)
            plot_close_event = ctx.Event()

            plot_process = ctx.Process(
                target=plot_worker,
                args=(plot_queue, plot_close_event),
                kwargs={
                    "refresh_hz": a.plot_rate,
                    "window_s": a.plot_window,
                    "expected_log_hz": a.rate,
                },
                daemon=True,
            )
            plot_process.start()

        print("[RECORD] Started. Ctrl+C or close plot window to stop.")

        period = 1.0 / a.rate
        start_time = time.perf_counter()
        next_tick = start_time
        next_flush = start_time + a.flush_interval

        rows = 0
        left_timeout_active = False
        right_timeout_active = False

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "elapsed_s",
                    "left_angle_x_deg",
                    "left_angular_velocity_x_dps",
                    "right_angle_x_deg",
                    "right_angular_velocity_x_dps",
                ]
            )

            while not stop_event.is_set():
                if (
                    plot_close_event is not None
                    and plot_close_event.is_set()
                ):
                    break

                now = time.perf_counter()
                elapsed = now - start_time

                if a.duration > 0 and elapsed >= a.duration:
                    break

                if now < next_tick:
                    sleep_s = next_tick - now
                    if sleep_s > 0.0005:
                        time.sleep(min(0.001, sleep_s * 0.5))
                    continue

                left_sample = left_reader.snapshot()
                right_sample = right_reader.snapshot()

                left_angle = math.nan
                left_gyro = math.nan
                right_angle = math.nan
                right_gyro = math.nan

                # LEFT
                if left_sample is not None:
                    left_age = now - left_sample.host_time

                    if left_age <= a.imu_timeout:
                        left_angle, left_gyro = zeroed_values(
                            left_sample,
                            left_zero,
                            LEFT_DIRECTION,
                        )

                        if left_timeout_active:
                            print("[IMU] LEFT communication recovered.")
                            left_timeout_active = False
                    elif not left_timeout_active:
                        print(
                            f"[WARNING] LEFT sample timeout: "
                            f"{left_age*1000.0:.1f} ms"
                        )
                        left_timeout_active = True

                # RIGHT
                if right_sample is not None:
                    right_age = now - right_sample.host_time

                    if right_age <= a.imu_timeout:
                        right_angle, right_gyro = zeroed_values(
                            right_sample,
                            right_zero,
                            RIGHT_DIRECTION,
                        )

                        if right_timeout_active:
                            print("[IMU] RIGHT communication recovered.")
                            right_timeout_active = False
                    elif not right_timeout_active:
                        print(
                            f"[WARNING] RIGHT sample timeout: "
                            f"{right_age*1000.0:.1f} ms"
                        )
                        right_timeout_active = True

                # Exactly 4 decimal places for all recorded numeric values.
                writer.writerow(
                    [
                        f"{elapsed:.4f}",
                        f"{left_angle:.4f}" if math.isfinite(left_angle) else "",
                        f"{left_gyro:.4f}" if math.isfinite(left_gyro) else "",
                        f"{right_angle:.4f}" if math.isfinite(right_angle) else "",
                        f"{right_gyro:.4f}" if math.isfinite(right_gyro) else "",
                    ]
                )
                rows += 1

                if plot_queue is not None:
                    push_plot_sample(
                        plot_queue,
                        (
                            elapsed,
                            left_angle,
                            right_angle,
                        ),
                    )

                if now >= next_flush:
                    f.flush()
                    next_flush = now + a.flush_interval

                next_tick += period

                # Prevent a long scheduling delay from causing a burst of rows.
                if now - next_tick > period:
                    next_tick = now + period

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C received.")

    finally:
        stop_event.set()

        if plot_queue is not None:
            try:
                plot_queue.put_nowait(None)
            except Exception:
                pass

        if plot_close_event is not None:
            plot_close_event.set()

        for reader in readers:
            reader.join(timeout=2.0)

        if plot_process is not None:
            plot_process.join(timeout=2.0)

            if plot_process.is_alive():
                plot_process.terminate()
                plot_process.join(timeout=1.0)

        print("=" * 92)
        print(f"CSV saved: {csv_path}")

        for reader in readers:
            hz, total, bad = reader.stats()
            print(
                f"{reader.side:5s}: "
                f"RX={hz:6.2f} Hz | samples={total} | bad={bad}"
                + (f" | ERROR={reader.error}" if reader.error else "")
            )

        print("=" * 92)


if __name__ == "__main__":
    mp.freeze_support()
    main()