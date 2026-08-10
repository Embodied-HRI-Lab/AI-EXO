"""
Samsung PC controller with independent dual IMU threads + Teensy
====================================================

Purpose
-------
This version is specifically designed to diagnose intermittent IMU timeout
without letting plotting or the other IMU reader block the control loop.

Thread/process architecture
---------------------------
Main thread:
    - 100 Hz Samsung calculation
    - 100 Hz command TX to Teensy
    - 100 Hz CSV logging

Thread 1:
    - LEFT IM948 only (COM7)

Thread 2:
    - RIGHT IM948 only (COM8)

Thread 3:
    - Teensy RX feedback (COM3)

Main thread:
    - 100 Hz Samsung calculation
    - 100 Hz torque TX to Teensy
    - 100 Hz compact CSV logging

Optional separate process:
    - Matplotlib realtime plot

Important timeout behavior
--------------------------
Each IMU has its own sample timestamp and sequence counter.

    age <= 50 ms       : OK
    50 ms < age <=150  : STALE warning, keep using latest valid sample
    age > 150 ms       : CONTROL TIMEOUT, Samsung command is forced to zero

The display angle is NOT replaced with NaN during timeout. The plot therefore
stays continuous and you can directly see which IMU is holding its last value.

Formal CSV keeps only the signals required for experiment recording:
    elapsed_s
    left_angle_x_deg / right_angle_x_deg
    left_angular_velocity_x_dps / right_angular_velocity_x_dps
    left_actual_torque_nm / right_actual_torque_nm

All CSV numeric values are written with exactly 4 decimal places.

Startup zero calibration removes:
    1) the standing X-angle offset;
    2) the static X-gyro bias.

Samsung control still uses ONLY the standing-zeroed X angle.

Safety
------
Default behavior is calculation/logging only:
    --enable is NOT set -> Teensy receives 0 Nm and enable=0.

Only add --enable after communication/timeout behavior is satisfactory.

Examples
--------
Safe print test:
    python samsung_pc_dual_imu_teensy.py --display print

Safe plot test:
    python samsung_pc_dual_imu_teensy.py --display plot

Real torque test (only after diagnostics are satisfactory):
    python samsung_pc_dual_imu_teensy.py --display plot --enable

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
# General configuration
# =============================================================================

HEADER: Final[bytes] = b"\xA5\x5A"

DEFAULT_LEFT_IMU_PORT = "COM8"
DEFAULT_RIGHT_IMU_PORT = "COM6"
DEFAULT_TEENSY_PORT = "COM7"
DEFAULT_BAUD = 115200

DEFAULT_CONTROL_HZ = 100.0
DEFAULT_PRINT_HZ = 10.0
DEFAULT_PLOT_HZ = 30.0
DEFAULT_PLOT_WINDOW_S = 10.0

DEFAULT_STALE_WARNING_S = 0.050
DEFAULT_IMU_TIMEOUT_S = 0.150
DEFAULT_TEENSY_TIMEOUT_S = 0.200

CMD_TORQUE = 0x54
CMD_STOP = 0x50
CMD_CLEAR_FAULT = 0x43
CMD_STATE = 0x44

TORQUE_PAYLOAD = struct.Struct("<HffB")
STATE_PAYLOAD = struct.Struct("<Hff")
TORQUE_FRAME_SIZE = 15
STATE_FRAME_SIZE = 14


# =============================================================================
# IM948 protocol
# =============================================================================

IMU_FRAME_BEGIN = 0x49
IMU_FRAME_END = 0x4D
IMU_BROADCAST_ADDRESS = 0xFF

CMD_WAKE = 0x03
CMD_REPORT = 0x11
CMD_SET_PARAMS = 0x12
CMD_REPORT_OFF = 0x18
CMD_REPORT_ON = 0x19

REPORT_TAG = 0x0044  # gyro XYZ + Euler XYZ
CONTROL_AXIS = "X"
# Current mounting: X direction is inverted so that the controller
# uses the desired hip flexion/extension convention.

ANGLE_SCALE_DEG = 180.0 / 32768.0
GYRO_SCALE_DPS = 2000.0 / 32768.0
MAX_IMU_DATA_LEN = 128


@dataclass(frozen=True)
class ImuSample:
    angle_x_deg: float
    gyro_x_dps: float
    host_time: float
    sequence: int


@dataclass(frozen=True)
class ImuZeroOffset:
    angle_x_deg: float
    gyro_x_dps: float


@dataclass
class ImuStats:
    hz: float = 0.0
    total_samples: int = 0
    bad_packets: int = 0


class ImuParser:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.bad = 0

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
                self.bad += 1
                del self.buffer[0]
                continue

            frame_len = data_len + 5

            if len(self.buffer) < frame_len:
                return bodies

            candidate = bytes(self.buffer[:frame_len])

            if candidate[-1] != IMU_FRAME_END:
                self.bad += 1
                del self.buffer[0]
                continue

            body = candidate[3:3 + data_len]
            recv_checksum = candidate[3 + data_len]
            calc_checksum = sum(candidate[1:3 + data_len]) & 0xFF

            if recv_checksum != calc_checksum:
                self.bad += 1
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
    return (
        b"\x00" * wake_prefix_bytes
        + core
        + bytes([checksum, IMU_FRAME_END])
    )


def imu_send(
    uart: serial.Serial,
    body: bytes,
    settle_s: float,
) -> None:
    uart.write(imu_pack_command(body))
    uart.flush()
    time.sleep(settle_s)


def configure_imu(uart: serial.Serial) -> None:
    """Force one IM948 to 100 Hz with report tag 0x0044."""
    imu_send(uart, bytes([CMD_REPORT_OFF]), 0.15)
    imu_send(uart, bytes([CMD_WAKE]), 0.20)

    params = bytes(
        [
            CMD_SET_PARAMS,
            5,      # accStill
            255,    # stillToZero
            0,      # moveToZero
            6,      # compass off + barometer filter
            100,    # report Hz
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


class SingleImuReader(threading.Thread):
    """One thread owns exactly one IMU serial port."""

    def __init__(
        self,
        *,
        name: str,
        port: str,
        baud: int,
        configure: bool,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=f"{name}ImuReader", daemon=True)

        self.side_name = name
        self.port = port
        self.baud = baud
        self.configure = configure
        self.stop_event = stop_event

        self._lock = threading.Lock()
        self._latest: ImuSample | None = None
        self._stats = ImuStats()

        self.error = ""

    def snapshot(self) -> tuple[ImuSample | None, ImuStats]:
        with self._lock:
            return self._latest, ImuStats(**vars(self._stats))

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
                print(
                    f"[IMU] configure {self.side_name} "
                    f"{self.port} -> 100 Hz / 0x0044"
                )
                configure_imu(uart)
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
                            parser.bad += 1
                            sample = None

                        if sample is None:
                            continue

                        rate_count += 1

                        with self._lock:
                            self._latest = sample
                            self._stats.total_samples += 1

                now = time.perf_counter()

                if now - rate_start >= 1.0:
                    elapsed = now - rate_start

                    with self._lock:
                        self._stats.hz = rate_count / elapsed
                        self._stats.bad_packets = parser.bad

                    rate_count = 0
                    rate_start = now

                # Keep this tiny; serial reads themselves are non-blocking.
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
# Teensy link
# =============================================================================

def crc8(data: bytes) -> int:
    crc = 0

    for byte in data:
        crc ^= byte

        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF

    return crc


def make_frame(cmd: int, payload: bytes = b"") -> bytes:
    body = bytes([cmd]) + payload
    return HEADER + body + bytes([crc8(body)])


@dataclass(frozen=True)
class MotorFeedback:
    sequence: int
    left_actual_nm: float
    right_actual_nm: float
    host_time: float


@dataclass
class TeensyStats:
    hz: float = 0.0
    packets: int = 0
    crc_errors: int = 0


class TeensyLink(threading.Thread):
    def __init__(
        self,
        *,
        port: str,
        baud: int,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="TeensyLink", daemon=True)

        self.port = port
        self.baud = baud
        self.stop_event = stop_event

        self._lock = threading.Lock()
        self._tx_lock = threading.Lock()

        self.uart: serial.Serial | None = None
        self.latest: MotorFeedback | None = None
        self.stats = TeensyStats()

        self.rx = bytearray()
        self.tx_seq = 0
        self.error = ""

    def snapshot(self) -> tuple[MotorFeedback | None, TeensyStats]:
        with self._lock:
            return self.latest, TeensyStats(**vars(self.stats))

    def send_torque(
        self,
        left_nm: float,
        right_nm: float,
        enable: bool,
    ) -> None:
        uart = self.uart

        if uart is None or not uart.is_open:
            return

        payload = TORQUE_PAYLOAD.pack(
            self.tx_seq,
            float(left_nm),
            float(right_nm),
            int(bool(enable)),
        )
        self.tx_seq = (self.tx_seq + 1) & 0xFFFF

        packet = make_frame(CMD_TORQUE, payload)

        with self._tx_lock:
            uart.write(packet)

    def send_stop(self) -> None:
        uart = self.uart

        if uart is None or not uart.is_open:
            return

        with self._tx_lock:
            uart.write(make_frame(CMD_STOP))

    def clear_fault(self) -> None:
        uart = self.uart

        if uart is None or not uart.is_open:
            return

        with self._tx_lock:
            uart.write(make_frame(CMD_CLEAR_FAULT))

    def _parse_rx(self) -> int:
        count = 0

        while True:
            if len(self.rx) < 3:
                return count

            sync_index = self.rx.find(HEADER)

            if sync_index < 0:
                if self.rx and self.rx[-1] == HEADER[0]:
                    self.rx[:] = self.rx[-1:]
                else:
                    self.rx.clear()
                return count

            if sync_index > 0:
                del self.rx[:sync_index]

            if len(self.rx) < STATE_FRAME_SIZE:
                return count

            if self.rx[2] != CMD_STATE:
                del self.rx[0]
                continue

            packet = bytes(self.rx[:STATE_FRAME_SIZE])
            payload = packet[3:-1]

            if packet[-1] != crc8(bytes([CMD_STATE]) + payload):
                with self._lock:
                    self.stats.crc_errors += 1
                del self.rx[0]
                continue

            del self.rx[:STATE_FRAME_SIZE]

            sequence, left_actual, right_actual = STATE_PAYLOAD.unpack(payload)

            feedback = MotorFeedback(
                sequence=sequence,
                left_actual_nm=left_actual,
                right_actual_nm=right_actual,
                host_time=time.perf_counter(),
            )

            with self._lock:
                self.latest = feedback
                self.stats.packets += 1

            count += 1

    def run(self) -> None:
        rate_count = 0
        rate_start = time.perf_counter()

        try:
            self.uart = serial.Serial(
                self.port,
                self.baud,
                timeout=0,
                write_timeout=0.05,
            )
            self.uart.reset_input_buffer()
            self.uart.reset_output_buffer()

            # Safe startup.
            for _ in range(3):
                self.send_stop()
                time.sleep(0.02)

            for _ in range(3):
                self.clear_fault()
                time.sleep(0.03)

            while not self.stop_event.is_set():
                n = self.uart.in_waiting

                if n > 0:
                    self.rx.extend(self.uart.read(n))
                    rate_count += self._parse_rx()

                now = time.perf_counter()

                if now - rate_start >= 1.0:
                    elapsed = now - rate_start

                    with self._lock:
                        self.stats.hz = rate_count / elapsed

                    rate_count = 0
                    rate_start = now

                time.sleep(0.0005)

        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.stop_event.set()

        finally:
            if self.uart is not None:
                try:
                    for _ in range(3):
                        self.send_stop()
                        time.sleep(0.01)
                except Exception:
                    pass

                try:
                    self.uart.close()
                except Exception:
                    pass


# =============================================================================
# Initial zero calibration
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


def relative_x_deg(
    raw_angle_deg: float,
    zero_offset_deg: float,
    direction_sign: float,
) -> float:
    """
    Unified hip-angle coordinate used everywhere in this program.

    Pipeline:
        IMU raw Euler-X
        -> subtract startup standing zero
        -> wrap to [-180, 180)
        -> apply side direction sign

    The returned value is the ONLY angle allowed to enter:
        1) Samsung controller
        2) realtime plot / terminal display
        3) CSV logging

    Expected convention after correct sign selection:
        standing     ~= 0 deg
        flexion      = one sign
        extension    = the opposite sign
    """
    rel = wrap_angle_deg(raw_angle_deg - zero_offset_deg)
    return direction_sign * rel


def relative_x_gyro_dps(
    raw_gyro_dps: float,
    zero_bias_dps: float,
    direction_sign: float,
) -> float:
    """Remove startup static gyro bias and use the same side sign as angle."""
    return direction_sign * (raw_gyro_dps - zero_bias_dps)


def calibrate_initial_x_zero(
    left_reader: SingleImuReader,
    right_reader: SingleImuReader,
    *,
    sample_count: int,
    timeout_s: float,
    stop_event: threading.Event,
) -> tuple[ImuZeroOffset, ImuZeroOffset]:
    """
    Simultaneously calibrate both IMUs while the wearer stands still.

    Angle:
        circular mean of Euler-X samples

    Angular velocity:
        arithmetic mean of gyro-X samples

    During runtime:
        zeroed_angle = wrap(raw_angle - angle_offset) * direction_sign
        zeroed_gyro  = (raw_gyro - gyro_bias) * direction_sign
    """
    left_angles: list[float] = []
    right_angles: list[float] = []
    left_gyros: list[float] = []
    right_gyros: list[float] = []

    last_left_seq = -1
    last_right_seq = -1

    deadline = time.perf_counter() + timeout_s
    next_print = 0.0

    print(
        f"[ZERO] Stand naturally and keep BOTH IMUs still. "
        f"Collecting {sample_count} fresh samples per side..."
    )

    while not stop_event.is_set():
        now = time.perf_counter()

        if now > deadline:
            raise RuntimeError(
                "IMU zeroing timeout: "
                f"L={len(left_angles)}/{sample_count}, "
                f"R={len(right_angles)}/{sample_count}"
            )

        left, _ = left_reader.snapshot()
        right, _ = right_reader.snapshot()

        if (
            left is not None
            and left.sequence != last_left_seq
            and len(left_angles) < sample_count
        ):
            left_angles.append(left.angle_x_deg)
            left_gyros.append(left.gyro_x_dps)
            last_left_seq = left.sequence

        if (
            right is not None
            and right.sequence != last_right_seq
            and len(right_angles) < sample_count
        ):
            right_angles.append(right.angle_x_deg)
            right_gyros.append(right.gyro_x_dps)
            last_right_seq = right.sequence

        if (
            len(left_angles) >= sample_count
            and len(right_angles) >= sample_count
        ):
            break

        if now >= next_print:
            print(
                f"\r[ZERO] L {len(left_angles):4d}/{sample_count} | "
                f"R {len(right_angles):4d}/{sample_count}",
                end="",
                flush=True,
            )
            next_print = now + 0.1

        time.sleep(0.001)

    if stop_event.is_set():
        raise RuntimeError("zeroing interrupted")

    left_zero = ImuZeroOffset(
        angle_x_deg=circular_mean_deg(left_angles),
        gyro_x_dps=sum(left_gyros) / len(left_gyros),
    )
    right_zero = ImuZeroOffset(
        angle_x_deg=circular_mean_deg(right_angles),
        gyro_x_dps=sum(right_gyros) / len(right_gyros),
    )

    print(
        f"\r[ZERO] L {len(left_angles):4d}/{sample_count} | "
        f"R {len(right_angles):4d}/{sample_count}"
    )
    print(
        "[ZERO OK] LEFT  | "
        f"angle={left_zero.angle_x_deg:+.4f} deg | "
        f"gyro bias={left_zero.gyro_x_dps:+.4f} deg/s"
    )
    print(
        "[ZERO OK] RIGHT | "
        f"angle={right_zero.angle_x_deg:+.4f} deg | "
        f"gyro bias={right_zero.gyro_x_dps:+.4f} deg/s"
    )

    return left_zero, right_zero


# =============================================================================
# Samsung controller
# =============================================================================

@dataclass
class SamsungConfig:
    rescaling: float = 5.0
    flex_gain: float = 1.0
    ext_gain: float = 1.0
    delay_index: int = 0
    filter_tau_s: float = 0.035
    max_command_nm: float = 0.20


class SamsungController:
    """
    Angle-based Samsung bilateral assistance law using standing-zeroed relative angles.

    gyro-X is not used by the torque law; it is recorded for diagnostics.
    """

    HISTORY_SIZE = 100

    def __init__(self, cfg: SamsungConfig) -> None:
        self.cfg = cfg

        self.left_filtered: float | None = None
        self.right_filtered: float | None = None

        self.phase_history = [0.0] * self.HISTORY_SIZE
        self.shape_history = [0.0] * self.HISTORY_SIZE

        self.write_index = 0
        self.valid_count = 0
        self.last_time: float | None = None

    def update(
        self,
        left_angle_deg: float,
        right_angle_deg: float,
        now: float,
    ) -> tuple[float, float]:
        # Inputs are already standing-zeroed, wrapped, and direction-normalized.
        left = math.radians(left_angle_deg)
        right = math.radians(right_angle_deg)

        if self.last_time is None:
            dt = 1.0 / DEFAULT_CONTROL_HZ
        else:
            dt = max(0.001, min(now - self.last_time, 0.05))

        self.last_time = now

        if self.left_filtered is None:
            self.left_filtered = left
            self.right_filtered = right
        else:
            alpha = 1.0 - math.exp(
                -dt / self.cfg.filter_tau_s
            )

            self.left_filtered += alpha * (
                left - self.left_filtered
            )
            assert self.right_filtered is not None
            self.right_filtered += alpha * (
                right - self.right_filtered
            )

        assert self.right_filtered is not None

        phase = self.right_filtered - self.left_filtered
        shape = (
            math.sin(self.right_filtered)
            - math.sin(self.left_filtered)
        )

        current_index = self.write_index
        self.phase_history[current_index] = phase
        self.shape_history[current_index] = shape

        self.write_index = (
            self.write_index + 1
        ) % self.HISTORY_SIZE

        self.valid_count = min(
            self.valid_count + 1,
            self.HISTORY_SIZE,
        )

        delay = max(
            0,
            min(self.cfg.delay_index, self.HISTORY_SIZE - 1),
        )

        if self.valid_count <= delay:
            return 0.0, 0.0

        delayed_index = (
            current_index - delay
        ) % self.HISTORY_SIZE

        delayed_phase = self.phase_history[delayed_index]
        delayed_shape = self.shape_history[delayed_index]

        phase_limit = math.radians(120.0)

        if 0.0 <= delayed_phase < phase_limit:
            left_tau = (
                -self.cfg.rescaling
                * self.cfg.ext_gain
                * delayed_shape
            )
            right_tau = (
                self.cfg.rescaling
                * self.cfg.flex_gain
                * delayed_shape
            )

        elif -phase_limit < delayed_phase < 0.0:
            right_tau = (
                self.cfg.rescaling
                * self.cfg.ext_gain
                * delayed_shape
            )
            left_tau = (
                -self.cfg.rescaling
                * self.cfg.flex_gain
                * delayed_shape
            )

        else:
            left_tau = 0.0
            right_tau = 0.0

        limit = abs(self.cfg.max_command_nm)

        left_tau = max(-limit, min(limit, left_tau))
        right_tau = max(-limit, min(limit, right_tau))

        return left_tau, right_tau


# =============================================================================
# Plot process
# =============================================================================

def sample_state(
    age_s: float,
    stale_s: float,
    timeout_s: float,
) -> str:
    if age_s > timeout_s:
        return "TIMEOUT"
    if age_s > stale_s:
        return "STALE"
    return "OK"


def plot_worker(
    data_queue,
    close_event,
    *,
    refresh_hz: float,
    window_s: float,
    stale_warning_s: float,
    imu_timeout_s: float,
) -> None:
    import matplotlib.pyplot as plt

    history_len = max(
        int(window_s * DEFAULT_CONTROL_HZ * 1.5),
        300,
    )

    t_hist = deque(maxlen=history_len)
    la_hist = deque(maxlen=history_len)
    ra_hist = deque(maxlen=history_len)

    lc_hist = deque(maxlen=history_len)
    rc_hist = deque(maxlen=history_len)
    lactual_hist = deque(maxlen=history_len)
    ractual_hist = deque(maxlen=history_len)

    fig, (ax_angle, ax_torque) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(11, 7),
    )

    line_la, = ax_angle.plot([], [], label="Left relative angle")
    line_ra, = ax_angle.plot([], [], label="Right relative angle")
    ax_angle.axhline(0.0, linewidth=0.8)
    ax_angle.set_ylabel("Relative hip angle (deg)")
    ax_angle.set_title("Standing-zeroed relative hip angle")
    ax_angle.legend(loc="upper right")
    ax_angle.grid(True, alpha=0.25)

    line_lc, = ax_torque.plot([], [], label="Left command")
    line_rc, = ax_torque.plot([], [], label="Right command")
    line_lactual, = ax_torque.plot([], [], label="Left actual")
    line_ractual, = ax_torque.plot([], [], label="Right actual")
    ax_torque.set_ylabel("Torque (Nm)")
    ax_torque.set_xlabel("Elapsed time (s)")
    ax_torque.legend(loc="upper right")
    ax_torque.grid(True, alpha=0.25)

    status_text = fig.suptitle("Waiting for samples...")

    latest_left_age = math.inf
    latest_right_age = math.inf
    latest_control_ok = False

    def on_close(_event) -> None:
        close_event.set()

    fig.canvas.mpl_connect("close_event", on_close)

    refresh_period = 1.0 / max(refresh_hz, 1.0)
    next_refresh = time.perf_counter()

    while not close_event.is_set():
        got_any = False

        while True:
            try:
                item = data_queue.get_nowait()
            except queue.Empty:
                break

            if item is None:
                close_event.set()
                break

            (
                elapsed,
                left_angle,
                right_angle,
                left_cmd,
                right_cmd,
                left_actual,
                right_actual,
                left_age_s,
                right_age_s,
                control_ok,
            ) = item

            t_hist.append(elapsed)
            la_hist.append(left_angle)
            ra_hist.append(right_angle)

            lc_hist.append(left_cmd)
            rc_hist.append(right_cmd)
            lactual_hist.append(left_actual)
            ractual_hist.append(right_actual)

            latest_left_age = left_age_s
            latest_right_age = right_age_s
            latest_control_ok = control_ok
            got_any = True

        now = time.perf_counter()

        if got_any and now >= next_refresh and len(t_hist) >= 2:
            x = list(t_hist)

            line_la.set_data(x, list(la_hist))
            line_ra.set_data(x, list(ra_hist))

            line_lc.set_data(x, list(lc_hist))
            line_rc.set_data(x, list(rc_hist))
            line_lactual.set_data(x, list(lactual_hist))
            line_ractual.set_data(x, list(ractual_hist))

            xmax = x[-1]
            xmin = max(x[0], xmax - window_s)

            ax_angle.set_xlim(xmin, max(xmax, xmin + 0.1))
            ax_torque.set_xlim(xmin, max(xmax, xmin + 0.1))

            # Autoscale y only; keep the requested rolling x-window.
            ax_angle.relim()
            ax_angle.autoscale_view(scalex=False, scaley=True)

            ax_torque.relim()
            ax_torque.autoscale_view(scalex=False, scaley=True)

            left_state = sample_state(
                latest_left_age,
                stale_warning_s,
                imu_timeout_s,
            )
            right_state = sample_state(
                latest_right_age,
                stale_warning_s,
                imu_timeout_s,
            )

            status_text.set_text(
                f"L age={latest_left_age * 1000:5.1f} ms [{left_state}]  |  "
                f"R age={latest_right_age * 1000:5.1f} ms [{right_state}]  |  "
                f"control={'OK' if latest_control_ok else 'ZERO'}"
            )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            next_refresh = now + refresh_period

        plt.pause(0.001)

    try:
        plt.close(fig)
    except Exception:
        pass


def push_plot_sample(data_queue, sample) -> None:
    """Plotting can lose display samples, but it must never block control."""
    try:
        data_queue.put_nowait(sample)
        return
    except queue.Full:
        pass

    try:
        data_queue.get_nowait()
    except queue.Empty:
        pass

    try:
        data_queue.put_nowait(sample)
    except queue.Full:
        pass


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Samsung controller with independent IMU threads "
            "and timeout diagnostics"
        )
    )

    p.add_argument(
        "--left-port",
        default=DEFAULT_LEFT_IMU_PORT,
    )
    p.add_argument(
        "--right-port",
        default=DEFAULT_RIGHT_IMU_PORT,
    )
    p.add_argument(
        "--teensy-port",
        default=DEFAULT_TEENSY_PORT,
    )
    p.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
    )

    p.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_CONTROL_HZ,
    )

    p.add_argument(
        "--display",
        choices=("print", "plot"),
        default="print",
        help="print or realtime plot; mutually exclusive",
    )
    p.add_argument(
        "--print-rate",
        type=float,
        default=DEFAULT_PRINT_HZ,
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
        "--stale-warning",
        type=float,
        default=DEFAULT_STALE_WARNING_S,
        help="seconds; default 0.050",
    )
    p.add_argument(
        "--imu-timeout",
        type=float,
        default=DEFAULT_IMU_TIMEOUT_S,
        help="seconds; default 0.150",
    )
    p.add_argument(
        "--teensy-timeout",
        type=float,
        default=DEFAULT_TEENSY_TIMEOUT_S,
    )

    p.add_argument(
        "--zero-samples",
        type=int,
        default=200,
    )
    p.add_argument(
        "--zero-timeout",
        type=float,
        default=10.0,
    )
    p.add_argument(
        "--skip-zero",
        action="store_true",
    )
    p.add_argument(
        "--no-configure-imu",
        action="store_true",
    )

    p.add_argument(
        "--rescaling",
        type=float,
        default=5.0,
    )
    p.add_argument(
        "--flex",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--ext",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--delay-index",
        type=int,
        default=0,
    )
    p.add_argument(
        "--filter-tau",
        type=float,
        default=0.035,
    )
    p.add_argument(
        "--max-command",
        type=float,
        default=5.0,
    )
    p.add_argument(
        "--left-angle-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=-1.0,
    )
    p.add_argument(
        "--right-angle-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=-1.0,
    )

    p.add_argument(
        "--enable",
        action="store_true",
        help=(
            "actually allow torque output; "
            "default is safe calculation/logging only"
        ),
    )

    p.add_argument(
        "--csv",
        type=Path,
        default=None,
    )

    return p


def validate_args(a: argparse.Namespace) -> None:
    ports = {
        a.left_port.upper(),
        a.right_port.upper(),
        a.teensy_port.upper(),
    }

    if len(ports) != 3:
        raise ValueError(
            "left IMU, right IMU, and Teensy must use different COM ports"
        )

    if a.rate <= 0:
        raise ValueError("--rate must be positive")
    if a.print_rate <= 0:
        raise ValueError("--print-rate must be positive")
    if a.plot_rate <= 0:
        raise ValueError("--plot-rate must be positive")
    if a.plot_window <= 0:
        raise ValueError("--plot-window must be positive")

    if a.stale_warning <= 0:
        raise ValueError("--stale-warning must be positive")
    if a.imu_timeout <= a.stale_warning:
        raise ValueError(
            "--imu-timeout must be larger than --stale-warning"
        )
    if a.teensy_timeout <= 0:
        raise ValueError("--teensy-timeout must be positive")

    if a.zero_samples <= 0:
        raise ValueError("--zero-samples must be positive")
    if a.zero_timeout <= 0:
        raise ValueError("--zero-timeout must be positive")

    if a.delay_index < 0 or a.delay_index >= 100:
        raise ValueError("--delay-index must be in [0, 99]")
    if a.filter_tau <= 0:
        raise ValueError("--filter-tau must be positive")
    if a.max_command <= 0:
        raise ValueError("--max-command must be positive")


def default_csv_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs") / f"samsung_formal_record_{stamp}.csv"


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    a = build_parser().parse_args()
    validate_args(a)

    if a.display == "plot":
        try:
            import matplotlib  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "--display plot requires matplotlib. "
                "Install with: python -m pip install matplotlib"
            ) from exc

    csv_path = (
        a.csv.expanduser().resolve()
        if a.csv is not None
        else default_csv_path().resolve()
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()

    left_imu = SingleImuReader(
        name="LEFT",
        port=a.left_port,
        baud=a.baud,
        configure=not a.no_configure_imu,
        stop_event=stop_event,
    )

    right_imu = SingleImuReader(
        name="RIGHT",
        port=a.right_port,
        baud=a.baud,
        configure=not a.no_configure_imu,
        stop_event=stop_event,
    )

    teensy = TeensyLink(
        port=a.teensy_port,
        baud=a.baud,
        stop_event=stop_event,
    )

    controller = SamsungController(
        SamsungConfig(
            rescaling=a.rescaling,
            flex_gain=a.flex,
            ext_gain=a.ext,
            delay_index=a.delay_index,
            filter_tau_s=a.filter_tau,
            max_command_nm=a.max_command,
        )
    )

    plot_queue = None
    plot_close_event = None
    plot_process = None

    if a.display == "plot":
        ctx = mp.get_context("spawn")
        plot_queue = ctx.Queue(maxsize=400)
        plot_close_event = ctx.Event()

        plot_process = ctx.Process(
            target=plot_worker,
            args=(plot_queue, plot_close_event),
            kwargs={
                "refresh_hz": a.plot_rate,
                "window_s": a.plot_window,
                "stale_warning_s": a.stale_warning,
                "imu_timeout_s": a.imu_timeout,
            },
            daemon=True,
        )

    print("=" * 104)
    print("Samsung PC / dual independent IMU / Teensy controller")
    print(
        f"LEFT IMU  : {a.left_port} @ {a.baud} "
        f"(independent thread)"
    )
    print(
        f"RIGHT IMU : {a.right_port} @ {a.baud} "
        f"(independent thread)"
    )
    print(
        f"Teensy    : {a.teensy_port} @ {a.baud} "
        f"(independent thread)"
    )
    print(f"Control   : {a.rate:.1f} Hz | CSV target={a.rate:.1f} Hz")

    if a.display == "print":
        print(f"Display   : PRINT @ {a.print_rate:.1f} Hz")
    else:
        print(
            f"Display   : PLOT @ {a.plot_rate:.1f} Hz redraw | "
            f"{a.plot_window:.1f}s window"
        )

    print(
        f"IMU age   : OK <= {a.stale_warning * 1000:.0f} ms | "
        f"STALE <= {a.imu_timeout * 1000:.0f} ms | "
        f"TIMEOUT > {a.imu_timeout * 1000:.0f} ms"
    )
    print(
        f"Samsung   : rescaling={a.rescaling:.3f}, "
        f"flex={a.flex:.2f}, ext={a.ext:.2f}, "
        f"delay={a.delay_index}, filter tau={a.filter_tau:.3f}s"
    )
    print(f"PC clamp  : ±{a.max_command:.3f} Nm")
    print(
        f"Angle coord: Euler-X, standing=0 deg | "
        f"L sign={a.left_angle_sign:+.0f}, "
        f"R sign={a.right_angle_sign:+.0f} | "
        f"CSV/plot/Samsung all use the same relative angle"
    )
    print(
        f"Output    : "
        f"{'ENABLED' if a.enable else 'DISABLED - calculate/log only'}"
    )
    print(
        "Zero      : "
        + (
            "SKIPPED"
            if a.skip_zero
            else f"{a.zero_samples} fresh samples per side"
        )
    )
    print(f"CSV       : {csv_path}")
    print("Ctrl+C or closing plot -> zero torque + STOP x3")
    print("=" * 104)

    # Start all communication threads.
    left_imu.start()
    right_imu.start()
    teensy.start()

    # Wait for one valid sample from each IMU.
    startup_deadline = time.perf_counter() + 7.0

    while (
        time.perf_counter() < startup_deadline
        and not stop_event.is_set()
    ):
        left_sample, _ = left_imu.snapshot()
        right_sample, _ = right_imu.snapshot()

        if left_sample is not None and right_sample is not None:
            break

        time.sleep(0.01)

    left_sample, _ = left_imu.snapshot()
    right_sample, _ = right_imu.snapshot()

    if left_sample is None or right_sample is None:
        stop_event.set()
        raise RuntimeError(
            "Failed to receive both IMUs. "
            f"LEFT={left_imu.error or 'no sample'}, "
            f"RIGHT={right_imu.error or 'no sample'}"
        )

    left_zero = ImuZeroOffset(0.0, 0.0)
    right_zero = ImuZeroOffset(0.0, 0.0)

    if a.skip_zero:
        print("[ZERO] skipped; angle offsets and gyro biases = 0")
    else:
        left_zero, right_zero = calibrate_initial_x_zero(
            left_imu,
            right_imu,
            sample_count=a.zero_samples,
            timeout_s=a.zero_timeout,
            stop_event=stop_event,
        )

    # Start GUI only after zero calibration.
    if plot_process is not None:
        plot_process.start()

    period = 1.0 / a.rate
    print_period = 1.0 / a.print_rate

    start_time = time.perf_counter()
    next_tick = start_time
    next_print = start_time

    last_left_seq = -1
    last_right_seq = -1

    rows = 0
    left_stale_rows = 0
    right_stale_rows = 0
    left_timeout_rows = 0
    right_timeout_rows = 0
    control_timeout_rows = 0

    max_left_age_s = 0.0
    max_right_age_s = 0.0

    lcmd = 0.0
    rcmd = 0.0
    enabled = False

    try:
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
                    "left_actual_torque_nm",
                    "right_actual_torque_nm",
                ]
            )

            while not stop_event.is_set():
                if (
                    plot_close_event is not None
                    and plot_close_event.is_set()
                ):
                    break

                now = time.perf_counter()

                if now >= next_tick:
                    left_sample, left_stats = left_imu.snapshot()
                    right_sample, right_stats = right_imu.snapshot()
                    feedback, teensy_stats = teensy.snapshot()

                    # Samples exist after startup. If a reader thread dies,
                    # the old sample remains and age reveals the problem.
                    left_age_s = (
                        now - left_sample.host_time
                        if left_sample is not None
                        else math.inf
                    )
                    right_age_s = (
                        now - right_sample.host_time
                        if right_sample is not None
                        else math.inf
                    )

                    max_left_age_s = max(
                        max_left_age_s,
                        left_age_s
                        if math.isfinite(left_age_s)
                        else 0.0,
                    )
                    max_right_age_s = max(
                        max_right_age_s,
                        right_age_s
                        if math.isfinite(right_age_s)
                        else 0.0,
                    )

                    left_new = int(
                        left_sample is not None
                        and left_sample.sequence != last_left_seq
                    )
                    right_new = int(
                        right_sample is not None
                        and right_sample.sequence != last_right_seq
                    )

                    if left_sample is not None:
                        last_left_seq = left_sample.sequence

                    if right_sample is not None:
                        last_right_seq = right_sample.sequence

                    left_stale = (
                        left_age_s > a.stale_warning
                    )
                    right_stale = (
                        right_age_s > a.stale_warning
                    )

                    left_timeout = (
                        left_age_s > a.imu_timeout
                    )
                    right_timeout = (
                        right_age_s > a.imu_timeout
                    )

                    if left_stale:
                        left_stale_rows += 1
                    if right_stale:
                        right_stale_rows += 1
                    if left_timeout:
                        left_timeout_rows += 1
                    if right_timeout:
                        right_timeout_rows += 1

                    imu_control_ok = (
                        left_sample is not None
                        and right_sample is not None
                        and not left_timeout
                        and not right_timeout
                    )

                    if not imu_control_ok:
                        control_timeout_rows += 1

                    # IMPORTANT:
                    # Display/log angles always use the latest valid sample.
                    # They are not replaced by NaN on timeout.
                    left_rel_deg = (
                        relative_x_deg(
                            left_sample.angle_x_deg,
                            left_zero.angle_x_deg,
                            a.left_angle_sign,
                        )
                        if left_sample is not None
                        else math.nan
                    )
                    right_rel_deg = (
                        relative_x_deg(
                            right_sample.angle_x_deg,
                            right_zero.angle_x_deg,
                            a.right_angle_sign,
                        )
                        if right_sample is not None
                        else math.nan
                    )

                    # Formal recorded angular velocity:
                    # remove the static startup gyro bias and apply the same
                    # direction convention used for the corresponding angle.
                    left_rel_gyro_dps = (
                        relative_x_gyro_dps(
                            left_sample.gyro_x_dps,
                            left_zero.gyro_x_dps,
                            a.left_angle_sign,
                        )
                        if left_sample is not None
                        else math.nan
                    )
                    right_rel_gyro_dps = (
                        relative_x_gyro_dps(
                            right_sample.gyro_x_dps,
                            right_zero.gyro_x_dps,
                            a.right_angle_sign,
                        )
                        if right_sample is not None
                        else math.nan
                    )

                    # Control keeps using latest valid data through the STALE
                    # warning region. Only true TIMEOUT forces Samsung to zero.
                    if imu_control_ok:
                        lcmd, rcmd = controller.update(
                            left_rel_deg,
                            right_rel_deg,
                            now,
                        )
                    else:
                        lcmd = 0.0
                        rcmd = 0.0

                    teensy_ok = (
                        feedback is not None
                        and now - feedback.host_time
                        <= a.teensy_timeout
                    )

                    enabled = bool(
                        a.enable
                        and imu_control_ok
                        and teensy_ok
                    )

                    # Default no --enable: send zero torque and enable=0.
                    teensy.send_torque(
                        lcmd if enabled else 0.0,
                        rcmd if enabled else 0.0,
                        enabled,
                    )

                    left_actual = (
                        feedback.left_actual_nm
                        if feedback is not None
                        else math.nan
                    )
                    right_actual = (
                        feedback.right_actual_nm
                        if feedback is not None
                        else math.nan
                    )

                    elapsed = now - start_time

                    writer.writerow(
                        [
                            f"{elapsed:.4f}",
                            (
                                f"{left_rel_deg:.4f}"
                                if math.isfinite(left_rel_deg)
                                else ""
                            ),
                            (
                                f"{left_rel_gyro_dps:.4f}"
                                if math.isfinite(left_rel_gyro_dps)
                                else ""
                            ),
                            (
                                f"{right_rel_deg:.4f}"
                                if math.isfinite(right_rel_deg)
                                else ""
                            ),
                            (
                                f"{right_rel_gyro_dps:.4f}"
                                if math.isfinite(right_rel_gyro_dps)
                                else ""
                            ),
                            (
                                f"{left_actual:.4f}"
                                if math.isfinite(left_actual)
                                else ""
                            ),
                            (
                                f"{right_actual:.4f}"
                                if math.isfinite(right_actual)
                                else ""
                            ),
                        ]
                    )
                    rows += 1

                    if plot_queue is not None:
                        push_plot_sample(
                            plot_queue,
                            (
                                elapsed,
                                left_rel_deg,
                                right_rel_deg,
                                lcmd,
                                rcmd,
                                left_actual,
                                right_actual,
                                left_age_s,
                                right_age_s,
                                imu_control_ok,
                            ),
                        )

                    next_tick += period

                    # Do not perform a long catch-up burst if Windows delayed
                    # this process. Resume from the current time.
                    if now - next_tick > period:
                        next_tick = now + period

                # Print and plot are mutually exclusive.
                if a.display == "print" and now >= next_print:
                    left_sample, left_stats = left_imu.snapshot()
                    right_sample, right_stats = right_imu.snapshot()
                    feedback, teensy_stats = teensy.snapshot()

                    left_age = (
                        time.perf_counter() - left_sample.host_time
                        if left_sample is not None
                        else math.inf
                    )
                    right_age = (
                        time.perf_counter() - right_sample.host_time
                        if right_sample is not None
                        else math.inf
                    )

                    lstate = sample_state(
                        left_age,
                        a.stale_warning,
                        a.imu_timeout,
                    )
                    rstate = sample_state(
                        right_age,
                        a.stale_warning,
                        a.imu_timeout,
                    )

                    left_angle = (
                        relative_x_deg(
                            left_sample.angle_x_deg,
                            left_zero.angle_x_deg,
                            a.left_angle_sign,
                        )
                        if left_sample is not None
                        else math.nan
                    )
                    right_angle = (
                        relative_x_deg(
                            right_sample.angle_x_deg,
                            right_zero.angle_x_deg,
                            a.right_angle_sign,
                        )
                        if right_sample is not None
                        else math.nan
                    )
                    left_gyro = (
                        relative_x_gyro_dps(
                            left_sample.gyro_x_dps,
                            left_zero.gyro_x_dps,
                            a.left_angle_sign,
                        )
                        if left_sample is not None
                        else math.nan
                    )
                    right_gyro = (
                        relative_x_gyro_dps(
                            right_sample.gyro_x_dps,
                            right_zero.gyro_x_dps,
                            a.right_angle_sign,
                        )
                        if right_sample is not None
                        else math.nan
                    )

                    phase_deg = (
                        right_angle - left_angle
                        if math.isfinite(left_angle)
                        and math.isfinite(right_angle)
                        else math.nan
                    )

                    actual_text = (
                        f"{feedback.left_actual_nm:+.3f}/"
                        f"{feedback.right_actual_nm:+.3f}"
                        if feedback is not None
                        else "NO_FB"
                    )

                    print(
                        f"L {left_stats.hz:5.1f}Hz "
                        f"X={left_angle:+7.2f}deg "
                        f"W={left_gyro:+7.2f}dps "
                        f"age={left_age * 1000:6.1f}ms [{lstate}] | "
                        f"R {right_stats.hz:5.1f}Hz "
                        f"X={right_angle:+7.2f}deg "
                        f"W={right_gyro:+7.2f}dps "
                        f"age={right_age * 1000:6.1f}ms [{rstate}] | "
                        f"PH={phase_deg:+7.2f}deg | "
                        f"CMD={lcmd:+.3f}/{rcmd:+.3f} | "
                        f"ACT={actual_text} | "
                        f"T={teensy_stats.hz:5.1f}Hz | "
                        f"{'ON' if enabled else 'OFF'}"
                    )

                    next_print = now + print_period

                sleep_s = next_tick - time.perf_counter()

                if sleep_s > 0.0004:
                    time.sleep(min(0.001, sleep_s * 0.5))

    except KeyboardInterrupt:
        print("\nCtrl+C received.")

    finally:
        # Always command a safe stop first.
        try:
            for _ in range(3):
                teensy.send_torque(0.0, 0.0, False)
                teensy.send_stop()
                time.sleep(0.02)
        except Exception:
            pass

        stop_event.set()

        if plot_queue is not None:
            try:
                plot_queue.put_nowait(None)
            except Exception:
                pass

        if plot_close_event is not None:
            plot_close_event.set()

        left_imu.join(timeout=2.0)
        right_imu.join(timeout=2.0)
        teensy.join(timeout=2.0)

        if plot_process is not None:
            plot_process.join(timeout=2.0)

            if plot_process.is_alive():
                plot_process.terminate()
                plot_process.join(timeout=1.0)

        _, left_stats = left_imu.snapshot()
        _, right_stats = right_imu.snapshot()
        _, teensy_stats = teensy.snapshot()

        duration = max(
            time.perf_counter() - start_time,
            1e-9,
        )
        csv_hz = rows / duration

        print("=" * 104)
        print(f"CSV saved     : {csv_path}")
        print(
            f"CSV rows/rate : {rows} / {csv_hz:.2f} Hz "
            f"(target {a.rate:.1f} Hz)"
        )
        print(
            "Zero angle    : "
            f"L={left_zero.angle_x_deg:+.4f} deg, "
            f"R={right_zero.angle_x_deg:+.4f} deg"
        )
        print(
            "Gyro bias     : "
            f"L={left_zero.gyro_x_dps:+.4f} deg/s, "
            f"R={right_zero.gyro_x_dps:+.4f} deg/s"
        )
        print(
            f"IMU final Hz  : "
            f"L={left_stats.hz:.1f}, "
            f"R={right_stats.hz:.1f}"
        )
        print(
            f"IMU bad       : "
            f"L={left_stats.bad_packets}, "
            f"R={right_stats.bad_packets}"
        )
        print(
            f"Max sample age: "
            f"L={max_left_age_s * 1000:.1f} ms, "
            f"R={max_right_age_s * 1000:.1f} ms"
        )
        print(
            f"Stale rows    : "
            f"L={left_stale_rows}, "
            f"R={right_stale_rows} "
            f"(>{a.stale_warning * 1000:.0f} ms)"
        )
        print(
            f"Timeout rows  : "
            f"L={left_timeout_rows}, "
            f"R={right_timeout_rows}, "
            f"control={control_timeout_rows} "
            f"(>{a.imu_timeout * 1000:.0f} ms)"
        )
        print(
            f"Teensy final  : "
            f"{teensy_stats.hz:.1f} Hz, "
            f"crc_errors={teensy_stats.crc_errors}"
        )

        if left_imu.error:
            print("LEFT IMU error :", left_imu.error)

        if right_imu.error:
            print("RIGHT IMU error:", right_imu.error)

        if teensy.error:
            print("Teensy error   :", teensy.error)

        print("=" * 104)


if __name__ == "__main__":
    mp.freeze_support()
    main()