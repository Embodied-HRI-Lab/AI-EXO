"""
Raspberry Pi Paper-aligned Parametric Delayed Output Feedback Controller
==========================================================================

Based on the user's dual-IM948 + Teensy Raspberry Pi architecture, but the high-level
Samsung control law is changed to follow Lim et al. (IEEE RA-L 2023):

    y_raw = sin(q_right) - sin(q_left)
    y     = (1-alpha) * y_prev + alpha * y_raw
    u     = gain * y(t-delay)

The two hip commands are equal-and-opposite. By default this file preserves
this project's existing motor-command polarity:

    left_cmd  = -u
    right_cmd = +u

Use --torque-polarity -1 ONLY after a low-torque bench / walking-direction
check if your motor coordinate requires the opposite sign.

Main changes from the previous implementation
---------------------------------------------
1. Explicit delay is specified in seconds, not control-loop samples.
2. Default walking parameters are paper-inspired: alpha=0.05, delay=0.25 s.
3. The raw bilateral output state is smoothed directly, as in the 2023 paper.
4. The old phase-sign branch and independent flex/ext gains are removed.
5. The filter/history advances only when at least one fresh IMU packet arrives;
   repeated Pi control ticks do not repeatedly filter a held measurement.
6. CSV adds command/state/power-proxy/pair-skew fields needed for tuning.

Runtime keys
------------
Up / Down    : feedback gain +/- 0.5
Left / Right : explicit delay +/- 0.01 s
[ / ]        : smoothing alpha -/+ 0.01 (clamped 0.01..0.50)

Safety
------
Without --enable, Teensy always receives zero torque / enable=0.
Start human testing with a small --max-command and low gain; do not copy the
paper's high gains directly because actuator, gearing, sensing and latency differ.
"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import queue
import select
import struct
import sys
import termios
import threading
import time
import tty
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

DEFAULT_LEFT_IMU_PORT = "/dev/ttyUSB1"
DEFAULT_RIGHT_IMU_PORT = "/dev/ttyUSB0"
DEFAULT_TEENSY_PORT = "/dev/serial0"
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


def start_imu_reporting(uart: serial.Serial) -> None:
    """Wake the IM948 and enable its existing report configuration."""
    imu_send(uart, bytes([CMD_WAKE]), 0.20)
    imu_send(uart, bytes([CMD_REPORT_ON]), 0.20)


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
            else:
                # Normal Raspberry Pi startup: do not rewrite a configuration
                # that has already been verified stable; just wake/report ON.
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
# Samsung controller - paper-aligned parametric DOFC
# =============================================================================

@dataclass
class SamsungConfig:
    feedback_gain: float = 5.0
    smoothing_alpha: float = 0.05
    delay_s: float = 0.25
    max_command_nm: float = 8.0
    # Preserve this project's previous command polarity by default:
    # left=-u, right=+u. Set -1 only after direction verification.
    torque_polarity: float = 1.0


class SamsungController:
    """
    Parametric delayed output feedback controller.

    Paper-form state:
        y_raw = sin(q_right) - sin(q_left)
        y     = (1-alpha) * y_prev + alpha * y_raw
        u     = kappa * y(t-delay)

    The input angles must already be standing-zeroed and direction-normalized.
    """

    MAX_DELAY_S = 0.60
    HISTORY_MARGIN_S = 0.50

    def __init__(self, cfg: SamsungConfig) -> None:
        self.cfg = cfg
        self.filtered_state: float | None = None
        self.last_measurement_time: float | None = None
        self.history: deque[tuple[float, float]] = deque(maxlen=512)

        self.last_y_raw = 0.0
        self.last_y_filtered = 0.0
        self.last_y_delayed = 0.0
        self.last_u = 0.0
        self.last_left_tau = 0.0
        self.last_right_tau = 0.0
        self.delay_ready = False

    def _delayed_value(self, target_time: float) -> tuple[float, bool]:
        if not self.history:
            return 0.0, False

        if self.cfg.delay_s <= 1e-9:
            return self.history[-1][1], True

        # Need history that reaches the requested target time.
        if self.history[0][0] > target_time:
            return 0.0, False

        prev_t, prev_y = self.history[0]
        for next_t, next_y in list(self.history)[1:]:
            if next_t >= target_time:
                dt = next_t - prev_t
                if dt <= 1e-9:
                    return next_y, True
                beta = (target_time - prev_t) / dt
                beta = max(0.0, min(1.0, beta))
                return prev_y + beta * (next_y - prev_y), True
            prev_t, prev_y = next_t, next_y

        # target_time is newer than the most recent history entry.
        return self.history[-1][1], True

    def update(
        self,
        left_angle_deg: float,
        right_angle_deg: float,
        measurement_time: float,
    ) -> tuple[float, float]:
        left = math.radians(left_angle_deg)
        right = math.radians(right_angle_deg)

        # 2023 paper: form the bilateral state first, then smooth that state.
        y_raw = math.sin(right) - math.sin(left)
        alpha_nominal = max(0.001, min(1.0, self.cfg.smoothing_alpha))

        # The paper reports alpha for a 100-Hz update. Convert it to an
        # equivalent time-aware alpha so occasional packet jitter/dropout does
        # not silently change the filter dynamics. At exactly 100 Hz this is
        # identical to the paper's discrete equation.
        if self.last_measurement_time is None:
            dt = 0.01
        else:
            dt = max(0.001, min(measurement_time - self.last_measurement_time, 0.05))
        self.last_measurement_time = measurement_time
        alpha = 1.0 - (1.0 - alpha_nominal) ** (dt / 0.01)

        if self.filtered_state is None:
            self.filtered_state = y_raw
        else:
            self.filtered_state = (
                (1.0 - alpha) * self.filtered_state
                + alpha * y_raw
            )

        y_filtered = self.filtered_state
        self.history.append((measurement_time, y_filtered))

        cutoff_time = measurement_time - (
            self.MAX_DELAY_S + self.HISTORY_MARGIN_S
        )
        while len(self.history) > 2 and self.history[1][0] < cutoff_time:
            self.history.popleft()

        delay_s = max(0.0, min(self.cfg.delay_s, self.MAX_DELAY_S))
        y_delayed, ready = self._delayed_value(measurement_time - delay_s)

        self.last_y_raw = y_raw
        self.last_y_filtered = y_filtered
        self.last_y_delayed = y_delayed
        self.delay_ready = ready

        if not ready:
            self.last_u = 0.0
            self.last_left_tau = 0.0
            self.last_right_tau = 0.0
            return 0.0, 0.0

        u = self.cfg.feedback_gain * y_delayed
        polarity = 1.0 if self.cfg.torque_polarity >= 0.0 else -1.0

        # Preserve the previous project's physical command convention.
        left_tau = -polarity * u
        right_tau = polarity * u

        limit = abs(self.cfg.max_command_nm)
        left_tau = max(-limit, min(limit, left_tau))
        right_tau = max(-limit, min(limit, right_tau))

        self.last_u = u
        self.last_left_tau = left_tau
        self.last_right_tau = right_tau
        return left_tau, right_tau

    def held_command(self) -> tuple[float, float]:
        """Return last command without advancing filter or delay history."""
        return self.last_left_tau, self.last_right_tau

# =============================================================================
# Runtime keyboard control
# =============================================================================

GAIN_STEP = 0.5
DELAY_STEP_S = 0.01
ALPHA_STEP = 0.01


class LinuxKeyReader:
    """
    Non-blocking Linux/SSH terminal key reader.

    Arrow keys:
        Up/Down    -> gain +/- 0.5
        Left/Right -> delay +/- 0.01 s

    Ordinary keys:
        [ -> alpha -0.01
        ] -> alpha +0.01

    The original terminal settings are restored in close().
    """

    _ARROWS = {
        b"\x1b[A": "up",
        b"\x1b[B": "down",
        b"\x1b[D": "left",
        b"\x1b[C": "right",
        b"\x1bOA": "up",
        b"\x1bOB": "down",
        b"\x1bOD": "left",
        b"\x1bOC": "right",
    }

    def __init__(self) -> None:
        self.fd: int | None = None
        self.old_settings = None
        self.buffer = bytearray()

    def open(self) -> bool:
        if not sys.stdin.isatty():
            return False

        try:
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            return True
        except Exception:
            self.fd = None
            self.old_settings = None
            return False

    def close(self) -> None:
        if self.fd is not None and self.old_settings is not None:
            try:
                termios.tcsetattr(
                    self.fd,
                    termios.TCSADRAIN,
                    self.old_settings,
                )
            except Exception:
                pass

        self.fd = None
        self.old_settings = None
        self.buffer.clear()

    def poll(self) -> list[str]:
        if self.fd is None:
            return []

        while True:
            try:
                readable, _, _ = select.select([self.fd], [], [], 0.0)
            except Exception:
                return []

            if not readable:
                break

            try:
                chunk = os.read(self.fd, 32)
            except (BlockingIOError, OSError):
                break

            if not chunk:
                break

            self.buffer.extend(chunk)

        keys: list[str] = []

        while self.buffer:
            matched = False

            for seq, key in self._ARROWS.items():
                if self.buffer.startswith(seq):
                    del self.buffer[:len(seq)]
                    keys.append(key)
                    matched = True
                    break

            if matched:
                continue

            if self.buffer[0] == 0x1B:
                # Preserve a possibly incomplete arrow escape sequence.
                if len(self.buffer) < 3:
                    break
                del self.buffer[0]
                continue

            ch = self.buffer[0]
            del self.buffer[0]

            if ch == ord("["):
                keys.append("alpha_down")
            elif ch == ord("]"):
                keys.append("alpha_up")

        return keys


def apply_runtime_key(
    controller: SamsungController,
    key: str,
    control_hz: float,
) -> str | None:
    del control_hz  # Delay is deliberately time-based, not sample-based.

    if key == "up":
        controller.cfg.feedback_gain += GAIN_STEP
    elif key == "down":
        controller.cfg.feedback_gain = max(
            0.0,
            controller.cfg.feedback_gain - GAIN_STEP,
        )
    elif key == "left":
        controller.cfg.delay_s = max(
            0.0,
            controller.cfg.delay_s - DELAY_STEP_S,
        )
    elif key == "right":
        controller.cfg.delay_s = min(
            controller.MAX_DELAY_S,
            controller.cfg.delay_s + DELAY_STEP_S,
        )
    elif key == "alpha_down":
        controller.cfg.smoothing_alpha = max(
            0.01,
            controller.cfg.smoothing_alpha - ALPHA_STEP,
        )
    elif key == "alpha_up":
        controller.cfg.smoothing_alpha = min(
            0.50,
            controller.cfg.smoothing_alpha + ALPHA_STEP,
        )
    else:
        return None

    return (
        f"[PARAM] gain={controller.cfg.feedback_gain:.2f} | "
        f"delay={controller.cfg.delay_s:.3f}s | "
        f"alpha={controller.cfg.smoothing_alpha:.2f}"
    )


def drain_plot_key_queue(control_queue) -> list[str]:
    if control_queue is None:
        return []

    keys: list[str] = []
    while True:
        try:
            keys.append(control_queue.get_nowait())
        except queue.Empty:
            break
    return keys


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
    control_queue,
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
    latest_gain = math.nan
    latest_delay_s = math.nan
    latest_alpha = math.nan

    def on_close(_event) -> None:
        close_event.set()

    def on_key(event) -> None:
        key = event.key
        if key == "[":
            key = "alpha_down"
        elif key == "]":
            key = "alpha_up"
        elif key not in ("up", "down", "left", "right"):
            return

        try:
            control_queue.put_nowait(key)
        except queue.Full:
            pass

    fig.canvas.mpl_connect("close_event", on_close)
    fig.canvas.mpl_connect("key_press_event", on_key)

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
                gain,
                delay_s,
                alpha,
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
            latest_gain = gain
            latest_delay_s = delay_s
            latest_alpha = alpha
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
                f"control={'OK' if latest_control_ok else 'ZERO'}  |  "
                f"gain={latest_gain:.2f}  delay={latest_delay_s:.3f}s  "
                f"alpha={latest_alpha:.2f}"
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
        "--pair-skew-warning",
        type=float,
        default=0.030,
        help="warn in display/log when left/right IMU timestamps differ by more than this many seconds",
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
        "--configure-imu",
        action="store_true",
        help=(
            "rewrite IM948 to 100 Hz / report tag 0x0044; "
            "default only wakes the already-configured IMUs"
        ),
    )

    p.add_argument(
        "--gain", "--rescaling",
        dest="feedback_gain",
        type=float,
        default=5.0,
        help="DOFC feedback gain kappa; --rescaling is kept as an alias",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="state smoothing factor; paper walking range is typically 0.05-0.10",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="explicit DOFC delay in seconds; paper walking starting point 0.20-0.25 s",
    )
    p.add_argument(
        "--torque-polarity",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="global command polarity; +1 preserves the previous project convention",
    )
    p.add_argument(
        "--max-command",
        type=float,
        default=8.0,
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
            "left IMU, right IMU, and Teensy must use different serial devices"
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

    if not (0.01 <= a.alpha <= 0.50):
        raise ValueError("--alpha must be in [0.01, 0.50]")
    if not (0.0 <= a.delay <= 0.60):
        raise ValueError("--delay must be in [0.0, 0.60] s")
    if a.pair_skew_warning <= 0:
        raise ValueError("--pair-skew-warning must be positive")
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
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print(
                "[PLOT] No DISPLAY/WAYLAND_DISPLAY detected; "
                "falling back to --display print."
            )
            a.display = "print"
        else:
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
        configure=a.configure_imu,
        stop_event=stop_event,
    )

    right_imu = SingleImuReader(
        name="RIGHT",
        port=a.right_port,
        baud=a.baud,
        configure=a.configure_imu,
        stop_event=stop_event,
    )

    teensy = TeensyLink(
        port=a.teensy_port,
        baud=a.baud,
        stop_event=stop_event,
    )

    controller = SamsungController(
        SamsungConfig(
            feedback_gain=a.feedback_gain,
            smoothing_alpha=a.alpha,
            delay_s=a.delay,
            max_command_nm=a.max_command,
            torque_polarity=a.torque_polarity,
        )
    )

    plot_queue = None
    plot_close_event = None
    plot_control_queue = None
    plot_process = None

    if a.display == "plot":
        ctx = mp.get_context("spawn")
        plot_queue = ctx.Queue(maxsize=400)
        plot_close_event = ctx.Event()
        plot_control_queue = ctx.Queue(maxsize=32)

        plot_process = ctx.Process(
            target=plot_worker,
            args=(plot_queue, plot_close_event, plot_control_queue),
            kwargs={
                "refresh_hz": a.plot_rate,
                "window_s": a.plot_window,
                "stale_warning_s": a.stale_warning,
                "imu_timeout_s": a.imu_timeout,
            },
            daemon=True,
        )

    print("=" * 104)
    print("Samsung Raspberry Pi / paper-aligned DOFC / dual IMU / Teensy controller")
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
        f"Samsung   : gain={a.feedback_gain:.3f}, alpha={a.alpha:.2f}, "
        f"delay={a.delay:.3f}s, polarity={a.torque_polarity:+.0f}"
    )
    print(f"Pi clamp  : ±{a.max_command:.3f} Nm")
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
    print(
        "IMU setup : "
        + (
            "FORCE 100 Hz / report tag 0x0044"
            if a.configure_imu
            else "wake + report ON only"
        )
    )
    print(f"CSV       : {csv_path}")
    print(
        "Keys      : Up/Down gain ±0.5 | Left/Right delay ±0.01 s | "
        "[/] alpha ±0.01"
    )
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

    key_reader = LinuxKeyReader()
    keyboard_ok = key_reader.open()

    if keyboard_ok:
        print("[KEY] SSH/Linux runtime tuning enabled.")
    else:
        print("[KEY] stdin is not an interactive TTY; terminal tuning disabled.")

    try:
        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.writer(f)

            # Compact formal CSV: keep only the signals needed for gait/torque
            # analysis, plus the two parameters that can be changed online.
            # Alpha is intentionally omitted because normal walking should keep
            # it fixed at the default 0.05 unless a separate alpha experiment
            # is being performed.
            writer.writerow(
                [
                    "elapsed_s",
                    "left_angle_x_deg",
                    "left_angular_velocity_x_dps",
                    "right_angle_x_deg",
                    "right_angular_velocity_x_dps",
                    "left_command_torque_nm",
                    "right_command_torque_nm",
                    "left_actual_torque_nm",
                    "right_actual_torque_nm",
                    "feedback_gain",
                    "delay_s",
                ]
            )

            while not stop_event.is_set():
                if (
                    plot_close_event is not None
                    and plot_close_event.is_set()
                ):
                    break

                # Runtime tuning. Console arrows work on Windows; when the
                # plot window has focus, matplotlib forwards the same keys.
                runtime_keys = key_reader.poll()
                runtime_keys.extend(
                    drain_plot_key_queue(plot_control_queue)
                )

                for key in runtime_keys:
                    param_message = apply_runtime_key(
                        controller,
                        key,
                        a.rate,
                    )
                    if param_message is not None:
                        print(param_message)

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

                    pair_skew_s = (
                        abs(left_sample.host_time - right_sample.host_time)
                        if left_sample is not None and right_sample is not None
                        else math.inf
                    )
                    measurement_is_new = bool(left_new or right_new)

                    # Advance the paper-style discrete smoother/history only on
                    # a fresh sensor update. Repeated 100-Hz PC ticks simply hold
                    # the latest command instead of filtering the same sample again.
                    if imu_control_ok and measurement_is_new:
                        measurement_time = 0.5 * (
                            left_sample.host_time + right_sample.host_time
                        )
                        lcmd, rcmd = controller.update(
                            left_rel_deg,
                            right_rel_deg,
                            measurement_time,
                        )
                    elif imu_control_ok:
                        lcmd, rcmd = controller.held_command()
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

                    left_power_proxy_w = (
                        left_actual * math.radians(left_rel_gyro_dps)
                        if math.isfinite(left_actual) and math.isfinite(left_rel_gyro_dps)
                        else math.nan
                    )
                    right_power_proxy_w = (
                        right_actual * math.radians(right_rel_gyro_dps)
                        if math.isfinite(right_actual) and math.isfinite(right_rel_gyro_dps)
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
                            f"{lcmd:.4f}",
                            f"{rcmd:.4f}",
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
                            f"{controller.cfg.feedback_gain:.4f}",
                            f"{controller.cfg.delay_s:.4f}",
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
                                controller.cfg.feedback_gain,
                                controller.cfg.delay_s,
                                controller.cfg.smoothing_alpha,
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
                        f"Y={controller.last_y_delayed:+.3f} | "
                        f"G={controller.cfg.feedback_gain:.1f} "
                        f"A={controller.cfg.smoothing_alpha:.2f} "
                        f"D={controller.cfg.delay_s:.2f}s | "
                        f"SK={pair_skew_s * 1000.0:4.1f}ms"
                        f"{'!' if pair_skew_s > a.pair_skew_warning else ''} | "
                        f"Pxy={(left_power_proxy_w + right_power_proxy_w):+.2f}W | "
                        f"{'ON' if enabled else 'OFF'}"
                    )

                    next_print = now + print_period

                sleep_s = next_tick - time.perf_counter()

                if sleep_s > 0.0004:
                    time.sleep(min(0.001, sleep_s * 0.5))

    except KeyboardInterrupt:
        print("\nCtrl+C received.")

    finally:
        key_reader.close()

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