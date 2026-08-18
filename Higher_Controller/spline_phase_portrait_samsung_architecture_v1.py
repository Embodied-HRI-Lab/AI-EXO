"""
Phase Portrait + Spline hip exoskeleton controller
==================================================

This version intentionally keeps the proven Samsung PC controller communication
and concurrency architecture:

Main thread:
    - 100 Hz Phase Portrait + spline calculation
    - 100 Hz torque TX to Teensy
    - 100 Hz CSV logging

Thread 1:
    - LEFT IM948 only

Thread 2:
    - RIGHT IM948 only

Thread 3:
    - Teensy RX feedback

Optional separate process:
    - Matplotlib realtime plot

The ONLY high-level control change is:
    Samsung bilateral angle law
        -> bilateral Phase Portrait gait phase
        -> paper-inspired periodic PCHIP spline torque profile

Coordinate-chain rule:
    The zeroed angles and gyros use exactly the Samsung code path.
    CSV, plot and controller receive the same values. No plot-only or
    controller-only sign inversion is permitted. Default L/R angle sign = -1.

Safety:
    Without --enable, Teensy always receives 0 Nm / enable=0.
    IMU timeout >150 ms or Teensy feedback timeout forces immediate zero.
    After zeroing, a configurable HISTORY stage runs before assistance.

Default spline timing:
    extension peak : 9% gait cycle
    zero window    : 25-33%
    flexion peak   : 61% gait cycle

Runtime keys:
    Up / Down      : torque scale +/-0.1
    Left / Right   : spline phase offset +/-1% gait cycle

Examples:
    python spline_phase_portrait_samsung_architecture_v1.py --display print
    python spline_phase_portrait_samsung_architecture_v1.py --display plot
    python spline_phase_portrait_samsung_architecture_v1.py --display plot --enable --max-command 1.0

Dependencies:
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

try:
    import msvcrt  # Windows console keyboard input
except ImportError:
    msvcrt = None


# =============================================================================
# General configuration
# =============================================================================

HEADER: Final[bytes] = b"\xA5\x5A"

DEFAULT_LEFT_IMU_PORT = "COM8"
DEFAULT_RIGHT_IMU_PORT = "COM9"
DEFAULT_TEENSY_PORT = "COM10"
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
# Phase Portrait + Spline controller
# =============================================================================


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def wrap180(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def wrap360(angle_deg: float) -> float:
    return angle_deg % 360.0


class ShapePreservingCubic:
    """Small PCHIP-style cubic interpolator; no SciPy dependency required."""

    def __init__(self, x: list[float], y: list[float]) -> None:
        if len(x) != len(y) or len(x) < 2:
            raise ValueError("x and y must have equal length >= 2")
        self.x = [float(v) for v in x]
        self.y = [float(v) for v in y]
        for a, b in zip(self.x[:-1], self.x[1:]):
            if b <= a:
                raise ValueError("x must be strictly increasing")
        self.d = self._compute_derivatives()

    @staticmethod
    def _same_sign(a: float, b: float) -> bool:
        return (a > 0.0 and b > 0.0) or (a < 0.0 and b < 0.0)

    @staticmethod
    def _endpoint_slope(
        h0: float,
        h1: float,
        delta0: float,
        delta1: float,
    ) -> float:
        d = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / (h0 + h1)
        if not ShapePreservingCubic._same_sign(d, delta0):
            return 0.0
        if (
            not ShapePreservingCubic._same_sign(delta0, delta1)
            and abs(d) > abs(3.0 * delta0)
        ):
            return 3.0 * delta0
        return d

    def _compute_derivatives(self) -> list[float]:
        n = len(self.x)
        if n == 2:
            slope = (self.y[1] - self.y[0]) / (self.x[1] - self.x[0])
            return [slope, slope]

        h = [self.x[i + 1] - self.x[i] for i in range(n - 1)]
        delta = [
            (self.y[i + 1] - self.y[i]) / h[i]
            for i in range(n - 1)
        ]
        d = [0.0] * n

        d[0] = self._endpoint_slope(h[0], h[1], delta[0], delta[1])
        d[-1] = self._endpoint_slope(
            h[-1], h[-2], delta[-1], delta[-2]
        )

        for k in range(1, n - 1):
            dm = delta[k - 1]
            dp = delta[k]
            if dm == 0.0 or dp == 0.0 or not self._same_sign(dm, dp):
                d[k] = 0.0
            else:
                w1 = 2.0 * h[k] + h[k - 1]
                w2 = h[k] + 2.0 * h[k - 1]
                d[k] = (w1 + w2) / (w1 / dm + w2 / dp)

        return d

    def __call__(self, xq: float) -> float:
        if xq <= self.x[0]:
            return self.y[0]
        if xq >= self.x[-1]:
            return self.y[-1]

        lo = 0
        hi = len(self.x) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.x[mid] <= xq:
                lo = mid
            else:
                hi = mid

        i = lo
        h = self.x[i + 1] - self.x[i]
        t = (xq - self.x[i]) / h
        t2 = t * t
        t3 = t2 * t

        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2

        return (
            h00 * self.y[i]
            + h10 * h * self.d[i]
            + h01 * self.y[i + 1]
            + h11 * h * self.d[i + 1]
        )


@dataclass
class SplineProfileConfig:
    """
    Paper-inspired bilateral hip spline profile.

    Gait percentage is defined relative to the controller's LEFT gait cycle.
    Extension is negative and flexion is positive at this high-level anatomical
    layer. Hardware direction can be changed independently with torque signs.
    """

    ext_start_pct: float = 84.0
    ext_peak_pct: float = 9.0
    ext_end_pct: float = 25.0
    flex_start_pct: float = 33.0
    flex_peak_pct: float = 61.0
    flex_end_pct: float = 84.0
    ext_peak_nm: float = -1.0
    flex_peak_nm: float = +1.0


class HipSplineProfile:
    """Periodic shape-preserving hip torque profile."""

    def __init__(self, cfg: SplineProfileConfig) -> None:
        self.cfg = cfg
        self._validate()
        self._spline = self._build_periodic_spline()

    def _validate(self) -> None:
        p = self.cfg
        timings = [
            p.ext_start_pct,
            p.ext_peak_pct,
            p.ext_end_pct,
            p.flex_start_pct,
            p.flex_peak_pct,
            p.flex_end_pct,
        ]
        if any(v < 0.0 or v > 100.0 for v in timings):
            raise ValueError("all spline timing values must lie in [0, 100]")

        ext_start_previous = p.ext_start_pct - 100.0
        if not (
            ext_start_previous
            < p.ext_peak_pct
            < p.ext_end_pct
            <= p.flex_start_pct
            < p.flex_peak_pct
            < p.flex_end_pct
            <= p.ext_start_pct
        ):
            raise ValueError(
                "invalid spline timing order: ext_start(prev) < ext_peak < "
                "ext_end <= flex_start < flex_peak < flex_end <= ext_start"
            )
        if p.ext_peak_nm >= 0.0:
            raise ValueError("--ext-peak-nm must be negative")
        if p.flex_peak_nm <= 0.0:
            raise ValueError("--flex-peak-nm must be positive")

    def _one_cycle_nodes(self, shift: float) -> list[tuple[float, float]]:
        p = self.cfg
        return [
            (p.ext_start_pct - 100.0 + shift, 0.0),
            (p.ext_peak_pct + shift, p.ext_peak_nm),
            (p.ext_end_pct + shift, 0.0),
            (p.flex_start_pct + shift, 0.0),
            (p.flex_peak_pct + shift, p.flex_peak_nm),
            (p.flex_end_pct + shift, 0.0),
            (p.ext_start_pct + shift, 0.0),
        ]

    def _build_periodic_spline(self) -> ShapePreservingCubic:
        nodes: list[tuple[float, float]] = []
        for shift in (-100.0, 0.0, 100.0, 200.0):
            nodes.extend(self._one_cycle_nodes(shift))
        nodes.sort(key=lambda item: item[0])

        xs: list[float] = []
        ys: list[float] = []
        for x, y in nodes:
            if xs and abs(x - xs[-1]) < 1e-9:
                if abs(y - ys[-1]) > 1e-9:
                    raise ValueError("conflicting spline nodes")
                continue
            xs.append(float(x))
            ys.append(float(y))

        return ShapePreservingCubic(xs, ys)

    def torque_nm(self, gait_percent: float) -> float:
        return self._spline(gait_percent % 100.0)


@dataclass
class PhasePortraitConfig:
    history_window_s: float = 1.5
    amplitude_floor_deg: float = 6.0
    portrait_min_p2p_deg: float = 12.0
    walk_enter_p2p_deg: float = 14.0
    walk_exit_p2p_deg: float = 12.0
    walk_exit_hold_s: float = 0.35
    default_stride_s: float = 1.10
    min_stride_s: float = 0.65
    max_stride_s: float = 1.60
    lock_events: int = 2
    event_refractory_s: float = 0.55
    phase_kp_per_s: float = 4.0
    phase_correction_rate_dps: float = 180.0
    max_correction_error_deg: float = 60.0
    min_radius: float = 0.08
    max_radius: float = 3.0


@dataclass
class PhaseState:
    gait_signal_deg: float = 0.0
    gait_velocity_dps: float = 0.0
    gait_p2p_deg: float = 0.0
    gait_phase_deg: float = 0.0
    portrait_phase_deg: float = 0.0
    phase_error_deg: float = 0.0
    portrait_radius: float = 0.0
    portrait_valid: bool = False
    stride_period_s: float = 1.10
    cadence_spm: float = 120.0 / 1.10
    walking_active: bool = False
    phase_locked: bool = False
    event_detected: bool = False


class PhasePortraitEstimator:
    """
    Bilateral thigh phase portrait with a continuously advancing phase oscillator.

    gait_signal = LEFT relative hip angle - RIGHT relative hip angle
    gait_velocity = LEFT gyro - RIGHT gyro

    Portrait phase convention:
        0 deg   : centered signal, positive-going
        90 deg  : positive maximum of LEFT-RIGHT gait signal
        180 deg : centered signal, negative-going
        270 deg : negative maximum

    The internal oscillator continues through short portrait dropouts; valid
    portrait measurements gently correct it. This prevents phase freeze and
    therefore prevents a spline torque from sticking at a constant value.
    """

    def __init__(self, cfg: PhasePortraitConfig) -> None:
        self.cfg = cfg
        self.history: deque[tuple[float, float]] = deque()
        self.phase_deg = 0.0
        self.stride_period_s = cfg.default_stride_s
        self.last_update_time: float | None = None
        self.prev_x_norm: float | None = None
        self.last_event_time: float | None = None
        self.event_intervals: deque[float] = deque(maxlen=6)
        self.good_event_count = 0
        self.walking_active = False
        self.phase_locked = False
        self.exit_low_since: float | None = None

    def _history_stats(
        self,
        now: float,
        signal: float,
    ) -> tuple[float, float, float]:
        self.history.append((now, signal))
        cutoff = now - self.cfg.history_window_s
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

        vals = [v for _, v in self.history]
        if not vals:
            return 0.0, 0.0, self.cfg.amplitude_floor_deg

        vmin = min(vals)
        vmax = max(vals)
        p2p = vmax - vmin
        center = 0.5 * (vmax + vmin)
        amplitude = max(0.5 * p2p, self.cfg.amplitude_floor_deg)
        return p2p, center, amplitude

    def _update_walking(
        self,
        now: float,
        p2p: float,
        data_fresh: bool,
    ) -> None:
        if not data_fresh:
            self.walking_active = False
            self.phase_locked = False
            self.good_event_count = 0
            self.exit_low_since = None
            return

        if not self.walking_active:
            if p2p >= self.cfg.walk_enter_p2p_deg:
                self.walking_active = True
                self.exit_low_since = None
        else:
            if p2p < self.cfg.walk_exit_p2p_deg:
                if self.exit_low_since is None:
                    self.exit_low_since = now
                elif now - self.exit_low_since >= self.cfg.walk_exit_hold_s:
                    self.walking_active = False
                    self.phase_locked = False
                    self.good_event_count = 0
                    self.last_event_time = None
                    self.event_intervals.clear()
                    self.exit_low_since = None
            else:
                self.exit_low_since = None

    def _register_event(self, now: float) -> None:
        if self.last_event_time is not None:
            interval = now - self.last_event_time
            if self.cfg.min_stride_s <= interval <= self.cfg.max_stride_s:
                self.event_intervals.append(interval)
                ordered = sorted(self.event_intervals)
                core = ordered[1:-1] if len(ordered) >= 3 else ordered
                measured = sum(core) / len(core)
                self.stride_period_s = (
                    0.75 * self.stride_period_s + 0.25 * measured
                )
        self.last_event_time = now

    def update(
        self,
        now: float,
        gait_signal_deg: float,
        gait_velocity_dps: float,
        *,
        data_fresh: bool,
    ) -> PhaseState:
        if self.last_update_time is None:
            dt = 0.0
        else:
            dt = clamp(now - self.last_update_time, 0.0, 0.05)
        self.last_update_time = now

        # Predictor: gait clock always advances; never freeze on one phase.
        if dt > 0.0:
            self.phase_deg = wrap360(
                self.phase_deg
                + 360.0 * dt / max(self.stride_period_s, 1e-3)
            )

        p2p, center, amplitude = self._history_stats(now, gait_signal_deg)
        omega = 2.0 * math.pi / max(self.stride_period_s, 1e-3)
        x_norm = (gait_signal_deg - center) / max(amplitude, 1e-6)
        y_norm = gait_velocity_dps / max(amplitude * omega, 1e-6)
        radius = math.hypot(x_norm, y_norm)

        portrait_phase = wrap360(
            math.degrees(math.atan2(x_norm, y_norm))
        )
        portrait_valid = (
            data_fresh
            and p2p >= self.cfg.portrait_min_p2p_deg
            and self.cfg.min_radius <= radius <= self.cfg.max_radius
        )

        self._update_walking(now, p2p, data_fresh)

        event_detected = False
        if (
            self.walking_active
            and portrait_valid
            and self.prev_x_norm is not None
        ):
            crossed = (
                self.prev_x_norm < 0.0 <= x_norm
                and gait_velocity_dps > 0.0
            )
            refractory_ok = (
                self.last_event_time is None
                or now - self.last_event_time >= self.cfg.event_refractory_s
            )
            if crossed and refractory_ok:
                event_detected = True
                self._register_event(now)
                self.good_event_count += 1
                if (
                    self.good_event_count >= self.cfg.lock_events
                    and not self.phase_locked
                ):
                    # No assistance is allowed before lock; first lock may snap.
                    self.phase_deg = portrait_phase
                    self.phase_locked = True

        self.prev_x_norm = x_norm

        phase_error = wrap180(portrait_phase - self.phase_deg)
        if (
            self.walking_active
            and self.phase_locked
            and portrait_valid
            and dt > 0.0
            and abs(phase_error) <= self.cfg.max_correction_error_deg
        ):
            correction_rate = clamp(
                self.cfg.phase_kp_per_s * phase_error,
                -self.cfg.phase_correction_rate_dps,
                +self.cfg.phase_correction_rate_dps,
            )
            self.phase_deg = wrap360(
                self.phase_deg + correction_rate * dt
            )
            phase_error = wrap180(portrait_phase - self.phase_deg)

        if self.phase_locked and self.last_event_time is not None:
            max_no_event = max(1.8, 2.2 * self.stride_period_s)
            if now - self.last_event_time > max_no_event:
                self.phase_locked = False
                self.good_event_count = 0

        return PhaseState(
            gait_signal_deg=gait_signal_deg,
            gait_velocity_dps=gait_velocity_dps,
            gait_p2p_deg=p2p,
            gait_phase_deg=self.phase_deg,
            portrait_phase_deg=portrait_phase,
            phase_error_deg=phase_error if portrait_valid else 0.0,
            portrait_radius=radius,
            portrait_valid=portrait_valid,
            stride_period_s=self.stride_period_s,
            cadence_spm=120.0 / max(self.stride_period_s, 1e-3),
            walking_active=self.walking_active,
            phase_locked=self.phase_locked,
            event_detected=event_detected,
        )


class AssistGainRamp:
    def __init__(self, rise_s: float, fall_s: float) -> None:
        self.rise_s = max(rise_s, 1e-6)
        self.fall_s = max(fall_s, 1e-6)
        self.gain = 0.0

    def update(self, active: bool, dt: float) -> float:
        if dt <= 0.0:
            return self.gain
        if active:
            self.gain = min(1.0, self.gain + dt / self.rise_s)
        else:
            self.gain = max(0.0, self.gain - dt / self.fall_s)
        return self.gain


class SlewLimiter:
    def __init__(self, rate_nm_s: float) -> None:
        self.rate_nm_s = max(0.0, rate_nm_s)
        self.value = 0.0

    def update(self, target: float, dt: float) -> float:
        if dt <= 0.0 or self.rate_nm_s <= 0.0:
            self.value = target
            return self.value
        max_step = self.rate_nm_s * dt
        self.value += clamp(target - self.value, -max_step, +max_step)
        return self.value

    def reset(self, value: float = 0.0) -> None:
        self.value = value


@dataclass
class SplineControllerConfig:
    torque_scale: float = 1.0
    # Phase Portrait 90 deg ~= positive LEFT-RIGHT maximum, approximately left
    # heel strike for the current flexion-positive thigh convention. Therefore
    # -25% maps portrait 90 deg to spline gait 0% by default.
    spline_phase_offset_pct: float = -25.0
    left_torque_sign: float = 1.0
    right_torque_sign: float = 1.0
    assist_rise_s: float = 0.50
    assist_fall_s: float = 0.25
    slew_rate_nm_s: float = 10.0
    max_command_nm: float = 8.0


@dataclass
class SplineControlOutput:
    phase: PhaseState
    history_ready: bool
    assistance_active: bool
    assist_gain: float
    left_gait_pct: float
    right_gait_pct: float
    left_spline_base_nm: float
    right_spline_base_nm: float
    left_planned_nm: float
    right_planned_nm: float
    left_command_nm: float
    right_command_nm: float


class PhasePortraitSplineController:
    """
    Drop-in replacement for SamsungController at the main-loop level.

    IMPORTANT coordinate rule:
        This controller receives exactly the SAME standing-zeroed angles and
        gyros used by Samsung plot/CSV. It performs no additional left/right
        sign inversion.
    """

    def __init__(
        self,
        controller_cfg: SplineControllerConfig,
        phase_cfg: PhasePortraitConfig,
        spline_cfg: SplineProfileConfig,
    ) -> None:
        self.cfg = controller_cfg
        self.phase_estimator = PhasePortraitEstimator(phase_cfg)
        self.spline = HipSplineProfile(spline_cfg)
        self.assist_ramp = AssistGainRamp(
            controller_cfg.assist_rise_s,
            controller_cfg.assist_fall_s,
        )
        self.left_slew = SlewLimiter(controller_cfg.slew_rate_nm_s)
        self.right_slew = SlewLimiter(controller_cfg.slew_rate_nm_s)
        self.last_time: float | None = None

    def update(
        self,
        *,
        left_angle_deg: float,
        left_gyro_dps: float,
        right_angle_deg: float,
        right_gyro_dps: float,
        now: float,
        history_ready: bool,
        data_fresh: bool,
    ) -> SplineControlOutput:
        if self.last_time is None:
            dt = 1.0 / DEFAULT_CONTROL_HZ
        else:
            dt = clamp(now - self.last_time, 0.001, 0.05)
        self.last_time = now

        # Use the Samsung-proven relative coordinates directly. NO extra sign.
        gait_signal = left_angle_deg - right_angle_deg
        gait_velocity = left_gyro_dps - right_gyro_dps

        phase = self.phase_estimator.update(
            now,
            gait_signal,
            gait_velocity,
            data_fresh=data_fresh,
        )

        assistance_active = bool(
            history_ready
            and data_fresh
            and phase.walking_active
            and phase.phase_locked
        )
        assist_gain = self.assist_ramp.update(assistance_active, dt)

        base_pct = phase.gait_phase_deg / 3.6
        left_pct = (
            base_pct + self.cfg.spline_phase_offset_pct
        ) % 100.0
        # Right leg is half a gait cycle later. A spline is not antisymmetric,
        # so never use right_torque = -left_torque.
        right_pct = (left_pct + 50.0) % 100.0

        left_base = self.spline.torque_nm(left_pct)
        right_base = self.spline.torque_nm(right_pct)

        left_planned = (
            self.cfg.left_torque_sign
            * self.cfg.torque_scale
            * assist_gain
            * left_base
        )
        right_planned = (
            self.cfg.right_torque_sign
            * self.cfg.torque_scale
            * assist_gain
            * right_base
        )

        limit = abs(self.cfg.max_command_nm)
        left_planned = clamp(left_planned, -limit, +limit)
        right_planned = clamp(right_planned, -limit, +limit)

        if data_fresh:
            left_command = self.left_slew.update(left_planned, dt)
            right_command = self.right_slew.update(right_planned, dt)
        else:
            # Match Samsung timeout safety: true timeout means immediate zero.
            self.left_slew.reset(0.0)
            self.right_slew.reset(0.0)
            left_command = 0.0
            right_command = 0.0

        return SplineControlOutput(
            phase=phase,
            history_ready=history_ready,
            assistance_active=assistance_active,
            assist_gain=assist_gain,
            left_gait_pct=left_pct,
            right_gait_pct=right_pct,
            left_spline_base_nm=left_base,
            right_spline_base_nm=right_base,
            left_planned_nm=left_planned,
            right_planned_nm=right_planned,
            left_command_nm=left_command,
            right_command_nm=right_command,
        )


# =============================================================================
# Runtime keyboard control
# =============================================================================

TORQUE_SCALE_STEP = 0.1
PHASE_OFFSET_STEP_PCT = 1.0


def poll_console_arrow_keys() -> list[str]:
    """Read pending Windows console arrow keys without blocking control."""
    if msvcrt is None:
        return []

    keys: list[str] = []
    key_map = {
        b"H": "up",
        b"P": "down",
        b"K": "left",
        b"M": "right",
    }

    while msvcrt.kbhit():
        first = msvcrt.getch()
        if first in (b"\x00", b"\xe0"):
            second = msvcrt.getch()
            key = key_map.get(second)
            if key is not None:
                keys.append(key)

    return keys


def apply_runtime_key(
    controller: PhasePortraitSplineController,
    key: str,
) -> str | None:
    if key == "up":
        controller.cfg.torque_scale += TORQUE_SCALE_STEP
    elif key == "down":
        controller.cfg.torque_scale = max(
            0.0,
            controller.cfg.torque_scale - TORQUE_SCALE_STEP,
        )
    elif key == "left":
        controller.cfg.spline_phase_offset_pct -= PHASE_OFFSET_STEP_PCT
    elif key == "right":
        controller.cfg.spline_phase_offset_pct += PHASE_OFFSET_STEP_PCT
    else:
        return None

    # Keep the displayed value compact while preserving periodic equivalence.
    while controller.cfg.spline_phase_offset_pct <= -100.0:
        controller.cfg.spline_phase_offset_pct += 100.0
    while controller.cfg.spline_phase_offset_pct >= 100.0:
        controller.cfg.spline_phase_offset_pct -= 100.0

    return (
        f"[PARAM] torque_scale={controller.cfg.torque_scale:.2f} | "
        f"spline_phase_offset={controller.cfg.spline_phase_offset_pct:+.1f}%"
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
    phase_hist = deque(maxlen=history_len)
    portrait_hist = deque(maxlen=history_len)

    lp_hist = deque(maxlen=history_len)
    rp_hist = deque(maxlen=history_len)
    ls_hist = deque(maxlen=history_len)
    rs_hist = deque(maxlen=history_len)
    lact_hist = deque(maxlen=history_len)
    ract_hist = deque(maxlen=history_len)

    fig, (ax_angle, ax_phase, ax_torque) = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(11, 9),
    )

    # IMPORTANT: plot EXACTLY the relative angles supplied to the controller.
    # There is NO plotting-only sign inversion here.
    line_la, = ax_angle.plot([], [], label="Left relative angle")
    line_ra, = ax_angle.plot([], [], label="Right relative angle")
    ax_angle.axhline(0.0, linewidth=0.8)
    ax_angle.set_ylabel("Hip angle (deg)")
    ax_angle.set_title("Samsung-coordinate relative hip angles")
    ax_angle.legend(loc="upper right")
    ax_angle.grid(True, alpha=0.25)

    line_phase, = ax_phase.plot([], [], label="Control gait phase")
    line_portrait, = ax_phase.plot([], [], label="Portrait phase")
    ax_phase.set_ylim(-10.0, 370.0)
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.legend(loc="upper right")
    ax_phase.grid(True, alpha=0.25)

    line_lp, = ax_torque.plot([], [], label="Left planned")
    line_rp, = ax_torque.plot([], [], label="Right planned")
    line_ls, = ax_torque.plot([], [], linestyle="--", label="Left sent")
    line_rs, = ax_torque.plot([], [], linestyle="--", label="Right sent")
    line_lact, = ax_torque.plot([], [], alpha=0.65, label="Left actual")
    line_ract, = ax_torque.plot([], [], alpha=0.65, label="Right actual")
    ax_torque.axhline(0.0, linewidth=0.8)
    ax_torque.set_ylabel("Torque (Nm)")
    ax_torque.set_xlabel("Elapsed time (s)")
    ax_torque.legend(loc="upper right", ncol=2)
    ax_torque.grid(True, alpha=0.25)

    status_text = fig.suptitle("Waiting for samples...")

    latest_left_age = math.inf
    latest_right_age = math.inf
    latest_control_ok = False
    latest_history_ready = False
    latest_walking = False
    latest_locked = False
    latest_scale = math.nan
    latest_offset = math.nan

    def on_close(_event) -> None:
        close_event.set()

    def on_key(event) -> None:
        if event.key not in ("up", "down", "left", "right"):
            return
        try:
            control_queue.put_nowait(event.key)
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
                gait_phase,
                portrait_phase,
                left_planned,
                right_planned,
                left_sent,
                right_sent,
                left_actual,
                right_actual,
                left_age_s,
                right_age_s,
                control_ok,
                history_ready,
                walking_active,
                phase_locked,
                torque_scale,
                phase_offset_pct,
            ) = item

            t_hist.append(elapsed)
            la_hist.append(left_angle)
            ra_hist.append(right_angle)
            phase_hist.append(gait_phase)
            portrait_hist.append(portrait_phase)

            lp_hist.append(left_planned)
            rp_hist.append(right_planned)
            ls_hist.append(left_sent)
            rs_hist.append(right_sent)
            lact_hist.append(left_actual)
            ract_hist.append(right_actual)

            latest_left_age = left_age_s
            latest_right_age = right_age_s
            latest_control_ok = control_ok
            latest_history_ready = history_ready
            latest_walking = walking_active
            latest_locked = phase_locked
            latest_scale = torque_scale
            latest_offset = phase_offset_pct
            got_any = True

        now = time.perf_counter()

        if got_any and now >= next_refresh and len(t_hist) >= 2:
            x = list(t_hist)

            line_la.set_data(x, list(la_hist))
            line_ra.set_data(x, list(ra_hist))
            line_phase.set_data(x, list(phase_hist))
            line_portrait.set_data(x, list(portrait_hist))

            line_lp.set_data(x, list(lp_hist))
            line_rp.set_data(x, list(rp_hist))
            line_ls.set_data(x, list(ls_hist))
            line_rs.set_data(x, list(rs_hist))
            line_lact.set_data(x, list(lact_hist))
            line_ract.set_data(x, list(ract_hist))

            xmax = x[-1]
            xmin = max(x[0], xmax - window_s)
            for ax in (ax_angle, ax_phase, ax_torque):
                ax.set_xlim(xmin, max(xmax, xmin + 0.1))

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

            if not latest_history_ready:
                stage = "HISTORY"
            elif not latest_control_ok:
                stage = "ZERO/TIMEOUT"
            elif latest_locked:
                stage = "LOCKED"
            elif latest_walking:
                stage = "WALK/ACQ"
            else:
                stage = "IDLE"

            status_text.set_text(
                f"{stage} | "
                f"L age={latest_left_age * 1000:5.1f} ms [{left_state}] | "
                f"R age={latest_right_age * 1000:5.1f} ms [{right_state}] | "
                f"scale={latest_scale:.2f} | offset={latest_offset:+.1f}%"
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
    """Plotting may drop display samples but must never block control."""
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
            "Phase Portrait + spline controller using the proven Samsung "
            "dual-IMU / Teensy threading architecture"
        )
    )

    # Keep Samsung communication defaults and options.
    p.add_argument("--left-port", default=DEFAULT_LEFT_IMU_PORT)
    p.add_argument("--right-port", default=DEFAULT_RIGHT_IMU_PORT)
    p.add_argument("--teensy-port", default=DEFAULT_TEENSY_PORT)
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--rate", type=float, default=DEFAULT_CONTROL_HZ)

    p.add_argument(
        "--display",
        choices=("print", "plot"),
        default="print",
        help="print or realtime plot; mutually exclusive",
    )
    p.add_argument("--print-rate", type=float, default=DEFAULT_PRINT_HZ)
    p.add_argument("--plot-rate", type=float, default=DEFAULT_PLOT_HZ)
    p.add_argument("--plot-window", type=float, default=DEFAULT_PLOT_WINDOW_S)

    p.add_argument(
        "--stale-warning",
        type=float,
        default=DEFAULT_STALE_WARNING_S,
    )
    p.add_argument(
        "--imu-timeout",
        type=float,
        default=DEFAULT_IMU_TIMEOUT_S,
    )
    p.add_argument(
        "--teensy-timeout",
        type=float,
        default=DEFAULT_TEENSY_TIMEOUT_S,
    )

    p.add_argument("--zero-samples", type=int, default=200)
    p.add_argument("--zero-timeout", type=float, default=10.0)
    p.add_argument("--skip-zero", action="store_true")
    p.add_argument("--no-configure-imu", action="store_true")

    # IMPORTANT: these are copied from the Samsung-proven coordinate chain.
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

    # Startup history/acquisition stage after zeroing.
    p.add_argument("--history-seconds", type=float, default=3.0)

    # Spline amplitude / timing.
    p.add_argument("--torque-scale", type=float, default=1.0)
    p.add_argument(
        "--spline-phase-offset-pct",
        type=float,
        default=-25.0,
        help=(
            "spline shift relative to Phase Portrait. Default -25%% maps "
            "portrait 90 deg (positive L-R maximum) to gait 0%%"
        ),
    )
    p.add_argument("--ext-start-pct", type=float, default=84.0)
    p.add_argument("--ext-peak-pct", type=float, default=9.0)
    p.add_argument("--ext-end-pct", type=float, default=25.0)
    p.add_argument("--flex-start-pct", type=float, default=33.0)
    p.add_argument("--flex-peak-pct", type=float, default=61.0)
    p.add_argument("--flex-end-pct", type=float, default=84.0)
    p.add_argument("--ext-peak-nm", type=float, default=-1.0)
    p.add_argument("--flex-peak-nm", type=float, default=+1.0)

    p.add_argument(
        "--left-torque-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
    )
    p.add_argument(
        "--right-torque-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
    )

    # Phase Portrait / gait acquisition.
    p.add_argument("--phase-history-window", type=float, default=1.5)
    p.add_argument("--portrait-min-p2p", type=float, default=12.0)
    p.add_argument("--walk-enter-p2p", type=float, default=14.0)
    p.add_argument("--walk-exit-p2p", type=float, default=12.0)
    p.add_argument("--walk-exit-hold", type=float, default=0.35)
    p.add_argument("--default-stride", type=float, default=1.10)
    p.add_argument("--min-stride", type=float, default=0.65)
    p.add_argument("--max-stride", type=float, default=1.60)
    p.add_argument("--lock-events", type=int, default=2)
    p.add_argument("--event-refractory", type=float, default=0.55)
    p.add_argument("--phase-kp", type=float, default=4.0)
    p.add_argument("--phase-correction-rate", type=float, default=180.0)
    p.add_argument("--max-phase-error", type=float, default=60.0)

    # Torque transition/safety at PC level.
    p.add_argument("--assist-rise", type=float, default=0.50)
    p.add_argument("--assist-fall", type=float, default=0.25)
    p.add_argument("--pc-slew-nm-s", type=float, default=10.0)
    p.add_argument("--max-command", type=float, default=8.0)

    p.add_argument(
        "--enable",
        action="store_true",
        help=(
            "actually allow torque output; default is safe calculation/logging only"
        ),
    )
    p.add_argument("--csv", type=Path, default=None)

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

    if a.rate <= 0 or a.print_rate <= 0 or a.plot_rate <= 0:
        raise ValueError("control/print/plot rates must be positive")
    if a.plot_window <= 0:
        raise ValueError("--plot-window must be positive")
    if a.stale_warning <= 0:
        raise ValueError("--stale-warning must be positive")
    if a.imu_timeout <= a.stale_warning:
        raise ValueError("--imu-timeout must exceed --stale-warning")
    if a.teensy_timeout <= 0:
        raise ValueError("--teensy-timeout must be positive")
    if a.zero_samples <= 0 or a.zero_timeout <= 0:
        raise ValueError("zero calibration parameters must be positive")
    if a.history_seconds < 0:
        raise ValueError("--history-seconds cannot be negative")
    if a.torque_scale < 0:
        raise ValueError("--torque-scale cannot be negative")
    if a.max_command <= 0:
        raise ValueError("--max-command must be positive")
    if a.assist_rise <= 0 or a.assist_fall <= 0:
        raise ValueError("assist rise/fall times must be positive")
    if a.pc_slew_nm_s <= 0:
        raise ValueError("--pc-slew-nm-s must be positive")
    if a.lock_events <= 0:
        raise ValueError("--lock-events must be positive")
    if not (0 < a.min_stride < a.max_stride):
        raise ValueError("require 0 < min stride < max stride")


def default_csv_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs") / f"spline_phase_portrait_record_{stamp}.csv"


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

    # EXACT Samsung communication-thread architecture.
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

    spline_cfg = SplineProfileConfig(
        ext_start_pct=a.ext_start_pct,
        ext_peak_pct=a.ext_peak_pct,
        ext_end_pct=a.ext_end_pct,
        flex_start_pct=a.flex_start_pct,
        flex_peak_pct=a.flex_peak_pct,
        flex_end_pct=a.flex_end_pct,
        ext_peak_nm=a.ext_peak_nm,
        flex_peak_nm=a.flex_peak_nm,
    )
    phase_cfg = PhasePortraitConfig(
        history_window_s=a.phase_history_window,
        portrait_min_p2p_deg=a.portrait_min_p2p,
        walk_enter_p2p_deg=a.walk_enter_p2p,
        walk_exit_p2p_deg=a.walk_exit_p2p,
        walk_exit_hold_s=a.walk_exit_hold,
        default_stride_s=a.default_stride,
        min_stride_s=a.min_stride,
        max_stride_s=a.max_stride,
        lock_events=a.lock_events,
        event_refractory_s=a.event_refractory,
        phase_kp_per_s=a.phase_kp,
        phase_correction_rate_dps=a.phase_correction_rate,
        max_correction_error_deg=a.max_phase_error,
    )
    controller = PhasePortraitSplineController(
        SplineControllerConfig(
            torque_scale=a.torque_scale,
            spline_phase_offset_pct=a.spline_phase_offset_pct,
            left_torque_sign=a.left_torque_sign,
            right_torque_sign=a.right_torque_sign,
            assist_rise_s=a.assist_rise,
            assist_fall_s=a.assist_fall,
            slew_rate_nm_s=a.pc_slew_nm_s,
            max_command_nm=a.max_command,
        ),
        phase_cfg,
        spline_cfg,
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

    print("=" * 110)
    print("Phase Portrait + Spline controller / Samsung communication architecture")
    print(f"LEFT IMU  : {a.left_port} @ {a.baud} (independent thread)")
    print(f"RIGHT IMU : {a.right_port} @ {a.baud} (independent thread)")
    print(f"Teensy    : {a.teensy_port} @ {a.baud} (independent RX thread)")
    print(f"Control   : {a.rate:.1f} Hz | CSV target={a.rate:.1f} Hz")
    print(
        f"Display   : {a.display.upper()} | "
        + (
            f"{a.print_rate:.1f} Hz"
            if a.display == "print"
            else f"{a.plot_rate:.1f} Hz redraw / {a.plot_window:.1f}s window"
        )
    )
    print(
        f"IMU age   : OK <= {a.stale_warning*1000:.0f} ms | "
        f"STALE <= {a.imu_timeout*1000:.0f} ms | "
        f"TIMEOUT > {a.imu_timeout*1000:.0f} ms"
    )
    print(
        f"Angle coord: SAME AS SAMSUNG | Euler-X, standing=0 | "
        f"L sign={a.left_angle_sign:+.0f}, R sign={a.right_angle_sign:+.0f}"
    )
    print(
        f"Spline    : ext peak {a.ext_peak_pct:.1f}%/{a.ext_peak_nm:+.2f}Nm | "
        f"zero {a.ext_end_pct:.1f}-{a.flex_start_pct:.1f}% | "
        f"flex peak {a.flex_peak_pct:.1f}%/{a.flex_peak_nm:+.2f}Nm"
    )
    print(
        f"Mapping   : phase offset={a.spline_phase_offset_pct:+.1f}% | "
        "RIGHT gait = LEFT gait + 50%"
    )
    print(
        f"Phase     : history={a.phase_history_window:.2f}s | "
        f"walk enter/exit={a.walk_enter_p2p:.1f}/{a.walk_exit_p2p:.1f}deg p2p | "
        f"lock events={a.lock_events}"
    )
    print(
        f"Startup   : zero -> HISTORY {a.history_seconds:.1f}s -> walk/acquire -> lock -> ramp"
    )
    print(
        f"Torque    : scale={a.torque_scale:.2f} | "
        f"PC clamp=±{a.max_command:.2f}Nm | slew={a.pc_slew_nm_s:.1f}Nm/s"
    )
    print(
        f"Output    : {'ENABLED' if a.enable else 'DISABLED - calculate/log only'}"
    )
    print(f"CSV       : {csv_path}")
    print(
        "Keys      : Up/Down torque scale ±0.1 | "
        "Left/Right spline phase offset ±1% gait cycle"
    )
    print("Ctrl+C or closing plot -> zero torque + STOP x3")
    print("=" * 110)

    left_imu.start()
    right_imu.start()
    teensy.start()

    startup_deadline = time.perf_counter() + 7.0
    while time.perf_counter() < startup_deadline and not stop_event.is_set():
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

    if plot_process is not None:
        plot_process.start()

    period = 1.0 / a.rate
    print_period = 1.0 / a.print_rate
    start_time = time.perf_counter()
    history_ready_time = start_time + a.history_seconds
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

    last_output = SplineControlOutput(
        phase=PhaseState(),
        history_ready=False,
        assistance_active=False,
        assist_gain=0.0,
        left_gait_pct=0.0,
        right_gait_pct=50.0,
        left_spline_base_nm=0.0,
        right_spline_base_nm=0.0,
        left_planned_nm=0.0,
        right_planned_nm=0.0,
        left_command_nm=0.0,
        right_command_nm=0.0,
    )
    left_sent = 0.0
    right_sent = 0.0
    enabled = False

    try:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "elapsed_s",
                    "left_angle_x_deg",
                    "left_angular_velocity_x_dps",
                    "right_angle_x_deg",
                    "right_angular_velocity_x_dps",
                    "left_imu_age_s",
                    "right_imu_age_s",
                    "gait_signal_deg",
                    "gait_velocity_dps",
                    "gait_p2p_deg",
                    "gait_phase_deg",
                    "portrait_phase_deg",
                    "phase_error_deg",
                    "portrait_radius",
                    "portrait_valid",
                    "stride_period_s",
                    "cadence_spm",
                    "walking_active",
                    "phase_locked",
                    "event_detected",
                    "history_ready",
                    "assist_gain",
                    "left_gait_percent",
                    "right_gait_percent",
                    "torque_scale",
                    "spline_phase_offset_pct",
                    "left_spline_base_nm",
                    "right_spline_base_nm",
                    "left_planned_torque_nm",
                    "right_planned_torque_nm",
                    "left_sent_torque_nm",
                    "right_sent_torque_nm",
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

                runtime_keys = poll_console_arrow_keys()
                runtime_keys.extend(
                    drain_plot_key_queue(plot_control_queue)
                )
                for key in runtime_keys:
                    msg = apply_runtime_key(controller, key)
                    if msg is not None:
                        print(msg)

                now = time.perf_counter()

                if now >= next_tick:
                    left_sample, left_stats = left_imu.snapshot()
                    right_sample, right_stats = right_imu.snapshot()
                    feedback, teensy_stats = teensy.snapshot()

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

                    if math.isfinite(left_age_s):
                        max_left_age_s = max(max_left_age_s, left_age_s)
                    if math.isfinite(right_age_s):
                        max_right_age_s = max(max_right_age_s, right_age_s)

                    left_new = int(
                        left_sample is not None
                        and left_sample.sequence != last_left_seq
                    )
                    right_new = int(
                        right_sample is not None
                        and right_sample.sequence != last_right_seq
                    )
                    # Kept for Samsung-style diagnostics / future use.
                    _ = left_new, right_new

                    if left_sample is not None:
                        last_left_seq = left_sample.sequence
                    if right_sample is not None:
                        last_right_seq = right_sample.sequence

                    left_stale = left_age_s > a.stale_warning
                    right_stale = right_age_s > a.stale_warning
                    left_timeout = left_age_s > a.imu_timeout
                    right_timeout = right_age_s > a.imu_timeout

                    if left_stale:
                        left_stale_rows += 1
                    if right_stale:
                        right_stale_rows += 1
                    if left_timeout:
                        left_timeout_rows += 1
                    if right_timeout:
                        right_timeout_rows += 1

                    imu_control_ok = bool(
                        left_sample is not None
                        and right_sample is not None
                        and not left_timeout
                        and not right_timeout
                    )
                    if not imu_control_ok:
                        control_timeout_rows += 1

                    # EXACT Samsung angle/gyro chain; no new sign conversion.
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

                    history_ready = now >= history_ready_time

                    if (
                        math.isfinite(left_rel_deg)
                        and math.isfinite(right_rel_deg)
                        and math.isfinite(left_rel_gyro_dps)
                        and math.isfinite(right_rel_gyro_dps)
                    ):
                        last_output = controller.update(
                            left_angle_deg=left_rel_deg,
                            left_gyro_dps=left_rel_gyro_dps,
                            right_angle_deg=right_rel_deg,
                            right_gyro_dps=right_rel_gyro_dps,
                            now=now,
                            history_ready=history_ready,
                            data_fresh=imu_control_ok,
                        )

                    teensy_ok = bool(
                        feedback is not None
                        and now - feedback.host_time <= a.teensy_timeout
                    )

                    # Keep Teensy enabled during a normal gait-stop fade-out.
                    # Hard IMU/Teensy timeout still forces immediate zero.
                    enabled = bool(
                        a.enable
                        and history_ready
                        and imu_control_ok
                        and teensy_ok
                    )

                    left_sent = (
                        last_output.left_command_nm if enabled else 0.0
                    )
                    right_sent = (
                        last_output.right_command_nm if enabled else 0.0
                    )
                    teensy.send_torque(left_sent, right_sent, enabled)

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
                    ph = last_output.phase

                    def fmt(v: float) -> str:
                        return f"{v:.4f}" if math.isfinite(v) else ""

                    writer.writerow(
                        [
                            fmt(elapsed),
                            fmt(left_rel_deg),
                            fmt(left_rel_gyro_dps),
                            fmt(right_rel_deg),
                            fmt(right_rel_gyro_dps),
                            fmt(left_age_s),
                            fmt(right_age_s),
                            fmt(ph.gait_signal_deg),
                            fmt(ph.gait_velocity_dps),
                            fmt(ph.gait_p2p_deg),
                            fmt(ph.gait_phase_deg),
                            fmt(ph.portrait_phase_deg),
                            fmt(ph.phase_error_deg),
                            fmt(ph.portrait_radius),
                            int(ph.portrait_valid),
                            fmt(ph.stride_period_s),
                            fmt(ph.cadence_spm),
                            int(ph.walking_active),
                            int(ph.phase_locked),
                            int(ph.event_detected),
                            int(last_output.history_ready),
                            fmt(last_output.assist_gain),
                            fmt(last_output.left_gait_pct),
                            fmt(last_output.right_gait_pct),
                            fmt(controller.cfg.torque_scale),
                            fmt(controller.cfg.spline_phase_offset_pct),
                            fmt(last_output.left_spline_base_nm),
                            fmt(last_output.right_spline_base_nm),
                            fmt(last_output.left_planned_nm),
                            fmt(last_output.right_planned_nm),
                            fmt(left_sent),
                            fmt(right_sent),
                            fmt(left_actual),
                            fmt(right_actual),
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
                                ph.gait_phase_deg,
                                ph.portrait_phase_deg,
                                last_output.left_planned_nm,
                                last_output.right_planned_nm,
                                left_sent,
                                right_sent,
                                left_actual,
                                right_actual,
                                left_age_s,
                                right_age_s,
                                imu_control_ok,
                                last_output.history_ready,
                                ph.walking_active,
                                ph.phase_locked,
                                controller.cfg.torque_scale,
                                controller.cfg.spline_phase_offset_pct,
                            ),
                        )

                    next_tick += period
                    if now - next_tick > period:
                        next_tick = now + period

                if a.display == "print" and now >= next_print:
                    left_sample_p, left_stats = left_imu.snapshot()
                    right_sample_p, right_stats = right_imu.snapshot()
                    feedback_p, teensy_stats = teensy.snapshot()

                    left_age = (
                        time.perf_counter() - left_sample_p.host_time
                        if left_sample_p is not None
                        else math.inf
                    )
                    right_age = (
                        time.perf_counter() - right_sample_p.host_time
                        if right_sample_p is not None
                        else math.inf
                    )
                    lstate = sample_state(
                        left_age, a.stale_warning, a.imu_timeout
                    )
                    rstate = sample_state(
                        right_age, a.stale_warning, a.imu_timeout
                    )

                    if not last_output.history_ready:
                        remaining = max(0.0, history_ready_time - now)
                        stage = f"HIST {remaining:3.1f}s"
                    elif not imu_control_ok:
                        stage = "IMU_TIMEOUT"
                    elif last_output.phase.phase_locked:
                        stage = "LOCKED"
                    elif last_output.phase.walking_active:
                        stage = "WALK/ACQ"
                    else:
                        stage = "IDLE"

                    actual_text = (
                        f"{feedback_p.left_actual_nm:+.3f}/"
                        f"{feedback_p.right_actual_nm:+.3f}"
                        if feedback_p is not None
                        else "NO_FB"
                    )

                    print(
                        f"[{stage:11s}] "
                        f"L {left_stats.hz:5.1f}Hz "
                        f"X={left_rel_deg:+7.2f} W={left_rel_gyro_dps:+7.2f} "
                        f"age={left_age*1000:5.1f}ms [{lstate}] | "
                        f"R {right_stats.hz:5.1f}Hz "
                        f"X={right_rel_deg:+7.2f} W={right_rel_gyro_dps:+7.2f} "
                        f"age={right_age*1000:5.1f}ms [{rstate}] | "
                        f"PH={last_output.phase.gait_phase_deg:6.1f}deg "
                        f"GC={last_output.left_gait_pct:5.1f}/"
                        f"{last_output.right_gait_pct:5.1f}% | "
                        f"PLAN={last_output.left_planned_nm:+.3f}/"
                        f"{last_output.right_planned_nm:+.3f} | "
                        f"SENT={left_sent:+.3f}/{right_sent:+.3f} | "
                        f"ACT={actual_text} | "
                        f"T={teensy_stats.hz:5.1f}Hz | "
                        f"SCL={controller.cfg.torque_scale:.2f} "
                        f"OFF={controller.cfg.spline_phase_offset_pct:+.1f}% | "
                        f"{'ON' if enabled else 'OFF'}"
                    )
                    next_print = now + print_period

                sleep_s = next_tick - time.perf_counter()
                if sleep_s > 0.0004:
                    time.sleep(min(0.001, sleep_s * 0.5))

    except KeyboardInterrupt:
        print("\nCtrl+C received.")

    finally:
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

        duration = max(time.perf_counter() - start_time, 1e-9)
        csv_hz = rows / duration

        print("=" * 110)
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
            f"IMU final Hz  : L={left_stats.hz:.1f}, R={right_stats.hz:.1f}"
        )
        print(
            f"IMU bad       : L={left_stats.bad_packets}, R={right_stats.bad_packets}"
        )
        print(
            f"Max sample age: L={max_left_age_s*1000:.1f} ms, "
            f"R={max_right_age_s*1000:.1f} ms"
        )
        print(
            f"Stale rows    : L={left_stale_rows}, R={right_stale_rows} "
            f"(>{a.stale_warning*1000:.0f} ms)"
        )
        print(
            f"Timeout rows  : L={left_timeout_rows}, R={right_timeout_rows}, "
            f"control={control_timeout_rows} (>{a.imu_timeout*1000:.0f} ms)"
        )
        print(
            f"Teensy final  : {teensy_stats.hz:.1f} Hz, "
            f"crc_errors={teensy_stats.crc_errors}"
        )
        print(
            f"Final spline  : scale={controller.cfg.torque_scale:.2f}, "
            f"phase offset={controller.cfg.spline_phase_offset_pct:+.1f}%"
        )

        if left_imu.error:
            print("LEFT IMU error :", left_imu.error)
        if right_imu.error:
            print("RIGHT IMU error:", right_imu.error)
        if teensy.error:
            print("Teensy error   :", teensy.error)

        print("=" * 110)


if __name__ == "__main__":
    mp.freeze_support()
    main()
