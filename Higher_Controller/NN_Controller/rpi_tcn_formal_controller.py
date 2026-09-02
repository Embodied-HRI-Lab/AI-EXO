"""Compact 100 Hz Raspberry Pi bilateral IMU -> TCN -> Teensy controller.

The policy implementation lives in ``tcn_controllor.py``.  This file owns
only hardware timing, IMU resampling/calibration/filtering, command filtering,
serial transport, logging, and safety state.

Timing contract
---------------
- The external policy call grid is fixed at 100 Hz.
- Target IMU time is scheduled control tick minus a fixed delay (15 ms by
  default).
- Both IMUs are sampled at that same target time by interpolation or a very
  short bounded extrapolation.
- A short interpolation miss repeats the last final network input and still
  executes the policy, so stateful TorchScript history keeps advancing.
- After the short-HOLD grace period, the external command ramps toward zero
  through the normal output LPF/slew limiter.
- Only a true IMU timeout resets policy history and immediately disables the
  motor command.

Existing deployment packages are called at 100 Hz.  Some packages sample
their learned history at 30 Hz internally; the Raspberry Pi must not perform
that downsampling a second time.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import select
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import serial
except ImportError:  # pragma: no cover - present on the deployment Pi.
    serial = None

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - non-POSIX development host.
    termios = tty = None


# =============================================================================
# Defaults
# =============================================================================

HEADER = b"\xA5\x5A"

LEFT_IMU_PORT = "/dev/ttyUSB0"
RIGHT_IMU_PORT = "/dev/ttyUSB1"
TEENSY_PORT = "/dev/serial0"
BAUD = 115200

CONTROL_HZ = 100.0
PRINT_HZ = 10.0

# IMU -> TCN
IMU_DELAY_MS = 15.0
IMU_INTERP_MAX_GAP_MS = 50.0
IMU_EXTRAP_MAX_MS = 8.0
IMU_SHORT_HOLD_MS = 50.0
IMU_TIMEOUT_S = 0.150
IMU_HISTORY_LEN = 64

# All four trained input channels use the same validated 10 Hz causal LPF.
ANGLE_CUTOFF_HZ = 10.0
GYRO_CUTOFF_HZ = 10.0

# TCN -> Teensy.  These are an additional hardware layer; stateful deployment
# packages already contain their mandatory 0.21 N m/100-Hz-call slew limit.
OUTPUT_CUTOFF_HZ = 2.0
MAX_DELTA_NM_PER_STEP = 0.4
MAX_TORQUE_NM = 8.0

TORQUE_SCALE = 1.0
TORQUE_SCALE_STEP = 0.1
MIN_TORQUE_SCALE = 0.0
MAX_TORQUE_SCALE = 2.0

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "flat_thigh_imu_tcn_balanced_4p2nm_100hz_lzn.pt"
)

EXPECTED_INPUTS = [
    "left_thigh_angle_rad",
    "left_thigh_velocity_rad_s",
    "right_thigh_angle_rad",
    "right_thigh_velocity_rad_s",
]


# =============================================================================
# Teensy protocol
# =============================================================================

CMD_TORQUE = 0x54
CMD_STOP = 0x50
CMD_CLEAR_FAULT = 0x43
CMD_STATE = 0x44

TORQUE_PAYLOAD = struct.Struct("<HffB")
STATE_PAYLOAD = struct.Struct("<Hffff")
STATE_FRAME_SIZE = 2 + 1 + STATE_PAYLOAD.size + 1  # 22 bytes


# =============================================================================
# IM948 protocol
# =============================================================================

IMU_BEGIN = 0x49
IMU_END = 0x4D
IMU_ADDR = 0xFF

IMU_CMD_WAKE = 0x03
IMU_CMD_REPORT = 0x11
IMU_CMD_SET_PARAMS = 0x12
IMU_CMD_REPORT_OFF = 0x18
IMU_CMD_REPORT_ON = 0x19

REPORT_TAG = 0x0044
ANGLE_SCALE_DEG = 180.0 / 32768.0
GYRO_SCALE_DPS = 2000.0 / 32768.0
MAX_IMU_DATA_LEN = 128


# =============================================================================
# Data and helpers
# =============================================================================


@dataclass(frozen=True)
class ImuSample:
    angle_deg: float
    gyro_dps: float
    t: float
    seq: int


@dataclass(frozen=True)
class ImuZero:
    angle_deg: float
    gyro_dps: float


@dataclass(frozen=True)
class Feedback:
    seq: int
    left_actual: float
    right_actual: float
    t: float


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_deg(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


def crc8(data: bytes) -> int:
    crc = 0
    for value in data:
        crc ^= value
        for _ in range(8):
            crc = (
                ((crc << 1) ^ 0x07) if (crc & 0x80) else (crc << 1)
            ) & 0xFF
    return crc


def make_frame(command: int, payload: bytes = b"") -> bytes:
    body = bytes([command]) + payload
    return HEADER + body + bytes([crc8(body)])


# =============================================================================
# IM948
# =============================================================================


class ImuParser:
    def __init__(self) -> None:
        self.buf = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        if data:
            self.buf.extend(data)
        result: list[bytes] = []
        while True:
            start = self.buf.find(bytes([IMU_BEGIN]))
            if start < 0:
                self.buf.clear()
                return result
            if start:
                del self.buf[:start]
            if len(self.buf) < 3:
                return result
            data_len = self.buf[2]
            if data_len <= 0 or data_len > MAX_IMU_DATA_LEN:
                del self.buf[0]
                continue
            frame_len = data_len + 5
            if len(self.buf) < frame_len:
                return result
            frame = bytes(self.buf[:frame_len])
            if (
                frame[-1] != IMU_END
                or frame[3 + data_len]
                != (sum(frame[1 : 3 + data_len]) & 0xFF)
            ):
                del self.buf[0]
                continue
            del self.buf[:frame_len]
            result.append(frame[3 : 3 + data_len])


def imu_packet(body: bytes) -> bytes:
    core = bytes([IMU_BEGIN, IMU_ADDR, len(body)]) + body
    return b"\x00" * 50 + core + bytes([sum(core[1:]) & 0xFF, IMU_END])


def imu_send(uart: object, body: bytes, delay: float) -> None:
    uart.write(imu_packet(body))
    uart.flush()
    time.sleep(delay)


def configure_imu(uart: object) -> None:
    imu_send(uart, bytes([IMU_CMD_REPORT_OFF]), 0.15)
    imu_send(uart, bytes([IMU_CMD_WAKE]), 0.20)
    params = bytes(
        [
            IMU_CMD_SET_PARAMS,
            5,
            255,
            0,
            6,
            100,
            2,
            4,
            9,
            REPORT_TAG & 0xFF,
            (REPORT_TAG >> 8) & 0xFF,
        ]
    )
    imu_send(uart, params, 0.30)
    imu_send(uart, bytes([IMU_CMD_REPORT_ON]), 0.20)


def parse_imu(body: bytes, seq: int, timestamp: float) -> ImuSample | None:
    if len(body) < 7 or body[0] != IMU_CMD_REPORT:
        return None
    tag = int.from_bytes(body[1:3], "little")
    offset = 7
    gyro = None
    angle = None

    def skip(count: int) -> None:
        nonlocal offset
        offset += count
        if offset > len(body):
            raise ValueError("short IMU report")

    if tag & 0x0001:
        skip(6)
    if tag & 0x0002:
        skip(6)
    if tag & 0x0004:
        gyro = struct.unpack_from("<hhh", body, offset)
        offset += 6
    if tag & 0x0008:
        skip(6)
    if tag & 0x0010:
        skip(8)
    if tag & 0x0020:
        skip(8)
    if tag & 0x0040:
        angle = struct.unpack_from("<hhh", body, offset)

    if gyro is None or angle is None:
        return None
    return ImuSample(
        angle_deg=float(angle[0]) * ANGLE_SCALE_DEG,
        gyro_dps=float(gyro[0]) * GYRO_SCALE_DPS,
        t=float(timestamp),
        seq=int(seq),
    )


class ImuReader(threading.Thread):
    def __init__(
        self,
        name: str,
        port: str,
        baud: int,
        configure: bool,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=f"{name}IMU", daemon=True)
        self.side = name
        self.port = port
        self.baud = int(baud)
        self.configure = bool(configure)
        self.stop_event = stop_event
        self.lock = threading.Lock()
        self.hist: deque[ImuSample] = deque(maxlen=IMU_HISTORY_LEN)
        self.error = ""

    def snapshot(self) -> tuple[ImuSample | None, tuple[ImuSample, ...]]:
        with self.lock:
            latest = self.hist[-1] if self.hist else None
            return latest, tuple(self.hist)

    def clear(self) -> None:
        with self.lock:
            self.hist.clear()

    def run(self) -> None:
        if serial is None:
            self.error = "pyserial is not installed"
            self.stop_event.set()
            return
        uart = None
        parser = ImuParser()
        sequence = 0
        try:
            uart = serial.Serial(
                self.port,
                self.baud,
                timeout=0,
                write_timeout=0.5,
            )
            uart.reset_input_buffer()
            if self.configure:
                print(f"[IMU] configure {self.side}: {self.port}")
                configure_imu(uart)
                uart.reset_input_buffer()
            while not self.stop_event.is_set():
                count = uart.in_waiting
                if count:
                    for body in parser.feed(uart.read(count)):
                        sequence += 1
                        try:
                            sample = parse_imu(
                                body,
                                sequence,
                                time.perf_counter(),
                            )
                        except (ValueError, struct.error):
                            sample = None
                        if sample is not None:
                            with self.lock:
                                self.hist.append(sample)
                else:
                    time.sleep(0.0004)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.stop_event.set()
        finally:
            if uart is not None:
                try:
                    uart.close()
                except Exception:
                    pass


def sample_at(
    hist: tuple[ImuSample, ...],
    target_t: float,
    *,
    max_interp_gap_s: float,
    max_extrap_s: float,
) -> ImuSample | None:
    """Sample one IMU at a common fixed-grid timestamp."""

    if len(hist) < 2:
        return None
    if hist[0].t <= target_t <= hist[-1].t:
        for index in range(len(hist) - 1, 0, -1):
            first = hist[index - 1]
            second = hist[index]
            if first.t <= target_t <= second.t:
                duration = second.t - first.t
                if duration <= 0.0 or duration > max_interp_gap_s:
                    return None
                ratio = clamp((target_t - first.t) / duration, 0.0, 1.0)
                delta_angle = wrap_deg(second.angle_deg - first.angle_deg)
                return ImuSample(
                    angle_deg=wrap_deg(first.angle_deg + ratio * delta_angle),
                    gyro_dps=first.gyro_dps
                    + ratio * (second.gyro_dps - first.gyro_dps),
                    t=target_t,
                    seq=second.seq,
                )

    latest = hist[-1]
    ahead = target_t - latest.t
    if 0.0 < ahead <= max_extrap_s:
        return ImuSample(
            angle_deg=wrap_deg(latest.angle_deg + latest.gyro_dps * ahead),
            gyro_dps=latest.gyro_dps,
            t=target_t,
            seq=latest.seq,
        )
    return None


def circular_mean_deg(values: list[float]) -> float:
    sine = sum(math.sin(math.radians(value)) for value in values)
    cosine = sum(math.cos(math.radians(value)) for value in values)
    return math.degrees(math.atan2(sine, cosine))


def calibrate_zero(
    left: ImuReader,
    right: ImuReader,
    sample_count: int,
    timeout_s: float,
    stop_event: threading.Event,
) -> tuple[ImuZero, ImuZero]:
    print(f"[ZERO] collecting {sample_count} unique samples per side...")
    left_angles: list[float] = []
    left_gyros: list[float] = []
    right_angles: list[float] = []
    right_gyros: list[float] = []
    left_seq = -1
    right_seq = -1
    deadline = time.perf_counter() + timeout_s
    while not stop_event.is_set() and time.perf_counter() < deadline:
        left_sample, _ = left.snapshot()
        right_sample, _ = right.snapshot()
        if (
            left_sample is not None
            and left_sample.seq != left_seq
            and len(left_angles) < sample_count
        ):
            left_angles.append(left_sample.angle_deg)
            left_gyros.append(left_sample.gyro_dps)
            left_seq = left_sample.seq
        if (
            right_sample is not None
            and right_sample.seq != right_seq
            and len(right_angles) < sample_count
        ):
            right_angles.append(right_sample.angle_deg)
            right_gyros.append(right_sample.gyro_dps)
            right_seq = right_sample.seq
        if len(left_angles) >= sample_count and len(right_angles) >= sample_count:
            break
        time.sleep(0.001)

    if len(left_angles) < sample_count or len(right_angles) < sample_count:
        raise RuntimeError(
            f"zero timeout: L={len(left_angles)}/{sample_count}, "
            f"R={len(right_angles)}/{sample_count}"
        )
    left_zero = ImuZero(
        circular_mean_deg(left_angles),
        sum(left_gyros) / len(left_gyros),
    )
    right_zero = ImuZero(
        circular_mean_deg(right_angles),
        sum(right_gyros) / len(right_gyros),
    )
    print(
        f"[ZERO] L angle={left_zero.angle_deg:+.3f} deg "
        f"gyro={left_zero.gyro_dps:+.3f} dps"
    )
    print(
        f"[ZERO] R angle={right_zero.angle_deg:+.3f} deg "
        f"gyro={right_zero.gyro_dps:+.3f} dps"
    )
    return left_zero, right_zero


# =============================================================================
# Filters
# =============================================================================


class LPF:
    """First-order causal LPF matching the deployment-data preprocessing."""

    def __init__(self, cutoff_hz: float, sample_hz: float, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.alpha = 1.0 - math.exp(
            -2.0 * math.pi * float(cutoff_hz) / float(sample_hz)
        )
        self.ready = False
        self.y = 0.0

    def reset(self) -> None:
        self.ready = False
        self.y = 0.0

    def update(self, value: float) -> float:
        if not self.enabled:
            return float(value)
        if not self.ready:
            self.y = float(value)
            self.ready = True
        else:
            self.y += self.alpha * (float(value) - self.y)
        return self.y


class ImuFilter:
    def __init__(self, angle_fc: float, gyro_fc: float, enabled: bool) -> None:
        self.filters = [
            LPF(angle_fc, CONTROL_HZ, enabled),
            LPF(gyro_fc, CONTROL_HZ, enabled),
            LPF(angle_fc, CONTROL_HZ, enabled),
            LPF(gyro_fc, CONTROL_HZ, enabled),
        ]

    def reset(self) -> None:
        for filter_ in self.filters:
            filter_.reset()

    def update(
        self,
        values: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        result = tuple(
            filter_.update(value)
            for filter_, value in zip(self.filters, values, strict=True)
        )
        return tuple(float(value) for value in result)  # type: ignore[return-value]


class TorqueFilter:
    def __init__(
        self,
        cutoff_hz: float,
        output_lpf: bool,
        slew: bool,
        max_delta: float,
        limit: float,
    ) -> None:
        self.left = LPF(cutoff_hz, CONTROL_HZ, output_lpf)
        self.right = LPF(cutoff_hz, CONTROL_HZ, output_lpf)
        self.slew = bool(slew)
        self.max_delta = float(max_delta)
        self.limit = float(limit)
        self.prev_left = 0.0
        self.prev_right = 0.0

    def reset(self) -> None:
        self.left.reset()
        self.right.reset()
        self.prev_left = 0.0
        self.prev_right = 0.0

    def update(self, left_nm: float, right_nm: float) -> tuple[float, float]:
        left = self.left.update(left_nm)
        right = self.right.update(right_nm)
        if self.slew:
            left = clamp(
                left,
                self.prev_left - self.max_delta,
                self.prev_left + self.max_delta,
            )
            right = clamp(
                right,
                self.prev_right - self.max_delta,
                self.prev_right + self.max_delta,
            )
        left = clamp(left, -self.limit, self.limit)
        right = clamp(right, -self.limit, self.limit)
        self.prev_left = float(left)
        self.prev_right = float(right)
        return float(left), float(right)


# =============================================================================
# Teensy
# =============================================================================


class TeensyLink(threading.Thread):
    def __init__(
        self,
        port: str,
        baud: int,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="TeensyLink", daemon=True)
        self.port = port
        self.baud = int(baud)
        self.stop_event = stop_event
        self.lock = threading.Lock()
        self.tx_lock = threading.Lock()
        self.latest: Feedback | None = None
        self.uart = None
        self.rx = bytearray()
        self.tx_seq = 0
        self.error = ""

    def snapshot(self) -> Feedback | None:
        with self.lock:
            return self.latest

    def _write(self, data: bytes) -> None:
        if self.uart is not None and self.uart.is_open:
            with self.tx_lock:
                self.uart.write(data)

    def send_torque(self, left_nm: float, right_nm: float, enable: bool) -> None:
        payload = TORQUE_PAYLOAD.pack(
            self.tx_seq,
            float(left_nm),
            float(right_nm),
            int(bool(enable)),
        )
        self.tx_seq = (self.tx_seq + 1) & 0xFFFF
        self._write(make_frame(CMD_TORQUE, payload))

    def stop_motors(self) -> None:
        self._write(make_frame(CMD_STOP))

    def clear_fault(self) -> None:
        self._write(make_frame(CMD_CLEAR_FAULT))

    def _parse_latest(self) -> Feedback | None:
        newest = None
        while True:
            start = self.rx.find(HEADER)
            if start < 0:
                if self.rx and self.rx[-1] == HEADER[0]:
                    self.rx[:] = self.rx[-1:]
                else:
                    self.rx.clear()
                return newest
            if start:
                del self.rx[:start]
            if len(self.rx) < 3:
                return newest
            if self.rx[2] != CMD_STATE:
                del self.rx[0]
                continue
            if len(self.rx) < STATE_FRAME_SIZE:
                return newest
            frame = bytes(self.rx[:STATE_FRAME_SIZE])
            body = frame[2:-1]
            if frame[-1] != crc8(body):
                del self.rx[0]
                continue
            del self.rx[:STATE_FRAME_SIZE]
            sequence, left_tau, right_tau, _left_pos, _right_pos = (
                STATE_PAYLOAD.unpack(frame[3:-1])
            )
            newest = Feedback(
                seq=int(sequence),
                left_actual=float(left_tau),
                right_actual=float(right_tau),
                t=time.perf_counter(),
            )

    def run(self) -> None:
        if serial is None:
            self.error = "pyserial is not installed"
            self.stop_event.set()
            return
        try:
            self.uart = serial.Serial(
                self.port,
                self.baud,
                timeout=0,
                write_timeout=0.5,
            )
            self.uart.reset_input_buffer()
            self.stop_motors()
            time.sleep(0.03)
            self.clear_fault()
            time.sleep(0.03)
            while not self.stop_event.is_set():
                count = self.uart.in_waiting
                if count:
                    self.rx.extend(self.uart.read(count))
                    newest = self._parse_latest()
                    if newest is not None:
                        with self.lock:
                            self.latest = newest
                else:
                    time.sleep(0.0004)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.stop_event.set()
        finally:
            if self.uart is not None:
                try:
                    for _ in range(3):
                        self.stop_motors()
                        time.sleep(0.01)
                except Exception:
                    pass
                try:
                    self.uart.close()
                except Exception:
                    pass


# =============================================================================
# Keyboard
# =============================================================================


class Keyboard(threading.Thread):
    """SSH-friendly W/S, +/-, and arrow-key torque-scale control."""

    def __init__(
        self,
        initial: float,
        step: float,
        minimum: float,
        maximum: float,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="Keyboard", daemon=True)
        self.scale = float(initial)
        self.step = float(step)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.stop_event = stop_event
        self.lock = threading.Lock()
        self.changed = False
        self.available = False
        self.source = "none"

    def snapshot(self) -> tuple[float, bool]:
        with self.lock:
            changed = self.changed
            self.changed = False
            return self.scale, changed

    def _change(self, delta: float) -> None:
        with self.lock:
            new_scale = clamp(
                self.scale + delta,
                self.minimum,
                self.maximum,
            )
            if abs(new_scale - self.scale) > 1.0e-12:
                self.scale = new_scale
                self.changed = True

    def _open_terminal(self) -> tuple[int | None, bool]:
        if termios is None or tty is None:
            return None, False
        try:
            if sys.stdin.isatty():
                self.source = "stdin"
                return sys.stdin.fileno(), False
        except Exception:
            pass
        try:
            descriptor = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
            self.source = "/dev/tty"
            return descriptor, True
        except OSError:
            return None, False

    def run(self) -> None:
        descriptor, close_descriptor = self._open_terminal()
        if descriptor is None:
            print(
                "\n[KEYBOARD] disabled: no interactive TTY; use ssh -t for "
                "remote key control."
            )
            return
        try:
            previous = termios.tcgetattr(descriptor)
        except Exception:
            if close_descriptor:
                os.close(descriptor)
            print("\n[KEYBOARD] disabled: cannot configure terminal.")
            return

        self.available = True
        print(
            f"\n[KEYBOARD] active via {self.source}: "
            "UP/W/+ increase, DOWN/S/- decrease, Q quit"
        )
        escape_state = 0
        try:
            tty.setcbreak(descriptor)
            while not self.stop_event.is_set():
                ready, _, _ = select.select([descriptor], [], [], 0.10)
                if not ready:
                    continue
                try:
                    data = os.read(descriptor, 64)
                except BlockingIOError:
                    continue
                for value in data:
                    character = chr(value)
                    if escape_state == 1:
                        if character == "[":
                            escape_state = 2
                            continue
                        escape_state = 0
                    elif escape_state == 2:
                        if character == "A":
                            self._change(+self.step)
                        elif character == "B":
                            self._change(-self.step)
                        escape_state = 0
                        continue
                    if value == 0x1B:
                        escape_state = 1
                    elif character in ("q", "Q"):
                        self.stop_event.set()
                        return
                    elif character in ("w", "W", "+", "="):
                        self._change(+self.step)
                    elif character in ("s", "S", "-", "_"):
                        self._change(-self.step)
        finally:
            try:
                termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)
            except Exception:
                pass
            if close_descriptor:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


# =============================================================================
# CLI and validation
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed-grid 100 Hz bilateral IMU/TCN/Teensy controller."
    )
    parser.add_argument("--no-policy", action="store_true")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--left-port", default=LEFT_IMU_PORT)
    parser.add_argument("--right-port", default=RIGHT_IMU_PORT)
    parser.add_argument("--teensy-port", default=TEENSY_PORT)
    parser.add_argument("--baud", type=int, default=BAUD)
    parser.add_argument("--print-rate", type=float, default=PRINT_HZ)
    parser.add_argument("--imu-delay-ms", type=float, default=IMU_DELAY_MS)
    parser.add_argument(
        "--imu-interp-max-gap-ms",
        type=float,
        default=IMU_INTERP_MAX_GAP_MS,
    )
    parser.add_argument(
        "--imu-extrap-max-ms",
        type=float,
        default=IMU_EXTRAP_MAX_MS,
    )
    parser.add_argument(
        "--imu-short-hold-ms",
        type=float,
        default=IMU_SHORT_HOLD_MS,
    )
    parser.add_argument("--imu-timeout", type=float, default=IMU_TIMEOUT_S)
    parser.add_argument("--no-configure-imu", action="store_true")
    parser.add_argument("--angle-cutoff-hz", type=float, default=ANGLE_CUTOFF_HZ)
    parser.add_argument("--gyro-cutoff-hz", type=float, default=GYRO_CUTOFF_HZ)
    parser.add_argument("--no-imu-filter", action="store_true")
    parser.add_argument(
        "--output-cutoff-hz",
        type=float,
        default=OUTPUT_CUTOFF_HZ,
    )
    parser.add_argument("--no-output-filter", action="store_true")
    parser.add_argument(
        "--max-delta-nm-per-step",
        type=float,
        default=MAX_DELTA_NM_PER_STEP,
    )
    parser.add_argument("--no-slew-limiter", action="store_true")
    parser.add_argument("--max-torque", type=float, default=MAX_TORQUE_NM)
    parser.add_argument("--torque-scale", type=float, default=TORQUE_SCALE)
    parser.add_argument(
        "--torque-scale-step",
        type=float,
        default=TORQUE_SCALE_STEP,
    )
    parser.add_argument("--min-torque-scale", type=float, default=MIN_TORQUE_SCALE)
    parser.add_argument("--max-torque-scale", type=float, default=MAX_TORQUE_SCALE)
    parser.add_argument("--zero-settle", type=float, default=3.0)
    parser.add_argument("--zero-samples", type=int, default=200)
    parser.add_argument("--zero-timeout", type=float, default=10.0)
    parser.add_argument("--skip-zero", action="store_true")
    parser.add_argument(
        "--left-imu-direction",
        type=float,
        choices=(-1.0, 1.0),
        default=-1.0,
    )
    parser.add_argument(
        "--right-imu-direction",
        type=float,
        choices=(-1.0, 1.0),
        default=-1.0,
    )
    parser.add_argument("--arm", action="store_true")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.print_rate <= 0:
        raise ValueError("--print-rate must be > 0")
    if args.imu_delay_ms < 0:
        raise ValueError("--imu-delay-ms must be >= 0")
    if args.imu_interp_max_gap_ms <= 0:
        raise ValueError("--imu-interp-max-gap-ms must be > 0")
    if args.imu_extrap_max_ms < 0:
        raise ValueError("--imu-extrap-max-ms must be >= 0")
    if args.imu_short_hold_ms < 0:
        raise ValueError("--imu-short-hold-ms must be >= 0")
    if args.imu_timeout <= 0:
        raise ValueError("--imu-timeout must be > 0")
    if not 0 < args.max_torque <= 10.0:
        raise ValueError("--max-torque must be in (0, 10]")
    if args.max_delta_nm_per_step <= 0:
        raise ValueError("--max-delta-nm-per-step must be > 0")
    if not 0.0 < args.angle_cutoff_hz < 0.5 * CONTROL_HZ:
        raise ValueError("--angle-cutoff-hz must be in (0, 50)")
    if not 0.0 < args.gyro_cutoff_hz < 0.5 * CONTROL_HZ:
        raise ValueError("--gyro-cutoff-hz must be in (0, 50)")
    if not 0.0 < args.output_cutoff_hz < 0.5 * CONTROL_HZ:
        raise ValueError("--output-cutoff-hz must be in (0, 50)")
    if args.torque_scale_step <= 0:
        raise ValueError("--torque-scale-step must be > 0")
    if not (
        0.0 <= args.min_torque_scale
        <= args.torque_scale
        <= args.max_torque_scale
    ):
        raise ValueError("initial torque scale is outside allowed range")


def default_csv(no_policy: bool) -> Path:
    prefix = "imu_fixedgrid" if no_policy else "tcn_fixedgrid"
    return Path("logs") / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.csv"


def load_policy(args: argparse.Namespace):
    if args.no_policy:
        return None
    import torch

    from tcn_controllor import UnifiedTCNPolicy

    torch.set_num_threads(1)
    policy = UnifiedTCNPolicy(args.model.expanduser().resolve())
    input_hz = int(getattr(policy, "input_hz", policy.sensor_hz))
    if input_hz != 100:
        raise RuntimeError(f"checkpoint input_hz={input_hz}, expected 100")
    if int(policy.control_hz) != 100:
        raise RuntimeError(
            f"checkpoint control_hz={policy.control_hz}, expected 100"
        )
    if list(policy.input_channel_names) != EXPECTED_INPUTS:
        raise RuntimeError(
            "unexpected TCN input order:\n"
            f"  checkpoint={policy.input_channel_names}\n"
            f"  expected={EXPECTED_INPUTS}"
        )

    expected_cutoff = getattr(policy, "input_filter_cutoff_hz", None)
    if expected_cutoff is not None and not args.no_imu_filter:
        if not math.isclose(
            float(args.angle_cutoff_hz),
            float(expected_cutoff),
            abs_tol=1.0e-9,
        ) or not math.isclose(
            float(args.gyro_cutoff_hz),
            float(expected_cutoff),
            abs_tol=1.0e-9,
        ):
            raise RuntimeError(
                "IMU filter differs from checkpoint contract: "
                f"model={expected_cutoff:g} Hz, "
                f"angle={args.angle_cutoff_hz:g} Hz, "
                f"gyro={args.gyro_cutoff_hz:g} Hz"
            )
    return policy


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    policy = load_policy(args)

    imu_filter = ImuFilter(
        args.angle_cutoff_hz,
        args.gyro_cutoff_hz,
        enabled=not args.no_imu_filter,
    )
    torque_filter = None
    if policy is not None:
        torque_filter = TorqueFilter(
            args.output_cutoff_hz,
            output_lpf=not args.no_output_filter,
            slew=not args.no_slew_limiter,
            max_delta=args.max_delta_nm_per_step,
            limit=args.max_torque,
        )

    csv_path = (
        args.csv.expanduser().resolve()
        if args.csv is not None
        else default_csv(args.no_policy).resolve()
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("100 Hz FIXED-GRID LOW-LATENCY IMU / TCN CONTROLLER")
    print(f"IMU grid delay      : {args.imu_delay_ms:g} ms")
    print(f"Max interp bracket  : {args.imu_interp_max_gap_ms:g} ms")
    print(f"Max extrapolation   : {args.imu_extrap_max_ms:g} ms")
    print(f"Short command hold  : {args.imu_short_hold_ms:g} ms")
    print(f"Hard IMU timeout    : {args.imu_timeout * 1000.0:g} ms")
    print(
        "IMU LPF             : "
        + (
            "OFF"
            if args.no_imu_filter
            else (
                f"angle {args.angle_cutoff_hz:g} Hz / "
                f"gyro {args.gyro_cutoff_hz:g} Hz"
            )
        )
    )
    if policy is None:
        print("Mode                : IMU only")
    else:
        policy.print_startup_summary()
        print(
            "External output     : "
            f"LPF={'OFF' if args.no_output_filter else f'{args.output_cutoff_hz:g} Hz'}, "
            f"slew={'OFF' if args.no_slew_limiter else f'{args.max_delta_nm_per_step:g} N m/step'}, "
            f"limit=+/-{args.max_torque:g} N m"
        )
        print("Teensy RX           : latest 22-byte STATE <Hffff>")
        print(f"Arm                 : {args.arm}")
        print("Keys                : UP/W/+ increase, DOWN/S/- decrease, Q quit")
    print(f"CSV                 : {csv_path}")
    print("=" * 96)

    stop_event = threading.Event()
    left = ImuReader(
        "LEFT",
        args.left_port,
        args.baud,
        not args.no_configure_imu,
        stop_event,
    )
    right = ImuReader(
        "RIGHT",
        args.right_port,
        args.baud,
        not args.no_configure_imu,
        stop_event,
    )
    left.start()
    right.start()

    teensy = None
    if policy is not None:
        teensy = TeensyLink(args.teensy_port, args.baud, stop_event)
        teensy.start()

    keyboard = Keyboard(
        args.torque_scale,
        args.torque_scale_step,
        args.min_torque_scale,
        args.max_torque_scale,
        stop_event,
    )
    keyboard.start()

    last_feedback_seq = -1
    soft_hold_count = 0
    hard_timeout_count = 0

    try:
        deadline = time.perf_counter() + 8.0
        while time.perf_counter() < deadline and not stop_event.is_set():
            if left.snapshot()[0] is not None and right.snapshot()[0] is not None:
                break
            time.sleep(0.01)
        if left.snapshot()[0] is None or right.snapshot()[0] is None:
            raise RuntimeError(
                "IMU startup failed. "
                f"LEFT='{left.error}', RIGHT='{right.error}'"
            )

        if args.skip_zero:
            left_zero = ImuZero(0.0, 0.0)
            right_zero = ImuZero(0.0, 0.0)
            print("[ZERO] skipped")
        else:
            if args.zero_settle > 0:
                print(f"[ZERO] keep still for {args.zero_settle:.1f} s...")
                time.sleep(args.zero_settle)
            left_zero, right_zero = calibrate_zero(
                left,
                right,
                args.zero_samples,
                args.zero_timeout,
                stop_event,
            )

        left.clear()
        right.clear()
        imu_filter.reset()
        if policy is not None:
            policy.reset()
            torque_filter.reset()
        time.sleep(max(0.06, args.imu_delay_ms * 0.001 + 0.04))

        period = 1.0 / CONTROL_HZ
        print_period = 1.0 / args.print_rate
        delay_s = args.imu_delay_ms * 0.001
        max_interp_gap_s = args.imu_interp_max_gap_ms * 0.001
        max_extrap_s = args.imu_extrap_max_ms * 0.001
        short_hold_s = args.imu_short_hold_ms * 0.001

        start = time.perf_counter()
        next_tick = start
        next_print = start
        obs = (math.nan, math.nan, math.nan, math.nan)
        last_good_resample_wall = -math.inf
        left_cmd = 0.0
        right_cmd = 0.0
        command_ready = False
        hard_timeout_active = False

        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "timestamp",
                    "left_angle_rad",
                    "left_velocity_rad_s",
                    "right_angle_rad",
                    "right_velocity_rad_s",
                    "left_raw_tcn_nm",
                    "right_raw_tcn_nm",
                    "left_cmd_nm",
                    "right_cmd_nm",
                    "left_actual_nm",
                    "right_actual_nm",
                    "teensy_seq",
                ]
            )

            while not stop_event.is_set():
                now = time.perf_counter()
                if args.duration is not None and now - start >= args.duration:
                    break
                if now < next_tick:
                    time.sleep(min(0.0005, next_tick - now))
                    continue

                scheduled_tick = next_tick
                tick_wall = time.perf_counter()
                elapsed = scheduled_tick - start
                target_t = scheduled_tick - delay_s
                next_tick += period

                left_latest, left_hist = left.snapshot()
                right_latest, right_hist = right.snapshot()
                left_age = (
                    tick_wall - left_latest.t
                    if left_latest is not None
                    else math.inf
                )
                right_age = (
                    tick_wall - right_latest.t
                    if right_latest is not None
                    else math.inf
                )
                hard_timeout = (
                    left_latest is None
                    or right_latest is None
                    or left_age > args.imu_timeout
                    or right_age > args.imu_timeout
                )
                fresh_resample = False
                use_short_hold = False
                raw_left_tcn_nm = math.nan
                raw_right_tcn_nm = math.nan

                if not hard_timeout:
                    left_sample = sample_at(
                        left_hist,
                        target_t,
                        max_interp_gap_s=max_interp_gap_s,
                        max_extrap_s=max_extrap_s,
                    )
                    right_sample = sample_at(
                        right_hist,
                        target_t,
                        max_interp_gap_s=max_interp_gap_s,
                        max_extrap_s=max_extrap_s,
                    )
                    if left_sample is not None and right_sample is not None:
                        left_angle = math.radians(
                            args.left_imu_direction
                            * wrap_deg(left_sample.angle_deg - left_zero.angle_deg)
                        )
                        right_angle = math.radians(
                            args.right_imu_direction
                            * wrap_deg(right_sample.angle_deg - right_zero.angle_deg)
                        )
                        left_velocity = math.radians(
                            args.left_imu_direction
                            * (left_sample.gyro_dps - left_zero.gyro_dps)
                        )
                        right_velocity = math.radians(
                            args.right_imu_direction
                            * (right_sample.gyro_dps - right_zero.gyro_dps)
                        )
                        obs = imu_filter.update(
                            (
                                left_angle,
                                left_velocity,
                                right_angle,
                                right_velocity,
                            )
                        )
                        last_good_resample_wall = tick_wall
                        fresh_resample = True
                        hard_timeout_active = False
                    elif math.isfinite(obs[0]):
                        use_short_hold = True
                        soft_hold_count += 1

                scale, changed = keyboard.snapshot()
                if changed and policy is not None:
                    print(f"\n[SCALE] {scale:.2f}")

                if policy is not None:
                    if hard_timeout:
                        if not hard_timeout_active:
                            policy.reset()
                            torque_filter.reset()
                            hard_timeout_count += 1
                            hard_timeout_active = True
                        obs = (math.nan, math.nan, math.nan, math.nan)
                        left_cmd = 0.0
                        right_cmd = 0.0
                        command_ready = False
                    elif fresh_resample or use_short_hold:
                        # This call is mandatory on every usable 100-Hz tick.
                        # For scripted policies, append_frame alone does not
                        # advance the package's internal dense history.
                        policy.append_frame(*obs)
                        output = policy.infer()
                        if output is not None:
                            _, raw_nm = output
                            raw_left_tcn_nm = float(raw_nm[0])
                            raw_right_tcn_nm = float(raw_nm[1])

                        miss_age = (
                            0.0
                            if fresh_resample
                            else tick_wall - last_good_resample_wall
                        )
                        if output is not None and miss_age <= short_hold_s:
                            left_cmd, right_cmd = torque_filter.update(
                                scale * raw_left_tcn_nm,
                                scale * raw_right_tcn_nm,
                            )
                            command_ready = True
                        elif command_ready:
                            # Keep executing the stateful policy above, but
                            # ignore its stale-input command after the grace
                            # period and smoothly drive the hardware layer down.
                            left_cmd, right_cmd = torque_filter.update(0.0, 0.0)
                    else:
                        left_cmd = 0.0
                        right_cmd = 0.0
                        command_ready = False

                feedback = teensy.snapshot() if teensy is not None else None
                if feedback is None:
                    left_actual = math.nan
                    right_actual = math.nan
                    teensy_seq = -1
                else:
                    left_actual = feedback.left_actual
                    right_actual = feedback.right_actual
                    teensy_seq = feedback.seq
                    last_feedback_seq = feedback.seq

                enabled = bool(
                    args.arm
                    and policy is not None
                    and not hard_timeout
                    and command_ready
                )
                if teensy is not None:
                    teensy.send_torque(
                        left_cmd if enabled else 0.0,
                        right_cmd if enabled else 0.0,
                        enabled,
                    )

                writer.writerow(
                    [
                        f"{elapsed:.6f}",
                        f"{obs[0]:.9f}" if math.isfinite(obs[0]) else "nan",
                        f"{obs[1]:.9f}" if math.isfinite(obs[1]) else "nan",
                        f"{obs[2]:.9f}" if math.isfinite(obs[2]) else "nan",
                        f"{obs[3]:.9f}" if math.isfinite(obs[3]) else "nan",
                        (
                            f"{raw_left_tcn_nm:.6f}"
                            if math.isfinite(raw_left_tcn_nm)
                            else "nan"
                        ),
                        (
                            f"{raw_right_tcn_nm:.6f}"
                            if math.isfinite(raw_right_tcn_nm)
                            else "nan"
                        ),
                        f"{left_cmd:.6f}",
                        f"{right_cmd:.6f}",
                        (
                            f"{left_actual:.6f}"
                            if math.isfinite(left_actual)
                            else "nan"
                        ),
                        (
                            f"{right_actual:.6f}"
                            if math.isfinite(right_actual)
                            else "nan"
                        ),
                        teensy_seq,
                    ]
                )

                if tick_wall >= next_print:
                    if hard_timeout:
                        imu_state = "TIMEOUT"
                    elif fresh_resample:
                        imu_state = "OK"
                    elif use_short_hold:
                        imu_state = "HOLD"
                    else:
                        imu_state = "WAIT"
                    if policy is None:
                        print(
                            f"\rt={elapsed:7.2f}s imu={imu_state:7s} "
                            f"L={obs[0]:+.3f}/{obs[1]:+.3f} "
                            f"R={obs[2]:+.3f}/{obs[3]:+.3f}",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\rt={elapsed:7.2f}s imu={imu_state:7s} "
                            f"cmd=({left_cmd:+5.2f},{right_cmd:+5.2f}) "
                            f"actual=({left_actual:+5.2f},{right_actual:+5.2f}) "
                            f"seq={teensy_seq:5d} scale={scale:.1f} "
                            f"arm={int(enabled)}",
                            end="",
                            flush=True,
                        )
                    next_print += print_period

                after = time.perf_counter()
                if after - next_tick > period:
                    skipped = int((after - next_tick) / period) + 1
                    next_tick += skipped * period

    except KeyboardInterrupt:
        print("\nCtrl+C")
    finally:
        stop_event.set()
        if teensy is not None:
            try:
                for _ in range(3):
                    teensy.send_torque(0.0, 0.0, False)
                    time.sleep(0.01)
                teensy.stop_motors()
            except Exception:
                pass
        left.join(timeout=1.0)
        right.join(timeout=1.0)
        if teensy is not None:
            teensy.join(timeout=1.0)
        keyboard.join(timeout=0.2)
        print()
        print("=" * 96)
        print(f"CSV                : {csv_path}")
        print(f"Short IMU holds    : {soft_hold_count}")
        print(f"Hard IMU timeouts  : {hard_timeout_count}")
        print(f"LEFT error         : {left.error or 'none'}")
        print(f"RIGHT error        : {right.error or 'none'}")
        if teensy is not None:
            print(f"Teensy error       : {teensy.error or 'none'}")
            print(f"Last Teensy RX seq : {last_feedback_seq}")
        if policy is not None and policy.last_error:
            print(f"TCN last error     : {policy.last_error}")
        print("=" * 96)


if __name__ == "__main__":
    main()
