"""
PC neural-network controller for bilateral hip exoskeleton
==========================================================

Architecture
------------
Thread 1:
    LEFT IM948 on COM8, independent serial reader

Thread 2:
    RIGHT IM948 on COM6, independent serial reader

Thread 3:
    Teensy on COM7, independent torque-feedback reader

Main thread:
    100 Hz
    - get latest zeroed X angle / X gyro from both IMUs
    - get latest actual motor torque from Teensy
    - convert deg / deg/s to rad / rad/s for the NN
    - run NN inference
    - send torque command to Teensy
    - write formal CSV

Optional isolated subprocess:
    - matplotlib realtime plot
    - default redraw 30 Hz
    - 10 s rolling window

Important units
---------------
CSV / plot:
    angle          = deg
    angular speed  = deg/s
    torque         = Nm

Neural-network input:
    angle          = rad
    angular speed  = rad/s
    torque         = Nm

This preserves the unit convention used by the original packaged NN models.

Formal CSV
----------
elapsed_s
left_angle_x_deg
left_angular_velocity_x_dps
right_angle_x_deg
right_angular_velocity_x_dps
left_actual_torque_nm
right_actual_torque_nm
left_nn_command_nm
right_nn_command_nm

All CSV numeric values are written with exactly 4 decimal places.

Safety
------
Without --arm:
    NN still runs and is plotted/printed, but Teensy receives zero torque.

With --arm:
    torque is sent only while both IMUs and Teensy feedback are fresh and the
    NN output is valid.

Examples
--------
Dry-run with terminal:
    python NN_PC_Controller.py --display print

Dry-run with realtime plot:
    python NN_PC_Controller.py --display plot

Select any packaged checkpoint (MLP, GRU, or MoE):
    python NN_PC_Controller.py --model models/slope_adam_lowtorque/uphill_direct_100hz.pt

Real torque:
    python NN_PC_Controller.py --display plot --arm

Dependencies
------------
    python -m pip install pyserial matplotlib numpy torch
"""

from __future__ import annotations

import argparse
import csv
import math
import queue
import struct
import subprocess
import sys
import threading
import time
from collections import deque

try:
    import ctypes  # Windows global keyboard state (works even when plot has focus)
except ImportError:
    ctypes = None

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Optional, Tuple

import numpy as np
import serial
import torch  
from torch import nn   

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

# Final command = clamp(nominal NN torque * scale, +/-max_torque).
DEFAULT_TORQUE_SCALE = 0.1
DEFAULT_TORQUE_SCALE_STEP = 0.1
MIN_TORQUE_SCALE = 0.1
MAX_TORQUE_SCALE = 1.0

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
    Unified zeroed X-angle coordinate used by NN, plot, and CSV.

    Pipeline:
        IMU raw Euler-X
        -> subtract startup standing zero
        -> wrap to [-180, 180)
        -> apply side direction sign

    The returned value is the ONLY angle allowed to enter:
        1) neural-network input (after rad conversion)
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



Observation = Tuple[float, float, float, float, float, float]
TorqueCommand = Tuple[float, float]   

# ============================================================
# 6. Neural-network policy
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_FILES = {
    "direct": SCRIPT_DIR / "models" / "flat22" / "direct_100hz.pt",
    "pd": SCRIPT_DIR / "models" / "flat22" / "target_pd_100hz.pt",
}


class PolicyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)  


class _FlatMLPTorqueInterface:
    """Stateful 100-Hz inference for either packaged Exo policy."""

    def __init__(
        self,
        policy_type: str,
        model_path: Optional[Path] = None,
    ) -> None:
        self.policy_type = policy_type
        self.model_path = model_path or DEFAULT_MODEL_FILES[policy_type]
        self.model: Optional[PolicyMLP] = None
        self.load_message = ""
        self.last_error = ""
        self.calls = 0
        self.valid_outputs = 0
        self.history: deque[np.ndarray] = deque()
        self.previous_nm = np.zeros(2, dtype=np.float32)
        self._load()

    @property
    def available(self) -> bool:
        return self.model is not None

    def _load(self) -> None:
        try:
            payload = torch.load(
                self.model_path, map_location="cpu", weights_only=False
            )
            self.history_steps = int(payload["history_steps"])
            self.input_mean = np.asarray(
                payload["input_mean"], dtype=np.float32
            )
            self.input_std = np.asarray(
                payload["input_std"], dtype=np.float32
            )
            self.torque_scale_nm = float(payload.get("torque_scale_nm", 10.0))
            self.max_delta_nm = float(payload["max_delta_nm_per_step"])
            output_dim = 2 if self.policy_type == "direct" else 1
            self.model = PolicyMLP(
                len(self.input_mean), int(payload["hidden_dim"]), output_dim
            )
            self.model.load_state_dict(payload["state_dict"])
            self.model.eval()
            if self.policy_type == "pd":
                self.kp = float(payload["kp_nm_per_rad"])
                self.kd = float(payload["kd_nm_s_per_rad"])
                self.offset_limit = float(payload["target_offset_limit_rad"])
                self.torque_limit_nm = float(payload["torque_limit_nm"])
        except Exception as exc:
            self.load_message = (
                f"Failed to load {self.policy_type} model "
                f"'{self.model_path}': {exc}. "
                "The controller will output zero torque."
            )
            self.model = None
            return
        self.load_message = (
            f"Loaded {self.policy_type} Exo policy: {self.model_path}"
        )

    def reset(self) -> None:
        self.history.clear()
        self.previous_nm.fill(0.0)

    def zero_state_test(self, steps: int = 40) -> Optional[TorqueCommand]:
        """
        Evaluate the packaged policy at an exact stationary zero state.

        The test deliberately runs multiple 100-Hz-equivalent inference steps
        because the packaged policy contains its own per-step torque limiter.
        The policy state and diagnostic counters are restored afterward.
        """
        old_calls = self.calls
        old_valid_outputs = self.valid_outputs
        old_error = self.last_error

        self.reset()  
        result: Optional[TorqueCommand] = None   

        for _ in range(max(int(steps), self.history_steps, 1)):
            result = self.get_torque((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            if result is None:
                break

        self.reset()
        self.calls = old_calls
        self.valid_outputs = old_valid_outputs
        test_error = self.last_error
        self.last_error = old_error

        if result is None and test_error:
            print(f"[NN ZERO TEST ERROR] {test_error}")

        return result

    def _append_history(self, frame: np.ndarray) -> np.ndarray:
        if not self.history:
            for _ in range(self.history_steps):
                self.history.append(frame.copy())
        else:
            self.history.append(frame.copy())
            while len(self.history) > self.history_steps:
                self.history.popleft()
        return np.stack(self.history)

    def _limit(self, desired_nm: np.ndarray) -> np.ndarray:
        applied = np.clip(
            desired_nm,
            self.previous_nm - self.max_delta_nm,
            self.previous_nm + self.max_delta_nm,
        )
        self.previous_nm = np.clip(applied, -10.0, 10.0).astype(np.float32)
        return self.previous_nm.copy()

    def _infer_right_left(self, observation: Observation) -> np.ndarray:
        (
            left_actual_nm,
            right_actual_nm,
            left_angle,
            left_velocity,
            right_angle,
            right_velocity,
        ) = observation
        
        hip4 = np.asarray(
            [right_angle, left_angle, right_velocity, left_velocity],
            dtype=np.float32,
        )
        # observation is already in rad and rad/s. The main loop converts the
        # zeroed IMU readings with math.radians() before calling get_torque().

        if self.policy_type == "direct":
            history = self._append_history(hip4)
            features = history.reshape(-1)
            normalized = torch.from_numpy(
                (features - self.input_mean) / self.input_std
            )[None]
            with torch.inference_mode(): 
                desired_nm = (
                    self.torque_scale_nm * self.model(normalized)[0].numpy()
                )  
            return self._limit(desired_nm)

        actual_normalized = np.asarray(
            [right_actual_nm, left_actual_nm], dtype=np.float32
        ) / self.torque_scale_nm
        history = self._append_history(
            np.concatenate((hip4, actual_normalized))
        )
        right_features = history.reshape(-1)
        left_features = history[:, [1, 0, 3, 2, 5, 4]].reshape(-1)  
        features = np.stack((right_features, left_features))  
        normalized = torch.from_numpy(
            (features - self.input_mean) / self.input_std
        )
        with torch.inference_mode():
            offset = self.offset_limit * self.model(normalized)[:, 0].numpy()
        desired_nm = np.clip(
            self.kp * offset + self.kd * hip4[2:],
            -self.torque_limit_nm,
            self.torque_limit_nm,
        )
        return self._limit(desired_nm)

    def get_torque(
        self,
        observation: Observation,
    ) -> Optional[TorqueCommand]:
        """
        Return a valid (left_nm, right_nm), or None.

        None means:
            no neural-network torque is currently available.
        """
        if self.model is None:
            return None

        self.calls += 1

        try:
            right_left = self._infer_right_left(observation)
            result = (float(right_left[1]), float(right_left[0]))
        except Exception as exc:
            self.last_error = (
                f"{type(exc).__name__}: {exc}"
            )   
            return None 

        if result is None:
            self.last_error = ""
            return None

        try:
            if len(result) != 2:
                raise ValueError(
                    "Output must contain exactly two torque values."
                )

            left = float(result[0])  
            right = float(result[1])  

            if not math.isfinite(left) or not math.isfinite(right):
                raise ValueError("Output contains NaN or Inf.")

        except Exception as exc:
            self.last_error = (
                f"Invalid neural output: {type(exc).__name__}: {exc}"
            )
            return None

        self.last_error = ""
        self.valid_outputs += 1
        return left, right


class RecurrentExoNetwork(nn.Module):
    """Causal GRU backend shared by slope experts and learned-gate MoE."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        expert_count: int,
        max_delta: float,
        output_mode: str,
    ) -> None:
        super().__init__()
        self.expert_count = int(expert_count)
        self.max_delta = float(max_delta)
        self.output_mode = str(output_mode)
        self.gru = nn.GRU(int(input_dim), int(hidden_dim), batch_first=True)
        self.expert_head = nn.Linear(int(hidden_dim), self.expert_count * 2)
        self.gate_head = (
            nn.Linear(int(hidden_dim), self.expert_count)
            if self.expert_count > 1
            else None
        )

    def step(
        self,
        normalized_input: torch.Tensor,
        previous_exo: torch.Tensor,
        hidden: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        features, hidden = self.gru(normalized_input[:, None], hidden)
        features = features[:, 0]
        raw = self.expert_head(features).reshape(-1, self.expert_count, 2)

        if self.output_mode == "delta":
            delta = self.max_delta * torch.tanh(raw)
        elif self.output_mode == "absolute_slew":
            target = torch.tanh(raw)
            delta = torch.clamp(
                target - previous_exo[:, None],
                -self.max_delta,
                self.max_delta,
            )
        else:
            raise ValueError(
                f"Unsupported recurrent output mode: {self.output_mode}"
            )

        expert_action = torch.clamp(
            previous_exo[:, None] + delta,
            -1.0,
            1.0,
        )
        if self.gate_head is None:
            return expert_action[:, 0], hidden, None

        gate_logits = self.gate_head(features)
        weights = torch.softmax(gate_logits, dim=-1)
        action = torch.sum(weights[:, :, None] * expert_action, dim=1)
        return action, hidden, gate_logits


class _RecurrentTorqueInterface:
    """Runtime adapter for recurrent Direct, target-PD, and learned MoE."""

    def __init__(self, policy_type: str, model_path: Path) -> None:
        self.policy_type = policy_type
        self.model_path = model_path
        self.model: Optional[RecurrentExoNetwork] = None
        self.backend = "gru_moe"
        self.model_type = ""
        self.requires_torque_feedback = False
        self.load_message = ""
        self.last_error = ""
        self.calls = 0
        self.valid_outputs = 0
        self.history_steps = 1
        self.previous_normalized = torch.zeros((1, 2), dtype=torch.float32)
        self.hidden: Optional[torch.Tensor] = None
        self.last_gate = np.ones(1, dtype=np.float32)
        self._load()

    @property
    def available(self) -> bool:
        return self.model is not None

    def _load(self) -> None:
        try:
            payload = torch.load(
                self.model_path, map_location="cpu", weights_only=False
            )
            self.model_type = str(payload.get("model_type", ""))
            expected = (
                "recurrent_exo_target_pd"
                if self.policy_type == "pd"
                else "recurrent_exo"
            )
            if self.model_type != expected:
                raise ValueError(
                    f"Expected {expected}, found {self.model_type or 'unknown'}"
                )
            if float(payload.get("control_hz", -1.0)) != 100.0:
                raise ValueError("Checkpoint is not explicitly marked as 100 Hz")
            if str(payload.get("exo_sensor_mode", "")) != "hip4_exo6":
                raise ValueError("Checkpoint must use hip4_exo6 input")
            if int(payload["proprio_dim"]) != 6:
                raise ValueError("Checkpoint must have proprio_dim=6")

            self.history_steps = int(payload.get("history_steps", 1))
            self.torque_scale_nm = float(payload.get("torque_scale_nm", 10.0))
            self.mean = torch.as_tensor(
                payload["proprio_mean"], dtype=torch.float32
            )[None]
            self.std = torch.as_tensor(
                payload["proprio_std"], dtype=torch.float32
            )[None]
            self.expert_count = int(payload["expert_count"])
            model = RecurrentExoNetwork(
                input_dim=int(payload["proprio_dim"]),
                hidden_dim=int(payload["hidden_dim"]),
                expert_count=self.expert_count,
                max_delta=float(payload["max_delta"]),
                output_mode=str(payload.get("output_mode", "delta")),
            )
            model.load_state_dict(
                payload["proprio_exo_state_dict"], strict=True
            )
            model.eval()
            self.model = model
            self.last_gate = np.full(
                self.expert_count,
                1.0 / self.expert_count,
                dtype=np.float32,
            )
            if self.policy_type == "pd":
                self.kp = float(payload["kp_nm_per_rad"])
                self.kd = float(payload["kd_nm_s_per_rad"])
                self.offset_limit = float(payload["target_offset_limit_rad"])
                self.torque_limit_nm = float(payload["torque_limit_nm"])
        except Exception as exc:
            self.load_message = (
                f"Failed to load {self.policy_type} recurrent model "
                f"'{self.model_path}': {exc}. "
                "The controller will output zero torque."
            )
            self.model = None
            return

        self.load_message = (
            f"Loaded {self.policy_type} recurrent Exo policy: "
            f"{self.model_path}"
        )

    def reset(self) -> None:
        # The training data used the previous nominal Exo command, not measured
        # motor torque. Runtime assistance scaling is therefore applied after
        # get_torque(), while this unscaled recurrence stays inside the model.
        self.previous_normalized.zero_()
        self.hidden = None
        if self.last_gate.size:
            self.last_gate.fill(1.0 / self.last_gate.size)

    def zero_state_test(self, steps: int = 40) -> Optional[TorqueCommand]:
        old_calls = self.calls
        old_valid_outputs = self.valid_outputs
        old_error = self.last_error
        self.reset()
        result: Optional[TorqueCommand] = None
        for _ in range(max(int(steps), self.history_steps, 1)):
            result = self.get_torque((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            if result is None:
                break
        self.reset()
        self.calls = old_calls
        self.valid_outputs = old_valid_outputs
        test_error = self.last_error
        self.last_error = old_error
        if result is None and test_error:
            print(f"[NN ZERO TEST ERROR] {test_error}")
        return result

    def get_torque(self, observation: Observation) -> Optional[TorqueCommand]:
        if self.model is None:
            return None
        self.calls += 1
        try:
            (
                _left_actual_nm,
                _right_actual_nm,
                left_angle,
                left_velocity,
                right_angle,
                right_velocity,
            ) = observation
            hip4 = np.asarray(
                [right_angle, left_angle, right_velocity, left_velocity],
                dtype=np.float32,
            )
            current = torch.cat(
                (torch.from_numpy(hip4)[None], self.previous_normalized),
                dim=1,
            )
            normalized = (current - self.mean) / self.std
            with torch.inference_mode():
                action, self.hidden, gate_logits = self.model.step(
                    normalized,
                    self.previous_normalized,
                    self.hidden,
                )
            self.previous_normalized = action.detach()
            if gate_logits is not None:
                self.last_gate = torch.softmax(
                    gate_logits[0], dim=-1
                ).numpy()
            right_left = action[0].numpy() * self.torque_scale_nm

            if self.policy_type == "pd":
                offset = (right_left - self.kd * hip4[2:]) / self.kp
                if np.any(np.abs(offset) > self.offset_limit + 1.0e-6):
                    raise RuntimeError("Requested recurrent PD offset exceeds limit")
                right_left = np.clip(
                    self.kp * offset + self.kd * hip4[2:],
                    -self.torque_limit_nm,
                    self.torque_limit_nm,
                )
            result = (float(right_left[1]), float(right_left[0]))
            if not all(math.isfinite(value) for value in result):
                raise ValueError("Output contains NaN or Inf")
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

        self.last_error = ""
        self.valid_outputs += 1
        return result


class NeuralTorqueInterface:
    """Auto-detecting facade for all packaged 100-Hz Exo checkpoints."""

    def __init__(
        self,
        policy_type: str = "auto",
        model_path: Optional[Path] = None,
    ) -> None:
        if policy_type not in {"auto", "direct", "pd"}:
            raise ValueError("policy_type must be auto, direct, or pd")
        requested = policy_type
        default_policy = "direct" if requested == "auto" else requested
        path = Path(model_path or DEFAULT_MODEL_FILES[default_policy])
        self._delegate: Optional[object] = None
        self.model_path = path
        self.load_message = ""
        self.model_type = ""
        self.backend = ""
        self.policy_type = default_policy
        self.requires_torque_feedback = False

        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.model_type = str(payload.get("model_type", ""))
            if self.model_type == "recurrent_exo":
                inferred = "direct"
                interface_type = _RecurrentTorqueInterface
            elif self.model_type == "recurrent_exo_target_pd":
                inferred = "pd"
                interface_type = _RecurrentTorqueInterface
            elif (
                self.model_type == "shared_leg_target_position_pd"
                or "kp_nm_per_rad" in payload
            ):
                inferred = "pd"
                interface_type = _FlatMLPTorqueInterface
            else:
                inferred = "direct"
                interface_type = _FlatMLPTorqueInterface

            if requested != "auto" and requested != inferred:
                raise ValueError(
                    f"Checkpoint is {inferred}, but --policy {requested} "
                    "was requested"
                )
            self.policy_type = inferred
            delegate = interface_type(inferred, path)
            if not delegate.available:
                raise ValueError(delegate.load_message)
            self._delegate = delegate
            self.backend = delegate.backend if hasattr(delegate, "backend") else "mlp_history"
            self.requires_torque_feedback = bool(
                getattr(delegate, "requires_torque_feedback", inferred == "pd")
            )
            self.load_message = delegate.load_message
        except Exception as exc:
            self.load_message = (
                f"Failed to load model '{path}': {exc}. "
                "The controller will output zero torque."
            )

    @property
    def available(self) -> bool:
        return self._delegate is not None

    @property
    def model(self) -> Optional[nn.Module]:
        return None if self._delegate is None else self._delegate.model

    @property
    def calls(self) -> int:
        return 0 if self._delegate is None else self._delegate.calls

    @property
    def valid_outputs(self) -> int:
        return 0 if self._delegate is None else self._delegate.valid_outputs

    @property
    def last_error(self) -> str:
        return "" if self._delegate is None else self._delegate.last_error

    @property
    def history_steps(self) -> int:
        return 1 if self._delegate is None else self._delegate.history_steps

    @property
    def last_gate(self) -> np.ndarray:
        return np.ones(1, dtype=np.float32) if self._delegate is None else getattr(
            self._delegate, "last_gate", np.ones(1, dtype=np.float32)
        )

    def reset(self) -> None:
        if self._delegate is not None:
            self._delegate.reset()

    def zero_state_test(self, steps: int = 40) -> Optional[TorqueCommand]:
        if self._delegate is None:
            return None
        return self._delegate.zero_state_test(steps)

    def get_torque(self, observation: Observation) -> Optional[TorqueCommand]:
        if self._delegate is None:
            return None
        return self._delegate.get_torque(observation)



# =============================================================================
# Realtime plot subprocess client
# =============================================================================

PLOT_HELPER_PATH = Path(__file__).resolve().with_name(
    "pc_nn_plot_worker.py"
)


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


class PlotSubprocess:
    """
    Realtime plotting isolated in a completely separate Python program.

    Why:
        Windows multiprocessing uses spawn, which re-imports this controller
        module in the child process. Since this module imports torch, that can
        initialize Intel OpenMP a second time and trigger OMP Error #15.

    This class launches pc_nn_plot_worker.py with subprocess.Popen instead.
    The helper imports matplotlib but never imports torch or this controller.

    Control-loop safety:
        - main control thread never writes to the pipe directly
        - samples enter a bounded queue with put_nowait()
        - a dedicated feeder thread writes text lines to the child stdin
        - if plotting falls behind, display samples are dropped
        - NN/control/TX/CSV remain independent
    """

    def __init__(
        self,
        *,
        refresh_hz: float,
        window_s: float,
        stale_warning_s: float,
        imu_timeout_s: float,
        teensy_timeout_s: float,
    ) -> None:
        self.refresh_hz = float(refresh_hz)
        self.window_s = float(window_s)
        self.stale_warning_s = float(stale_warning_s)
        self.imu_timeout_s = float(imu_timeout_s)
        self.teensy_timeout_s = float(teensy_timeout_s)

        self.queue: queue.Queue = queue.Queue(maxsize=400)
        self.process: Optional[subprocess.Popen] = None
        self.feeder_thread: Optional[threading.Thread] = None
        self.feeder_error: Optional[str] = None

    @property
    def closed(self) -> bool:
        return (
            self.process is not None
            and self.process.poll() is not None
        )

    def start(self) -> None:
        if not PLOT_HELPER_PATH.exists():
            raise FileNotFoundError(
                "Realtime plot helper not found: "
                f"{PLOT_HELPER_PATH}. "
                "Keep pc_nn_plot_worker.py beside this controller."
            )

        command = [
            sys.executable,
            str(PLOT_HELPER_PATH),
            "--refresh-hz",
            str(self.refresh_hz),
            "--window-s",
            str(self.window_s),
            "--stale-warning-s",
            str(self.stale_warning_s),
            "--imu-timeout-s",
            str(self.imu_timeout_s),
            "--teensy-timeout-s",
            str(self.teensy_timeout_s),
        ]

        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self.feeder_thread = threading.Thread(
            target=self._feeder_loop,
            name="PlotPipeFeeder",
            daemon=True,
        )
        self.feeder_thread.start()

    def _feeder_loop(self) -> None:
        try:
            while True:
                item = self.queue.get()

                if item is None:
                    break

                process = self.process
                if process is None or process.poll() is not None:
                    break

                if process.stdin is None:
                    break

                # Tab-separated lightweight streaming protocol:
                # elapsed, Langle, Rangle, Lcmd, Rcmd, Lact, Ract,
                # Lage, Rage, Tage, control_ok, enabled
                line = "\t".join(
                    (
                        f"{float(item[0]):.9f}",
                        f"{float(item[1]):.9f}",
                        f"{float(item[2]):.9f}",
                        f"{float(item[3]):.9f}",
                        f"{float(item[4]):.9f}",
                        f"{float(item[5]):.9f}",
                        f"{float(item[6]):.9f}",
                        f"{float(item[7]):.9f}",
                        f"{float(item[8]):.9f}",
                        f"{float(item[9]):.9f}",
                        "1" if bool(item[10]) else "0",
                        "1" if bool(item[11]) else "0",
                    )
                )

                process.stdin.write(line + "\n")
                process.stdin.flush()

        except (BrokenPipeError, OSError) as exc:
            # Normal when the user closes the plot window.
            self.feeder_error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            self.feeder_error = f"{type(exc).__name__}: {exc}"

    def push(self, sample) -> None:
        """
        Non-blocking from the 100-Hz main control loop.
        Drop the oldest display sample if the plot queue is full.
        """
        if self.closed:
            return

        try:
            self.queue.put_nowait(sample)
            return
        except queue.Full:
            pass

        try:
            self.queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self.queue.put_nowait(sample)
        except queue.Full:
            pass

    def stop(self) -> None:
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                pass

        if (
            self.feeder_thread is not None
            and self.feeder_thread.is_alive()
        ):
            self.feeder_thread.join(timeout=1.0)

        process = self.process
        if process is None:
            return

        try:
            if process.stdin is not None:
                process.stdin.close()
        except Exception:
            pass

        if process.poll() is None:
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)


# =============================================================================
# Runtime keyboard control
# =============================================================================

# Windows virtual-key codes. GetAsyncKeyState() is used instead of msvcrt so
# Up/Down still work when the matplotlib plot window has keyboard focus.
VK_UP = 0x26
VK_DOWN = 0x28

_prev_up_pressed = False
_prev_down_pressed = False


def poll_torque_scale_keys(
    current_scale: float,
    step: float,
) -> Tuple[float, bool]:
    """
    Read Windows Up/Down keys globally without blocking the 100-Hz loop.

    A rising-edge detector is used so one physical key press changes the
    assistance scale exactly once, even though this function is polled at
    100 Hz. The plot window may have focus.

        Up   -> scale + step
        Down -> scale - step
    """
    global _prev_up_pressed, _prev_down_pressed

    if ctypes is None or sys.platform != "win32":
        return current_scale, False

    try:
        user32 = ctypes.windll.user32
        up_pressed = bool(user32.GetAsyncKeyState(VK_UP) & 0x8000)
        down_pressed = bool(user32.GetAsyncKeyState(VK_DOWN) & 0x8000)
    except Exception:
        # Keyboard control is optional; never let it disturb the control loop.
        return current_scale, False

    updated = float(current_scale)
    changed = False

    # Rising edge only. Holding a key does not repeatedly change scale.
    if up_pressed and not _prev_up_pressed:
        new_scale = min(MAX_TORQUE_SCALE, updated + step)
        new_scale = round(new_scale, 4)
        if abs(new_scale - updated) > 1.0e-12:
            updated = new_scale
            changed = True

    if down_pressed and not _prev_down_pressed:
        new_scale = max(MIN_TORQUE_SCALE, updated - step)
        new_scale = round(new_scale, 4)
        if abs(new_scale - updated) > 1.0e-12:
            updated = new_scale
            changed = True

    _prev_up_pressed = up_pressed
    _prev_down_pressed = down_pressed

    return updated, changed


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "100-Hz PC neural-network controller with independent dual IMUs "
            "and simplified 14-byte Teensy torque feedback"
        )
    )

    p.add_argument("--left-port", default=DEFAULT_LEFT_IMU_PORT)
    p.add_argument("--right-port", default=DEFAULT_RIGHT_IMU_PORT)
    p.add_argument("--teensy-port", default=DEFAULT_TEENSY_PORT)
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)

    p.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_CONTROL_HZ,
        help="NN/control/TX/CSV rate; packaged policies require 100 Hz",
    )

    p.add_argument(
        "--display",
        choices=("print", "plot"),
        default="print",
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

    p.add_argument(
        "--zero-settle",
        type=float,
        default=3.0,
        help=(
            "Seconds to let the IMU attitude solution settle before zero "
            "calibration (default: 3.0)"
        ),
    )
    p.add_argument("--zero-samples", type=int, default=200)
    p.add_argument("--zero-timeout", type=float, default=10.0)
    p.add_argument("--skip-zero", action="store_true")
    p.add_argument("--no-configure-imu", action="store_true")

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
        "--policy",
        choices=("auto", "direct", "pd"),
        default="auto",
        help=(
            "Checkpoint interface validation; auto infers it from --model "
            "(default: auto, or flat Direct when --model is omitted)"
        ),
    )
    p.add_argument(
        "--model",
        type=Path,
        default=None,
    )

    p.add_argument(
        "--torque-scale",
        type=float,
        default=DEFAULT_TORQUE_SCALE,
        help=(
            "Initial nominal NN torque scale in [0.1, 1.0]; "
            "Windows Up/Down changes it at runtime"
        ),
    )
    p.add_argument(
        "--torque-scale-step",
        type=float,
        default=DEFAULT_TORQUE_SCALE_STEP,
        help="Runtime Up/Down torque-scale step (default: 0.1)",
    )

    # Current Teensy rejects inputs beyond +/-5 Nm, so PC default is matched.
    p.add_argument(
        "--max-torque",
        type=float,
        default=8.0,
        help="PC NN torque clamp in Nm; current Teensy input limit is +/-5 Nm",
    )
    p.add_argument(
        "--arm",
        action="store_true",
        help="Allow valid NN torque to be sent. Default is dry-run zero torque.",
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

    if abs(a.rate - 100.0) > 1.0e-6:
        raise ValueError(
            "The packaged neural policies require --rate 100."
        )

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

    if a.zero_settle < 0:
        raise ValueError("--zero-settle must be >= 0")
    if a.zero_samples <= 0:
        raise ValueError("--zero-samples must be positive")
    if a.zero_timeout <= 0:
        raise ValueError("--zero-timeout must be positive")

    if not (MIN_TORQUE_SCALE <= a.torque_scale <= MAX_TORQUE_SCALE):
        raise ValueError(
            f"--torque-scale must be in "
            f"[{MIN_TORQUE_SCALE:.1f}, {MAX_TORQUE_SCALE:.1f}]"
        )
    if a.torque_scale_step <= 0:
        raise ValueError("--torque-scale-step must be positive")

    if a.max_torque <= 0 or a.max_torque > 10.0:
        raise ValueError(
            "--max-torque must be in (0, 10] to match the current Teensy."
        )


def default_csv_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs") / f"pc_nn_formal_{stamp}.csv"


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

    torch.set_num_threads(1)

    policy = NeuralTorqueInterface(
        a.policy,
        a.model.expanduser().resolve() if a.model is not None else None,
    )  

    if policy.available:
        zero_test = policy.zero_state_test()
        if zero_test is not None:
            print(
                "[NN ZERO TEST] exact zero state -> "
                f"L={zero_test[0]:+.4f} Nm, "
                f"R={zero_test[1]:+.4f} Nm"
            )

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

    plotter = None

    if a.display == "plot":
        plotter = PlotSubprocess(
            refresh_hz=a.plot_rate,
            window_s=a.plot_window,
            stale_warning_s=a.stale_warning,
            imu_timeout_s=a.imu_timeout,
            teensy_timeout_s=a.teensy_timeout,
        )

    print("=" * 108)
    print("PC NN / dual independent IMU / simplified Teensy controller")
    print(f"LEFT IMU  : {a.left_port} @ {a.baud} | independent thread")
    print(f"RIGHT IMU : {a.right_port} @ {a.baud} | independent thread")
    print(f"Teensy    : {a.teensy_port} @ {a.baud} | independent thread")
    print("NN/control/TX/CSV : 100.0 Hz")
    print(
        f"Display   : {a.display.upper()} | "
        + (
            f"{a.print_rate:.1f} Hz"
            if a.display == "print"
            else f"{a.plot_rate:.1f} Hz redraw, {a.plot_window:.1f}s window"
        )
    )
    print(
        f"IMU age   : OK <= {a.stale_warning*1000:.0f} ms | "
        f"STALE <= {a.imu_timeout*1000:.0f} ms | "
        f"TIMEOUT > {a.imu_timeout*1000:.0f} ms"
    )
    print(
        f"Teensy FB : timeout > {a.teensy_timeout*1000:.0f} ms"
    )
    print(
        f"X signs   : L={a.left_angle_sign:+.0f}, "
        f"R={a.right_angle_sign:+.0f}"
    )
    print(
        "NN units  : angle rad, angular velocity rad/s; "
        "CSV/plot remain deg and deg/s"
    )
    print(
        f"Policy    : {policy.policy_type} | backend={policy.backend or 'none'} | "
        f"{'READY' if policy.available else 'MISSING'}"
    )
    print(policy.load_message)
    print(
        f"NN scale  : {a.torque_scale:.1f} "
        f"(range {MIN_TORQUE_SCALE:.1f}-{MAX_TORQUE_SCALE:.1f}, "
        f"step {a.torque_scale_step:.1f})"
    )
    print("Keys      : Up = scale +step | Down = scale -step | global Windows hotkeys")
    print(f"PC clamp  : +/-{a.max_torque:.3f} Nm")
    print(f"Output    : {'ARMED' if a.arm else 'DRY RUN - zero torque sent'}")
    print(f"CSV       : {csv_path}")
    print("Ctrl+C or closing plot -> zero torque + STOP x3")
    print("=" * 108)

    left_imu.start()
    right_imu.start()
    teensy.start()

    # Wait until all three data sources have at least one sample.
    startup_deadline = time.perf_counter() + 7.0

    while (
        time.perf_counter() < startup_deadline
        and not stop_event.is_set()
    ):
        left_sample, _ = left_imu.snapshot()
        right_sample, _ = right_imu.snapshot()
        feedback, _ = teensy.snapshot()

        if (
            left_sample is not None
            and right_sample is not None
            and feedback is not None
        ):
            break

        time.sleep(0.01)

    left_sample, _ = left_imu.snapshot()
    right_sample, _ = right_imu.snapshot()
    feedback, _ = teensy.snapshot()

    if (
        left_sample is None
        or right_sample is None
        or feedback is None
    ):
        stop_event.set()
        raise RuntimeError(
            "Startup data missing. "
            f"LEFT={left_imu.error or ('OK' if left_sample else 'no sample')}, "
            f"RIGHT={right_imu.error or ('OK' if right_sample else 'no sample')}, "
            f"Teensy={teensy.error or ('OK' if feedback else 'no feedback')}"
        )

    left_zero = ImuZeroOffset(0.0, 0.0)
    right_zero = ImuZeroOffset(0.0, 0.0)

    if not a.skip_zero and a.zero_settle > 0:
        print(
            f"[ZERO SETTLE] Keep still for {a.zero_settle:.1f} s "
            "while IMU attitude settles..."
        )
        settle_end = time.perf_counter() + a.zero_settle
        next_settle_print = 0.0

        while (
            time.perf_counter() < settle_end
            and not stop_event.is_set()
        ):
            now_settle = time.perf_counter()

            if now_settle >= next_settle_print:
                ls, _ = left_imu.snapshot()
                rs, _ = right_imu.snapshot()

                if ls is not None and rs is not None:
                    remaining = max(settle_end - now_settle, 0.0)
                    print(
                        f"\r[ZERO SETTLE] "
                        f"L raw X={ls.angle_x_deg:+8.3f} deg | "
                        f"R raw X={rs.angle_x_deg:+8.3f} deg | "
                        f"{remaining:4.1f}s remaining",
                        end="",
                        flush=True,
                    )

                next_settle_print = now_settle + 0.1

            time.sleep(0.002)

        print()

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

    if plotter is not None:
        plotter.start()

    period = 1.0 / a.rate
    print_period = 1.0 / a.print_rate

    start_time = time.perf_counter()
    next_tick = start_time
    next_print = start_time

    rows = 0
    lcmd = 0.0
    rcmd = 0.0
    enabled = False
    torque_scale = float(a.torque_scale)

    prev_control_ok = False
    last_nn_error_print = ""

    max_left_age_s = 0.0
    max_right_age_s = 0.0
    max_teensy_age_s = 0.0

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
                    "left_nn_command_nm",
                    "right_nn_command_nm",
                ]
            )

            while not stop_event.is_set():
                if plotter is not None and plotter.closed:
                    print("\nPlot window closed.")
                    break

                torque_scale, scale_changed = poll_torque_scale_keys(
                    torque_scale,
                    a.torque_scale_step,
                )
                if scale_changed:
                    print(f"\n[SCALE] NN torque scale -> {torque_scale:.2f}")

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
                    teensy_age_s = (
                        now - feedback.host_time
                        if feedback is not None
                        else math.inf
                    )

                    if math.isfinite(left_age_s):
                        max_left_age_s = max(max_left_age_s, left_age_s)
                    if math.isfinite(right_age_s):
                        max_right_age_s = max(max_right_age_s, right_age_s)
                    if math.isfinite(teensy_age_s):
                        max_teensy_age_s = max(
                            max_teensy_age_s,
                            teensy_age_s,
                        )

                    left_timeout = left_age_s > a.imu_timeout
                    right_timeout = right_age_s > a.imu_timeout
                    teensy_timeout = teensy_age_s > a.teensy_timeout

                    left_angle_deg = (
                        relative_x_deg(
                            left_sample.angle_x_deg,
                            left_zero.angle_x_deg,
                            a.left_angle_sign,
                        )
                        if left_sample is not None
                        else math.nan
                    )
                    right_angle_deg = (
                        relative_x_deg(
                            right_sample.angle_x_deg,
                            right_zero.angle_x_deg,
                            a.right_angle_sign,
                        )
                        if right_sample is not None
                        else math.nan
                    )

                    left_gyro_dps = (
                        relative_x_gyro_dps(
                            left_sample.gyro_x_dps,
                            left_zero.gyro_x_dps,
                            a.left_angle_sign,
                        )
                        if left_sample is not None
                        else math.nan
                    )
                    right_gyro_dps = (
                        relative_x_gyro_dps(
                            right_sample.gyro_x_dps,
                            right_zero.gyro_x_dps,
                            a.right_angle_sign,
                        )
                        if right_sample is not None
                        else math.nan
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

                    imu_inputs_finite = all(
                        math.isfinite(v)
                        for v in (
                            left_angle_deg,
                            right_angle_deg,
                            left_gyro_dps,
                            right_gyro_dps,
                        )
                    )

                    imu_ok = (
                        imu_inputs_finite
                        and not left_timeout
                        and not right_timeout
                    )

                    teensy_ok = (
                        feedback is not None
                        and math.isfinite(left_actual)
                        and math.isfinite(right_actual)
                        and not teensy_timeout
                    )

                    # Only the flat target-PD MLP uses measured motor torque as
                    # a network input. Recurrent checkpoints retain their own
                    # unscaled nominal command state.
                    nn_input_ok = (
                        imu_ok
                        and policy.available
                        and (not a.arm or teensy_ok)
                        and (
                            not policy.requires_torque_feedback
                            or teensy_ok
                        )
                    )

                    nn_valid = False

                    if nn_input_ok:
                        # For direct policy these first two values are ignored.
                        # Supplying zero if feedback is stale avoids propagating
                        # stale motor data into the observation tuple.
                        obs_left_actual = (
                            float(left_actual)
                            if teensy_ok
                            else 0.0
                        )
                        obs_right_actual = (
                            float(right_actual)
                            if teensy_ok
                            else 0.0
                        )

                        observation: Observation = (
                            obs_left_actual,
                            obs_right_actual,
                            math.radians(left_angle_deg),
                            math.radians(left_gyro_dps),
                            math.radians(right_angle_deg),
                            math.radians(right_gyro_dps),
                        )

                        neural_output = policy.get_torque(observation)

                        if neural_output is None:
                            lcmd = 0.0
                            rcmd = 0.0
                            policy.reset()
                        else:
                            scaled_left_nm = -neural_output[0] * torque_scale
                            scaled_right_nm = -neural_output[1] * torque_scale
                            lcmd = max(
                                -a.max_torque,
                                min(a.max_torque, scaled_left_nm),
                            )
                            rcmd = max(
                                -a.max_torque,
                                min(a.max_torque, scaled_right_nm),
                            )
                            nn_valid = True
                    else:
                        lcmd = 0.0
                        rcmd = 0.0

                        if prev_control_ok:
                            policy.reset()

                    # Keep the existing variable name for plotting/status:
                    # here it means "NN command is currently valid".
                    control_ok = nn_valid
                    prev_control_ok = nn_valid

                    # Safety remains strict: real torque transmission requires
                    # BOTH a valid NN command and fresh Teensy feedback.
                    enabled = bool(
                        a.arm
                        and nn_valid
                        and teensy_ok
                    )

                    teensy.send_torque(
                        lcmd if enabled else 0.0,
                        rcmd if enabled else 0.0,
                        enabled,
                    )

                    elapsed = now - start_time

                    # Strict 100-Hz main-tick logging. Latest samples are used.
                    writer.writerow(
                        [
                            f"{elapsed:.4f}",
                            (
                                f"{left_angle_deg:.4f}"
                                if math.isfinite(left_angle_deg)
                                else ""
                            ),
                            (
                                f"{left_gyro_dps:.4f}"
                                if math.isfinite(left_gyro_dps)
                                else ""
                            ),
                            (
                                f"{right_angle_deg:.4f}"
                                if math.isfinite(right_angle_deg)
                                else ""
                            ),
                            (
                                f"{right_gyro_dps:.4f}"
                                if math.isfinite(right_gyro_dps)
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
                            (
                                f"{lcmd:.4f}"
                                if math.isfinite(lcmd)
                                else ""
                            ),
                            (
                                f"{rcmd:.4f}"
                                if math.isfinite(rcmd)
                                else ""
                            ),
                        ]
                    )
                    rows += 1

                    if plotter is not None:
                        plotter.push(
                            (
                                elapsed,
                                left_angle_deg,
                                right_angle_deg,
                                lcmd,
                                rcmd,
                                left_actual,
                                right_actual,
                                left_age_s,
                                right_age_s,
                                teensy_age_s,
                                control_ok,
                                enabled,
                            )
                        )

                    next_tick += period

                    if now - next_tick > period:
                        next_tick = now + period

                if a.display == "print" and now >= next_print:
                    left_sample, left_stats = left_imu.snapshot()
                    right_sample, right_stats = right_imu.snapshot()
                    feedback, teensy_stats = teensy.snapshot()

                    now2 = time.perf_counter()

                    left_age = (
                        now2 - left_sample.host_time
                        if left_sample is not None
                        else math.inf
                    )
                    right_age = (
                        now2 - right_sample.host_time
                        if right_sample is not None
                        else math.inf
                    )
                    teensy_age = (
                        now2 - feedback.host_time
                        if feedback is not None
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

                    actual_text = (
                        f"{feedback.left_actual_nm:+.4f}/"
                        f"{feedback.right_actual_nm:+.4f}"
                        if feedback is not None
                        else "NO_FB"
                    )
                    teensy_state = (
                        "OK"
                        if teensy_age <= a.teensy_timeout
                        else "TIMEOUT"
                    )

                    print(
                        f"L {left_stats.hz:5.1f}Hz "
                        f"age={left_age*1000:6.1f}ms [{lstate}] | "
                        f"R {right_stats.hz:5.1f}Hz "
                        f"age={right_age*1000:6.1f}ms [{rstate}] | "
                        f"T {teensy_stats.hz:5.1f}Hz "
                        f"age={teensy_age*1000:6.1f}ms "
                        f"[{teensy_state}] | "
                        f"SCALE={torque_scale:.2f} | "
                        f"CMD={lcmd:+.4f}/{rcmd:+.4f} | "
                        f"ACT={actual_text} | "
                        f"NN={'OK' if control_ok else 'ZERO'} | "
                        f"{'ON' if enabled else 'OFF'}"
                    )

                    if (
                        policy.last_error
                        and policy.last_error != last_nn_error_print
                    ):
                        print("[NN ERROR]", policy.last_error)
                        last_nn_error_print = policy.last_error

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

        if plotter is not None:
            plotter.stop()

        left_imu.join(timeout=2.0)
        right_imu.join(timeout=2.0)
        teensy.join(timeout=2.0)

        _, left_stats = left_imu.snapshot()
        _, right_stats = right_imu.snapshot()
        _, teensy_stats = teensy.snapshot()

        duration = max(time.perf_counter() - start_time, 1e-9)
        csv_hz = rows / duration

        print("=" * 108)
        print(f"CSV saved     : {csv_path}")
        print(
            f"CSV rows/rate : {rows} / {csv_hz:.2f} Hz "
            f"(target 100.0 Hz)"
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
            f"L={left_stats.hz:.1f}, R={right_stats.hz:.1f}"
        )
        print(
            f"Teensy final  : "
            f"{teensy_stats.hz:.1f} Hz, "
            f"crc_errors={teensy_stats.crc_errors}"
        )
        print(
            "Max data age  : "
            f"L={max_left_age_s*1000:.1f} ms, "
            f"R={max_right_age_s*1000:.1f} ms, "
            f"T={max_teensy_age_s*1000:.1f} ms"
        )
        print(
            f"NN inference  : calls={policy.calls}, "
            f"valid={policy.valid_outputs}"
        )
        print(f"Final NN scale: {torque_scale:.2f}")

        if left_imu.error:
            print("LEFT IMU error :", left_imu.error)
        if right_imu.error:
            print("RIGHT IMU error:", right_imu.error)
        if teensy.error:
            print("Teensy error   :", teensy.error)
        if policy.last_error:
            print("NN last error  :", policy.last_error)

        print("=" * 108)  


if __name__ == "__main__":
    main()   