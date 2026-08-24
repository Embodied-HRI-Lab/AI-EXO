"""
100 Hz Unified causal-TCN controller with IMU and torque-output filtering.

This file is intentionally self-contained for policy inference: keep it beside
unified_tcn_latest_100hz_deploy.pt on the control PC. Hardware is opened only inside
main() and never at import time. The default first-order 10 Hz causal low-pass
filters angle and gyro before they enter the TCN. A first-order 5 Hz causal
low-pass then smooths the TCN torque command, followed by a 0.8 Nm/cycle slew
limit. Use --dry-run for no motor output.
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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    import serial
except ImportError:  # pragma: no cover - hardware dependency may be absent in dry lab hosts.
    serial = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - Windows-only keyboard helper.
    msvcrt = None


HEADER: Final[bytes] = b"\xA5\x5A"
DEFAULT_LEFT_IMU_PORT = "/dev/ttyUSB0"
DEFAULT_RIGHT_IMU_PORT = "/dev/ttyUSB1"
DEFAULT_TEENSY_PORT = "/dev/serial0"
DEFAULT_BAUD = 115200
DEFAULT_CONTROL_HZ = 100.0
DEFAULT_PRINT_HZ = 10.0
DEFAULT_IMU_CUTOFF_HZ = 10.0
DEFAULT_OUTPUT_CUTOFF_HZ = 5.0
DEFAULT_MAX_DELTA_NM_PER_STEP = 0.8
DEFAULT_STALE_WARNING_S = 0.050
DEFAULT_IMU_TIMEOUT_S = 0.150
DEFAULT_TEENSY_TIMEOUT_S = 0.200
DEFAULT_MODEL_PATH = Path(__file__).resolve().with_name("steady_unique_100hz_20260824_124007_100hz_deploy.pt")

CMD_TORQUE = 0x54
CMD_STOP = 0x50
CMD_CLEAR_FAULT = 0x43
CMD_STATE = 0x44
TORQUE_PAYLOAD = struct.Struct("<HffB")
STATE_PAYLOAD = struct.Struct("<Hff")
STATE_FRAME_SIZE = 14

IMU_FRAME_BEGIN = 0x49
IMU_FRAME_END = 0x4D
IMU_BROADCAST_ADDRESS = 0xFF
CMD_WAKE = 0x03
CMD_REPORT = 0x11
CMD_SET_PARAMS = 0x12
CMD_REPORT_OFF = 0x18
CMD_REPORT_ON = 0x19
REPORT_TAG = 0x0044
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


def get_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"unsupported activation: {name}")


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int, dilation: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, hidden_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(hidden_channels, hidden_channels, kernel_size, dilation)
        self.activation1 = get_activation(activation)
        self.activation2 = get_activation(activation)
        self.output_activation = get_activation(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.residual = nn.Identity() if in_channels == hidden_channels else nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        y = self.dropout1(self.activation1(self.conv1(x)))
        y = self.dropout2(self.activation2(self.conv2(y)))
        return self.output_activation(residual + y)


class CausalTCN(nn.Module):
    def __init__(
        self,
        input_channels: int = 4,
        hidden_channels: int = 32,
        output_channels: int = 2,
        kernel_size: int = 4,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
        activation: str = "silu",
        receptive_field_samples: int | None = None,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_ch = int(input_channels)
        for dilation in dilations:
            blocks.append(TCNResidualBlock(in_ch, int(hidden_channels), int(kernel_size), int(dilation), float(dropout), activation))
            in_ch = int(hidden_channels)
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(int(hidden_channels), int(output_channels))
        self.output_activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected x=[B,C,T], got {tuple(x.shape)}")
        y = x
        for block in self.blocks:
            y = block(y)
        return self.output_activation(self.head(y[:, :, -1]))


class UnifiedTCNPolicy:
    """Stateful raw Unified TCN policy: IMU history -> normalized command -> Nm command scale."""

    def __init__(self, model_path: Path, *, device: str = "cpu") -> None:
        self.model_path = model_path.expanduser().resolve()
        self.device = torch.device(device)
        self.payload = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model = CausalTCN(**dict(self.payload["model_config"])).to(self.device)
        self.model.load_state_dict(self.payload["state_dict"])
        self.model.eval()
        self.history_steps = int(self.payload["history_steps"])
        self.sensor_hz = int(self.payload["sensor_hz"])
        self.control_hz = int(self.payload.get("control_hz", self.sensor_hz))
        self.input_channel_names = list(self.payload["input_channel_names"])
        self.input_mean = torch.tensor(self.payload["input_mean"], dtype=torch.float32, device=self.device).view(1, 4, 1)
        std = torch.tensor(self.payload["input_std"], dtype=torch.float32, device=self.device).view(1, 4, 1)
        if torch.any(std <= 0):
            raise ValueError("input_std contains non-positive values")
        self.input_std = torch.clamp(std, min=1e-8)
        self.normalization_scheme = str(self.payload["normalization_scheme"])
        self.torque_scale_nm = float(self.payload["torque_scale_nm"])
        self.history: deque[np.ndarray] = deque(maxlen=self.history_steps)
        self.calls = 0
        self.valid_outputs = 0
        self.last_error = ""
        self.last_inference_time_ms = math.nan

    @property
    def history_ready(self) -> bool:
        return len(self.history) >= self.history_steps

    def reset(self) -> None:
        self.history.clear()
        self.last_inference_time_ms = math.nan

    def append_frame(self, left_angle_rad: float, left_velocity_rad_s: float, right_angle_rad: float, right_velocity_rad_s: float) -> None:
        frame = np.asarray([left_angle_rad, left_velocity_rad_s, right_angle_rad, right_velocity_rad_s], dtype=np.float32)
        self.history.append(frame)

    def infer(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.history_ready:
            return None
        self.calls += 1
        try:
            hist = np.stack(tuple(self.history), axis=0).astype(np.float32, copy=False)
            x = torch.from_numpy(hist.T[None]).to(self.device)
            x = (x - self.input_mean) / self.input_std
            start = time.perf_counter()
            with torch.inference_mode():
                action = self.model(x)[0].detach().cpu().numpy().astype(np.float32)
            self.last_inference_time_ms = (time.perf_counter() - start) * 1000.0
            action = np.clip(action, -1.0, 1.0)
            command_nm = action * np.float32(self.torque_scale_nm)
            command_nm = np.clip(command_nm, -self.torque_scale_nm, self.torque_scale_nm).astype(np.float32)
            self.valid_outputs += 1
            self.last_error = ""
            return action, command_nm
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def print_startup_summary(self) -> None:
        arch = self.payload.get("architecture", {})
        safety = self.payload.get("safety", {})
        print("=" * 108)
        print("MODEL: causal TCN")
        print(f"MODE: {self.payload.get('mode')}")
        print(f"CONTROL: {self.control_hz} Hz")
        print(f"HISTORY: {self.history_steps} samples / {self.history_steps / self.sensor_hz:.1f} s")
        print("INPUT: L angle | L angular velocity | R angle | R angular velocity")
        print(f"INPUT ORDER: {self.input_channel_names}")
        print(f"NORMALIZATION: {self.normalization_scheme}")
        print(f"TORQUE COMMAND SCALE: {self.torque_scale_nm:g} Nm")
        print(f"MODEL PARAMETERS: {arch.get('parameter_count', 'unknown')}")
        print("HISTORY STARTUP: zero assistance until 100 samples")
        print(f"MODEL CHECKPOINT: {self.model_path}")
        print(f"LEFT_IMU_DIRECTION and RIGHT_IMU_DIRECTION are CLI-configurable; forward thigh flexion must be positive.")
        print(f"SAFETY METADATA: {safety}")
        print("=" * 108)


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


def imu_pack_command(body: bytes, address: int = IMU_BROADCAST_ADDRESS, wake_prefix_bytes: int = 50) -> bytes:
    core = bytes([IMU_FRAME_BEGIN, address, len(body)]) + body
    checksum = sum(core[1:]) & 0xFF
    return b"\x00" * wake_prefix_bytes + core + bytes([checksum, IMU_FRAME_END])


def imu_send(uart, body: bytes, settle_s: float) -> None:
    uart.write(imu_pack_command(body))
    uart.flush()
    time.sleep(settle_s)


def configure_imu(uart) -> None:
    imu_send(uart, bytes([CMD_REPORT_OFF]), 0.15)
    imu_send(uart, bytes([CMD_WAKE]), 0.20)
    params = bytes([CMD_SET_PARAMS, 5, 255, 0, 6, 100, 2, 4, 9, REPORT_TAG & 0xFF, (REPORT_TAG >> 8) & 0xFF])
    imu_send(uart, params, 0.30)
    imu_send(uart, bytes([CMD_REPORT_ON]), 0.20)


def parse_imu_body(body: bytes, sequence: int, host_time: float) -> ImuSample | None:
    if len(body) < 7 or body[0] != CMD_REPORT:
        return None
    tag = int.from_bytes(body[1:3], "little")
    offset = 7
    gyro = None
    angle = None

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
        offset += 6
    if gyro is None or angle is None:
        return None
    return ImuSample(angle_x_deg=angle[0] * ANGLE_SCALE_DEG, gyro_x_dps=gyro[0] * GYRO_SCALE_DPS, host_time=host_time, sequence=sequence)


class SingleImuReader(threading.Thread):
    def __init__(self, *, name: str, port: str, baud: int, configure: bool, stop_event: threading.Event) -> None:
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
        if serial is None:
            self.error = "pyserial is not installed"
            self.stop_event.set()
            return
        uart = None
        parser = ImuParser()
        sequence = 0
        rate_count = 0
        rate_start = time.perf_counter()
        try:
            uart = serial.Serial(self.port, self.baud, timeout=0, write_timeout=0.5)
            uart.reset_input_buffer()
            if self.configure:
                print(f"[IMU] configure {self.side_name} {self.port} -> 100 Hz / 0x0044")
                configure_imu(uart)
                uart.reset_input_buffer()
            while not self.stop_event.is_set():
                n = uart.in_waiting
                if n > 0:
                    for body in parser.feed(uart.read(n)):
                        try:
                            sequence += 1
                            sample = parse_imu_body(body, sequence, time.perf_counter())
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
                    with self._lock:
                        self._stats.hz = rate_count / (now - rate_start)
                        self._stats.bad_packets = parser.bad
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


def crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def make_frame(cmd: int, payload: bytes = b"") -> bytes:
    body = bytes([cmd]) + payload
    return HEADER + body + bytes([crc8(body)])


class TeensyLink(threading.Thread):
    def __init__(self, *, port: str, baud: int, stop_event: threading.Event, dry_run: bool) -> None:
        super().__init__(name="TeensyLink", daemon=True)
        self.port = port
        self.baud = baud
        self.stop_event = stop_event
        self.dry_run = dry_run
        self._lock = threading.Lock()
        self._tx_lock = threading.Lock()
        self.uart = None
        self.latest: MotorFeedback | None = None
        self.stats = TeensyStats()
        self.rx = bytearray()
        self.tx_seq = 0
        self.error = ""

    def snapshot(self) -> tuple[MotorFeedback | None, TeensyStats]:
        with self._lock:
            return self.latest, TeensyStats(**vars(self.stats))

    def send_torque(self, left_nm: float, right_nm: float, enable: bool) -> None:
        if self.dry_run or self.uart is None or not self.uart.is_open:
            return
        payload = TORQUE_PAYLOAD.pack(self.tx_seq, float(left_nm), float(right_nm), int(bool(enable)))
        self.tx_seq = (self.tx_seq + 1) & 0xFFFF
        with self._tx_lock:
            self.uart.write(make_frame(CMD_TORQUE, payload))

    def send_stop(self) -> None:
        if self.dry_run or self.uart is None or not self.uart.is_open:
            return
        with self._tx_lock:
            self.uart.write(make_frame(CMD_STOP))

    def clear_fault(self) -> None:
        if self.dry_run or self.uart is None or not self.uart.is_open:
            return
        with self._tx_lock:
            self.uart.write(make_frame(CMD_CLEAR_FAULT))

    def _parse_rx(self) -> int:
        count = 0
        while True:
            if len(self.rx) < STATE_FRAME_SIZE:
                return count
            sync_index = self.rx.find(HEADER)
            if sync_index < 0:
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
            with self._lock:
                self.latest = MotorFeedback(sequence, left_actual, right_actual, time.perf_counter())
                self.stats.packets += 1
            count += 1

    def run(self) -> None:
        if self.dry_run:
            return
        if serial is None:
            self.error = "pyserial is not installed"
            self.stop_event.set()
            return
        rate_count = 0
        rate_start = time.perf_counter()
        try:
            self.uart = serial.Serial(self.port, self.baud, timeout=0, write_timeout=0.05)
            self.uart.reset_input_buffer()
            self.uart.reset_output_buffer()
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
                    with self._lock:
                        self.stats.hz = rate_count / (now - rate_start)
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
                    self.uart.close()
                except Exception:
                    pass


def wrap_angle_deg(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def circular_mean_deg(values: list[float]) -> float:
    sin_sum = sum(math.sin(math.radians(v)) for v in values)
    cos_sum = sum(math.cos(math.radians(v)) for v in values)
    return math.degrees(math.atan2(sin_sum, cos_sum)) if abs(sin_sum) + abs(cos_sum) > 1e-12 else sum(values) / len(values)


def relative_x_deg(raw_angle_deg: float, zero_offset_deg: float, direction_sign: float) -> float:
    return direction_sign * wrap_angle_deg(raw_angle_deg - zero_offset_deg)


def relative_x_gyro_dps(raw_gyro_dps: float, zero_bias_dps: float, direction_sign: float) -> float:
    return direction_sign * (raw_gyro_dps - zero_bias_dps)


class ImuInputLowPass:
    """Single-pole causal low-pass for the four TCN input channels."""

    def __init__(self, *, cutoff_hz: float, sample_hz: float, enabled: bool) -> None:
        self.cutoff_hz = float(cutoff_hz)
        self.sample_hz = float(sample_hz)
        self.enabled = bool(enabled)
        self.alpha = 1.0 - math.exp(-2.0 * math.pi * self.cutoff_hz / self.sample_hz)
        self.time_constant_s = 1.0 / (2.0 * math.pi * self.cutoff_hz)
        self._state: np.ndarray | None = None

    def reset(self) -> None:
        self._state = None

    def update(
        self,
        left_angle_rad: float,
        left_gyro_rad_s: float,
        right_angle_rad: float,
        right_gyro_rad_s: float,
    ) -> tuple[float, float, float, float]:
        frame = np.asarray(
            [left_angle_rad, left_gyro_rad_s, right_angle_rad, right_gyro_rad_s],
            dtype=np.float64,
        )
        if not self.enabled:
            return tuple(float(value) for value in frame)
        if self._state is None:
            self._state = frame.copy()
        else:
            self._state += self.alpha * (frame - self._state)
        return tuple(float(value) for value in self._state)

    def print_startup_summary(self) -> None:
        if not self.enabled:
            print("IMU INPUT FILTER: DISABLED")
            return
        print(
            "IMU INPUT FILTER: first-order causal low-pass | "
            f"cutoff={self.cutoff_hz:g} Hz | alpha={self.alpha:.4f} | "
            f"time_constant={self.time_constant_s * 1000.0:.1f} ms"
        )


class TorqueCommandFilter:
    """Causal output LPF followed by a hard per-cycle torque slew limit."""

    def __init__(
        self,
        *,
        cutoff_hz: float,
        sample_hz: float,
        max_delta_nm_per_step: float,
        output_filter_enabled: bool,
        slew_limiter_enabled: bool,
        torque_limit_nm: float,
    ) -> None:
        self.cutoff_hz = float(cutoff_hz)
        self.sample_hz = float(sample_hz)
        self.max_delta_nm_per_step = float(max_delta_nm_per_step)
        self.output_filter_enabled = bool(output_filter_enabled)
        self.slew_limiter_enabled = bool(slew_limiter_enabled)
        self.torque_limit_nm = float(torque_limit_nm)
        self.alpha = 1.0 - math.exp(-2.0 * math.pi * self.cutoff_hz / self.sample_hz)
        self.time_constant_s = 1.0 / (2.0 * math.pi * self.cutoff_hz)
        self._lpf_state = np.zeros(2, dtype=np.float64)
        self._command_state = np.zeros(2, dtype=np.float64)

    def reset(self) -> None:
        # Restart from zero assistance after timeout, fault, or history refill.
        self._lpf_state.fill(0.0)
        self._command_state.fill(0.0)

    def update(self, raw_command_nm: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_command_nm, dtype=np.float64)
        if raw.shape != (2,) or not np.all(np.isfinite(raw)):
            raise ValueError(f"invalid torque command: {raw_command_nm!r}")
        if self.output_filter_enabled:
            self._lpf_state += self.alpha * (raw - self._lpf_state)
            target = self._lpf_state
        else:
            self._lpf_state = raw.copy()
            target = raw
        if self.slew_limiter_enabled:
            delta = np.clip(
                target - self._command_state,
                -self.max_delta_nm_per_step,
                self.max_delta_nm_per_step,
            )
            self._command_state += delta
        else:
            self._command_state = target.copy()
        self._command_state = np.clip(
            self._command_state, -self.torque_limit_nm, self.torque_limit_nm
        )
        return self._command_state.astype(np.float32, copy=True)

    def print_startup_summary(self) -> None:
        if self.output_filter_enabled:
            print(
                "TORQUE OUTPUT FILTER: first-order causal low-pass | "
                f"cutoff={self.cutoff_hz:g} Hz | alpha={self.alpha:.4f} | "
                f"time_constant={self.time_constant_s * 1000.0:.1f} ms"
            )
        else:
            print("TORQUE OUTPUT FILTER: DISABLED")
        if self.slew_limiter_enabled:
            print(
                "TORQUE SLEW LIMITER: ENABLED | "
                f"max_delta={self.max_delta_nm_per_step:g} Nm/step | "
                f"equivalent={self.max_delta_nm_per_step * self.sample_hz:g} Nm/s"
            )
        else:
            print("TORQUE SLEW LIMITER: DISABLED")


def calibrate_initial_x_zero(left_reader: SingleImuReader, right_reader: SingleImuReader, *, sample_count: int, timeout_s: float, stop_event: threading.Event) -> tuple[ImuZeroOffset, ImuZeroOffset]:
    left_angles: list[float] = []
    right_angles: list[float] = []
    left_gyros: list[float] = []
    right_gyros: list[float] = []
    last_left_seq = -1
    last_right_seq = -1
    deadline = time.perf_counter() + timeout_s
    print(f"[ZERO] Stand naturally. Collecting {sample_count} fresh samples per side...")
    while not stop_event.is_set():
        if time.perf_counter() > deadline:
            raise RuntimeError(f"IMU zeroing timeout: L={len(left_angles)}/{sample_count}, R={len(right_angles)}/{sample_count}")
        left, _ = left_reader.snapshot()
        right, _ = right_reader.snapshot()
        if left is not None and left.sequence != last_left_seq and len(left_angles) < sample_count:
            left_angles.append(left.angle_x_deg)
            left_gyros.append(left.gyro_x_dps)
            last_left_seq = left.sequence
        if right is not None and right.sequence != last_right_seq and len(right_angles) < sample_count:
            right_angles.append(right.angle_x_deg)
            right_gyros.append(right.gyro_x_dps)
            last_right_seq = right.sequence
        if len(left_angles) >= sample_count and len(right_angles) >= sample_count:
            break
        time.sleep(0.001)
    left_zero = ImuZeroOffset(circular_mean_deg(left_angles), sum(left_gyros) / len(left_gyros))
    right_zero = ImuZeroOffset(circular_mean_deg(right_angles), sum(right_gyros) / len(right_gyros))
    print(f"[ZERO OK] LEFT standing offset angle={left_zero.angle_x_deg:+.4f} deg gyro={left_zero.gyro_x_dps:+.4f} deg/s")
    print(f"[ZERO OK] RIGHT standing offset angle={right_zero.angle_x_deg:+.4f} deg gyro={right_zero.gyro_x_dps:+.4f} deg/s")
    return left_zero, right_zero


def poll_stop_key() -> bool:
    if msvcrt is None:
        return False
    while msvcrt.kbhit():
        key = msvcrt.getwch().lower()
        if key in ("q", "\x1b"):
            return True
    return False


def sample_state(age_s: float, stale_s: float, timeout_s: float) -> str:
    if age_s > timeout_s:
        return "TIMEOUT"
    if age_s > stale_s:
        return "STALE"
    return "OK"


def default_csv_path() -> Path:
    return Path("logs") / f"pc_tcn_formal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="100 Hz Unified causal-TCN formal controller.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--left-port", default=DEFAULT_LEFT_IMU_PORT)
    p.add_argument("--right-port", default=DEFAULT_RIGHT_IMU_PORT)
    p.add_argument("--teensy-port", default=DEFAULT_TEENSY_PORT)
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--rate", type=float, default=DEFAULT_CONTROL_HZ)
    p.add_argument("--print-rate", type=float, default=DEFAULT_PRINT_HZ)
    p.add_argument("--imu-cutoff-hz", type=float, default=DEFAULT_IMU_CUTOFF_HZ, help="First-order causal IMU low-pass cutoff; default 10 Hz.")
    p.add_argument("--no-imu-filter", action="store_true", help="Disable the IMU input low-pass for A/B testing.")
    p.add_argument("--output-cutoff-hz", type=float, default=DEFAULT_OUTPUT_CUTOFF_HZ, help="First-order causal torque-output low-pass cutoff; default 5 Hz.")
    p.add_argument("--no-output-filter", action="store_true", help="Disable torque-output low-pass for A/B testing.")
    p.add_argument("--max-delta-nm-per-step", type=float, default=DEFAULT_MAX_DELTA_NM_PER_STEP, help="Maximum command change per 10 ms; default 0.8 Nm.")
    p.add_argument("--no-slew-limiter", action="store_true", help="Disable torque slew limiter for A/B testing.")
    p.add_argument("--stale-warning", type=float, default=DEFAULT_STALE_WARNING_S)
    p.add_argument("--imu-timeout", type=float, default=DEFAULT_IMU_TIMEOUT_S)
    p.add_argument("--teensy-timeout", type=float, default=DEFAULT_TEENSY_TIMEOUT_S)
    p.add_argument("--zero-settle", type=float, default=3.0)
    p.add_argument("--zero-samples", type=int, default=200)
    p.add_argument("--zero-timeout", type=float, default=10.0)
    p.add_argument("--skip-zero", action="store_true")
    p.add_argument("--no-configure-imu", action="store_true")
    p.add_argument("--left-imu-direction", type=float, choices=(-1.0, 1.0), default=-1.0)
    p.add_argument("--right-imu-direction", type=float, choices=(-1.0, 1.0), default=-1.0)
    p.add_argument("--max-torque", type=float, default=12.0, help="PC command clamp in Nm; default follows deployment checkpoint scale.")
    p.add_argument("--arm", action="store_true", help="Allow valid TCN command to be sent to Teensy.")
    p.add_argument("--dry-run", action="store_true", help="Do not open Teensy motor output and do not send torque.")
    p.add_argument("--mock-imu", action="store_true", help="Use synthetic thigh input; implies no IMU hardware.")
    p.add_argument("--duration", type=float, default=None, help="Optional run duration in seconds for dry-run smoke.")
    p.add_argument("--csv", type=Path, default=None)
    return p


def validate_args(a: argparse.Namespace) -> None:
    if abs(a.rate - 100.0) > 1e-9:
        raise ValueError("Unified TCN deployment requires --rate 100")
    if a.print_rate <= 0:
        raise ValueError("--print-rate must be positive")
    if a.imu_cutoff_hz <= 0 or a.imu_cutoff_hz >= 0.5 * a.rate:
        raise ValueError("--imu-cutoff-hz must be in (0, Nyquist)")
    if a.output_cutoff_hz <= 0 or a.output_cutoff_hz >= 0.5 * a.rate:
        raise ValueError("--output-cutoff-hz must be in (0, Nyquist)")
    if a.max_delta_nm_per_step <= 0:
        raise ValueError("--max-delta-nm-per-step must be positive")
    if a.max_torque <= 0 or a.max_torque > 12.0:
        raise ValueError("--max-torque must be in (0, 12]")
    if not a.dry_run and not a.arm:
        print("[SAFETY] --arm not set: Teensy output will remain zero.")
    if a.dry_run:
        print("[DRY-RUN] Hardware motor output disabled.")
    if a.mock_imu and not a.dry_run:
        raise ValueError("--mock-imu is allowed only with --dry-run")


def synthetic_frame(t: float) -> tuple[float, float, float, float]:
    freq = 1.0
    amp = 0.25
    left_angle = amp * math.sin(2 * math.pi * freq * t)
    right_angle = amp * math.sin(2 * math.pi * freq * t + math.pi)
    left_vel = amp * 2 * math.pi * freq * math.cos(2 * math.pi * freq * t)
    right_vel = amp * 2 * math.pi * freq * math.cos(2 * math.pi * freq * t + math.pi)
    return left_angle, left_vel, right_angle, right_vel


def main() -> None:
    a = build_parser().parse_args()
    validate_args(a)
    torch.set_num_threads(1)
    policy = UnifiedTCNPolicy(a.model)
    imu_filter = ImuInputLowPass(cutoff_hz=a.imu_cutoff_hz, sample_hz=a.rate, enabled=not a.no_imu_filter)
    torque_filter = TorqueCommandFilter(
        cutoff_hz=a.output_cutoff_hz,
        sample_hz=a.rate,
        max_delta_nm_per_step=a.max_delta_nm_per_step,
        output_filter_enabled=not a.no_output_filter,
        slew_limiter_enabled=not a.no_slew_limiter,
        torque_limit_nm=a.max_torque,
    )
    policy.print_startup_summary()
    imu_filter.print_startup_summary()
    torque_filter.print_startup_summary()
    print(f"LEFT_IMU_DIRECTION={a.left_imu_direction:+.0f}; RIGHT_IMU_DIRECTION={a.right_imu_direction:+.0f}")
    print("Convention check: forward thigh flexion and forward angular motion must be positive after sign and standing-zero calibration.")

    csv_path = (a.csv.expanduser().resolve() if a.csv is not None else default_csv_path().resolve())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    left_imu = right_imu = None
    teensy = TeensyLink(port=a.teensy_port, baud=a.baud, stop_event=stop_event, dry_run=a.dry_run)
    left_zero = ImuZeroOffset(0.0, 0.0)
    right_zero = ImuZeroOffset(0.0, 0.0)

    if not a.mock_imu:
        left_imu = SingleImuReader(name="LEFT", port=a.left_port, baud=a.baud, configure=not a.no_configure_imu, stop_event=stop_event)
        right_imu = SingleImuReader(name="RIGHT", port=a.right_port, baud=a.baud, configure=not a.no_configure_imu, stop_event=stop_event)
        left_imu.start()
        right_imu.start()
        teensy.start()
        deadline = time.perf_counter() + 7.0
        while time.perf_counter() < deadline and not stop_event.is_set():
            ls, _ = left_imu.snapshot()
            rs, _ = right_imu.snapshot()
            fb, _ = teensy.snapshot()
            if ls is not None and rs is not None and (a.dry_run or fb is not None):
                break
            time.sleep(0.01)
        if not a.skip_zero and a.zero_settle > 0:
            print(f"[ZERO SETTLE] Keep still for {a.zero_settle:.1f} s...")
            time.sleep(a.zero_settle)
        if not a.skip_zero:
            left_zero, right_zero = calibrate_initial_x_zero(left_imu, right_imu, sample_count=a.zero_samples, timeout_s=a.zero_timeout, stop_event=stop_event)
        else:
            print("[ZERO] skipped; offsets are zero.")
    else:
        print("[MOCK-IMU] Synthetic input active. No serial IMU ports opened.")

    period = 1.0 / a.rate
    print_period = 1.0 / a.print_rate
    start_time = time.perf_counter()
    next_tick = start_time
    next_print = start_time
    rows = 0
    overrun_count = 0
    loop_times_ms: list[float] = []
    lcmd = rcmd = 0.0
    lrawcmd = rrawcmd = 0.0
    laction = raction = 0.0
    enabled = False

    try:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "loop_index", "left_raw_angle_deg", "right_raw_angle_deg",
                "left_calibrated_angle_rad", "right_calibrated_angle_rad",
                "left_angular_velocity_rad_s", "right_angular_velocity_rad_s",
                "left_filtered_angle_rad", "right_filtered_angle_rad",
                "left_filtered_angular_velocity_rad_s", "right_filtered_angular_velocity_rad_s",
                "left_actual_torque_nm", "right_actual_torque_nm",
                "left_action_norm", "right_action_norm",
                "left_raw_tcn_command_nm", "right_raw_tcn_command_nm",
                "left_command_nm", "right_command_nm",
                "history_ready", "inference_time_ms", "loop_time_ms", "overrun", "enabled",
            ])
            while not stop_event.is_set():
                now = time.perf_counter()
                if a.duration is not None and now - start_time >= a.duration:
                    break
                if poll_stop_key():
                    print("Stop key received.")
                    break
                if now >= next_tick:
                    tick_start = time.perf_counter()
                    elapsed = tick_start - start_time
                    left_raw = right_raw = math.nan
                    left_actual = right_actual = math.nan
                    if a.mock_imu:
                        left_angle_rad, left_vel_rad_s, right_angle_rad, right_vel_rad_s = synthetic_frame(elapsed)
                    else:
                        assert left_imu is not None and right_imu is not None
                        left_sample, _ = left_imu.snapshot()
                        right_sample, _ = right_imu.snapshot()
                        feedback, _ = teensy.snapshot()
                        left_age = tick_start - left_sample.host_time if left_sample is not None else math.inf
                        right_age = tick_start - right_sample.host_time if right_sample is not None else math.inf
                        fb_age = tick_start - feedback.host_time if feedback is not None else math.inf
                        imu_ok = left_sample is not None and right_sample is not None and left_age <= a.imu_timeout and right_age <= a.imu_timeout
                        teensy_ok = a.dry_run or (feedback is not None and fb_age <= a.teensy_timeout)
                        if not imu_ok:
                            policy.reset()
                            imu_filter.reset()
                            torque_filter.reset()
                            lcmd = rcmd = laction = raction = 0.0
                            lrawcmd = rrawcmd = 0.0
                            next_tick += period
                            time.sleep(0.0005)
                            continue
                        left_raw = left_sample.angle_x_deg
                        right_raw = right_sample.angle_x_deg
                        left_angle_deg = relative_x_deg(left_sample.angle_x_deg, left_zero.angle_x_deg, a.left_imu_direction)
                        right_angle_deg = relative_x_deg(right_sample.angle_x_deg, right_zero.angle_x_deg, a.right_imu_direction)
                        left_gyro_dps = relative_x_gyro_dps(left_sample.gyro_x_dps, left_zero.gyro_x_dps, a.left_imu_direction)
                        right_gyro_dps = relative_x_gyro_dps(right_sample.gyro_x_dps, right_zero.gyro_x_dps, a.right_imu_direction)
                        left_angle_rad = math.radians(left_angle_deg)
                        right_angle_rad = math.radians(right_angle_deg)
                        left_vel_rad_s = math.radians(left_gyro_dps)
                        right_vel_rad_s = math.radians(right_gyro_dps)
                        if feedback is not None:
                            left_actual = feedback.left_actual_nm
                            right_actual = feedback.right_actual_nm
                        if not teensy_ok:
                            enabled = False
                    left_unfiltered_angle_rad = left_angle_rad
                    right_unfiltered_angle_rad = right_angle_rad
                    left_unfiltered_vel_rad_s = left_vel_rad_s
                    right_unfiltered_vel_rad_s = right_vel_rad_s
                    left_angle_rad, left_vel_rad_s, right_angle_rad, right_vel_rad_s = imu_filter.update(
                        left_unfiltered_angle_rad,
                        left_unfiltered_vel_rad_s,
                        right_unfiltered_angle_rad,
                        right_unfiltered_vel_rad_s,
                    )
                    policy.append_frame(left_angle_rad, left_vel_rad_s, right_angle_rad, right_vel_rad_s)
                    out = policy.infer()
                    if out is None:
                        laction = raction = 0.0
                        lcmd = rcmd = 0.0
                        lrawcmd = rrawcmd = 0.0
                        torque_filter.reset()
                    else:
                        action, command_nm = out
                        laction, raction = float(action[0]), float(action[1])
                        lrawcmd, rrawcmd = float(command_nm[0]), float(command_nm[1])
                        filtered_command_nm = torque_filter.update(command_nm)
                        lcmd, rcmd = float(filtered_command_nm[0]), float(filtered_command_nm[1])
                    enabled = bool(a.arm and not a.dry_run and out is not None)
                    teensy.send_torque(lcmd if enabled else 0.0, rcmd if enabled else 0.0, enabled)
                    loop_time_ms = (time.perf_counter() - tick_start) * 1000.0
                    overrun = loop_time_ms > period * 1000.0
                    overrun_count += int(overrun)
                    loop_times_ms.append(loop_time_ms)
                    writer.writerow([
                        f"{elapsed:.6f}", rows, f"{left_raw:.6f}", f"{right_raw:.6f}",
                        f"{left_unfiltered_angle_rad:.9f}", f"{right_unfiltered_angle_rad:.9f}",
                        f"{left_unfiltered_vel_rad_s:.9f}", f"{right_unfiltered_vel_rad_s:.9f}",
                        f"{left_angle_rad:.9f}", f"{right_angle_rad:.9f}",
                        f"{left_vel_rad_s:.9f}", f"{right_vel_rad_s:.9f}",
                        f"{left_actual:.6f}", f"{right_actual:.6f}",
                        f"{laction:.9f}", f"{raction:.9f}",
                        f"{lrawcmd:.6f}", f"{rrawcmd:.6f}", f"{lcmd:.6f}", f"{rcmd:.6f}",
                        int(policy.history_ready), f"{policy.last_inference_time_ms:.6f}",
                        f"{loop_time_ms:.6f}", int(overrun), int(enabled),
                    ])
                    rows += 1
                    next_tick += period
                    if tick_start - next_tick > period:
                        next_tick = tick_start + period
                    if tick_start >= next_print:
                        print(f"t={elapsed:7.3f}s history={'READY' if policy.history_ready else 'FILL '} cmd={lcmd:+.3f}/{rcmd:+.3f} Nm action={laction:+.3f}/{raction:+.3f} inf={policy.last_inference_time_ms:.3f} ms {'ON' if enabled else 'OFF'}")
                        next_print = tick_start + print_period
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
        if left_imu is not None:
            left_imu.join(timeout=2.0)
        if right_imu is not None:
            right_imu.join(timeout=2.0)
        if teensy.ident is not None:
            teensy.join(timeout=2.0)
        duration = max(time.perf_counter() - start_time, 1e-9)
        mean_loop = float(np.mean(loop_times_ms)) if loop_times_ms else math.nan
        p95_loop = float(np.percentile(loop_times_ms, 95)) if loop_times_ms else math.nan
        print("=" * 108)
        print(f"CSV saved: {csv_path}")
        print(f"Rows/rate: {rows} / {rows / duration:.2f} Hz")
        print(f"Zero angle: L={left_zero.angle_x_deg:+.4f} deg, R={right_zero.angle_x_deg:+.4f} deg")
        print(f"Gyro bias: L={left_zero.gyro_x_dps:+.4f} deg/s, R={right_zero.gyro_x_dps:+.4f} deg/s")
        print(f"TCN inference: calls={policy.calls}, valid={policy.valid_outputs}, last={policy.last_inference_time_ms:.3f} ms")
        print(f"Loop timing: mean={mean_loop:.3f} ms, p95={p95_loop:.3f} ms, overruns={overrun_count}")
        if policy.last_error:
            print("TCN last error:", policy.last_error)
        print("=" * 108)


if __name__ == "__main__":
    main()
