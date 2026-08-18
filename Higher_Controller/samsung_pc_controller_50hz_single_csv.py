"""
Samsung PC controller - dual IM948 + Teensy, paired raw IMU, single CSV logger
=============================================================================

Data flow
---------
LEFT IM948 100 Hz thread ----\
                              > raw-packet device-time pairing -> Samsung update
RIGHT IM948 100 Hz thread ---/                                  |
                                                                  +-> desired torque
Teensy RX thread -------------------------------------------------+-> actual torque
                                                                  |
                                                                  +-> single CSV logger

Important behavior
------------------
1. Both IMUs are read independently and paired with their device timestamps.
2. Samsung control is updated once for every paired raw IMU sample (~100 Hz).
3. Teensy torque transmission is 50 Hz by default.
4. CSV logging is asynchronous and contains ONLY experiment signals:
       time_s,
       left/right angle,
       left/right angular velocity,
       left/right commanded torque,
       left/right actual torque.
5. Commanded torque in CSV/plot means the Samsung controller DESIRED torque for
   that IMU sample. It is calculated and recorded even without --enable.
6. Without --enable, the PC still calculates/logs/plots desired torque, but the
   torque actually sent to Teensy is forced to zero with enable=0.
7. Actual torque is the newest Teensy feedback received no later than the paired
   IMU host timestamp. This avoids using future feedback for an older IMU row.

Examples
--------
Safe calculation + print + CSV (motor output remains zero):
    python samsung_pc_controller_optimized.py --display print

Safe calculation + realtime plot + CSV (cmd curve is still visible):
    python samsung_pc_controller_optimized.py --display plot

Real assistance after bench verification:
    python samsung_pc_controller_optimized.py --display plot --enable

Runtime keys
------------
Up / Down    : rescaling +/- 0.5, minimum 0
Left / Right : delay_index +/- 1 sample; at 100 Hz, 1 sample ~= 10 ms
Ctrl+C or closing the plot window: STOP
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
from typing import Final, Iterable

import serial

try:
    import msvcrt
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

DEFAULT_IMU_HZ = 100.0
DEFAULT_CONTROL_HZ = 100.0
DEFAULT_TX_HZ = 50.0
DEFAULT_PRINT_HZ = 10.0
DEFAULT_PLOT_HZ = 30.0
DEFAULT_PLOT_WINDOW_S = 10.0

DEFAULT_STALE_WARNING_S = 0.050
DEFAULT_IMU_TIMEOUT_S = 0.150
DEFAULT_TEENSY_TIMEOUT_S = 0.200

# Keep current controller dynamics by default. Test 0.020 s separately if desired.
DEFAULT_FILTER_TAU_S = 0.025
DEFAULT_MAX_COMMAND_NM = 8.0

# Pairing uses device timestamps. At 100 Hz samples should be 10 ms apart.
PAIR_DEVICE_TOLERANCE_MS = 4.0
PAIR_OFFSET_ADAPT_ALPHA = 0.01
PAIR_ANCHOR_SEARCH_DEPTH = 8

# Reader queues only buffer raw packets between the serial thread and main loop.
IMU_PENDING_MAXLEN = 2000
FEEDBACK_HISTORY_MAXLEN = 4000
LOGGER_QUEUE_SIZE = 20000

# Teensy direct torque protocol.
CMD_TORQUE = 0x54
CMD_STOP = 0x50
CMD_CLEAR_FAULT = 0x43
CMD_STATE = 0x44
TORQUE_PAYLOAD = struct.Struct("<HffB")
STATE_PAYLOAD = struct.Struct("<Hff")
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
ANGLE_SCALE_DEG = 180.0 / 32768.0
GYRO_SCALE_DPS = 2000.0 / 32768.0
MAX_IMU_DATA_LEN = 128


@dataclass(frozen=True)
class ImuSample:
    angle_x_deg: float
    gyro_x_dps: float
    device_ms: int
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
    pending_overflow: int = 0


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
    return b"\x00" * wake_prefix_bytes + core + bytes([checksum, IMU_FRAME_END])


def imu_send(uart: serial.Serial, body: bytes, settle_s: float) -> None:
    uart.write(imu_pack_command(body))
    uart.flush()
    time.sleep(settle_s)


def configure_imu(uart: serial.Serial) -> None:
    """Force IM948 to active 100 Hz output with gyro XYZ + Euler XYZ."""
    imu_send(uart, bytes([CMD_REPORT_OFF]), 0.15)
    imu_send(uart, bytes([CMD_WAKE]), 0.20)

    params = bytes(
        [
            CMD_SET_PARAMS,
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
    imu_send(uart, bytes([CMD_REPORT_ON]), 0.20)


def parse_imu_body(body: bytes, sequence: int, host_time: float) -> ImuSample | None:
    """
    Report body layout used here:
        body[0]   : CMD_REPORT 0x11
        body[1:3] : report tag uint16
        body[3:7] : IMU device timestamp uint32, milliseconds
        body[7:]  : fields selected by tag
    """
    if len(body) < 7 or body[0] != CMD_REPORT:
        return None

    tag = int.from_bytes(body[1:3], "little")
    device_ms = int.from_bytes(body[3:7], "little", signed=False)
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
        device_ms=device_ms,
        host_time=host_time,
        sequence=sequence,
    )


class SingleImuReader(threading.Thread):
    """One thread owns one IM948 serial port and never blocks the other IMU."""

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
        self._pending: deque[ImuSample] = deque(maxlen=IMU_PENDING_MAXLEN)
        self._stats = ImuStats()
        self.error = ""

    def snapshot(self) -> tuple[ImuSample | None, ImuStats]:
        with self._lock:
            return self._latest, ImuStats(**vars(self._stats))

    def drain_pending(self) -> list[ImuSample]:
        with self._lock:
            out = list(self._pending)
            self._pending.clear()
            return out

    def clear_pending(self) -> None:
        with self._lock:
            self._pending.clear()

    def run(self) -> None:
        uart: serial.Serial | None = None
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
                            if len(self._pending) == self._pending.maxlen:
                                self._stats.pending_overflow += 1
                            self._pending.append(sample)
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
# Device-time pairing
# =============================================================================


def signed_u32_diff(a: int, b: int) -> int:
    """Signed modular difference a-b for uint32 millisecond counters."""
    d = (int(a) - int(b)) & 0xFFFFFFFF
    if d & 0x80000000:
        d -= 0x100000000
    return d


def u32_elapsed_ms(now_ms: int, start_ms: int) -> int:
    return (int(now_ms) - int(start_ms)) & 0xFFFFFFFF


@dataclass(frozen=True)
class ImuPair:
    left: ImuSample
    right: ImuSample
    device_residual_ms: float


class ImuPairer:
    """
    Pair raw samples by device timestamp.

    The first pair anchors the constant difference between the two independent
    IMU clocks. The anchor is chosen among the first few queued samples by the
    smallest host-arrival separation. After that, host arrival time is ignored
    for pairing; only corrected device time is used.
    """

    def __init__(self, tolerance_ms: float = PAIR_DEVICE_TOLERANCE_MS) -> None:
        self.tolerance_ms = tolerance_ms
        self.left_buf: deque[ImuSample] = deque()
        self.right_buf: deque[ImuSample] = deque()
        self.device_offset_ms: float | None = None
        self.left_unmatched = 0
        self.right_unmatched = 0
        self.pairs = 0

    def add(self, left: Iterable[ImuSample], right: Iterable[ImuSample]) -> None:
        self.left_buf.extend(left)
        self.right_buf.extend(right)

    def _anchor(self) -> bool:
        if not self.left_buf or not self.right_buf:
            return False

        left_candidates = list(self.left_buf)[:PAIR_ANCHOR_SEARCH_DEPTH]
        right_candidates = list(self.right_buf)[:PAIR_ANCHOR_SEARCH_DEPTH]

        best: tuple[float, int, int] | None = None
        for li, ls in enumerate(left_candidates):
            for ri, rs in enumerate(right_candidates):
                cost = abs(ls.host_time - rs.host_time)
                if best is None or cost < best[0]:
                    best = (cost, li, ri)

        assert best is not None
        _, li, ri = best

        for _ in range(li):
            self.left_buf.popleft()
            self.left_unmatched += 1
        for _ in range(ri):
            self.right_buf.popleft()
            self.right_unmatched += 1

        l0 = self.left_buf[0]
        r0 = self.right_buf[0]
        self.device_offset_ms = float(signed_u32_diff(l0.device_ms, r0.device_ms))
        return True

    def pop_pairs(self) -> list[ImuPair]:
        out: list[ImuPair] = []
        if self.device_offset_ms is None and not self._anchor():
            return out

        assert self.device_offset_ms is not None

        while self.left_buf and self.right_buf:
            left = self.left_buf[0]
            right = self.right_buf[0]
            raw_diff = float(signed_u32_diff(left.device_ms, right.device_ms))
            residual = raw_diff - self.device_offset_ms

            if abs(residual) <= self.tolerance_ms:
                self.left_buf.popleft()
                self.right_buf.popleft()
                out.append(ImuPair(left, right, residual))
                self.pairs += 1

                # Very slow offset adaptation handles small independent clock drift.
                self.device_offset_ms = (
                    (1.0 - PAIR_OFFSET_ADAPT_ALPHA) * self.device_offset_ms
                    + PAIR_OFFSET_ADAPT_ALPHA * raw_diff
                )
                continue

            if residual < -self.tolerance_ms:
                # Left corrected device time is older.
                self.left_buf.popleft()
                self.left_unmatched += 1
            else:
                self.right_buf.popleft()
                self.right_unmatched += 1

        return out


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
    def __init__(self, *, port: str, baud: int, stop_event: threading.Event) -> None:
        super().__init__(name="TeensyLink", daemon=True)
        self.port = port
        self.baud = baud
        self.stop_event = stop_event

        self._lock = threading.Lock()
        self._tx_lock = threading.Lock()
        self.uart: serial.Serial | None = None
        self.latest: MotorFeedback | None = None
        self.history: deque[MotorFeedback] = deque(maxlen=FEEDBACK_HISTORY_MAXLEN)
        self.stats = TeensyStats()
        self.rx = bytearray()
        self.tx_seq = 0
        self.error = ""

    def snapshot(self) -> tuple[MotorFeedback | None, TeensyStats]:
        with self._lock:
            return self.latest, TeensyStats(**vars(self.stats))

    def latest_before(self, host_time: float) -> MotorFeedback | None:
        with self._lock:
            for item in reversed(self.history):
                if item.host_time <= host_time:
                    return item
            return None

    def send_torque(self, left_nm: float, right_nm: float, enable: bool) -> int | None:
        uart = self.uart
        if uart is None or not uart.is_open:
            return None

        seq = self.tx_seq
        payload = TORQUE_PAYLOAD.pack(seq, float(left_nm), float(right_nm), int(bool(enable)))
        packet = make_frame(CMD_TORQUE, payload)

        with self._tx_lock:
            uart.write(packet)

        self.tx_seq = (self.tx_seq + 1) & 0xFFFF
        return seq

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
                self.history.append(feedback)
                self.stats.packets += 1
            count += 1

    def run(self) -> None:
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
# Zeroing and joint coordinate
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


def relative_x_deg(raw: float, zero: float, sign: float) -> float:
    return sign * wrap_angle_deg(raw - zero)


def relative_x_gyro_dps(raw: float, bias: float, sign: float) -> float:
    return sign * (raw - bias)


def calibrate_initial_x_zero(
    left_reader: SingleImuReader,
    right_reader: SingleImuReader,
    *,
    sample_count: int,
    timeout_s: float,
    stop_event: threading.Event,
) -> tuple[ImuZeroOffset, ImuZeroOffset]:
    left_angles: list[float] = []
    right_angles: list[float] = []
    left_gyros: list[float] = []
    right_gyros: list[float] = []
    last_left_seq = -1
    last_right_seq = -1
    deadline = time.perf_counter() + timeout_s
    next_print = 0.0

    print(f"[ZERO] Stand naturally and keep BOTH IMUs still. Collecting {sample_count} fresh samples/side...")

    while not stop_event.is_set():
        now = time.perf_counter()
        if now > deadline:
            raise RuntimeError(
                f"IMU zero timeout: L={len(left_angles)}/{sample_count}, "
                f"R={len(right_angles)}/{sample_count}"
            )

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

        if now >= next_print:
            print(
                f"\r[ZERO] L {len(left_angles):4d}/{sample_count} | R {len(right_angles):4d}/{sample_count}",
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

    print(f"\r[ZERO] L {sample_count:4d}/{sample_count} | R {sample_count:4d}/{sample_count}")
    print(f"[ZERO OK] LEFT  angle={left_zero.angle_x_deg:+.4f} deg | gyro bias={left_zero.gyro_x_dps:+.4f} deg/s")
    print(f"[ZERO OK] RIGHT angle={right_zero.angle_x_deg:+.4f} deg | gyro bias={right_zero.gyro_x_dps:+.4f} deg/s")
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
    filter_tau_s: float = DEFAULT_FILTER_TAU_S
    max_command_nm: float = DEFAULT_MAX_COMMAND_NM


class SamsungController:
    HISTORY_SIZE = 100

    def __init__(self, cfg: SamsungConfig) -> None:
        self.cfg = cfg
        self.left_filtered: float | None = None
        self.right_filtered: float | None = None
        self.phase_history = [0.0] * self.HISTORY_SIZE
        self.shape_history = [0.0] * self.HISTORY_SIZE
        self.write_index = 0
        self.valid_count = 0
        self.last_sample_time_s: float | None = None

    def update(self, left_angle_deg: float, right_angle_deg: float, sample_time_s: float) -> tuple[float, float]:
        left = math.radians(left_angle_deg)
        right = math.radians(right_angle_deg)

        if self.last_sample_time_s is None:
            dt = 1.0 / DEFAULT_CONTROL_HZ
        else:
            dt = max(0.001, min(sample_time_s - self.last_sample_time_s, 0.05))
        self.last_sample_time_s = sample_time_s

        if self.left_filtered is None:
            self.left_filtered = left
            self.right_filtered = right
        else:
            alpha = 1.0 - math.exp(-dt / self.cfg.filter_tau_s)
            self.left_filtered += alpha * (left - self.left_filtered)
            assert self.right_filtered is not None
            self.right_filtered += alpha * (right - self.right_filtered)

        assert self.right_filtered is not None
        phase = self.right_filtered - self.left_filtered
        shape = math.sin(self.right_filtered) - math.sin(self.left_filtered)

        current_index = self.write_index
        self.phase_history[current_index] = phase
        self.shape_history[current_index] = shape
        self.write_index = (self.write_index + 1) % self.HISTORY_SIZE
        self.valid_count = min(self.valid_count + 1, self.HISTORY_SIZE)

        delay = max(0, min(self.cfg.delay_index, self.HISTORY_SIZE - 1))
        if self.valid_count <= delay:
            return 0.0, 0.0

        delayed_index = (current_index - delay) % self.HISTORY_SIZE
        delayed_phase = self.phase_history[delayed_index]
        delayed_shape = self.shape_history[delayed_index]
        phase_limit = math.radians(120.0)

        if 0.0 <= delayed_phase < phase_limit:
            left_tau = -self.cfg.rescaling * self.cfg.ext_gain * delayed_shape
            right_tau = self.cfg.rescaling * self.cfg.flex_gain * delayed_shape
        elif -phase_limit < delayed_phase < 0.0:
            right_tau = self.cfg.rescaling * self.cfg.ext_gain * delayed_shape
            left_tau = -self.cfg.rescaling * self.cfg.flex_gain * delayed_shape
        else:
            left_tau = 0.0
            right_tau = 0.0

        limit = abs(self.cfg.max_command_nm)
        return (
            max(-limit, min(limit, left_tau)),
            max(-limit, min(limit, right_tau)),
        )


# =============================================================================
# Runtime keyboard control
# =============================================================================

RESCALING_STEP = 0.5
DELAY_STEP = 1


def poll_console_arrow_keys() -> list[str]:
    if msvcrt is None:
        return []
    keys: list[str] = []
    key_map = {b"H": "up", b"P": "down", b"K": "left", b"M": "right"}
    while msvcrt.kbhit():
        first = msvcrt.getch()
        if first in (b"\x00", b"\xe0"):
            key = key_map.get(msvcrt.getch())
            if key is not None:
                keys.append(key)
    return keys


def apply_runtime_key(controller: SamsungController, key: str, control_hz: float) -> str | None:
    if key == "up":
        controller.cfg.rescaling += RESCALING_STEP
    elif key == "down":
        controller.cfg.rescaling = max(0.0, controller.cfg.rescaling - RESCALING_STEP)
    elif key == "left":
        controller.cfg.delay_index = max(0, controller.cfg.delay_index - DELAY_STEP)
    elif key == "right":
        controller.cfg.delay_index = min(controller.HISTORY_SIZE - 1, controller.cfg.delay_index + DELAY_STEP)
    else:
        return None

    delay_ms = 1000.0 * controller.cfg.delay_index / max(control_hz, 1e-9)
    return (
        f"[PARAM] rescaling={controller.cfg.rescaling:.1f} | "
        f"delay={controller.cfg.delay_index} samples (~{delay_ms:.0f} ms)"
    )


def drain_plot_key_queue(control_queue) -> list[str]:
    if control_queue is None:
        return []
    out: list[str] = []
    while True:
        try:
            out.append(control_queue.get_nowait())
        except queue.Empty:
            break
    return out


# =============================================================================
# Single CSV logging
# =============================================================================


CSV_FIELDS = [
    "time_s",
    "left_angle_x_deg",
    "left_angular_velocity_x_dps",
    "right_angle_x_deg",
    "right_angular_velocity_x_dps",
    "left_cmd_torque_nm",
    "right_cmd_torque_nm",
    "left_actual_torque_nm",
    "right_actual_torque_nm",
]


def csv_value(v: object) -> object:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return f"{v:.4f}"
    return v


class CsvLogger(threading.Thread):
    """Asynchronous single-file CSV writer; never blocks the control path."""

    def __init__(self, path: Path) -> None:
        super().__init__(name="CsvLogger", daemon=True)
        self.path = path
        self.q: queue.Queue[dict[str, object] | None] = queue.Queue(maxsize=LOGGER_QUEUE_SIZE)
        self._lock = threading.Lock()
        self._dropped_rows = 0
        self.error = ""

    @property
    def dropped_rows(self) -> int:
        with self._lock:
            return self._dropped_rows

    def submit(self, row: dict[str, object]) -> bool:
        try:
            self.q.put_nowait(row)
            return True
        except queue.Full:
            with self._lock:
                self._dropped_rows += 1
            return False

    def close(self) -> None:
        # Drain queued data before stopping the writer.
        while True:
            try:
                self.q.put(None, timeout=0.1)
                break
            except queue.Full:
                if not self.is_alive():
                    break
        self.join(timeout=3.0)

    def run(self) -> None:
        try:
            with self.path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()

                rows_since_flush = 0
                last_flush = time.perf_counter()

                while True:
                    item = self.q.get()
                    if item is None:
                        break

                    writer.writerow({k: csv_value(item.get(k, "")) for k in CSV_FIELDS})
                    rows_since_flush += 1

                    now = time.perf_counter()
                    if rows_since_flush >= 100 or now - last_flush >= 1.0:
                        f.flush()
                        rows_since_flush = 0
                        last_flush = now

                f.flush()

        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"


# =============================================================================
# Plot process
# =============================================================================


def plot_worker(data_queue, close_event, control_queue, *, refresh_hz: float, window_s: float) -> None:
    import matplotlib.pyplot as plt

    history_len = max(int(window_s * DEFAULT_IMU_HZ * 1.5), 300)
    t_hist = deque(maxlen=history_len)
    la_hist = deque(maxlen=history_len)
    ra_hist = deque(maxlen=history_len)
    lv_hist = deque(maxlen=history_len)
    rv_hist = deque(maxlen=history_len)
    lc_hist = deque(maxlen=history_len)
    rc_hist = deque(maxlen=history_len)
    lact_hist = deque(maxlen=history_len)
    ract_hist = deque(maxlen=history_len)

    fig, (ax_angle, ax_vel, ax_torque) = plt.subplots(3, 1, sharex=True, figsize=(11, 8.5))
    line_la, = ax_angle.plot([], [], label="Left angle")
    line_ra, = ax_angle.plot([], [], label="Right angle")
    line_lv, = ax_vel.plot([], [], label="Left angular velocity")
    line_rv, = ax_vel.plot([], [], label="Right angular velocity")
    line_lc, = ax_torque.plot([], [], label="Left cmd")
    line_rc, = ax_torque.plot([], [], label="Right cmd")
    line_lact, = ax_torque.plot([], [], label="Left actual")
    line_ract, = ax_torque.plot([], [], label="Right actual")

    ax_angle.set_ylabel("Angle (deg)")
    ax_vel.set_ylabel("Angular velocity (deg/s)")
    ax_torque.set_ylabel("Torque (Nm)")
    ax_torque.set_xlabel("IMU device time (s)")

    for ax in (ax_angle, ax_vel, ax_torque):
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    title = fig.suptitle("Waiting for data...")

    def on_close(_event) -> None:
        close_event.set()

    def on_key(event) -> None:
        if event.key in ("up", "down", "left", "right"):
            try:
                control_queue.put_nowait(event.key)
            except queue.Full:
                pass

    fig.canvas.mpl_connect("close_event", on_close)
    fig.canvas.mpl_connect("key_press_event", on_key)

    next_refresh = time.perf_counter()
    refresh_period = 1.0 / max(refresh_hz, 1.0)
    latest_meta = (0.0, 0)

    while not close_event.is_set():
        got = False
        while True:
            try:
                item = data_queue.get_nowait()
            except queue.Empty:
                break

            if item is None:
                close_event.set()
                break

            (
                t,
                la,
                ra,
                lv,
                rv,
                lc,
                rc,
                lact,
                ract,
                rescaling,
                delay_index,
            ) = item

            t_hist.append(t)
            la_hist.append(la)
            ra_hist.append(ra)
            lv_hist.append(lv)
            rv_hist.append(rv)
            lc_hist.append(lc)
            rc_hist.append(rc)
            lact_hist.append(lact)
            ract_hist.append(ract)
            latest_meta = (rescaling, delay_index)
            got = True

        now = time.perf_counter()
        if got and now >= next_refresh and len(t_hist) >= 2:
            x = list(t_hist)
            line_la.set_data(x, list(la_hist))
            line_ra.set_data(x, list(ra_hist))
            line_lv.set_data(x, list(lv_hist))
            line_rv.set_data(x, list(rv_hist))
            line_lc.set_data(x, list(lc_hist))
            line_rc.set_data(x, list(rc_hist))
            line_lact.set_data(x, list(lact_hist))
            line_ract.set_data(x, list(ract_hist))

            xmax = x[-1]
            xmin = max(x[0], xmax - window_s)
            for ax in (ax_angle, ax_vel, ax_torque):
                ax.set_xlim(xmin, max(xmax, xmin + 0.1))
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)

            rescaling, delay_index = latest_meta
            title.set_text(f"rescaling={rescaling:.1f} | delay={delay_index} samples")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            next_refresh = now + refresh_period

        plt.pause(0.001)

    try:
        plt.close(fig)
    except Exception:
        pass


def push_plot_sample(data_queue, sample) -> None:
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
    p = argparse.ArgumentParser(description="Samsung dual-IMU controller with device-time pairing and single CSV logging")
    p.add_argument("--left-port", default=DEFAULT_LEFT_IMU_PORT)
    p.add_argument("--right-port", default=DEFAULT_RIGHT_IMU_PORT)
    p.add_argument("--teensy-port", default=DEFAULT_TEENSY_PORT)
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)

    p.add_argument("--rate", type=float, default=DEFAULT_CONTROL_HZ, help="expected paired IMU/control sample rate; keep 100 Hz")
    p.add_argument("--tx-rate", type=float, default=DEFAULT_TX_HZ, help="PC -> Teensy torque command rate; default 50 Hz")
    p.add_argument("--display", choices=("print", "plot"), default="print")
    p.add_argument("--print-rate", type=float, default=DEFAULT_PRINT_HZ)
    p.add_argument("--plot-rate", type=float, default=DEFAULT_PLOT_HZ)
    p.add_argument("--plot-window", type=float, default=DEFAULT_PLOT_WINDOW_S)

    p.add_argument("--stale-warning", type=float, default=DEFAULT_STALE_WARNING_S)
    p.add_argument("--imu-timeout", type=float, default=DEFAULT_IMU_TIMEOUT_S)
    p.add_argument("--teensy-timeout", type=float, default=DEFAULT_TEENSY_TIMEOUT_S)

    p.add_argument("--zero-samples", type=int, default=200)
    p.add_argument("--zero-timeout", type=float, default=10.0)
    p.add_argument("--skip-zero", action="store_true")
    p.add_argument("--no-configure-imu", action="store_true")

    p.add_argument("--rescaling", type=float, default=5.0)
    p.add_argument("--flex", type=float, default=1.0)
    p.add_argument("--ext", type=float, default=1.0)
    p.add_argument("--delay-index", type=int, default=0)
    p.add_argument("--filter-tau", type=float, default=DEFAULT_FILTER_TAU_S)
    p.add_argument("--max-command", type=float, default=DEFAULT_MAX_COMMAND_NM)
    p.add_argument("--left-angle-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    p.add_argument("--right-angle-sign", type=float, choices=(-1.0, 1.0), default=-1.0)

    p.add_argument("--enable", action="store_true", help="allow non-zero torque output")
    p.add_argument("--log-dir", type=Path, default=Path("logs"))
    p.add_argument("--name", type=str, default="samsung")
    return p


def validate_args(a: argparse.Namespace) -> None:
    ports = {a.left_port.upper(), a.right_port.upper(), a.teensy_port.upper()}
    if len(ports) != 3:
        raise ValueError("left IMU, right IMU and Teensy must use different COM ports")
    if a.rate <= 0 or a.tx_rate <= 0:
        raise ValueError("--rate and --tx-rate must be positive")
    if a.tx_rate > a.rate:
        raise ValueError("--tx-rate should not exceed --rate")
    if a.print_rate <= 0 or a.plot_rate <= 0 or a.plot_window <= 0:
        raise ValueError("print/plot rates and plot window must be positive")
    if a.stale_warning <= 0 or a.imu_timeout <= a.stale_warning:
        raise ValueError("--imu-timeout must be larger than --stale-warning")
    if a.teensy_timeout <= 0:
        raise ValueError("--teensy-timeout must be positive")
    if a.zero_samples <= 0 or a.zero_timeout <= 0:
        raise ValueError("zero parameters must be positive")
    if not 0 <= a.delay_index < SamsungController.HISTORY_SIZE:
        raise ValueError("--delay-index must be in [0, 99]")
    if a.filter_tau <= 0 or a.max_command <= 0:
        raise ValueError("--filter-tau and --max-command must be positive")


def make_log_path(log_dir: Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{name}_{stamp}.csv"


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
            raise RuntimeError("--display plot requires matplotlib") from exc

    csv_path = make_log_path(a.log_dir, a.name)
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
    teensy = TeensyLink(port=a.teensy_port, baud=a.baud, stop_event=stop_event)
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
    pairer = ImuPairer()
    logger = CsvLogger(csv_path)

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
            kwargs={"refresh_hz": a.plot_rate, "window_s": a.plot_window},
            daemon=True,
        )

    print("=" * 106)
    print("Samsung PC controller | raw IMU device-time pairing | Teensy TX | single CSV")
    print(f"LEFT / RIGHT IMU : {a.left_port} / {a.right_port} @ {a.baud}, target 100 Hz")
    print(f"Teensy           : {a.teensy_port} @ {a.baud}, TX={a.tx_rate:.1f} Hz")
    print(f"Samsung          : rescaling={a.rescaling:.1f}, delay={a.delay_index}, filter tau={a.filter_tau:.3f}s")
    print(f"PC clamp         : +/-{a.max_command:.2f} Nm")
    print(f"Torque output    : {'ENABLED' if a.enable else 'SAFE ZERO (use --enable for real torque)'}")
    print(f"CSV              : {csv_path}")
    print("CSV content      : time + angle + angular velocity + cmd torque + actual torque")
    print("Keys             : Up/Down rescaling +/-0.5 | Left/Right delay +/-1 | Ctrl+C STOP")
    print("=" * 106)

    left_zero = ImuZeroOffset(0.0, 0.0)
    right_zero = ImuZeroOffset(0.0, 0.0)

    start_left_device_ms: int | None = None
    pair_index = 0
    last_pair_host_time = -math.inf
    latest_desired_left = 0.0
    latest_desired_right = 0.0
    latest_left_angle = math.nan
    latest_right_angle = math.nan
    latest_left_gyro = math.nan
    latest_right_gyro = math.nan
    latest_output_enabled = False
    next_tx = 0.0
    next_print = 0.0
    tx_period = 1.0 / a.tx_rate
    print_period = 1.0 / a.print_rate

    try:
        left_imu.start()
        right_imu.start()
        teensy.start()
        logger.start()
        if plot_process is not None:
            plot_process.start()

        # Wait briefly for first valid IMU samples.
        startup_deadline = time.perf_counter() + 5.0
        while not stop_event.is_set():
            l, _ = left_imu.snapshot()
            r, _ = right_imu.snapshot()
            if l is not None and r is not None:
                break
            if time.perf_counter() > startup_deadline:
                raise RuntimeError("IMU startup timeout")
            time.sleep(0.005)

        if not a.skip_zero:
            left_zero, right_zero = calibrate_initial_x_zero(
                left_imu,
                right_imu,
                sample_count=a.zero_samples,
                timeout_s=a.zero_timeout,
                stop_event=stop_event,
            )
        else:
            print("[ZERO] skipped; raw angle/gyro biases are not removed")

        # Discard zero-calibration samples from raw queues. Formal time starts on
        # the first fresh pair after calibration.
        left_imu.clear_pending()
        right_imu.clear_pending()
        time.sleep(0.02)
        left_imu.clear_pending()
        right_imu.clear_pending()

        run_start = time.perf_counter()
        next_tx = run_start
        next_print = run_start


        while not stop_event.is_set():
            if plot_close_event is not None and plot_close_event.is_set():
                break

            for key in poll_console_arrow_keys() + drain_plot_key_queue(plot_control_queue):
                msg = apply_runtime_key(controller, key, a.rate)
                if msg:
                    print(msg)

            # Drain raw packets quickly. The pairer reconstructs the 100 Hz
            # device-time order even when Windows delivers packets in bursts.
            pairer.add(left_imu.drain_pending(), right_imu.drain_pending())
            pairs = pairer.pop_pairs()

            for pair in pairs:
                pair_host_time = max(pair.left.host_time, pair.right.host_time)
                last_pair_host_time = pair_host_time

                if start_left_device_ms is None:
                    start_left_device_ms = pair.left.device_ms
                time_s = u32_elapsed_ms(pair.left.device_ms, start_left_device_ms) / 1000.0

                left_angle = relative_x_deg(pair.left.angle_x_deg, left_zero.angle_x_deg, a.left_angle_sign)
                right_angle = relative_x_deg(pair.right.angle_x_deg, right_zero.angle_x_deg, a.right_angle_sign)
                left_gyro = relative_x_gyro_dps(pair.left.gyro_x_dps, left_zero.gyro_x_dps, a.left_angle_sign)
                right_gyro = relative_x_gyro_dps(pair.right.gyro_x_dps, right_zero.gyro_x_dps, a.right_angle_sign)

                # Desired command belongs directly to this paired IMU sample.
                # It is recorded/plotted regardless of --enable.
                cmd_left, cmd_right = controller.update(left_angle, right_angle, time_s)
                latest_desired_left = cmd_left
                latest_desired_right = cmd_right
                latest_left_angle = left_angle
                latest_right_angle = right_angle
                latest_left_gyro = left_gyro
                latest_right_gyro = right_gyro

                # Teensy feedback has no device timestamp, so use the newest
                # feedback that had already arrived by this paired IMU timestamp.
                feedback = teensy.latest_before(pair_host_time)
                if feedback is None:
                    left_actual = right_actual = math.nan
                else:
                    left_actual = feedback.left_actual_nm
                    right_actual = feedback.right_actual_nm

                row = {
                    "time_s": time_s,
                    "left_angle_x_deg": left_angle,
                    "left_angular_velocity_x_dps": left_gyro,
                    "right_angle_x_deg": right_angle,
                    "right_angular_velocity_x_dps": right_gyro,
                    "left_cmd_torque_nm": cmd_left,
                    "right_cmd_torque_nm": cmd_right,
                    "left_actual_torque_nm": left_actual,
                    "right_actual_torque_nm": right_actual,
                }
                logger.submit(row)

                if plot_queue is not None:
                    push_plot_sample(
                        plot_queue,
                        (
                            time_s,
                            left_angle,
                            right_angle,
                            left_gyro,
                            right_gyro,
                            cmd_left,
                            cmd_right,
                            left_actual,
                            right_actual,
                            controller.cfg.rescaling,
                            controller.cfg.delay_index,
                        ),
                    )

                pair_index += 1

            now = time.perf_counter()

            # No catch-up TX burst: if Windows stalls, send only the latest
            # reconstructed controller state once when execution resumes.
            if now >= next_tx:
                latest_feedback, _ = teensy.snapshot()
                imu_fresh = now - last_pair_host_time <= a.imu_timeout
                teensy_fresh = (
                    latest_feedback is not None
                    and now - latest_feedback.host_time <= a.teensy_timeout
                )
                enabled = bool(a.enable and imu_fresh and teensy_fresh)
                latest_output_enabled = enabled

                if enabled:
                    tx_left = latest_desired_left
                    tx_right = latest_desired_right
                else:
                    tx_left = 0.0
                    tx_right = 0.0

                teensy.send_torque(tx_left, tx_right, enabled)

                # Keep periodic phase when possible, but never send catch-up bursts.
                next_tx += tx_period
                if next_tx <= now:
                    next_tx = now + tx_period

            if a.display == "print" and now >= next_print:
                l_latest, l_stats = left_imu.snapshot()
                r_latest, r_stats = right_imu.snapshot()
                feedback, t_stats = teensy.snapshot()
                l_age = (now - l_latest.host_time) * 1000.0 if l_latest else math.inf
                r_age = (now - r_latest.host_time) * 1000.0 if r_latest else math.inf
                f_age = (now - feedback.host_time) * 1000.0 if feedback else math.inf
                actual_text = (
                    f"{feedback.left_actual_nm:+6.2f}/{feedback.right_actual_nm:+6.2f}Nm"
                    if feedback is not None
                    else "  n/a /   n/a Nm"
                )
                max_imu_age_ms = max(l_age, r_age)
                if max_imu_age_ms <= a.stale_warning * 1000.0:
                    imu_status = "OK"
                elif max_imu_age_ms <= a.imu_timeout * 1000.0:
                    imu_status = "STALE"
                else:
                    imu_status = "TIMEOUT"
                output_text = "ON" if latest_output_enabled else ("SAFE" if not a.enable else "BLOCKED")
                print(
                    f"[RUN] pair={pair_index:6d} | "
                    f"L {latest_left_angle:+7.2f}deg {latest_left_gyro:+8.1f}dps "
                    f"R {latest_right_angle:+7.2f}deg {latest_right_gyro:+8.1f}dps | "
                    f"cmd {latest_desired_left:+6.2f}/{latest_desired_right:+6.2f}Nm | "
                    f"actual {actual_text} | output={output_text} imu={imu_status} | "
                    f"Hz L/R/T={l_stats.hz:5.1f}/{r_stats.hz:5.1f}/{t_stats.hz:5.1f} | "
                    f"age={l_age:5.1f}/{r_age:5.1f}/{f_age:5.1f}ms | "
                    f"pairDrop={pairer.left_unmatched}/{pairer.right_unmatched} "
                    f"logQ={logger.q.qsize()} drop={logger.dropped_rows}"
                )
                next_print += print_period
                if next_print <= now:
                    next_print = now + print_period

            if left_imu.error:
                raise RuntimeError(f"LEFT IMU: {left_imu.error}")
            if right_imu.error:
                raise RuntimeError(f"RIGHT IMU: {right_imu.error}")
            if teensy.error:
                raise RuntimeError(f"Teensy: {teensy.error}")
            if logger.error:
                raise RuntimeError(f"Logger: {logger.error}")

            time.sleep(0.0005)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C")
    finally:
        stop_event.set()
        try:
            for _ in range(3):
                teensy.send_stop()
                time.sleep(0.01)
        except Exception:
            pass

        if plot_queue is not None:
            try:
                plot_queue.put_nowait(None)
            except Exception:
                pass
        if plot_close_event is not None:
            plot_close_event.set()

        logger.close()
        left_imu.join(timeout=1.0)
        right_imu.join(timeout=1.0)
        teensy.join(timeout=1.0)
        if plot_process is not None:
            plot_process.join(timeout=1.0)
            if plot_process.is_alive():
                plot_process.terminate()

        print("=" * 106)
        print(f"[DONE] paired rows       : {pair_index}")
        print(f"[DONE] pairing unmatched : L={pairer.left_unmatched}, R={pairer.right_unmatched}")
        print(f"[DONE] logger dropped    : {logger.dropped_rows}")
        print(f"[DONE] CSV               : {csv_path}")
        if logger.error:
            print(f"[LOGGER ERROR] {logger.error}")
        print("=" * 106)


if __name__ == "__main__":
    mp.freeze_support()
    main()
