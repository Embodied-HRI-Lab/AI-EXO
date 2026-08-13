"""
PC <-> Teensy communication for bilateral hip exoskeleton.

Current UART protocol
---------------------
UART:
    /dev/serial0
    115200 baud
    8N1

Raspberry Pi -> Teensy
    Torque command (CMD 0x54), 15 bytes:
        A5 5A
        54
        sequence            uint16
        left_torque         float32   Nm
        right_torque        float32   Nm
        enable              uint8
        crc8                uint8

    STOP:
        A5 5A 50 crc8

    CLEAR_FAULT:
        A5 5A 43 crc8

Teensy -> Raspberry Pi
    High-rate state (CMD 0x44), 30 bytes, nominal 200 Hz:
        A5 5A
        44
        state_sequence      uint16
        left_tau_actual     float32   Nm
        right_tau_actual    float32   Nm
        left_imu_angle_z    float32   rad
        left_imu_gyro_z     float32   rad/s
        right_imu_angle_z   float32   rad
        right_imu_gyro_z    float32   rad/s
        crc8                uint8

    Low-rate status (CMD 0x53), 10 bytes, nominal 20 Hz:
        A5 5A
        53
        last_cmd_sequence   uint16
        status_flags        uint32
        crc8                uint8

Design goal
-----------
The serial communication layer is isolated in TeensyExoLink.

For future learning control, normally you only need to replace:

    controller(state)

with your learned policy.

The policy receives an ExoState and returns:
    left_torque_Nm, right_torque_Nm

or:
    left_torque_Nm, right_torque_Nm, enable

No serial parsing code needs to change.
"""

from __future__ import annotations

import argparse
import math
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import serial

from samsung_controller import (
    SamsungAssistConfig,
    SamsungAssistController,
)


# ============================================================
# 1. Protocol constants
# ============================================================

DEFAULT_PORT = "/dev/serial0"
DEFAULT_BAUD = 115200

HEADER = b"\xA5\x5A"

CMD_TORQUE = 0x54
CMD_STOP = 0x50
CMD_CLEAR_FAULT = 0x43
CMD_STATE = 0x44
CMD_STATUS = 0x53

# Little-endian formats. "<" also disables native padding.
TORQUE_PAYLOAD_STRUCT = struct.Struct("<HffB")
STATE_PAYLOAD_STRUCT = struct.Struct("<Hffffff")
STATUS_PAYLOAD_STRUCT = struct.Struct("<HI")

TORQUE_FRAME_SIZE = 2 + 1 + TORQUE_PAYLOAD_STRUCT.size + 1   # 15
STATE_FRAME_SIZE = 2 + 1 + STATE_PAYLOAD_STRUCT.size + 1     # 30
STATUS_FRAME_SIZE = 2 + 1 + STATUS_PAYLOAD_STRUCT.size + 1   # 10

FRAME_SIZE_BY_CMD = {
    CMD_STATE: STATE_FRAME_SIZE,
    CMD_STATUS: STATUS_FRAME_SIZE,
}


# ============================================================
# 2. Status flags -- must match Teensy
# ============================================================

STATUS_COMMAND_ENABLED = 1 << 0
STATUS_COMMAND_TIMEOUT = 1 << 1
STATUS_LEFT_FEEDBACK_TIMEOUT = 1 << 2
STATUS_RIGHT_FEEDBACK_TIMEOUT = 1 << 3
STATUS_LEFT_POS_LIMIT = 1 << 4
STATUS_RIGHT_POS_LIMIT = 1 << 5
STATUS_LEFT_VEL_LIMIT = 1 << 6
STATUS_RIGHT_VEL_LIMIT = 1 << 7
STATUS_BAD_PACKET = 1 << 8
STATUS_BAD_TORQUE_VALUE = 1 << 9
STATUS_LATCHED_FAULT = 1 << 10
STATUS_IMU_VALID = 1 << 11
STATUS_LEFT_DRIVE_FAULT = 1 << 12
STATUS_RIGHT_DRIVE_FAULT = 1 << 13
STATUS_MOTOR_PARAMETERS_INVALID = 1 << 14
STATUS_CAN_TX_FAIL = 1 << 15
STATUS_UART_TX_OVERFLOW = 1 << 16

STATUS_FLAG_NAMES = {
    STATUS_COMMAND_ENABLED: "COMMAND_ENABLED",
    STATUS_COMMAND_TIMEOUT: "COMMAND_TIMEOUT",
    STATUS_LEFT_FEEDBACK_TIMEOUT: "LEFT_FEEDBACK_TIMEOUT",
    STATUS_RIGHT_FEEDBACK_TIMEOUT: "RIGHT_FEEDBACK_TIMEOUT",
    STATUS_LEFT_POS_LIMIT: "LEFT_POS_LIMIT",
    STATUS_RIGHT_POS_LIMIT: "RIGHT_POS_LIMIT",
    STATUS_LEFT_VEL_LIMIT: "LEFT_VEL_LIMIT",
    STATUS_RIGHT_VEL_LIMIT: "RIGHT_VEL_LIMIT",
    STATUS_BAD_PACKET: "BAD_PACKET",
    STATUS_BAD_TORQUE_VALUE: "BAD_TORQUE_VALUE",
    STATUS_LATCHED_FAULT: "LATCHED_FAULT",
    STATUS_IMU_VALID: "IMU_VALID",
    STATUS_LEFT_DRIVE_FAULT: "LEFT_DRIVE_FAULT",
    STATUS_RIGHT_DRIVE_FAULT: "RIGHT_DRIVE_FAULT",
    STATUS_MOTOR_PARAMETERS_INVALID: "MOTOR_PARAMETERS_INVALID",
    STATUS_CAN_TX_FAIL: "CAN_TX_FAIL",
    STATUS_UART_TX_OVERFLOW: "UART_TX_OVERFLOW",
}


# ============================================================
# 3. Data objects exposed to the learning/controller layer
# ============================================================

@dataclass(frozen=True)
class ExoState:
    """One valid 200 Hz state packet from Teensy."""

    sequence: int

    left_tau_actual: float
    right_tau_actual: float

    left_imu_angle_z: float
    left_imu_gyro_z: float

    right_imu_angle_z: float
    right_imu_gyro_z: float

    # Raspberry Pi local monotonic timestamp at packet reception.
    rx_time: float

    def as_vector(self) -> Tuple[float, float, float, float, float, float]:
        """
        Observation vector for a future learning algorithm.

        Order is fixed:
            [left_tau_actual,
             right_tau_actual,
             left_imu_angle_z,
             left_imu_gyro_z,
             right_imu_angle_z,
             right_imu_gyro_z]
        """
        return (
            self.left_tau_actual,
            self.right_tau_actual,
            self.left_imu_angle_z,
            self.left_imu_gyro_z,
            self.right_imu_angle_z,
            self.right_imu_gyro_z,
        )


@dataclass(frozen=True)
class ExoStatus:
    """One valid 20 Hz status packet from Teensy."""

    last_command_sequence: int
    flags: int
    rx_time: float

    @property
    def command_enabled(self) -> bool:
        return bool(self.flags & STATUS_COMMAND_ENABLED)

    @property
    def imu_valid(self) -> bool:
        return bool(self.flags & STATUS_IMU_VALID)

    @property
    def latched_fault(self) -> bool:
        return bool(self.flags & STATUS_LATCHED_FAULT)

    def active_names(self) -> Tuple[str, ...]:
        names = [
            name
            for bit, name in STATUS_FLAG_NAMES.items()
            if self.flags & bit
        ]
        return tuple(names) if names else ("NONE",)


# ============================================================
# 4. CRC-8
# ============================================================

def crc8_update(crc: int, data: int) -> int:
    """
    CRC-8:
        polynomial = 0x07
        init       = 0x00

    This exactly matches the Teensy implementation.
    """
    crc ^= data

    for _ in range(8):
        if crc & 0x80:
            crc = ((crc << 1) ^ 0x07) & 0xFF
        else:
            crc = (crc << 1) & 0xFF

    return crc


def crc8_compute(cmd: int, payload: bytes = b"") -> int:
    # CRC covers CMD + PAYLOAD, excluding A5 5A.
    crc = crc8_update(0x00, cmd)

    for byte in payload:
        crc = crc8_update(crc, byte)

    return crc


# ============================================================
# 5. Raspberry Pi <-> Teensy communication class
# ============================================================

class TeensyExoLink:
    """
    Thread-safe serial interface.

    Receiving:
        A background thread continuously reads and parses Teensy packets.

    Control:
        Any high-level controller can simply call:
            link.send_torque(left_nm, right_nm)

    Observation:
        The latest valid state is available through:
            link.get_latest_state()

    This separation is intentional so a future neural-network/RL controller
    does not need to know anything about packet framing or CRC.
    """

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baud: int = DEFAULT_BAUD,
        serial_timeout_s: float = 0.01,
    ) -> None:
        self.port = port
        self.baud = baud

        self.ser = serial.Serial(
            port=port,
            baudrate=baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=serial_timeout_s,
            write_timeout=0.05,
        )

        # Drop a possible partial packet left in the Linux UART buffer.
        self.ser.reset_input_buffer()

        self._rx_buffer = bytearray()

        self._data_lock = threading.Lock()
        self._tx_lock = threading.Lock()

        self._latest_state: Optional[ExoState] = None
        self._latest_status: Optional[ExoStatus] = None

        self._command_sequence = 0
        self._last_state_sequence: Optional[int] = None

        self.state_packets = 0
        self.status_packets = 0
        self.state_packets_dropped = 0
        self.crc_errors = 0
        self.unknown_frames = 0

        self._running = threading.Event()
        self._running.set()

        self._reader_exception: Optional[BaseException] = None

        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="TeensySerialReader",
            daemon=True,
        )
        self._reader_thread.start()

    # --------------------------------------------------------
    # Context-manager support
    # --------------------------------------------------------

    def __enter__(self) -> "TeensyExoLink":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # --------------------------------------------------------
    # TX: high-level controller -> Teensy
    # --------------------------------------------------------

    def send_torque(
        self,
        left_torque_nm: float,
        right_torque_nm: float,
        enable: bool = True,
    ) -> int:
        """
        Send one left/right torque command.

        Returns:
            The uint16 sequence number used in this command.

        Important:
            This function intentionally does NOT impose a high-level torque
            limit. Teensy remains the final safety layer and applies its own
            validation, slew limit, LPF, torque limit, current limit, etc.

            A learning policy may optionally apply its own tighter limit before
            calling this function.
        """

        left = float(left_torque_nm)
        right = float(right_torque_nm)

        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("Torque command must be finite.")

        sequence = self._command_sequence
        self._command_sequence = (self._command_sequence + 1) & 0xFFFF

        payload = TORQUE_PAYLOAD_STRUCT.pack(
            sequence,
            left,
            right,
            1 if enable else 0,
        )

        frame = self._make_frame(CMD_TORQUE, payload)
        self._write_frame(frame)

        return sequence

    def stop(self) -> None:
        """Immediately request command disable on Teensy."""
        self._write_frame(self._make_frame(CMD_STOP))

    def clear_fault(self) -> None:
        """Ask Teensy to clear its latched fault if safe conditions allow it."""
        self._write_frame(self._make_frame(CMD_CLEAR_FAULT))

    # --------------------------------------------------------
    # RX data exposed to the controller / learning algorithm
    # --------------------------------------------------------

    def get_latest_state(self) -> Optional[ExoState]:
        with self._data_lock:
            return self._latest_state

    def get_latest_status(self) -> Optional[ExoStatus]:
        with self._data_lock:
            return self._latest_status

    def wait_for_state(self, timeout_s: float = 2.0) -> Optional[ExoState]:
        """Wait for the first valid state packet."""
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            self.check_reader()
            state = self.get_latest_state()
            if state is not None:
                return state
            time.sleep(0.002)

        return None

    def check_reader(self) -> None:
        """Raise if the background serial-reader thread has failed."""
        if self._reader_exception is not None:
            raise RuntimeError(
                "Teensy serial reader stopped unexpectedly."
            ) from self._reader_exception

    # --------------------------------------------------------
    # Close
    # --------------------------------------------------------

    def close(self) -> None:
        if not self._running.is_set():
            return

        # Explicitly stop the actuator command before closing.
        try:
            for _ in range(3):
                self.stop()
                time.sleep(0.01)
        except Exception:
            pass

        self._running.clear()

        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=0.5)

        if self.ser.is_open:
            self.ser.close()

    # --------------------------------------------------------
    # Internal TX helpers
    # --------------------------------------------------------

    @staticmethod
    def _make_frame(cmd: int, payload: bytes = b"") -> bytes:
        crc = crc8_compute(cmd, payload)
        return HEADER + bytes((cmd,)) + payload + bytes((crc,))

    def _write_frame(self, frame: bytes) -> None:
        with self._tx_lock:
            self.check_reader()
            written = self.ser.write(frame)

            if written != len(frame):
                raise IOError(
                    f"UART short write: {written}/{len(frame)} bytes"
                )

    # --------------------------------------------------------
    # Background RX
    # --------------------------------------------------------

    def _reader_loop(self) -> None:
        try:
            while self._running.is_set():
                # Read whatever is currently available.
                # If nothing is available, read(1) blocks only up to timeout.
                n = self.ser.in_waiting
                data = self.ser.read(n if n > 0 else 1)

                if data:
                    self._rx_buffer.extend(data)
                    self._parse_rx_buffer()

        except BaseException as exc:
            self._reader_exception = exc
            self._running.clear()

    def _parse_rx_buffer(self) -> None:
        """
        Parse a mixed stream containing 0x44 state and 0x53 status packets.

        Parser behavior:
            - searches for A5 5A
            - determines length from CMD
            - validates CRC
            - on corruption, advances one byte and resynchronizes
        """

        while True:
            # Need at least header + cmd.
            if len(self._rx_buffer) < 3:
                return

            header_index = self._rx_buffer.find(HEADER)

            if header_index < 0:
                # Preserve a trailing A5 because it might be the first byte
                # of a header split across serial reads.
                if self._rx_buffer and self._rx_buffer[-1] == HEADER[0]:
                    self._rx_buffer[:] = self._rx_buffer[-1:]
                else:
                    self._rx_buffer.clear()
                return

            if header_index > 0:
                del self._rx_buffer[:header_index]

            if len(self._rx_buffer) < 3:
                return

            cmd = self._rx_buffer[2]
            frame_size = FRAME_SIZE_BY_CMD.get(cmd)

            if frame_size is None:
                self.unknown_frames += 1
                # Drop only the first byte so the parser can quickly
                # resynchronize if another A5 follows.
                del self._rx_buffer[0]
                continue

            if len(self._rx_buffer) < frame_size:
                return

            frame = bytes(self._rx_buffer[:frame_size])
            payload = frame[3:-1]
            received_crc = frame[-1]
            expected_crc = crc8_compute(cmd, payload)

            if received_crc != expected_crc:
                self.crc_errors += 1
                del self._rx_buffer[0]
                continue

            # Frame is valid. Remove it before processing.
            del self._rx_buffer[:frame_size]

            if cmd == CMD_STATE:
                self._process_state(payload)
            elif cmd == CMD_STATUS:
                self._process_status(payload)

    def _process_state(self, payload: bytes) -> None:
        (
            sequence,
            left_tau_actual,
            right_tau_actual,
            left_imu_angle_z,
            left_imu_gyro_z,
            right_imu_angle_z,
            right_imu_gyro_z,
        ) = STATE_PAYLOAD_STRUCT.unpack(payload)

        # Detect missing telemetry packets using uint16 wraparound.
        if self._last_state_sequence is not None:
            expected = (self._last_state_sequence + 1) & 0xFFFF

            if sequence != expected:
                gap = (sequence - expected) & 0xFFFF

                # Ignore a large backwards jump, which is more likely to mean
                # Teensy restarted than that ~65535 packets were lost.
                if gap < 0x8000:
                    self.state_packets_dropped += gap

        self._last_state_sequence = sequence

        state = ExoState(
            sequence=sequence,
            left_tau_actual=left_tau_actual,
            right_tau_actual=right_tau_actual,
            left_imu_angle_z=left_imu_angle_z,
            left_imu_gyro_z=left_imu_gyro_z,
            right_imu_angle_z=right_imu_angle_z,
            right_imu_gyro_z=right_imu_gyro_z,
            rx_time=time.monotonic(),
        )

        with self._data_lock:
            self._latest_state = state

        self.state_packets += 1

    def _process_status(self, payload: bytes) -> None:
        last_command_sequence, flags = STATUS_PAYLOAD_STRUCT.unpack(payload)

        status = ExoStatus(
            last_command_sequence=last_command_sequence,
            flags=flags,
            rx_time=time.monotonic(),
        )

        with self._data_lock:
            self._latest_status = status

        self.status_packets += 1


# ============================================================
# 6. Controller interface
# ============================================================

ControllerReturn = Union[
    Tuple[float, float],
    Tuple[float, float, bool],
]

Controller = Callable[[ExoState], ControllerReturn]


_samsung_controller = SamsungAssistController(
    SamsungAssistConfig()
)


def configure_controller(config: SamsungAssistConfig) -> None:
    """
    Configure the stateful Samsung assistance controller.

    This function changes only the high-level policy. The existing UART,
    CRC, packet parser, receiver thread, and run_control_loop remain intact.
    """
    global _samsung_controller  
    _samsung_controller = SamsungAssistController(config)   


def controller(state: ExoState) -> ControllerReturn:
    """
    Existing high-level controller(state) interface.

    Input:
        the same ExoState already exposed by this program

    Output:
        left_desired_torque_Nm,
        right_desired_torque_Nm, 
        enable 

    The actual Samsung assistance algorithm is implemented in the small
    helper file samsung_controller.py.
    """
    return _samsung_controller(state) 


def controller_debug_text() -> str:
    return _samsung_controller.debug_text()


# Attach an optional debug method to preserve run_control_loop's generic
# callable-policy interface. 
controller.debug_text = controller_debug_text  # type: ignore[attr-defined]


# ============================================================
# 7. Example real-time control loop
# ============================================================

def run_control_loop(
    link: TeensyExoLink,
    policy: Controller,
    control_rate_hz: float = 200.0,
    print_rate_hz: float = 10.0,
    max_state_age_s: float = 0.05,
) -> None:
    """
    Generic high-level loop.

    Every control tick:
        1) read the latest Teensy state
        2) call policy(state)
        3) send left/right desired torque

    The policy can later be a neural network, RL policy, adaptive controller,
    etc. The serial layer does not need to change.

    max_state_age_s:
        If Pi has not received a fresh Teensy state within this time,
        disable torque commands instead of controlling from stale data.
    """

    if control_rate_hz <= 0:
        raise ValueError("control_rate_hz must be > 0")

    control_period = 1.0 / control_rate_hz
    print_period = 1.0 / max(print_rate_hz, 0.1)

    next_control = time.perf_counter()
    next_print = next_control

    commanded_left = 0.0
    commanded_right = 0.0
    commanded_enable = False

    print(
        f"Control loop started: {control_rate_hz:.1f} Hz\n"
        "Press Ctrl+C to stop."
    )

    try:
        while True:
            link.check_reader()

            now_perf = time.perf_counter()

            # ------------------------------------------------
            # High-level control tick
            # ------------------------------------------------
            if now_perf >= next_control:
                state = link.get_latest_state()
                now_mono = time.monotonic()

                if (
                    state is None
                    or (now_mono - state.rx_time) > max_state_age_s
                ):
                    # No fresh sensor data -> disable actuation.
                    commanded_left = 0.0
                    commanded_right = 0.0
                    commanded_enable = False

                else:
                    result = policy(state)

                    if len(result) == 2:
                        commanded_left, commanded_right = result
                        commanded_enable = True
                    elif len(result) == 3:
                        (
                            commanded_left,
                            commanded_right,
                            commanded_enable,
                        ) = result
                    else:
                        raise ValueError(
                            "controller(state) must return "
                            "(left_tau, right_tau) or "
                            "(left_tau, right_tau, enable)"
                        )

                    commanded_left = float(commanded_left)
                    commanded_right = float(commanded_right)
                    commanded_enable = bool(commanded_enable)

                link.send_torque(
                    commanded_left,
                    commanded_right,
                    enable=commanded_enable,
                )

                # Keep a stable target cadence while also recovering cleanly
                # if Python/policy execution temporarily overruns.
                next_control += control_period

                if now_perf - next_control > control_period:
                    next_control = now_perf + control_period

            # ------------------------------------------------
            # Human-readable console output at low frequency.
            # Never print at 200 Hz.
            # ------------------------------------------------
            if now_perf >= next_print:
                state = link.get_latest_state()
                status = link.get_latest_status()

                if state is None:
                    print("Waiting for Teensy state...")
                else:
                    age_ms = (time.monotonic() - state.rx_time) * 1000.0

                    status_text = (
                        "|".join(status.active_names())
                        if status is not None
                        else "NO_STATUS_YET"
                    )

                    debug_method = getattr(
                        policy,
                        "debug_text",
                        None,
                    )
                    policy_debug = (
                        f" | {debug_method()}"
                        if callable(debug_method)
                        else ""
                    )

                    print(
                        f"seq={state.sequence:5d} "
                        f"age={age_ms:5.1f} ms | "
                        f"tau L={state.left_tau_actual:+.3f} "
                        f"R={state.right_tau_actual:+.3f} Nm | "
                        f"IMU L=({state.left_imu_angle_z:+.3f} rad, "
                        f"{state.left_imu_gyro_z:+.3f} rad/s) "
                        f"R=({state.right_imu_angle_z:+.3f} rad, "
                        f"{state.right_imu_gyro_z:+.3f} rad/s) | "
                        f"cmd L={commanded_left:+.3f} "
                        f"R={commanded_right:+.3f} Nm | "
                        f"{status_text} | "
                        f"drop={link.state_packets_dropped} "
                        f"crc={link.crc_errors}"
                        f"{policy_debug}"
                    )

                next_print = now_perf + print_period

            # Short sleep only to avoid spinning one CPU core at 100%.
            # Timing is still governed by perf_counter above.
            sleep_s = next_control - time.perf_counter()

            if sleep_s > 0.0005:
                time.sleep(min(sleep_s * 0.5, 0.001))

    except KeyboardInterrupt:
        print("\nCtrl+C received. Sending STOP...")

    finally:
        # Redundant STOP packets are deliberate.
        for _ in range(3):
            try:
                link.stop()
            except Exception:
                pass
            time.sleep(0.01)


# ============================================================
# 8. Simple manual constant-torque test
# ============================================================

def make_constant_torque_controller(
    left_nm: float,
    right_nm: float,
) -> Controller:
    """
    Useful before integrating the learning algorithm.

    Example:
        --left 0.05 --right 0.05

    Start with very small torque and no human wearer.
    """

    def _constant_controller(state: ExoState) -> ControllerReturn:
        del state
        return left_nm, right_nm

    return _constant_controller


# ============================================================
# 9. Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PC <-> Teensy exoskeleton UART controller"
    )

    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help=f"UART device (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"UART baud rate (default: {DEFAULT_BAUD})",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=200.0,
        help="Pi -> Teensy command rate in Hz (default: 200)",
    )
    parser.add_argument(
        "--left",
        type=float,
        default=None,
        help="Manual constant left torque in Nm",
    )
    parser.add_argument(
        "--right",
        type=float,
        default=None,
        help="Manual constant right torque in Nm",
    )
    parser.add_argument(
        "--clear-fault",
        action="store_true",
        help="Send CLEAR_FAULT once after opening the UART",
    )

    # --------------------------------------------------------
    # Samsung assistance-controller parameters
    # --------------------------------------------------------
    parser.add_argument(
        "--arm",
        action="store_true",
        help=(
            "Send the calculated nonzero Samsung torque. "
            "Without --arm, calculate only and send zero/disabled."
        ),
    )
    parser.add_argument(
        "--rescaling",
        type=float,
        default=5.0,
        help="Original Rescaling_gain (safe default: 0.0)",
    )
    parser.add_argument(
        "--flex-gain",
        type=float,
        default=1.0,
        help="Original Flex_Assist_gain (default: 1.0)",
    )
    parser.add_argument(
        "--ext-gain",
        type=float,
        default=1.0,
        help="Original Ext_Assist_gain (default: 1.0)",
    )
    parser.add_argument(
        "--delay-samples",
        type=int,
        default=0,
        help=(
            "Original Assist_delay_gain in 28 Hz controller samples "
            "(0-99, default: 0)"
        ),
    )
    parser.add_argument(
        "--calibration",
        type=float,
        default=2.0,
        help="Startup standing-still calibration time (default: 2 s)",
    )
    parser.add_argument(
        "--max-torque",
        type=float,
        default=5.0,
        help=(
            "High-level two-side safety threshold in Nm "
            "(must be <= 0.25, default: 0.20)"
        ),
    )
    parser.add_argument(
        "--left-angle-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
    )
    parser.add_argument(
        "--right-angle-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help=(
            "Keep +1 because the tested right leg angle increases "
            "during leg raising"
        ),
    )
    parser.add_argument(
        "--left-torque-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
    )
    parser.add_argument(
        "--right-torque-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
    )
    parser.add_argument(
        "--legacy-hold-out-of-range",
        action="store_true",
        help=(
            "Hold the previous torque when |R-L| >= 120 deg, "
            "matching the original commented else behavior. "
            "Default is safer zero."
        ),
    )

    args = parser.parse_args()

    configure_controller(
        SamsungAssistConfig(
            controller_rate_hz=28.0,
            calibration_s=args.calibration,
            assist_delay_samples=args.delay_samples,
            rescaling_gain=args.rescaling,
            flex_assist_gain=args.flex_gain,
            ext_assist_gain=args.ext_gain,
            max_torque_nm=args.max_torque,
            left_angle_sign=args.left_angle_sign,
            right_angle_sign=args.right_angle_sign,
            left_torque_sign=args.left_torque_sign,
            right_torque_sign=args.right_torque_sign,
            armed=args.arm,
            legacy_hold_out_of_range=(
                args.legacy_hold_out_of_range
            ),
        )
    )

    print(f"Opening {args.port} @ {args.baud} baud")

    with TeensyExoLink(
        port=args.port,
        baud=args.baud,
    ) as link:

        if args.clear_fault:
            print("Sending CLEAR_FAULT")
            link.clear_fault()
            time.sleep(0.1)

        print("Waiting for first valid Teensy state packet...")
        first_state = link.wait_for_state(timeout_s=3.0)

        if first_state is None:
            raise RuntimeError(
                "No valid Teensy state received within 3 seconds. "
                "Check TX/RX crossing, common GND, baud rate, "
                "Teensy firmware, and /dev/serial0."
            )   

        print(
            "First valid state received:\n"
            f"  seq          = {first_state.sequence}\n"
            f"  tau L/R      = {first_state.left_tau_actual:+.3f}, "
            f"{first_state.right_tau_actual:+.3f} Nm\n"
            f"  IMU L z/wz   = {first_state.left_imu_angle_z:+.3f} rad, "
            f"{first_state.left_imu_gyro_z:+.3f} rad/s\n"
            f"  IMU R z/wz   = {first_state.right_imu_angle_z:+.3f} rad, "
            f"{first_state.right_imu_gyro_z:+.3f} rad/s"
        )

        # ----------------------------------------------------
        # Two operating modes:
        #
        # 1) No --left/--right:
        #       use controller(state)
        #       -> this is where the future learning policy goes.
        #
        # 2) --left/--right provided:
        #       simple constant-torque bench test.
        # ----------------------------------------------------
        if args.left is None and args.right is None:
            active_controller = controller
            print(
                "Using controller(state) interface with the "
                "Samsung assistance policy."
            )
            print(
                f"  Pi -> Teensy command rate: {args.rate:.1f} Hz\n"
                "  Samsung internal update:    28.0 Hz\n"
                f"  arm:                        {args.arm}\n"
                f"  rescaling/flex/ext:         "
                f"{args.rescaling}, {args.flex_gain}, "
                f"{args.ext_gain}\n"
                f"  delay samples:              "
                f"{args.delay_samples}\n"
                f"  high-level max torque:      "
                f"{args.max_torque:.3f} Nm"
            )

        else:
            left_nm = 0.0 if args.left is None else args.left
            right_nm = 0.0 if args.right is None else args.right

            active_controller = make_constant_torque_controller(
                left_nm,
                right_nm,
            )

            print(
                "Using constant-torque test controller: "
                f"L={left_nm:+.3f} Nm, R={right_nm:+.3f} Nm"
            )

        run_control_loop(
            link=link,
            policy=active_controller,
            control_rate_hz=args.rate,
        )


if __name__ == "__main__":
    main()