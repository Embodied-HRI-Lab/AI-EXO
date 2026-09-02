from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# =============================================================================
# TCN MODEL
# =============================================================================


def get_activation(name: str) -> nn.Module:
    name = str(name).lower()

    if name == "silu":
        return nn.SiLU()

    if name == "relu":
        return nn.ReLU()

    raise ValueError(
        f"unsupported activation: {name}"
    )


class CausalConv1d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:

        super().__init__()

        self.left_padding = (
            kernel_size - 1
        ) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.conv(
            F.pad(
                x,
                (
                    self.left_padding,
                    0,
                ),
            )
        )


class TCNResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        activation: str,
    ) -> None:

        super().__init__()

        self.conv1 = CausalConv1d(
            in_channels,
            hidden_channels,
            kernel_size,
            dilation,
        )

        self.conv2 = CausalConv1d(
            hidden_channels,
            hidden_channels,
            kernel_size,
            dilation,
        )

        self.activation1 = get_activation(
            activation
        )

        self.activation2 = get_activation(
            activation
        )

        self.output_activation = get_activation(
            activation
        )

        self.dropout1 = nn.Dropout(
            dropout
        )

        self.dropout2 = nn.Dropout(
            dropout
        )

        self.residual = (
            nn.Identity()
            if in_channels == hidden_channels
            else nn.Conv1d(
                in_channels,
                hidden_channels,
                kernel_size=1,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = self.residual(
            x
        )

        y = self.dropout1(
            self.activation1(
                self.conv1(
                    x
                )
            )
        )

        y = self.dropout2(
            self.activation2(
                self.conv2(
                    y
                )
            )
        )

        return self.output_activation(
            residual + y
        )


class CausalTCN(nn.Module):

    def __init__(
        self,
        input_channels: int = 4,
        hidden_channels: int = 32,
        output_channels: int = 2,
        kernel_size: int = 4,
        dilations: list[int] | tuple[int, ...] = (
            1,
            2,
            4,
            8,
        ),
        dropout: float = 0.1,
        activation: str = "silu",
        receptive_field_samples: int | None = None,
    ) -> None:

        super().__init__()

        blocks: list[nn.Module] = []

        in_ch = int(
            input_channels
        )

        for dilation in dilations:

            blocks.append(
                TCNResidualBlock(
                    in_ch,
                    int(hidden_channels),
                    int(kernel_size),
                    int(dilation),
                    float(dropout),
                    activation,
                )
            )

            in_ch = int(
                hidden_channels
            )

        self.blocks = nn.ModuleList(
            blocks
        )

        self.head = nn.Linear(
            int(hidden_channels),
            int(output_channels),
        )

        self.output_activation = nn.Tanh()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                f"expected x=[B,C,T], "
                f"got {tuple(x.shape)}"
            )

        y = x

        for block in self.blocks:
            y = block(
                y
            )

        return self.output_activation(
            self.head(
                y[:, :, -1]
            )
        )


# =============================================================================
# TCN POLICY
# =============================================================================


class _RawUnifiedTCNPolicy:
    """
    Stateful causal-TCN policy.

    Expected input order:

        1. left thigh angle [rad]
        2. left thigh angular velocity [rad/s]
        3. right thigh angle [rad]
        4. right thigh angular velocity [rad/s]

    Output:

        action:
            normalized TCN output in [-1, +1]

        command_nm:
            raw TCN torque command after checkpoint torque scaling

    Notes
    -----
    This class contains NO hardware logic.

    It can therefore be used both by:

        real-time exoskeleton controller

    and:

        offline CSV replay
    """

    def __init__(
        self,
        model_path: Path | str,
        *,
        device: str = "cpu",
    ) -> None:

        self.model_path = (
            Path(
                model_path
            )
            .expanduser()
            .resolve()
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"model checkpoint not found: "
                f"{self.model_path}"
            )

        self.device = torch.device(
            device
        )

        # -------------------------------------------------------------
        # Load deployment checkpoint
        # -------------------------------------------------------------

        self.payload = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )

        # -------------------------------------------------------------
        # Build TCN
        # -------------------------------------------------------------

        self.model = CausalTCN(
            **dict(
                self.payload[
                    "model_config"
                ]
            )
        ).to(
            self.device
        )

        self.model.load_state_dict(
            self.payload[
                "state_dict"
            ]
        )

        self.model.eval()

        # -------------------------------------------------------------
        # Deployment metadata
        # -------------------------------------------------------------

        self.history_steps = int(
            self.payload[
                "history_steps"
            ]
        )

        self.sensor_hz = int(
            self.payload[
                "sensor_hz"
            ]
        )

        self.control_hz = int(
            self.payload.get(
                "control_hz",
                self.sensor_hz,
            )
        )

        self.input_channel_names = list(
            self.payload[
                "input_channel_names"
            ]
        )

        input_channels = len(
            self.input_channel_names
        )

        # -------------------------------------------------------------
        # Normalization
        # -------------------------------------------------------------

        self.input_mean = torch.tensor(
            self.payload[
                "input_mean"
            ],
            dtype=torch.float32,
            device=self.device,
        ).view(
            1,
            input_channels,
            1,
        )

        std = torch.tensor(
            self.payload[
                "input_std"
            ],
            dtype=torch.float32,
            device=self.device,
        ).view(
            1,
            input_channels,
            1,
        )

        if torch.any(
            std <= 0
        ):
            raise ValueError(
                "input_std contains "
                "non-positive values"
            )

        self.input_std = torch.clamp(
            std,
            min=1e-8,
        )

        self.normalization_scheme = str(
            self.payload[
                "normalization_scheme"
            ]
        )

        # -------------------------------------------------------------
        # Torque scaling stored in checkpoint
        # -------------------------------------------------------------

        self.torque_scale_nm = float(
            self.payload[
                "torque_scale_nm"
            ]
        )

        # -------------------------------------------------------------
        # History buffer
        # -------------------------------------------------------------

        self.history: deque[
            np.ndarray
        ] = deque(
            maxlen=self.history_steps
        )

        # -------------------------------------------------------------
        # Runtime diagnostics
        # -------------------------------------------------------------

        self.calls = 0

        self.valid_outputs = 0

        self.last_error = ""

        self.last_inference_time_ms = (
            math.nan
        )

    # -----------------------------------------------------------------
    # History status
    # -----------------------------------------------------------------

    @property
    def history_ready(
        self,
    ) -> bool:

        return (
            len(
                self.history
            )
            >= self.history_steps
        )

    # -----------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------

    def reset(
        self,
    ) -> None:

        self.history.clear()

        self.last_inference_time_ms = (
            math.nan
        )

    # -----------------------------------------------------------------
    # Append one 100-Hz input frame
    # -----------------------------------------------------------------

    def append_frame(
        self,
        left_angle_rad: float,
        left_angular_velocity_rad_s: float,
        right_angle_rad: float,
        right_angular_velocity_rad_s: float,
    ) -> None:

        frame = np.asarray(
            [
                left_angle_rad,
                left_angular_velocity_rad_s,
                right_angle_rad,
                right_angular_velocity_rad_s,
            ],
            dtype=np.float32,
        )

        if not np.all(
            np.isfinite(
                frame
            )
        ):
            raise ValueError(
                f"non-finite TCN input: "
                f"{frame}"
            )

        self.history.append(
            frame
        )

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    def infer(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ] | None:

        if not self.history_ready:
            return None

        self.calls += 1

        try:

            # ---------------------------------------------------------
            # [T, 4]
            # ---------------------------------------------------------

            hist = np.stack(
                tuple(
                    self.history
                ),
                axis=0,
            ).astype(
                np.float32,
                copy=False,
            )

            # ---------------------------------------------------------
            # [T, 4]
            #     ↓
            # [1, 4, T]
            # ---------------------------------------------------------

            x = torch.from_numpy(
                hist.T[
                    None
                ]
            ).to(
                self.device
            )

            # ---------------------------------------------------------
            # Normalize exactly as training/deployment checkpoint
            # ---------------------------------------------------------

            x = (
                x
                - self.input_mean
            ) / self.input_std

            # ---------------------------------------------------------
            # TCN inference
            # ---------------------------------------------------------

            start = (
                time.perf_counter()
            )

            with torch.inference_mode():

                action = (
                    self.model(
                        x
                    )[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32
                    )
                )

            self.last_inference_time_ms = (
                (
                    time.perf_counter()
                    - start
                )
                * 1000.0
            )

            # ---------------------------------------------------------
            # Normalized action
            # ---------------------------------------------------------

            action = np.clip(
                action,
                -1.0,
                1.0,
            )

            # ---------------------------------------------------------
            # Action -> Nm
            # ---------------------------------------------------------

            command_nm = (
                action
                * np.float32(
                    self.torque_scale_nm
                )
            )

            command_nm = np.clip(
                command_nm,
                -self.torque_scale_nm,
                self.torque_scale_nm,
            ).astype(
                np.float32
            )

            self.valid_outputs += 1

            self.last_error = ""

            return (
                action,
                command_nm,
            )

        except Exception as exc:

            self.last_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            return None

    # -----------------------------------------------------------------
    # Model information
    # -----------------------------------------------------------------

    def print_startup_summary(
        self,
    ) -> None:

        arch = self.payload.get(
            "architecture",
            {},
        )

        safety = self.payload.get(
            "safety",
            {},
        )

        print(
            "=" * 108
        )

        print(
            "MODEL: causal TCN"
        )

        print(
            f"MODE: "
            f"{self.payload.get('mode')}"
        )

        print(
            f"CONTROL: "
            f"{self.control_hz} Hz"
        )

        print(
            f"SENSOR: "
            f"{self.sensor_hz} Hz"
        )

        print(
            f"HISTORY: "
            f"{self.history_steps} samples / "
            f"{self.history_steps / self.sensor_hz:.3f} s"
        )

        print(
            "INPUT: "
            "L angle | "
            "L angular velocity | "
            "R angle | "
            "R angular velocity"
        )

        print(
            f"INPUT ORDER: "
            f"{self.input_channel_names}"
        )

        print(
            f"NORMALIZATION: "
            f"{self.normalization_scheme}"
        )

        print(
            f"TORQUE COMMAND SCALE: "
            f"{self.torque_scale_nm:g} Nm"
        )

        print(
            f"MODEL PARAMETERS: "
            f"{arch.get('parameter_count', 'unknown')}"
        )

        print(
            "HISTORY STARTUP: "
            f"zero output until "
            f"{self.history_steps} samples"
        )

        print(
            f"MODEL CHECKPOINT: "
            f"{self.model_path}"
        )

        print(
            f"SAFETY METADATA: "
            f"{safety}"
        )

        print(
            "=" * 108
        )


# =============================================================================
# VALIDATED TORCHSCRIPT POLICY
# =============================================================================


SCRIPTED_METADATA_FILE = "deployment.json"
SCRIPTED_DEPLOYMENT_FORMAT = "scripted_tcn_policy_v1"
CANONICAL_INPUT_CHANNEL_NAMES = [
    "left_thigh_angle_rad",
    "left_thigh_angular_velocity_rad_s",
    "right_thigh_angle_rad",
    "right_thigh_angular_velocity_rad_s",
]
PACKAGED_INPUT_CHANNEL_NAMES = [
    "left_thigh_angle_rad",
    "left_thigh_velocity_rad_s",
    "right_thigh_angle_rad",
    "right_thigh_velocity_rad_s",
]


def _canonical_input_names(names: list[str]) -> list[str]:
    """Accept the historical velocity spelling without changing channel order."""

    values = [str(value) for value in names]
    if values == PACKAGED_INPUT_CHANNEL_NAMES:
        return list(CANONICAL_INPUT_CHANNEL_NAMES)
    return values


class _ScriptedTCNPolicy:
    """Adapter for the stateful TorchScript packages stored under ``models/``.

    The package owns normalization, 100-to-30 Hz history sampling, standstill
    gating, and its mandatory physical slew limiter.  Consequently ``infer``
    returns an already projected physical command in N m; this adapter must not
    normalize, rescale, or build a second learned history around it.
    """

    def __init__(
        self,
        model_path: Path,
        model: torch.jit.ScriptModule,
        metadata: dict[str, object],
        *,
        device: str = "cpu",
    ) -> None:
        self.model_path = model_path.expanduser().resolve()
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.payload = metadata

        if metadata.get("deployment_format") != SCRIPTED_DEPLOYMENT_FORMAT:
            raise ValueError(
                "unsupported scripted deployment format: "
                f"{metadata.get('deployment_format')!r}"
            )
        if int(metadata.get("control_hz", -1)) != 100:
            raise ValueError("scripted policy must expose a 100 Hz call interface")
        if metadata.get("history_startup") != "repeat_first_valid_frame":
            raise ValueError("unsupported scripted history startup contract")
        if not hasattr(self.model, "reset"):
            raise ValueError("scripted policy does not expose reset()")

        checkpoint_names = [
            str(value) for value in metadata.get("input_channel_names", [])
        ]
        if _canonical_input_names(checkpoint_names) != CANONICAL_INPUT_CHANNEL_NAMES:
            raise ValueError(
                "scripted policy input order differs from the bilateral thigh-IMU "
                f"contract: {checkpoint_names}"
            )

        self.checkpoint_input_channel_names = checkpoint_names
        self.input_channel_names = list(CANONICAL_INPUT_CHANNEL_NAMES)
        self.control_hz = int(metadata["control_hz"])
        # ``sensor_hz`` is the external frame rate expected by the Raspberry Pi
        # controller.  A cross-rate package may sample its learned history at
        # 30 Hz internally; keep that separate so it is not rejected as a
        # non-100-Hz input policy.
        self.input_hz = self.control_hz
        self.sensor_hz = self.input_hz
        self.history_sampling_hz = int(
            metadata.get("history_sampling_hz", self.control_hz)
        )
        if not 0 < self.history_sampling_hz <= self.input_hz:
            raise ValueError("invalid scripted history sampling rate")
        self.history_steps = int(metadata["history_steps"])
        self.history_dense_steps = int(
            metadata.get("history_dense_steps", self.history_steps)
        )
        self.normalization_scheme = "embedded_in_torchscript"
        self.torque_scale_nm = float(metadata["torque_scale_nm"])
        self.mandatory_delta_nm = float(metadata["max_delta_nm_per_step"])
        self.input_filter_cutoff_hz = (
            None
            if metadata.get("input_filter_cutoff_hz") is None
            else float(metadata["input_filter_cutoff_hz"])
        )
        if not 0.0 < self.torque_scale_nm <= 12.0:
            raise ValueError("scripted torque scale must be in (0, 12] N m")
        if not 0.0 < self.mandatory_delta_nm <= self.torque_scale_nm:
            raise ValueError("scripted mandatory slew metadata is invalid")

        self.latest_frame: np.ndarray | None = None
        self.calls = 0
        self.valid_outputs = 0
        self.last_error = ""
        self.last_inference_time_ms = math.nan
        self.reset()

    @property
    def history_ready(self) -> bool:
        # The first valid frame is repeated inside the validated package.
        return self.latest_frame is not None

    def reset(self) -> None:
        self.latest_frame = None
        self.model.reset()
        self.last_inference_time_ms = math.nan

    def append_frame(
        self,
        left_angle_rad: float,
        left_angular_velocity_rad_s: float,
        right_angle_rad: float,
        right_angular_velocity_rad_s: float,
    ) -> None:
        frame = np.asarray(
            [
                left_angle_rad,
                left_angular_velocity_rad_s,
                right_angle_rad,
                right_angular_velocity_rad_s,
            ],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(frame)):
            raise ValueError(f"non-finite TCN input: {frame}")
        self.latest_frame = frame

    def infer(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.history_ready:
            return None
        self.calls += 1
        try:
            frame = torch.from_numpy(self.latest_frame).to(self.device)
            start = time.perf_counter()
            with torch.inference_mode():
                command = self.model(frame, self.mandatory_delta_nm)
            self.last_inference_time_ms = (
                time.perf_counter() - start
            ) * 1000.0
            command_nm = command.detach().cpu().numpy().astype(np.float32)
            if command_nm.shape != (2,) or not np.all(np.isfinite(command_nm)):
                raise ValueError("scripted policy output must be two finite torques")
            if np.any(np.abs(command_nm) > self.torque_scale_nm + 1.0e-5):
                raise ValueError("scripted policy exceeded its torque scale")
            action = command_nm / np.float32(self.torque_scale_nm)
            self.valid_outputs += 1
            self.last_error = ""
            return action.astype(np.float32), command_nm
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.latest_frame = None
            self.model.reset()
            return None

    def print_startup_summary(self) -> None:
        print("=" * 108)
        print("MODEL: validated stateful scripted thigh-IMU TCN")
        print(f"CONTROL/INPUT: {self.control_hz} Hz")
        print(
            f"HISTORY: {self.history_steps} samples at "
            f"{self.history_sampling_hz} Hz "
            f"({self.history_steps / self.history_sampling_hz:.3f} s), "
            f"dense 100-Hz buffer={self.history_dense_steps}"
        )
        print(f"INPUT ORDER: {self.input_channel_names}")
        print("NORMALIZATION: embedded in TorchScript package")
        print(f"TORQUE COMMAND SCALE: +/-{self.torque_scale_nm:g} N m")
        print(
            f"MANDATORY PACKAGE SLEW: {self.mandatory_delta_nm:.3f} "
            "N m/100-Hz call"
        )
        print("HISTORY STARTUP: repeat first valid frame inside package")
        print(f"STANDSTILL GATE: {self.payload.get('standstill_gate', {})}")
        print(f"MODEL CHECKPOINT: {self.model_path}")
        print("=" * 108)


def UnifiedTCNPolicy(
    model_path: Path | str,
    *,
    device: str = "cpu",
) -> _RawUnifiedTCNPolicy | _ScriptedTCNPolicy:
    """Load either the legacy raw TCN or a validated stateful package."""

    resolved = Path(model_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"model checkpoint not found: {resolved}")

    extra_files: dict[str, str | bytes] = {SCRIPTED_METADATA_FILE: ""}
    try:
        scripted = torch.jit.load(
            str(resolved),
            map_location=device,
            _extra_files=extra_files,
        )
    except RuntimeError:
        scripted = None

    if scripted is not None:
        raw_metadata = extra_files[SCRIPTED_METADATA_FILE]
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8")
        if not raw_metadata:
            raise ValueError(
                f"scripted checkpoint lacks {SCRIPTED_METADATA_FILE}: {resolved}"
            )
        metadata = json.loads(raw_metadata)
        if not isinstance(metadata, dict):
            raise ValueError("scripted deployment metadata must be a JSON object")
        return _ScriptedTCNPolicy(
            resolved,
            scripted,
            metadata,
            device=device,
        )

    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("model_type") != "causal_tcn":
        model_type = payload.get("model_type") if isinstance(payload, dict) else None
        raise ValueError(
            f"unsupported checkpoint model_type={model_type or 'missing'!r}: "
            f"{resolved}"
        )
    policy = _RawUnifiedTCNPolicy(resolved, device=device)
    policy.checkpoint_input_channel_names = list(policy.input_channel_names)
    policy.input_channel_names = _canonical_input_names(policy.input_channel_names)
    policy.input_hz = int(policy.sensor_hz)
    policy.history_sampling_hz = int(policy.sensor_hz)
    policy.history_dense_steps = int(policy.history_steps)
    policy.input_filter_cutoff_hz = None
    return policy


# =============================================================================
# OFFLINE CSV INFERENCE
# =============================================================================


def run_csv_inference(
    *,
    model_path: Path | str,
    csv_path: Path | str,
    output_path: Path | str | None = None,
    device: str = "cpu",
    input_mode: str = "filtered",
) -> Path:
    """
    Replay angle / angular velocity data from CSV through the TCN.

    The CSV is processed sequentially without sleeping. Each row is treated
    as one sensor sample. Therefore the CSV sampling frequency should match
    the model's expected sensor_hz.

    --------------------------------------------------------------------------
    input_mode = "filtered"
    --------------------------------------------------------------------------

    Uses:

        left_filtered_angle_rad
        left_filtered_angular_velocity_rad_s
        right_filtered_angle_rad
        right_filtered_angular_velocity_rad_s

    This is the recommended mode for reproducing exactly what the real-time
    TCN controller received after input filtering.

    --------------------------------------------------------------------------
    input_mode = "calibrated"
    --------------------------------------------------------------------------

    Uses:

        left_calibrated_angle_rad
        left_angular_velocity_rad_s
        right_calibrated_angle_rad
        right_angular_velocity_rad_s

    This bypasses the controller's IMU input low-pass filter and allows
    comparison between filtered and unfiltered input.
    """

    csv_path = (
        Path(
            csv_path
        )
        .expanduser()
        .resolve()
    )

    if not csv_path.exists():

        raise FileNotFoundError(
            f"input CSV not found: "
            f"{csv_path}"
        )

    if output_path is None:

        output_path = (
            csv_path.with_name(
                csv_path.stem
                + "_tcn_inference.csv"
            )
        )

    else:

        output_path = (
            Path(
                output_path
            )
            .expanduser()
            .resolve()
        )

    # -------------------------------------------------------------------------
    # Create policy
    # -------------------------------------------------------------------------

    policy = UnifiedTCNPolicy(
        model_path,
        device=device,
    )

    policy.print_startup_summary()

    # -------------------------------------------------------------------------
    # Select CSV columns
    # -------------------------------------------------------------------------

    if input_mode == "filtered":

        left_angle_col = (
            "left_filtered_angle_rad"
        )

        left_vel_col = (
            "left_filtered_angular_velocity_rad_s"
        )

        right_angle_col = (
            "right_filtered_angle_rad"
        )

        right_vel_col = (
            "right_filtered_angular_velocity_rad_s"
        )

    elif input_mode == "calibrated":

        left_angle_col = (
            "left_calibrated_angle_rad"
        )

        left_vel_col = (
            "left_angular_velocity_rad_s"
        )

        right_angle_col = (
            "right_calibrated_angle_rad"
        )

        right_vel_col = (
            "right_angular_velocity_rad_s"
        )

    else:

        raise ValueError(
            "input_mode must be either "
            "'filtered' or 'calibrated'"
        )

    required_columns = [
        left_angle_col,
        left_vel_col,
        right_angle_col,
        right_vel_col,
    ]

    # -------------------------------------------------------------------------
    # Information
    # -------------------------------------------------------------------------

    print()
    print(
        "=" * 108
    )

    print(
        "OFFLINE CSV -> TCN INFERENCE"
    )

    print(
        f"INPUT CSV : "
        f"{csv_path}"
    )

    print(
        f"OUTPUT CSV: "
        f"{output_path}"
    )

    print(
        f"INPUT MODE: "
        f"{input_mode}"
    )

    print()

    print(
        "TCN INPUT COLUMNS"
    )

    print(
        f"  LEFT ANGLE : "
        f"{left_angle_col}"
    )

    print(
        f"  LEFT VEL   : "
        f"{left_vel_col}"
    )

    print(
        f"  RIGHT ANGLE: "
        f"{right_angle_col}"
    )

    print(
        f"  RIGHT VEL  : "
        f"{right_vel_col}"
    )

    print(
        "=" * 108
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    input_rows = 0

    written_rows = 0

    invalid_rows = 0

    valid_outputs = 0

    timestamps: list[
        float
    ] = []

    inference_times: list[
        float
    ] = []

    # -------------------------------------------------------------------------
    # Open input CSV
    # -------------------------------------------------------------------------

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as fin:

        reader = csv.DictReader(
            fin
        )

        if reader.fieldnames is None:

            raise RuntimeError(
                "CSV has no header"
            )

        # ---------------------------------------------------------------------
        # Check columns
        # ---------------------------------------------------------------------

        missing = [
            column
            for column in required_columns
            if column
            not in reader.fieldnames
        ]

        if missing:

            print()
            print(
                "CSV AVAILABLE COLUMNS:"
            )

            for column in reader.fieldnames:
                print(
                    f"  {column}"
                )

            raise RuntimeError(
                "CSV missing required columns: "
                + ", ".join(
                    missing
                )
            )

        # ---------------------------------------------------------------------
        # Preserve original CSV columns
        # ---------------------------------------------------------------------

        output_fields = list(
            reader.fieldnames
        )

        # ---------------------------------------------------------------------
        # Add offline inference columns
        # ---------------------------------------------------------------------

        extra_fields = [
            "offline_tcn_history_ready",
            "offline_left_action_norm",
            "offline_right_action_norm",
            "offline_left_tcn_command_nm",
            "offline_right_tcn_command_nm",
            "offline_inference_time_ms",
        ]

        for field in extra_fields:

            if field not in output_fields:
                output_fields.append(
                    field
                )

        # ---------------------------------------------------------------------
        # Open output CSV
        # ---------------------------------------------------------------------

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as fout:

            writer = csv.DictWriter(
                fout,
                fieldnames=output_fields,
            )

            writer.writeheader()

            # -----------------------------------------------------------------
            # Replay CSV sequentially
            # -----------------------------------------------------------------

            for row in reader:

                input_rows += 1

                # -------------------------------------------------------------
                # Parse TCN inputs
                # -------------------------------------------------------------

                try:

                    left_angle = float(
                        row[
                            left_angle_col
                        ]
                    )

                    left_velocity = float(
                        row[
                            left_vel_col
                        ]
                    )

                    right_angle = float(
                        row[
                            right_angle_col
                        ]
                    )

                    right_velocity = float(
                        row[
                            right_vel_col
                        ]
                    )

                except (
                    ValueError,
                    TypeError,
                    KeyError,
                ):

                    invalid_rows += 1

                    continue

                values = np.asarray(
                    [
                        left_angle,
                        left_velocity,
                        right_angle,
                        right_velocity,
                    ],
                    dtype=np.float64,
                )

                if not np.all(
                    np.isfinite(
                        values
                    )
                ):

                    invalid_rows += 1

                    continue

                # -------------------------------------------------------------
                # Timestamp statistics
                # -------------------------------------------------------------

                if (
                    "timestamp"
                    in row
                ):

                    try:

                        timestamp = float(
                            row[
                                "timestamp"
                            ]
                        )

                        if math.isfinite(
                            timestamp
                        ):
                            timestamps.append(
                                timestamp
                            )

                    except (
                        ValueError,
                        TypeError,
                    ):
                        pass

                # -------------------------------------------------------------
                # Append one frame
                # -------------------------------------------------------------

                policy.append_frame(
                    left_angle,
                    left_velocity,
                    right_angle,
                    right_velocity,
                )

                # -------------------------------------------------------------
                # TCN inference
                # -------------------------------------------------------------

                result = (
                    policy.infer()
                )

                row[
                    "offline_tcn_history_ready"
                ] = int(
                    policy.history_ready
                )

                # -------------------------------------------------------------
                # History not full yet
                # -------------------------------------------------------------

                if result is None:

                    row[
                        "offline_left_action_norm"
                    ] = (
                        "0.000000000"
                    )

                    row[
                        "offline_right_action_norm"
                    ] = (
                        "0.000000000"
                    )

                    row[
                        "offline_left_tcn_command_nm"
                    ] = (
                        "0.000000"
                    )

                    row[
                        "offline_right_tcn_command_nm"
                    ] = (
                        "0.000000"
                    )

                    row[
                        "offline_inference_time_ms"
                    ] = (
                        "nan"
                    )

                # -------------------------------------------------------------
                # Valid TCN output
                # -------------------------------------------------------------

                else:

                    (
                        action,
                        command_nm,
                    ) = result

                    left_action = float(
                        action[
                            0
                        ]
                    )

                    right_action = float(
                        action[
                            1
                        ]
                    )

                    left_command = float(
                        command_nm[
                            0
                        ]
                    )

                    right_command = float(
                        command_nm[
                            1
                        ]
                    )

                    row[
                        "offline_left_action_norm"
                    ] = (
                        f"{left_action:.9f}"
                    )

                    row[
                        "offline_right_action_norm"
                    ] = (
                        f"{right_action:.9f}"
                    )

                    row[
                        "offline_left_tcn_command_nm"
                    ] = (
                        f"{left_command:.6f}"
                    )

                    row[
                        "offline_right_tcn_command_nm"
                    ] = (
                        f"{right_command:.6f}"
                    )

                    row[
                        "offline_inference_time_ms"
                    ] = (
                        f"{policy.last_inference_time_ms:.6f}"
                    )

                    if math.isfinite(
                        policy.last_inference_time_ms
                    ):

                        inference_times.append(
                            policy.last_inference_time_ms
                        )

                    valid_outputs += 1

                # -------------------------------------------------------------
                # Write row
                # -------------------------------------------------------------

                writer.writerow(
                    row
                )

                written_rows += 1

    # =========================================================================
    # Final statistics
    # =========================================================================

    print()
    print(
        "=" * 108
    )

    print(
        "CSV TCN INFERENCE COMPLETE"
    )

    print(
        f"Input rows     : "
        f"{input_rows}"
    )

    print(
        f"Written rows   : "
        f"{written_rows}"
    )

    print(
        f"Invalid rows   : "
        f"{invalid_rows}"
    )

    print(
        f"History steps  : "
        f"{policy.history_steps}"
    )

    print(
        f"Valid outputs  : "
        f"{valid_outputs}"
    )

    # -------------------------------------------------------------------------
    # Sampling-rate diagnostics
    # -------------------------------------------------------------------------

    if len(
        timestamps
    ) >= 2:

        timestamps_np = np.asarray(
            timestamps,
            dtype=np.float64,
        )

        dt = np.diff(
            timestamps_np
        )

        valid_dt = dt[
            (
                np.isfinite(
                    dt
                )
            )
            & (
                dt > 0
            )
        ]

        if len(
            valid_dt
        ) > 0:

            median_dt = float(
                np.median(
                    valid_dt
                )
            )

            median_hz = (
                1.0
                / median_dt
            )

            total_duration = (
                timestamps_np[-1]
                - timestamps_np[0]
            )

            if total_duration > 0:

                average_hz = (
                    (
                        len(
                            timestamps_np
                        )
                        - 1
                    )
                    / total_duration
                )

            else:

                average_hz = (
                    math.nan
                )

            print(
                f"CSV median Hz  : "
                f"{median_hz:.3f}"
            )

            print(
                f"CSV average Hz : "
                f"{average_hz:.3f}"
            )

            print(
                f"Model sensor Hz: "
                f"{policy.sensor_hz}"
            )

            frequency_error = abs(
                median_hz
                - policy.sensor_hz
            )

            if frequency_error > (
                0.05
                * policy.sensor_hz
            ):

                print(
                    "[WARNING] CSV sampling "
                    "frequency differs by more "
                    "than 5% from model sensor_hz."
                )

                print(
                    "[WARNING] The TCN assumes "
                    "each CSV row represents one "
                    "model sampling interval."
                )

    # -------------------------------------------------------------------------
    # Inference diagnostics
    # -------------------------------------------------------------------------

    if inference_times:

        inference_np = np.asarray(
            inference_times,
            dtype=np.float64,
        )

        print(
            f"Inference mean : "
            f"{np.mean(inference_np):.3f} ms"
        )

        print(
            f"Inference p50  : "
            f"{np.percentile(inference_np, 50):.3f} ms"
        )

        print(
            f"Inference p95  : "
            f"{np.percentile(inference_np, 95):.3f} ms"
        )

        print(
            f"Inference max  : "
            f"{np.max(inference_np):.3f} ms"
        )

    # -------------------------------------------------------------------------

    print(
        f"Saved          : "
        f"{output_path}"
    )

    if policy.last_error:

        print(
            f"TCN last error : "
            f"{policy.last_error}"
        )

    print(
        "=" * 108
    )

    return output_path


# =============================================================================
# COMMAND LINE
# =============================================================================


def build_parser(
) -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Unified causal-TCN policy. "
            "Can also replay IMU CSV data "
            "through the trained TCN."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help=(
            "Path to TCN deployment "
            "checkpoint (.pt)."
        ),
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help=(
            "Input CSV containing thigh "
            "angle and angular velocity."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV. "
            "Default: "
            "<input>_tcn_inference.csv"
        ),
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "PyTorch device. "
            "Default: cpu"
        ),
    )

    parser.add_argument(
        "--input-mode",
        choices=[
            "filtered",
            "calibrated",
        ],
        default="filtered",
        help=(
            "filtered: use the exact "
            "filtered inputs normally sent "
            "to the TCN; "
            "calibrated: use calibrated "
            "but unfiltered IMU inputs."
        ),
    )

    return parser


# =============================================================================
# OFFLINE MAIN
# =============================================================================


def main(
) -> None:

    args = (
        build_parser()
        .parse_args()
    )

    # Raspberry Pi / CPU deployment:
    # avoid unnecessary PyTorch threads.
    torch.set_num_threads(
        1
    )

    run_csv_inference(
        model_path=args.model,
        csv_path=args.csv,
        output_path=args.output,
        device=args.device,
        input_mode=args.input_mode,
    )


if __name__ == "__main__":
    main()
