#!/usr/bin/env python3
"""Environment-independent inference for the two packaged hip Exo policies."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent

# Coordinate convention used during training and validation:
#   input  = [right hip angle, left hip angle,
#             right hip angular velocity, left hip angular velocity]
#   units  = [rad, rad, rad/s, rad/s]
#   positive angle/velocity = hip flexion (lifting the thigh forward)
#   output = [right hip torque, left hip torque] in Nm
#   positive torque = hip-flexion torque


class MLP(nn.Module):
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


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _slew_limit(
    desired_nm: np.ndarray, previous_nm: np.ndarray, max_delta_nm: float
) -> np.ndarray:
    return np.clip(
        desired_nm,
        previous_nm - max_delta_nm,
        previous_nm + max_delta_nm,
    )


class DirectExoController:
    """8-frame hip kinematics/optional measured torque -> bilateral torque."""

    def __init__(self, path: Path = HERE / "direct_exo_8frame.pt") -> None:
        payload = _load(path)
        self.steps = int(payload["history_steps"])
        self.mean = np.asarray(payload["input_mean"], dtype=np.float32)
        self.std = np.asarray(payload["input_std"], dtype=np.float32)
        self.scale = float(payload["torque_scale_nm"])
        self.max_delta = float(payload["max_delta_nm_per_step"])
        self.per_frame_dim = len(self.mean) // self.steps
        if self.per_frame_dim not in (4, 6):
            raise ValueError(
                f"unsupported Direct Exo input dimension: {self.per_frame_dim} per frame"
            )
        self.model = MLP(len(self.mean), int(payload["hidden_dim"]), 2)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.history: deque[np.ndarray] = deque(maxlen=self.steps)
        self.previous_nm = np.zeros(2, dtype=np.float32)

    def _frame(
        self,
        hip_state4: np.ndarray | list[float],
        measured_exo_nm: np.ndarray | list[float] | None,
    ) -> np.ndarray:
        state = np.asarray(hip_state4, dtype=np.float32)
        if state.shape != (4,):
            raise ValueError("hip_state4 must have shape (4,)")
        if self.per_frame_dim == 4:
            return state
        measured = (
            self.previous_nm
            if measured_exo_nm is None
            else np.asarray(measured_exo_nm, dtype=np.float32)
        )
        if measured.shape != (2,):
            raise ValueError("measured_exo_nm must have shape (2,)")
        return np.concatenate((state, measured / self.scale))

    def reset(
        self,
        hip_state4: np.ndarray | list[float],
        measured_exo_nm: np.ndarray | list[float] | None = None,
    ) -> None:
        frame = self._frame(hip_state4, measured_exo_nm)
        self.history = deque([frame.copy() for _ in range(self.steps)], self.steps)
        self.previous_nm = np.zeros(2, dtype=np.float32)

    def step(
        self,
        hip_state4: np.ndarray | list[float],
        measured_exo_nm: np.ndarray | list[float] | None = None,
    ) -> np.ndarray:
        frame = self._frame(hip_state4, measured_exo_nm)
        if not self.history:
            self.reset(hip_state4, measured_exo_nm)
        else:
            self.history.append(frame.copy())
        features = np.stack(self.history).reshape(-1)
        normalized = torch.from_numpy((features - self.mean) / self.std)[None]
        with torch.no_grad():
            desired_nm = self.scale * self.model(normalized)[0].numpy()
        self.previous_nm = _slew_limit(
            desired_nm, self.previous_nm, self.max_delta
        ).astype(np.float32)
        return self.previous_nm.copy()


class TargetPDExoController:
    """8-frame hip kinematics/command history -> target-angle PD torque."""

    def __init__(self, path: Path = HERE / "target_pd_exo_8frame.pt") -> None:
        payload = _load(path)
        self.steps = int(payload["history_steps"])
        self.mean = np.asarray(payload["input_mean"], dtype=np.float32)
        self.std = np.asarray(payload["input_std"], dtype=np.float32)
        self.kp = float(payload["kp_nm_per_rad"])
        self.kd = float(payload["kd_nm_s_per_rad"])
        self.offset_limit = float(payload["target_offset_limit_rad"])
        self.torque_limit = float(payload["torque_limit_nm"])
        self.scale = float(payload["torque_scale_nm"])
        self.max_delta = float(payload["max_delta_nm_per_step"])
        self.model = MLP(len(self.mean), int(payload["hidden_dim"]), 1)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.history: deque[np.ndarray] = deque(maxlen=self.steps)
        self.previous_nm = np.zeros(2, dtype=np.float32)

    def reset(self, hip_state4: np.ndarray | list[float]) -> None:
        state = np.asarray(hip_state4, dtype=np.float32)
        if state.shape != (4,):
            raise ValueError("hip_state4 must have shape (4,)")
        frame = np.concatenate((state, np.zeros(2, dtype=np.float32)))
        self.history = deque([frame.copy() for _ in range(self.steps)], self.steps)
        self.previous_nm = np.zeros(2, dtype=np.float32)

    def step(self, hip_state4: np.ndarray | list[float]) -> np.ndarray:
        state = np.asarray(hip_state4, dtype=np.float32)
        if not self.history:
            self.reset(state)
        command_normalized = self.previous_nm / self.scale
        self.history.append(np.concatenate((state, command_normalized)))
        history = np.stack(self.history)

        # The same leg network is used for right and mirrored-left histories.
        right = history.reshape(-1)
        left = history[:, [1, 0, 3, 2, 5, 4]].reshape(-1)
        features = np.stack((right, left))
        normalized = torch.from_numpy((features - self.mean) / self.std)
        with torch.no_grad():
            offset = self.offset_limit * self.model(normalized)[:, 0].numpy()

        angle = state[:2]
        velocity = state[2:]
        target_angle = angle + offset
        desired_nm = np.clip(
            self.kp * (target_angle - angle) + self.kd * velocity,
            -self.torque_limit,
            self.torque_limit,
        )
        self.previous_nm = _slew_limit(
            desired_nm, self.previous_nm, self.max_delta
        ).astype(np.float32)
        return self.previous_nm.copy()


if __name__ == "__main__":
    # A new trajectory must reset each controller before the first step.
    hip = np.array([0.20, 0.18, 0.10, -0.12], dtype=np.float32)
    direct = DirectExoController()
    target_pd = TargetPDExoController()
    direct.reset(hip)
    target_pd.reset(hip)

    for frame in range(5):
        direct_nm = direct.step(hip)
        pd_nm = target_pd.step(hip)
        print(
            f"{frame=:02d} direct={direct_nm} Nm  "
            f"target_pd={pd_nm} Nm"
        )
