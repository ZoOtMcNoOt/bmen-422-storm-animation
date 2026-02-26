"""Data models for STORM simulation and localization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SimulationParams:
    n_emitters: int = 120
    frame_count: int = 180
    canvas_size_px: tuple[int, int] = (64, 64)
    pixel_size_nm: float = 100.0
    psf_sigma_px: float = 1.2
    background_lambda: float = 2.0
    photon_mean: float = 550.0
    on_off_rates: tuple[float, float] = (0.05, 0.25)
    drift_per_frame_px: tuple[float, float] = (0.02, -0.01)
    camera_gain: float = 1.0
    seed: int = 4

    def validate(self) -> None:
        height, width = self.canvas_size_px
        if self.n_emitters <= 0:
            raise ValueError("n_emitters must be positive")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if height <= 8 or width <= 8:
            raise ValueError("canvas_size_px must be larger than 8x8")
        if self.psf_sigma_px <= 0:
            raise ValueError("psf_sigma_px must be positive")
        if self.photon_mean <= 0:
            raise ValueError("photon_mean must be positive")
        if self.background_lambda < 0:
            raise ValueError("background_lambda must be non-negative")
        if any(rate < 0 for rate in self.on_off_rates):
            raise ValueError("on_off_rates must be non-negative")


@dataclass(frozen=True)
class DetectionThreshold:
    sigma_multiplier: float = 2.5
    absolute_floor: float = 12.0
    min_distance_px: int = 2

    def validate(self) -> None:
        if self.sigma_multiplier <= 0:
            raise ValueError("sigma_multiplier must be positive")
        if self.absolute_floor < 0:
            raise ValueError("absolute_floor must be non-negative")
        if self.min_distance_px < 1:
            raise ValueError("min_distance_px must be >= 1")


@dataclass(frozen=True)
class FrameStack:
    frames: np.ndarray
    ground_truth_xy: np.ndarray
    active_masks: np.ndarray
    frame_drift_xy: np.ndarray
    emitter_photons: np.ndarray

    def __post_init__(self) -> None:
        if self.frames.ndim != 3:
            raise ValueError("frames must have shape (T, H, W)")
        if self.ground_truth_xy.ndim != 2 or self.ground_truth_xy.shape[1] != 2:
            raise ValueError("ground_truth_xy must have shape (N, 2)")
        if self.active_masks.shape != (self.frames.shape[0], self.ground_truth_xy.shape[0]):
            raise ValueError("active_masks shape mismatch")
        if self.frame_drift_xy.shape != (self.frames.shape[0], 2):
            raise ValueError("frame_drift_xy shape mismatch")
        if self.emitter_photons.shape != (self.frames.shape[0], self.ground_truth_xy.shape[0]):
            raise ValueError("emitter_photons shape mismatch")


@dataclass(frozen=True)
class LocalizationResult:
    estimated_xy: np.ndarray
    photon_estimates: np.ndarray
    uncertainty_nm: np.ndarray
    failure_flags: np.ndarray
    frame_index: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    def successful_xy(self) -> np.ndarray:
        if self.estimated_xy.size == 0:
            return self.estimated_xy
        return self.estimated_xy[~self.failure_flags]


@dataclass(frozen=True)
class SweepResult:
    parameter_name: str
    parameter_values: list[Any]
    localization_rmse_nm: list[float]
    failure_rate: list[float]
    merge_rate: list[float]
    effective_resolution_nm: list[float]
