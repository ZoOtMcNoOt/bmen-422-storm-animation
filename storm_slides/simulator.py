"""Monte Carlo STORM simulator and lightweight localization pipeline."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

from storm_slides.models import (
    DetectionThreshold,
    FrameStack,
    LocalizationResult,
    SimulationParams,
    SweepResult,
)


def _add_gaussian_spot(
    image: np.ndarray,
    center_xy: tuple[float, float],
    total_photons: float,
    sigma_px: float,
) -> None:
    """Add a normalized Gaussian spot with a finite support window."""
    center_x, center_y = center_xy
    radius = max(2, int(np.ceil(4 * sigma_px)))
    x_min = max(0, int(np.floor(center_x)) - radius)
    x_max = min(image.shape[1] - 1, int(np.floor(center_x)) + radius)
    y_min = max(0, int(np.floor(center_y)) - radius)
    y_max = min(image.shape[0] - 1, int(np.floor(center_y)) + radius)
    if x_min > x_max or y_min > y_max:
        return

    ys = np.arange(y_min, y_max + 1, dtype=np.float64)
    xs = np.arange(x_min, x_max + 1, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    kernel = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma_px**2))
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0:
        return
    image[y_min : y_max + 1, x_min : x_max + 1] += total_photons * kernel / kernel_sum


def render_frame_from_emitters(
    canvas_size_px: tuple[int, int],
    emitter_xy: np.ndarray,
    photon_totals: np.ndarray,
    psf_sigma_px: float,
    background_lambda: float,
    rng: np.random.Generator | None = None,
    *,
    apply_poisson: bool = True,
    camera_gain: float = 1.0,
) -> np.ndarray:
    """Render one frame from emitter coordinates and total photons per emitter."""
    if rng is None:
        rng = np.random.default_rng()
    height, width = canvas_size_px
    frame = np.full((height, width), background_lambda, dtype=np.float64)

    for idx in range(emitter_xy.shape[0]):
        x, y = emitter_xy[idx]
        photons = float(photon_totals[idx])
        if photons > 0:
            _add_gaussian_spot(frame, (x, y), photons, psf_sigma_px)

    if apply_poisson:
        frame = rng.poisson(np.clip(frame, 1e-9, None)).astype(np.float64)
    return frame * camera_gain


def simulate_storm_frames(params: SimulationParams) -> FrameStack:
    """Simulate stochastic blinking emitters and noisy image frames."""
    params.validate()
    rng = np.random.default_rng(params.seed)
    height, width = params.canvas_size_px

    margin = max(4, int(np.ceil(4 * params.psf_sigma_px)))
    gt_xy = np.column_stack(
        [
            rng.uniform(margin, width - margin, size=params.n_emitters),
            rng.uniform(margin, height - margin, size=params.n_emitters),
        ]
    )

    frames = np.zeros((params.frame_count, height, width), dtype=np.float64)
    active = np.zeros((params.frame_count, params.n_emitters), dtype=bool)
    drift_xy = np.zeros((params.frame_count, 2), dtype=np.float64)
    emitter_photons = np.zeros((params.frame_count, params.n_emitters), dtype=np.float64)

    on_rate, off_rate = params.on_off_rates
    steady_state_on = on_rate / (on_rate + off_rate + 1e-12)
    states = rng.random(params.n_emitters) < steady_state_on

    drift_step = np.asarray(params.drift_per_frame_px, dtype=np.float64)
    for frame_idx in range(params.frame_count):
        if frame_idx > 0:
            drift_xy[frame_idx] = drift_xy[frame_idx - 1] + drift_step

        turn_on = (~states) & (rng.random(params.n_emitters) < on_rate)
        turn_off = states & (rng.random(params.n_emitters) < off_rate)
        states = (states | turn_on) & (~turn_off)
        active[frame_idx] = states

        active_indices = np.flatnonzero(states)
        if active_indices.size:
            sampled_photons = np.maximum(
                1.0, rng.normal(loc=params.photon_mean, scale=0.15 * params.photon_mean, size=active_indices.size)
            )
            emitter_photons[frame_idx, active_indices] = sampled_photons

        frame_emitters = gt_xy + drift_xy[frame_idx]
        frames[frame_idx] = render_frame_from_emitters(
            params.canvas_size_px,
            frame_emitters[active[frame_idx]],
            emitter_photons[frame_idx, active[frame_idx]],
            params.psf_sigma_px,
            params.background_lambda,
            rng,
            apply_poisson=True,
            camera_gain=params.camera_gain,
        )

    return FrameStack(
        frames=frames,
        ground_truth_xy=gt_xy,
        active_masks=active,
        frame_drift_xy=drift_xy,
        emitter_photons=emitter_photons,
    )


def detect_spots(frame: np.ndarray, threshold_cfg: DetectionThreshold | None = None) -> np.ndarray:
    """Return candidate local maxima as (x, y) coordinates."""
    cfg = threshold_cfg or DetectionThreshold()
    cfg.validate()
    if frame.ndim != 2:
        raise ValueError("frame must be 2D")

    threshold = max(cfg.absolute_floor, float(frame.mean() + cfg.sigma_multiplier * frame.std()))
    candidate = frame >= threshold
    if not candidate.any():
        return np.empty((0, 2), dtype=np.float64)

    padded = np.pad(frame, 1, mode="edge")
    neighborhood = [
        padded[1:-1, 1:-1],
        padded[:-2, :-2],
        padded[:-2, 1:-1],
        padded[:-2, 2:],
        padded[1:-1, :-2],
        padded[1:-1, 2:],
        padded[2:, :-2],
        padded[2:, 1:-1],
        padded[2:, 2:],
    ]
    local_max_mask = candidate & (frame >= np.maximum.reduce(neighborhood))

    local_max_mask[[0, -1], :] = False
    local_max_mask[:, [0, -1]] = False
    yx = np.argwhere(local_max_mask)
    if yx.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Greedy non-maximum suppression in descending intensity order.
    order = np.argsort(frame[yx[:, 0], yx[:, 1]])[::-1]
    selected: list[tuple[int, int]] = []
    min_sq = cfg.min_distance_px**2
    for idx in order:
        y, x = int(yx[idx, 0]), int(yx[idx, 1])
        if all((x - sx) ** 2 + (y - sy) ** 2 >= min_sq for sy, sx in selected):
            selected.append((y, x))

    xy = np.asarray([[x, y] for y, x in selected], dtype=np.float64)
    return xy


def localize_spots_mle(
    frame: np.ndarray,
    candidates_xy: np.ndarray,
    psf_sigma_px: float,
    pixel_size_nm: float = 100.0,
    *,
    patch_radius: int = 3,
    min_signal_photons: float = 20.0,
) -> LocalizationResult:
    """Approximate Gaussian MLE localization on local patches."""
    if candidates_xy.size == 0:
        empty = np.empty((0,), dtype=np.float64)
        return LocalizationResult(
            estimated_xy=np.empty((0, 2), dtype=np.float64),
            photon_estimates=empty.copy(),
            uncertainty_nm=empty.copy(),
            failure_flags=np.empty((0,), dtype=bool),
        )

    estimated = np.full((candidates_xy.shape[0], 2), np.nan, dtype=np.float64)
    photons = np.zeros(candidates_xy.shape[0], dtype=np.float64)
    uncertainty = np.full(candidates_xy.shape[0], np.inf, dtype=np.float64)
    failures = np.ones(candidates_xy.shape[0], dtype=bool)

    height, width = frame.shape
    for idx, (x_raw, y_raw) in enumerate(candidates_xy):
        x = int(np.round(x_raw))
        y = int(np.round(y_raw))
        x_min = x - patch_radius
        x_max = x + patch_radius
        y_min = y - patch_radius
        y_max = y + patch_radius
        if x_min < 0 or y_min < 0 or x_max >= width or y_max >= height:
            continue

        patch = frame[y_min : y_max + 1, x_min : x_max + 1].astype(np.float64)
        edge_pixels = np.concatenate([patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]])
        background = float(np.median(edge_pixels))
        signal = np.clip(patch - background, a_min=0.0, a_max=None)
        total_signal = float(signal.sum())
        if total_signal < min_signal_photons:
            continue

        yy, xx = np.indices(signal.shape, dtype=np.float64)
        xx += x_min
        yy += y_min
        x_hat = float((signal * xx).sum() / total_signal)
        y_hat = float((signal * yy).sum() / total_signal)

        estimated[idx] = [x_hat, y_hat]
        photons[idx] = total_signal

        # Thompson-like behavior proxy: sigma/sqrt(N) worsened by background.
        background_penalty = 1.0 + background / max(total_signal, 1.0)
        uncertainty[idx] = (psf_sigma_px * pixel_size_nm / np.sqrt(total_signal)) * background_penalty
        failures[idx] = False

    return LocalizationResult(
        estimated_xy=estimated,
        photon_estimates=photons,
        uncertainty_nm=uncertainty,
        failure_flags=failures,
    )


def localize_frame_stack(
    frame_stack: FrameStack,
    psf_sigma_px: float,
    pixel_size_nm: float,
    threshold_cfg: DetectionThreshold | None = None,
) -> LocalizationResult:
    """Run detection + localization over every frame in a stack."""
    all_xy: list[np.ndarray] = []
    all_photons: list[np.ndarray] = []
    all_unc: list[np.ndarray] = []
    all_fail: list[np.ndarray] = []
    all_frame_index: list[np.ndarray] = []

    for frame_idx, frame in enumerate(frame_stack.frames):
        candidates = detect_spots(frame, threshold_cfg)
        result = localize_spots_mle(frame, candidates, psf_sigma_px, pixel_size_nm)
        if result.estimated_xy.size == 0:
            continue
        all_xy.append(result.estimated_xy)
        all_photons.append(result.photon_estimates)
        all_unc.append(result.uncertainty_nm)
        all_fail.append(result.failure_flags)
        all_frame_index.append(np.full(result.estimated_xy.shape[0], frame_idx, dtype=np.int64))

    if not all_xy:
        return LocalizationResult(
            estimated_xy=np.empty((0, 2), dtype=np.float64),
            photon_estimates=np.empty((0,), dtype=np.float64),
            uncertainty_nm=np.empty((0,), dtype=np.float64),
            failure_flags=np.empty((0,), dtype=bool),
            frame_index=np.empty((0,), dtype=np.int64),
        )

    return LocalizationResult(
        estimated_xy=np.vstack(all_xy),
        photon_estimates=np.concatenate(all_photons),
        uncertainty_nm=np.concatenate(all_unc),
        failure_flags=np.concatenate(all_fail),
        frame_index=np.concatenate(all_frame_index),
    )


def apply_drift_correction(
    localizations_xy: np.ndarray,
    frame_index: np.ndarray,
    drift_per_frame_px: tuple[float, float],
) -> np.ndarray:
    """Subtract linear frame-to-frame drift from localization coordinates."""
    if localizations_xy.shape[0] != frame_index.shape[0]:
        raise ValueError("localizations_xy and frame_index length mismatch")
    drift_step = np.asarray(drift_per_frame_px, dtype=np.float64)
    correction = frame_index[:, None].astype(np.float64) * drift_step[None, :]
    return localizations_xy - correction


def reconstruct_density_map(
    localizations_xy: np.ndarray,
    canvas_size_px: tuple[int, int],
    *,
    upsample_factor: int = 8,
    mode: str = "2d",
    z_values: np.ndarray | None = None,
    z_bins: int = 20,
) -> np.ndarray:
    """Aggregate localizations into a 2D or 3D super-resolution histogram."""
    height, width = canvas_size_px
    sr_height = height * upsample_factor
    sr_width = width * upsample_factor

    if mode not in {"2d", "3d"}:
        raise ValueError("mode must be either '2d' or '3d'")

    if mode == "2d":
        density = np.zeros((sr_height, sr_width), dtype=np.float64)
    else:
        density = np.zeros((z_bins, sr_height, sr_width), dtype=np.float64)

    if localizations_xy.size == 0:
        return density

    x_sr = np.round(localizations_xy[:, 0] * upsample_factor).astype(np.int64)
    y_sr = np.round(localizations_xy[:, 1] * upsample_factor).astype(np.int64)
    valid = (x_sr >= 0) & (x_sr < sr_width) & (y_sr >= 0) & (y_sr < sr_height)
    x_sr = x_sr[valid]
    y_sr = y_sr[valid]
    if x_sr.size == 0:
        return density

    if mode == "2d":
        np.add.at(density, (y_sr, x_sr), 1.0)
        return density

    if z_values is None or z_values.shape[0] != localizations_xy.shape[0]:
        raise ValueError("z_values must be provided for mode='3d' with matching length")

    z_valid = z_values[valid]
    z_min = float(np.min(z_valid))
    z_max = float(np.max(z_valid))
    if z_max == z_min:
        z_bin = np.zeros_like(z_valid, dtype=np.int64)
    else:
        z_bin = np.floor((z_valid - z_min) / (z_max - z_min) * (z_bins - 1)).astype(np.int64)
    np.add.at(density, (z_bin, y_sr, x_sr), 1.0)
    return density


def _nearest_neighbor_rmse_nm(
    estimated_xy: np.ndarray,
    truth_xy: np.ndarray,
    pixel_size_nm: float,
) -> float:
    if estimated_xy.size == 0 or truth_xy.size == 0:
        return float("nan")

    # Brute-force nearest-neighbor error is enough for this educational simulator.
    diffs = estimated_xy[:, None, :] - truth_xy[None, :, :]
    sq_dist = np.sum(diffs**2, axis=2)
    nearest_sq = np.min(sq_dist, axis=1)
    rmse_px = float(np.sqrt(np.mean(nearest_sq)))
    return rmse_px * pixel_size_nm


def _effective_resolution_nm(localizations_xy: np.ndarray, pixel_size_nm: float) -> float:
    if localizations_xy.shape[0] < 2:
        return float("nan")
    sample = localizations_xy
    if sample.shape[0] > 2000:
        rng = np.random.default_rng(0)
        sample = sample[rng.choice(sample.shape[0], size=2000, replace=False)]
    diffs = sample[:, None, :] - sample[None, :, :]
    sq = np.sum(diffs**2, axis=2)
    np.fill_diagonal(sq, np.inf)
    nn = np.sqrt(np.min(sq, axis=1))
    return float(np.median(nn) * pixel_size_nm)


def run_parameter_sweep(
    base_params: SimulationParams,
    sweep_spec: dict[str, Iterable[object]],
    threshold_cfg: DetectionThreshold | None = None,
) -> dict[str, SweepResult]:
    """Run parameter sweeps and compute quality metrics."""
    results: dict[str, SweepResult] = {}
    for param_name, iterable in sweep_spec.items():
        values = list(iterable)
        rmse_nm: list[float] = []
        failure_rate: list[float] = []
        merge_rate: list[float] = []
        effective_res_nm: list[float] = []

        for value in values:
            varied = replace(base_params, **{param_name: value})
            stack = simulate_storm_frames(varied)
            localized = localize_frame_stack(
                stack, psf_sigma_px=varied.psf_sigma_px, pixel_size_nm=varied.pixel_size_nm, threshold_cfg=threshold_cfg
            )

            if localized.estimated_xy.size == 0:
                rmse_nm.append(float("nan"))
                failure_rate.append(1.0)
                merge_rate.append(1.0)
                effective_res_nm.append(float("nan"))
                continue

            successful = ~localized.failure_flags
            success_xy = localized.estimated_xy[successful]
            success_frame_idx = localized.frame_index[successful]
            corrected_xy = apply_drift_correction(success_xy, success_frame_idx, varied.drift_per_frame_px)

            rmse_nm.append(_nearest_neighbor_rmse_nm(corrected_xy, stack.ground_truth_xy, varied.pixel_size_nm))
            failure_rate.append(float(np.mean(localized.failure_flags)))

            detected_per_frame = np.bincount(success_frame_idx, minlength=varied.frame_count)
            mean_detected = float(np.mean(detected_per_frame))
            mean_active = float(np.mean(stack.active_masks.sum(axis=1)))
            overlap_penalty = 1.0 - (mean_detected / max(mean_active, 1e-9))
            merge_rate.append(float(np.clip(overlap_penalty, 0.0, 1.0)))

            effective_res_nm.append(_effective_resolution_nm(corrected_xy, varied.pixel_size_nm))

        results[param_name] = SweepResult(
            parameter_name=param_name,
            parameter_values=values,
            localization_rmse_nm=rmse_nm,
            failure_rate=failure_rate,
            merge_rate=merge_rate,
            effective_resolution_nm=effective_res_nm,
        )
    return results


# ------------------------------------------------------------------
# Convenience helpers for slide visualisations
# ------------------------------------------------------------------

#: Fast-running default params for in-slide demos (small canvas, few frames).
QUICK_SIM_PARAMS = SimulationParams(
    n_emitters=30,
    frame_count=40,
    canvas_size_px=(32, 32),
    pixel_size_nm=100.0,
    psf_sigma_px=1.2,
    background_lambda=2.0,
    photon_mean=550.0,
    on_off_rates=(0.08, 0.3),
    drift_per_frame_px=(0.0, 0.0),
    camera_gain=1.0,
    seed=42,
)


def quick_sim(
    *,
    n_emitters: int | None = None,
    photon_mean: float | None = None,
    drift: tuple[float, float] | None = None,
    seed: int | None = None,
) -> tuple[FrameStack, LocalizationResult]:
    """Run a fast simulation with sensible defaults, returning frames + localisations."""
    overrides: dict = {}
    if n_emitters is not None:
        overrides["n_emitters"] = n_emitters
    if photon_mean is not None:
        overrides["photon_mean"] = photon_mean
    if drift is not None:
        overrides["drift_per_frame_px"] = drift
    if seed is not None:
        overrides["seed"] = seed
    params = replace(QUICK_SIM_PARAMS, **overrides)
    stack = simulate_storm_frames(params)
    locs = localize_frame_stack(stack, params.psf_sigma_px, params.pixel_size_nm)
    return stack, locs


def frame_to_image_array(frame: np.ndarray) -> np.ndarray:
    """Normalise a single frame to uint8 grayscale (H, W) for visualisation."""
    f = frame.astype(np.float64)
    mn, mx = f.min(), f.max()
    if mx == mn:
        return np.zeros_like(f, dtype=np.uint8)
    return ((f - mn) / (mx - mn) * 255).astype(np.uint8)
