# STORM End-to-End Manim Slides Deck

A 3Blue1Brown-quality `manim-slides` presentation (~25-30 minutes) covering the complete STORM (Stochastic Optical Reconstruction Microscopy) pipeline:

- Electromagnetic foundations (Maxwell → wave equation → Helmholtz)
- Fourier optics, NA-limited pupil, PSF, and diffraction limit
- STORM blinking mechanism and temporal sparsity
- Microscope hardware: objective, dichroic, filters, camera
- Photoswitching energy-level diagram (Jablonski)
- Poisson camera model and photon-counting statistics
- Gaussian MLE localization pipeline and precision formula
- Monte Carlo simulator with photon / density / drift sweeps
- 3D STORM via astigmatic PSF and z-calibration curves
- Biological examples: diffraction-limited vs STORM comparison
- Current limitations and research frontier

## Repository Layout

- `storm_slides/` — presentation scenes, simulator, custom Mobjects & animations
- `notes/speaker_notes.md` — per-slide narration and timing cues
- `scripts/` — render / present PowerShell helpers
- `tests/` — simulator and timing-contract tests
- `environment.yml` — conda environment specification

## Environment Setup

Requires [Miniforge](https://github.com/conda-forge/miniforge) (or any conda distribution).

1. **Create the conda environment:**

```powershell
conda env create -f environment.yml -y
```

2. **Activate it and install the project in dev mode:**

```powershell
conda activate storm
pip install -e ".[dev]"
```

3. **Verify scene list:**

```powershell
python -m storm_slides
```

> **Note:** MiKTeX (or another LaTeX distribution) must be on PATH for equation rendering.

## Rendering

Preview render (1080p30, fast):

```powershell
.\scripts\render_preview.ps1
```

Final render (4K30, production):

```powershell
.\scripts\render_full.ps1
```

Present slides interactively:

```powershell
.\scripts\present.ps1
```

## Testing

```powershell
python -m pytest
```

## Notes

- Target runtime is 25-30 minutes (contract enforced in `storm_slides/schedule.py`).
- Slide modules include no-op fallbacks so simulator tests run even without `manim` installed.
- The conda environment pins Python ≥3.11, <3.14 for full manim compatibility.
