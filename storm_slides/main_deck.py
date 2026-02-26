"""Master deck module with all ordered STORM slide scenes."""

from __future__ import annotations

from storm_slides.algorithm_scenes import (
    BiologicalExamplesSlide as _BiologicalExamplesSlide,
    ConclusionChecklistSlide as _ConclusionChecklistSlide,
    LimitationsFrontierSlide as _LimitationsFrontierSlide,
    LocalizationAlgorithmSlide as _LocalizationAlgorithmSlide,
    SimulatorLabSlide as _SimulatorLabSlide,
    ThreeDExtensionSlide as _ThreeDExtensionSlide,
)
from storm_slides.instrument_scenes import (
    CameraStatisticsSlide as _CameraStatisticsSlide,
    MicroscopeArchitectureSlide as _MicroscopeArchitectureSlide,
)
from storm_slides.physics_scenes import (
    FourierOpticsSlide as _FourierOpticsSlide,
    MaxwellToHelmholtzSlide as _MaxwellToHelmholtzSlide,
    OpeningRoadmapSlide as _OpeningRoadmapSlide,
    TemporalSparsitySlide as _TemporalSparsitySlide,
)


class OpeningRoadmapSlide(_OpeningRoadmapSlide):
    pass


class MaxwellToHelmholtzSlide(_MaxwellToHelmholtzSlide):
    pass


class FourierOpticsSlide(_FourierOpticsSlide):
    pass


class TemporalSparsitySlide(_TemporalSparsitySlide):
    pass


class MicroscopeArchitectureSlide(_MicroscopeArchitectureSlide):
    pass


class CameraStatisticsSlide(_CameraStatisticsSlide):
    pass


class LocalizationAlgorithmSlide(_LocalizationAlgorithmSlide):
    pass


class SimulatorLabSlide(_SimulatorLabSlide):
    pass


class ThreeDExtensionSlide(_ThreeDExtensionSlide):
    pass


class BiologicalExamplesSlide(_BiologicalExamplesSlide):
    pass


class LimitationsFrontierSlide(_LimitationsFrontierSlide):
    pass


class ConclusionChecklistSlide(_ConclusionChecklistSlide):
    pass

DECK_SCENES = [
    OpeningRoadmapSlide,
    MaxwellToHelmholtzSlide,
    FourierOpticsSlide,
    TemporalSparsitySlide,
    MicroscopeArchitectureSlide,
    CameraStatisticsSlide,
    LocalizationAlgorithmSlide,
    SimulatorLabSlide,
    ThreeDExtensionSlide,
    BiologicalExamplesSlide,
    LimitationsFrontierSlide,
    ConclusionChecklistSlide,
]

SCENE_NAME_ORDER = [scene.__name__ for scene in DECK_SCENES]

__all__ = ["DECK_SCENES", "SCENE_NAME_ORDER", *SCENE_NAME_ORDER]
