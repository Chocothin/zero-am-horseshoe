"""T3-cap: 평면 3체(축소 좌표) Poincaré 사상 CAP 도구 사슬. 공개 API 는 api·verdict·export·model·exact 에 있다."""
from .api import (
    C0Enclosure,
    C1Enclosure,
    ChartEnclosure,
    Crossings,
    Failure,
    TubeBounds,
    VectorFieldEnclosure,
    chart,
    face_boxes,
    poincare_c0,
    poincare_c1,
    vector_field,
)
from .exact import Box, Interval, box_around, point_box
from .model import COORD_NAMES, PR, PX, PY, R, X, Y, CoordinateSection, Frame, Settings, SystemParams

__all__ = [
    "Box", "C0Enclosure", "C1Enclosure", "COORD_NAMES", "ChartEnclosure", "CoordinateSection", "Crossings", "Failure",
    "Frame",
    "Interval", "PR", "PX", "PY", "R", "Settings", "SystemParams", "TubeBounds", "VectorFieldEnclosure", "X", "Y",
    "box_around", "chart", "face_boxes", "point_box", "poincare_c0", "poincare_c1", "vector_field",
]
