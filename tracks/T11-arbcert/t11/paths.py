"""T11 경로 상수(한 곳에서 선언)."""
from __future__ import annotations

from pathlib import Path

TRACK = Path(__file__).resolve().parents[1]
REPO = TRACK.parents[1]
RESULTS = REPO / "results" / "2026-10-17-T11-arbcert"
DESIGN = REPO / "results" / "2026-10-15-T6-horseshoe" / "graph_design.json"
TUBES = RESULTS / "tubes"
BOXES = RESULTS / "stage2"
RAW = RESULTS / "raw-local"
PYTHON = REPO / ".venv" / "bin" / "python"
BLAS_ONE = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1"}


def edge_stem(edge: str) -> str:
    return edge.replace(">", "__")
