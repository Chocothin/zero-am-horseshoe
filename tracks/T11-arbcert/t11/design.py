"""graph_design.json 읽기(h-set 정의와 변 목록). h-set 은 감사 사본(v5/data.py)과 같은 정본 정의를 정확 유리수로 다시 읽는다."""
from __future__ import annotations

import json
from fractions import Fraction
from functools import lru_cache

from .paths import DESIGN

EDGE_TYPES = ("cycle", "junction_out", "back_tail", "excursion", "front_tail", "junction_in")


@lru_cache(maxsize=1)
def design() -> dict:
    return json.loads(DESIGN.read_text())


def edges() -> list[str]:
    return [f"{e['source']}>{e['target']}" for e in design()["edges"]]


def hset(name: str):
    """중심 c(4), 행렬 M = F·diag(r)(4×4), 모두 정확 유리수(16진 double 과 double 반경의 정확값)."""
    v = design()["vertices"][name]
    c = [Fraction(float.fromhex(h)) for h in v["center_hex"]]
    f = [[Fraction(float.fromhex(h)) for h in row] for row in v["frame_hex"]]
    r = [Fraction(float(x)) for x in v["radii"]]
    return c, [[f[i][j] * r[j] for j in range(4)] for i in range(4)]


def chain_index(vertex: str) -> int | None:
    return int(vertex[1:]) if vertex.startswith("M") else None


def edge_type(edge: str) -> str:
    """T6 REPORT §5 의 구간 이름: cycle 16, 출발 접합 1, 뒤쪽 꼬리 129, 소풍 39, 앞쪽 꼬리 128, 도착 접합 1."""
    s, t = edge.split(">")
    if s.startswith("N") and t.startswith("N"):
        return "cycle"
    if s.startswith("N"):
        return "junction_out"
    if t.startswith("N"):
        return "junction_in"
    k = chain_index(s)
    if k <= -20:
        return "back_tail"
    if k <= 19:
        return "excursion"
    return "front_tail"
