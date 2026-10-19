"""T6 원자료 JSON 읽기(T6·T3 코드를 쓰지 않는다). 끝점은 16진 double 또는 정확 유리수 문자열이다."""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from flint import arb, fmpq

REPO = Path(__file__).resolve().parents[3]
R6 = REPO / "results" / "2026-10-15-T6-horseshoe"
R7 = REPO / "results" / "2026-10-16-T7-hyperbolic"


def endpoint(tok: str) -> Fraction:
    t = tok.strip()
    if t.lower().lstrip("+-").startswith("0x"):
        return Fraction(float.fromhex(t))
    return Fraction(t)


def interval(d: dict) -> tuple[Fraction, Fraction]:
    return endpoint(d["lo"]), endpoint(d["hi"])


def q2arb(q: Fraction) -> arb:
    return arb(fmpq(q.numerator, q.denominator))


def iv2arb(lo: Fraction, hi: Fraction) -> arb:
    """[lo, hi] 를 담는 공(끝점을 바깥으로 감싼다)."""
    a, b = q2arb(lo), q2arb(hi)
    return a.union(b)


def design():
    return json.loads((R6 / "graph_design.json").read_text())


def hset(vertex: dict):
    """중심 c(4), 행렬 M = F·diag(r)(4×4), 모두 정확 유리수."""
    c = [Fraction(float.fromhex(h)) for h in vertex["center_hex"]]
    f = [[Fraction(float.fromhex(h)) for h in row] for row in vertex["frame_hex"]]
    r = [Fraction(float(v)) for v in vertex["radii"]]
    M = [[f[i][j] * r[j] for j in range(4)] for i in range(4)]
    return c, M


def edge_file(edge: str) -> Path:
    src, dst = edge.split(">")
    return R6 / "coverings" / f"{src}__{dst}.json"


def record(edge: str) -> dict:
    return json.loads(edge_file(edge).read_text())


def edges_from_design(d) -> list[str]:
    return [f"{e['source']}>{e['target']}" for e in d["edges"]]
