"""제 엄밀 감싸기(arb 공)와 저장된 CAPD 감싸기(정확 유리수 구간)를 정확 유리수로 비교한다.

서로소(disjoint)면 두 엄밀 감싸기 가운데 하나가 틀렸다는 증명이다. 포함(contained)은 제 감싸기가 저장 구간 안에 있다는 뜻이다.
"""
from __future__ import annotations

from fractions import Fraction

from flint import arb


def _exact(a: arb) -> Fraction:
    """반경 0 인 arb(정확 이진수)를 Fraction 으로."""
    m, e = a.mid().man_exp()
    m, e = int(m), int(e)
    return Fraction(m) * (Fraction(2) ** e)


def bounds(a: arb) -> tuple[Fraction, Fraction]:
    mid = _exact(arb(a.mid()))
    rad = _exact(arb(a.rad()))
    return mid - rad, mid + rad


def relation(mine: arb, lo: Fraction, hi: Fraction) -> dict:
    a, b = bounds(mine)
    out = {
        "disjoint": b < lo or a > hi,
        "contained": a >= lo and b <= hi,
        "rel_pos": float(((a + b) / 2 - lo) / (hi - lo)) if hi > lo else None,
        "gap_to_edge": float(min((a + b) / 2 - lo, hi - (a + b) / 2)),
        "mine_rad": float((b - a) / 2),
        "theirs_width": float(hi - lo),
    }
    return out


def fmt(a: arb, digits: int = 25) -> dict:
    lo, hi = bounds(a)
    return {"mid": a.mid().str(digits, radius=False), "rad": f"{float((hi - lo) / 2):.3e}"}
