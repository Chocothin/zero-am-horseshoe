"""저장 기록에서 덮개 여유와 원뿔(Gershgorin) 상수를 정확 유리수로 다시 잰다(T6·T7 코드를 쓰지 않는다).

덮개(ZG2004 Thm 15 의 충분조건, T6 REPORT §4.1 이 적은 판정): 상자 상에서 「불안정 성분 하나가 통째로 |w'|>1」 또는
「안정 성분 둘이 통째로 (−1,1) 안」, 출구 면 {w_j=σ} 상에서 s_j·σ·w'_j > 1. 여유는 문턱 1 까지의 거리다.
원뿔: S = DᵀJD (J=diag(1,1,−1,−1)), 대각은 제곱 감싸기, R_i = Σ_{j≠i}|S_ij|,
a* = max_{i∈{2,3}}(R_i − S_ii.lo), b* = min_{i∈{0,1}}(S_ii.lo − R_i).
"""
from __future__ import annotations

from fractions import Fraction

from . import data

ONE = Fraction(1)
SIG = (1, 1, -1, -1)
MU_S = Fraction("0.8944")
MU_U = Fraction("1.1167")


def iv_mul(a, b):
    p = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return (min(p), max(p))


def iv_sq(a):
    lo, hi = a
    if lo >= 0:
        return (lo * lo, hi * hi)
    if hi <= 0:
        return (hi * hi, lo * lo)
    return (Fraction(0), max(lo * lo, hi * hi))


def iv_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def mag(a):
    return max(abs(a[0]), abs(a[1]))


def gershgorin(Dm):
    """Dm: 4×4 (lo,hi) 정확 유리수. (a*, b*, S대각 하한, R) 반환."""
    S = [[None] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(i, 4):
            acc = (Fraction(0), Fraction(0))
            for k in range(4):
                term = iv_sq(Dm[k][i]) if i == j else iv_mul(Dm[k][i], Dm[k][j])
                if SIG[k] < 0:
                    term = (-term[1], -term[0])
                acc = iv_add(acc, term)
            S[i][j] = S[j][i] = acc
    R = [sum((mag(S[i][j]) for j in range(4) if j != i), Fraction(0)) for i in range(4)]
    a_star = max(R[i] - S[i][i][0] for i in (2, 3))
    b_star = min(S[i][i][0] - R[i] for i in (0, 1))
    return a_star, b_star, [S[i][i][0] for i in range(4)], R


def derivative_of(rec):
    return [[data.interval(rec["derivative"][i][j]) for j in range(4)] for i in range(4)]


def covering_margin(rec):
    """상자·출구 면 판정 여유의 최솟값과 항목별 값."""
    det = rec["pieces_detail"]
    signs = rec["signs"]
    whole = [data.interval(v) for v in det["whole"][0]["image"]]
    out = max(max(whole[j][0] - ONE, -ONE - whole[j][1]) for j in (0, 1))
    inside = min(min(whole[k][0] + ONE, ONE - whole[k][1]) for k in (2, 3))
    items = {"whole": max(out, inside), "whole_stable_inside": inside}
    for j in (0, 1):
        for s in (-1, 1):
            img = [data.interval(v) for v in det[f"face_w{j}{s:+d}"][0]["image"]]
            items[f"face_w{j}{s:+d}"] = img[j][0] - ONE if signs[j] * s > 0 else -ONE - img[j][1]
    return min(items.values()), items
