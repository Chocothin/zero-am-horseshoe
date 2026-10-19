"""G8 독립 판정(G7 감사 대상): 원뿔 조건을 주 판정층(끝점 구간 곱 + Gershgorin)과 다른 감싸기·다른 양정치 판정으로 다시 본다.

감싸기는 중점·반경 꼴이다. A = D_c + E, |E| ≤ D_r 이면 |AᵀJA − D_cᵀJD_c| ≤ |D_c|ᵀD_r + D_rᵀ|D_c| + D_rᵀD_r =: Δ(성분별, |J|=I).
양정치는 Weyl 부등식 λ_min(C+F) ≥ λ_min(C) − ‖F‖₂ 와 ‖F‖₂ ≤ ‖Δ‖₂ ≤ max 행합(Δ) 로 본다. C − tI 의 양정치는 정확 유리수
Cholesky(피벗이 모두 양수)로 판정한다. 그래서 C=D_cᵀJD_c−μJ, t=max 행합(Δ) 에서 C−tI 가 양정치면 V_μ 의 모든 원소가 양정치다.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from t3cap.exact import Interval

SIGNATURE = (1, 1, -1, -1)


def midpoint_radius(d: Sequence[Sequence[Interval]]) -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    if len(d) != 4 or any(len(row) != 4 for row in d):
        raise ValueError("midpoint_radius: derivative enclosure must be 4x4")
    return ([[iv.mid for iv in row] for row in d], [[iv.radius for iv in row] for row in d])


def cholesky_positive(m: Sequence[Sequence[Fraction]]) -> bool:
    """정확 유리수 LDLᵀ: 피벗이 모두 양수 ⟺ 대칭 행렬이 양정치."""
    a = [list(row) for row in m]
    n = len(a)
    for c in range(n):
        if a[c][c] <= 0:
            return False
        for r in range(c + 1, n):
            f = a[r][c] / a[c][c]
            a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return True


def certifies(d: Sequence[Sequence[Interval]], mu: Fraction) -> bool:
    c, r = midpoint_radius(d)
    ac = [[abs(x) for x in row] for row in c]
    centre = [[sum(SIGNATURE[k] * c[k][i] * c[k][j] for k in range(4)) - (mu * SIGNATURE[i] if i == j else 0)
               for j in range(4)] for i in range(4)]
    delta = [[sum(ac[k][i] * r[k][j] + r[k][i] * ac[k][j] + r[k][i] * r[k][j] for k in range(4)) for j in range(4)]
             for i in range(4)]
    t = max(sum(row) for row in delta)
    return cholesky_positive([[centre[i][j] - (t if i == j else 0) for j in range(4)] for i in range(4)])
