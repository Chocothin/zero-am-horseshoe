"""원뿔 조건 판정층(G7 감사 대상). 정규 좌표 도함수 감싸기 D 에 대해 V_μ = DᵀJD − μJ 의 양정치를 정확 유리수 구간 산술과
Gershgorin 원판으로 판정한다. J = diag(1,1,−1,−1) 이고 μ=1 이 Z2009 Lemma 8 의 V 다.

D 안의 모든 실행렬 A 에 대해 AᵀJA − μJ 의 성분은 구간 S − μJ 안에 있으므로, 행마다 대각 하한 − 비대각 크기 합이 양수면
그 행렬들의 최소 고윳값이 그 최솟값 이상이다. J 가 대각이라 μ 는 대각에만 들어가서, 인증되는 μ 는 열린 구간 (a*, b*) 다.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from t3cap.exact import Interval

UNSTABLE = (0, 1)
STABLE = (2, 3)
SIGNATURE = (1, 1, -1, -1)
ZERO = Interval(Fraction(0), Fraction(0))


def square(iv: Interval) -> Interval:
    """{a² : a∈iv} 의 정확한 감싸기 — iv·iv 는 0 을 품을 때 음의 하한을 내므로 따로 둔다."""
    if iv.lo >= 0:
        return Interval(iv.lo * iv.lo, iv.hi * iv.hi)
    if iv.hi <= 0:
        return Interval(iv.hi * iv.hi, iv.lo * iv.lo)
    return Interval(Fraction(0), max(iv.lo * iv.lo, iv.hi * iv.hi))


def gram(d: Sequence[Sequence[Interval]]) -> tuple[tuple[Interval, ...], ...]:
    """S = DᵀJD 를 위 삼각만 계산해 대칭으로 채운다(같은 식이라 S_ij = S_ji)."""
    if len(d) != 4 or any(len(row) != 4 for row in d):
        raise ValueError("gram: derivative enclosure must be 4x4")
    s = [[ZERO] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(i, 4):
            acc = ZERO
            for k in range(4):
                term = square(d[k][i]) if i == j else d[k][i] * d[k][j]
                acc = acc + term if SIGNATURE[k] > 0 else acc - term
            s[i][j] = s[j][i] = acc
    return tuple(tuple(row) for row in s)


@dataclass(frozen=True)
class ConeBounds:
    diag_lower: tuple[Fraction, ...]   # S_ii.lo
    radii: tuple[Fraction, ...]        # R_i = Σ_{j≠i} |S_ij| 상계

    def margin(self, mu: Fraction) -> Fraction:
        """V_μ 의 Gershgorin 여유 = D 안 모든 행렬에서 λ_min(AᵀJA − μJ) 의 하한. 양수일 때만 양정치가 인증된다."""
        return min(self.diag_lower[i] - mu * SIGNATURE[i] - self.radii[i] for i in range(4))

    @property
    def lower(self) -> Fraction:
        """a*: 안정 행 여유가 양수이려면 μ > a* 여야 한다."""
        return max(self.radii[i] - self.diag_lower[i] for i in STABLE)

    @property
    def upper(self) -> Fraction:
        """b*: 불안정 행 여유가 양수이려면 μ < b* 여야 한다."""
        return min(self.diag_lower[i] - self.radii[i] for i in UNSTABLE)


def cone_bounds(d: Sequence[Sequence[Interval]]) -> ConeBounds:
    s = gram(d)
    radii = tuple(sum((s[i][j].magnitude() for j in range(4) if j != i), Fraction(0)) for i in range(4))
    return ConeBounds(tuple(s[i][i].lo for i in range(4)), radii)


@dataclass(frozen=True)
class ConeVerdict:
    proven: bool
    margin: Fraction       # μ 에서의 Gershgorin 여유(문턱을 빼기 전)
    bounds: ConeBounds


def cone_verdict(d: Sequence[Sequence[Interval]], mu: Fraction = Fraction(1),
                 threshold: Fraction = Fraction(0)) -> ConeVerdict:
    """문턱은 정리의 조건에서 0 이다(엄격 양정치). 다른 값은 음성 대조(문턱 강화)에서만 쓴다."""
    b = cone_bounds(d)
    m = b.margin(mu)
    return ConeVerdict(m > threshold, m, b)
