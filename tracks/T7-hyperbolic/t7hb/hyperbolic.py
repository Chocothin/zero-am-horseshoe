"""정리 A′ 조립(G7 감사 대상): 변마다 원뿔 판정을 모으고 cycle h-set 16개의 서로소(A6)를 잰 뒤, 보고서가 적은 정리 수준
[CAP] 주장(A5 전 변 원뿔, μ_u, μ_s, A6, 쌍곡 상수 4개)을 그 값으로 판정한다(judge_claims). 판정 결과는 Assembly·ClaimCheck 가 운반한다.

μ=1 의 Gershgorin 여유는 min(b*−1, 1−a*) 라서, μ_s<1<μ_u 와 a*<μ_s, b*>μ_u 가 성립하면 ε=min(μ_u−1, 1−μ_s) 이상이다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

from t6hs import assembly as t6assembly

from .cone import ConeBounds, cone_bounds
from .stored import ConeInput

PERIOD = 16
ONE = Fraction(1)


def cycle_names() -> tuple[str, ...]:
    return tuple(f"N{i}" for i in range(PERIOD))


@dataclass(frozen=True)
class EdgeCone:
    edge: str
    usable: bool
    problems: tuple[str, ...]
    bounds: ConeBounds | None

    @property
    def certified(self) -> bool:
        return self.usable and self.bounds is not None and self.bounds.margin(ONE) > 0


@dataclass(frozen=True)
class Assembly:
    edges: tuple[EdgeCone, ...]
    missing: tuple[str, ...]            # 설계 변 가운데 판정 입력이 없는 것
    lower_max: Fraction | None          # max_e a*_e
    lower_edge: str | None
    upper_min: Fraction | None          # min_e b*_e
    upper_edge: str | None
    cycle_gap: Fraction | None          # cycle h-set 외피 사이 분리 간격의 최솟값(양수일 때만 서로소)
    cycle_pair: tuple[str, str] | None

    @property
    def uncertified(self) -> tuple[str, ...]:
        return tuple(e.edge for e in self.edges if not e.certified)

    @property
    def certified(self) -> bool:
        return not self.missing and not self.uncertified and self.cycle_gap is not None and self.cycle_gap > 0


@dataclass(frozen=True)
class Claimed:
    """보고서가 정리 A′ 에 적은 상수(데이터와 독립으로 고정한 값)."""

    mu_s: Fraction       # 모든 변에서 V_{μ_s} 양정치(안정 원뿔 |Q| 축소율 상한)
    mu_u: Fraction       # 모든 변에서 V_{μ_u} 양정치(불안정 원뿔 Q 팽창률 하한)
    lambda_s: Fraction   # 안정 방향 걸음당 축소율 상한, λ_s² ≥ μ_s
    lambda_u: Fraction   # 불안정 방향 걸음당 팽창률 하한, λ_u² ≤ μ_u
    c_s: Fraction        # 안정 방향 상수, c_s² ≥ 1/ε
    c_u: Fraction        # 불안정 방향 상수, c_u² ≥ μ_u/ε


@dataclass(frozen=True)
class ClaimCheck:
    name: str
    holds: bool
    margin: Fraction | None   # 다시 잰 값과 주장의 부호 있는 차(성립하는 쪽이 양). 잴 값이 없으면 None


def _extreme(pairs, pick):
    pairs = [p for p in pairs if p[1] is not None]
    if not pairs:
        return None, None
    key, value = pick(pairs, key=lambda p: p[1])
    return value, key


def assemble(inputs: Iterable[ConeInput], design_edges: Iterable[str], hulls: dict[str, list]) -> Assembly:
    cones = tuple(EdgeCone(c.edge, c.usable, c.problems, None if c.derivative is None else cone_bounds(c.derivative))
                  for c in inputs)
    missing = tuple(sorted(set(design_edges) - {c.edge for c in cones}))
    with_bounds = [c for c in cones if c.bounds is not None]
    lower, lower_edge = _extreme([(c.edge, c.bounds.lower) for c in with_bounds], max)
    upper, upper_edge = _extreme([(c.edge, c.bounds.upper) for c in with_bounds], min)
    names = cycle_names()
    pairs = [((a, b), t6assembly.separation(hulls[a], hulls[b])) for i, a in enumerate(names) for b in names[i + 1:]]
    gap, pair = _extreme(pairs, min)
    return Assembly(cones, missing, lower, lower_edge, upper, upper_edge, gap, pair)


def judge_claims(a: Assembly, claimed: Claimed) -> tuple[ClaimCheck, ...]:
    """정리 A′ 의 [CAP] 주장 8개. μ 주장은 Gershgorin 이 엄격 부등식이라 a*<μ_s, b*>μ_u 를 엄격하게 본다.
    쌍곡 상수 4개는 §5.5 의 추정식(√μ_s, √μ_u, ε^{-1/2}, (μ_u/ε)^{1/2})을 받치는지 정확 유리수로 보고, μ 주장에 기댄다."""
    every = not a.missing and not a.uncertified and len(a.edges) > 0
    upper = None if a.upper_min is None else a.upper_min - claimed.mu_u
    lower = None if a.lower_max is None else claimed.mu_s - a.lower_max
    mu_u_holds = upper is not None and claimed.mu_u > ONE and upper > 0
    mu_s_holds = lower is not None and 0 < claimed.mu_s < ONE and lower > 0
    # 두 μ 주장이 성립하면 μ_s<1<μ_u 라서 ε>0 이므로 ε 부호는 따로 보지 않는다
    eps = derived_eps(claimed)
    both = mu_u_holds and mu_s_holds
    rate_s = claimed.lambda_s ** 2 - claimed.mu_s
    rate_u = claimed.mu_u - claimed.lambda_u ** 2
    const_s = claimed.c_s ** 2 * eps - ONE
    const_u = claimed.c_u ** 2 * eps - claimed.mu_u
    return (
        ClaimCheck("A5_all_edges_cone", every, Fraction(-(len(a.missing) + len(a.uncertified)))),
        ClaimCheck("mu_u", mu_u_holds, upper),
        ClaimCheck("mu_s", mu_s_holds, lower),
        ClaimCheck("A6_cycle_disjoint", a.cycle_gap is not None and a.cycle_gap > 0, a.cycle_gap),
        ClaimCheck("lambda_s", mu_s_holds and 0 < claimed.lambda_s < ONE and rate_s >= 0, rate_s),
        ClaimCheck("lambda_u", mu_u_holds and claimed.lambda_u > ONE and rate_u >= 0, rate_u),
        ClaimCheck("C_s", both and const_s >= 0, const_s),
        ClaimCheck("C_u", both and const_u >= 0, const_u),
    )


def eps_of(mu_s: Fraction, mu_u: Fraction) -> Fraction:
    """μ=1 에서 모든 변의 λ_min(AᵀJA − J) 하한(μ 주장이 성립할 때)."""
    return min(mu_u - ONE, ONE - mu_s)


def derived_eps(claimed: Claimed) -> Fraction:
    return eps_of(claimed.mu_s, claimed.mu_u)


def sqrt_bound(q: Fraction, digits: int, upper: bool) -> Fraction:
    """√q 를 10^-digits 격자에서 위(upper)나 아래로 자른 정확 유리수. ⌊√x⌋ = isqrt(⌊x⌋) 로 정수만 쓴다."""
    if q < 0:
        raise ValueError("sqrt_bound: negative argument")
    scaled = q * 10 ** (2 * digits)
    n = math.isqrt(scaled.numerator // scaled.denominator)
    exact = Fraction(n * n) == scaled
    return Fraction(n + (1 if upper and not exact else 0), 10 ** digits)


def rate_constants(mu_s: Fraction, mu_u: Fraction, rate_digits: int = 4, const_digits: int = 3) -> dict[str, Fraction]:
    """보고서에 적을 쌍곡 상수를 안전한 쪽으로 방향 반올림한다(축소율·상수는 위로, 팽창률은 아래로)."""
    eps = eps_of(mu_s, mu_u)
    return {"lambda_s": sqrt_bound(mu_s, rate_digits, upper=True),
            "lambda_u": sqrt_bound(mu_u, rate_digits, upper=False),
            "c_s": sqrt_bound(ONE / eps, const_digits, upper=True),
            "c_u": sqrt_bound(mu_u / eps, const_digits, upper=True)}
