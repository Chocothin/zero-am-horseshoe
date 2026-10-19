"""정리 A 가정 조립(G7 감사 대상): 저장 감싸기의 재판정·그래프 구조·M₀ 분리·흐름 관 상수·엔트로피 하한을 한 곳에서 모으고,
보고서가 적은 정리 수준 [CAP] 주장 7개를 그 값으로 판정한다(judge_claims).

덮개 증명 여부는 저장된 proven 표시가 아니라 records.covering_input 으로 꺼낸 감싸기를 covering_verdict 로 다시 판정한
결과를 쓴다. 판정 결과는 Assembly·ClaimCheck 가 운반한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

from .covering import covering_verdict
from .records import CoveringInput

PERIOD = 16


@dataclass(frozen=True)
class Assembly:
    unproven: tuple[str, ...]        # 재판정이 실패했거나 저장값 사용 조건이 깨진 변
    missing: tuple[str, ...]
    graph_diff: int                  # 설계 변 집합과 기대 그래프의 대칭차 크기
    m0_overlaps: tuple[str, ...]
    m0_gap: Fraction | None          # M₀ 외피와 다른 외피 사이 분리 간격의 최솟값(양수일 때만 서로소)
    delta: Fraction | None           # 흐름 관 r_ij 하한의 최솟값
    rmax: Fraction | None            # 흐름 관 r_ij 상한의 최댓값
    tmin: Fraction | None            # 귀환 시각 하한의 최솟값
    loops: tuple[int, int]
    rho_lower: Fraction

    @property
    def graph_matches(self) -> bool:
        return self.graph_diff == 0

    @property
    def certified(self) -> bool:
        return (not self.unproven and not self.missing and self.graph_matches
                and self.m0_gap is not None and self.m0_gap > 0
                and self.delta is not None and self.delta > 0 and self.tmin is not None and self.tmin > 0)


@dataclass(frozen=True)
class Claimed:
    """보고서가 정리 A 에 적은 상수(데이터와 독립으로 고정한 값)."""

    delta: Fraction      # r_ij ≥ delta
    rmax: Fraction       # r_ij ≤ rmax
    tmin: Fraction       # 연이은 syzygy 사이 시간 ≥ tmin
    rho: Fraction        # ρ ≥ rho
    ln_rho: Fraction     # ln ρ ≥ ln_rho


@dataclass(frozen=True)
class ClaimCheck:
    name: str
    holds: bool
    margin: Fraction | None   # 다시 잰 값과 주장의 부호 있는 차(성립하는 쪽이 양). 잴 값이 없으면 None


def hull(center, matrix) -> list[tuple[Fraction, Fraction]]:
    """c + M·[−1,1]^4 의 축 정렬 외피(정확)."""
    return [(center[i] - sum(abs(x) for x in matrix[i]), center[i] + sum(abs(x) for x in matrix[i]))
            for i in range(len(center))]


def separation(a, b) -> Fraction:
    """외피 둘이 어느 한 축에서 떨어진 거리의 최댓값. 양수일 때만 서로소다(닿으면 0, 겹치면 음수)."""
    return max(max(y[0] - x[1], x[0] - y[1]) for x, y in zip(a, b))


def disjoint(a, b) -> bool:
    return separation(a, b) > 0


def expected_edges(k: int, departure: int, arrival: int) -> set[tuple[str, str]]:
    edges = {(f"N{i}", f"N{(i + 1) % PERIOD}") for i in range(PERIOD)}
    edges |= {(f"N{departure}", f"M{-k + 1:+d}"), (f"M{k - 1:+d}", f"N{arrival}")}
    edges |= {(f"M{j:+d}", f"M{j + 1:+d}") for j in range(-k + 1, k - 1)}
    return edges


def loop_lengths(k: int, departure: int, arrival: int) -> tuple[int, int]:
    """(주기 순환, 호모클리닉 순환): N_jd → M₋ₖ₊₁ … Mₖ₋₁ → N_ja 가 2k 변, N_ja 에서 N_jd 까지 cycle 을 돈다."""
    return PERIOD, 2 * k + (departure - arrival) % PERIOD


def rho_lower_bound(l1: int, l2: int, iterations: int = 80) -> Fraction:
    """x^{−l1}+x^{−l2}=1 의 근 ρ∈(1,2]: F(lo)>1 을 유지하는 정확 유리수 이분법이라 lo ≤ ρ 가 보장된다."""
    lo, hi = Fraction(1), Fraction(2)
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if mid ** (-l1) + mid ** (-l2) > 1:
            lo = mid
        else:
            hi = mid
    return lo


def ln_lower_bound(rho: Fraction, terms: int = 30) -> Fraction:
    """ln ρ = 2·Σ z^{2n+1}/(2n+1), z=(ρ−1)/(ρ+1): ρ>1 이면 항이 모두 양수라 정확 유리수 부분합이 하한이다."""
    z = (rho - 1) / (rho + 1)
    return 2 * sum(z ** (2 * n + 1) / (2 * n + 1) for n in range(terms))


def floor_decimal(x: Fraction, digits: int = 25) -> str:
    """정확 유리수 하한을 아래로 자른 소수 문자열로(반올림하면 하한이 깨질 수 있다). x ≥ 0 에서만 쓴다."""
    n = math.floor(x * 10**digits)
    s = str(n).rjust(digits + 1, "0")
    return f"{s[:-digits]}.{s[-digits:]}"


def assemble(inputs: Iterable[CoveringInput], design_edges: set[tuple[str, str]], k: int, departure: int,
             arrival: int, hulls: dict[str, list], marker: str = "M+0") -> Assembly:
    inputs = list(inputs)
    unproven = tuple(c.edge for c in inputs
                     if c.problems or not covering_verdict(c.whole, c.faces, c.signs).proven)
    done = {tuple(c.edge.split(">")) for c in inputs}
    missing = tuple(sorted(f"{s}>{t}" for s, t in design_edges - done))
    graph_diff = len(design_edges ^ expected_edges(k, departure, arrival))
    gaps = {n: separation(hulls[marker], h) for n, h in hulls.items() if n != marker}
    overlaps = tuple(n for n, g in gaps.items() if not g > 0)
    delta = min((c.tube_lower for c in inputs), default=None)
    rmax = max((c.tube_upper for c in inputs), default=None)
    tmin = min((c.return_time_lower for c in inputs), default=None)
    loops = loop_lengths(k, departure, arrival)
    return Assembly(unproven, missing, graph_diff, overlaps, min(gaps.values(), default=None), delta, rmax, tmin,
                    loops, rho_lower_bound(*loops))


def judge_claims(a: Assembly, claimed: Claimed) -> tuple[ClaimCheck, ...]:
    """정리 A 의 [CAP] 주장 7개. ρ 는 F(x)=x^{−l1}+x^{−l2} 가 (1,∞) 에서 감소하므로 F(rho) ≥ 1 ⟺ rho ≤ 근이다."""
    delta = None if a.delta is None else a.delta - claimed.delta
    rmax = None if a.rmax is None else claimed.rmax - a.rmax
    tmin = None if a.tmin is None else a.tmin - claimed.tmin
    f_rho = claimed.rho ** (-a.loops[0]) + claimed.rho ** (-a.loops[1]) - 1
    rho_ok = claimed.rho > 1 and f_rho >= 0
    ln_margin = ln_lower_bound(claimed.rho) - claimed.ln_rho
    return (
        ClaimCheck("delta", delta is not None and claimed.delta > 0 and delta >= 0, delta),
        ClaimCheck("R", rmax is not None and rmax >= 0, rmax),
        ClaimCheck("t_min", tmin is not None and claimed.tmin > 0 and tmin >= 0, tmin),
        ClaimCheck("A3_m0_disjoint", a.m0_gap is not None and a.m0_gap > 0, a.m0_gap),
        ClaimCheck("A4_graph", a.graph_matches, Fraction(-a.graph_diff)),
        ClaimCheck("rho", rho_ok, f_rho),
        ClaimCheck("ln_rho", rho_ok and ln_margin >= 0, ln_margin),
    )
