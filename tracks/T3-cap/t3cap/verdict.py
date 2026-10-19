"""부등식 판정(G7 mutation 감사 대상). 엔진 감싸기를 정확 유리수로 받아 반올림 없이 판정하고, 결과는 Verdict 가 운반한다."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from .api import C0Enclosure, C1Enclosure
from .exact import Box, Interval, IntervalMatrix, RationalMatrix, identity, rmat_imat, rmat_ivec


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class Verdict:
    proven: bool
    checks: tuple[Check, ...]

    @property
    def failed(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if not c.passed)


def lower_bound_exceeds(iv: Interval, threshold: Fraction) -> bool:
    """iv 의 모든 원소가 threshold 보다 엄격히 크다."""
    return iv.lo > threshold


def subset_interior(inner: Sequence[Interval], outer: Sequence[Interval]) -> bool:
    """inner 가 outer 의 내부(각 성분 열린 구간)에 들어간다 — Krawczyk·덮개 판정의 핵심 부등식."""
    if len(inner) != len(outer):
        raise ValueError("dimension mismatch")
    return all(o.lo < i.lo and i.hi < o.hi for i, o in zip(inner, outer))


def verdict_of(checks: Sequence[Check]) -> Verdict:
    return Verdict(all(c.passed for c in checks) and len(checks) > 0, tuple(checks))


def certify_return(result: C0Enclosure | C1Enclosure, *, min_distance: Fraction) -> Verdict:
    """감싸기를 증명 근거로 쓸 수 있는지: 출발 차트 정의(⑤), 도착 차트 근(⑦ 게이지 포함), 반환 시간, 흐름 관 무충돌(④)."""
    tube = result.tube
    checks = [
        Check("chart_discriminant_positive", lower_bound_exceeds(result.discriminant, Fraction(0)),
              f"D in {result.discriminant}"),
        Check("target_branch_positive", lower_bound_exceeds(result.branch_out, Fraction(0)),
              f"branch*dH/dy_m in {result.branch_out}"),
        Check("return_time_positive", lower_bound_exceeds(result.return_time, Fraction(0)),
              f"T in {result.return_time}"),
        Check("tube_covers_return", tube.t_end.lo >= result.return_time.hi,
              f"t_end {tube.t_end} vs T {result.return_time}"),
        Check("tube_r12_clear", lower_bound_exceeds(tube.r12, min_distance), f"r12 in {tube.r12}"),
        Check("tube_r13_clear", lower_bound_exceeds(tube.r13, min_distance), f"r13 in {tube.r13}"),
        Check("tube_r23_clear", lower_bound_exceeds(tube.r23, min_distance), f"r23 in {tube.r23}"),
    ]
    return verdict_of(checks)


@dataclass(frozen=True)
class KrawczykVerdict:
    proven: bool
    image: Box               # K(center, box)
    checks: tuple[Check, ...]


def krawczyk(
    center: Sequence[Fraction],
    box: Box,
    f_center: Box,
    df_box: IntervalMatrix,
    preconditioner: RationalMatrix,
) -> KrawczykVerdict:
    """K = c - C F(c) + (I - C DF(X))(X - c). c 가 X 안에 있고 K 가 X 의 내부에 들어가면 X 안에 F 의 영점이 유일하게 있다.

    F(c)·DF(X) 감싸기의 엄밀성(certify_return 통과)은 호출자 책임이다. C 는 아무 정확 행렬이어도 결론이 엄밀하다.
    """
    n = len(box)
    square = len(df_box) == n and all(len(row) == n for row in df_box)
    square_c = len(preconditioner) == n and all(len(row) == n for row in preconditioner)
    # 차원 조건은 여기 한 곳에서만 본다 — 아래 계산은 길이가 맞다고 가정한다(어긋나면 조용히 잘려 거짓 증명이 된다)
    if len(center) != n or len(f_center) != n or not square or not square_c:
        raise ValueError("krawczyk: center, box, F(c), DF(X), C must share dimension n")
    c_box = tuple(Interval(q, q) for q in center)
    deviation = tuple(x - c for x, c in zip(box, c_box))
    eye = identity(n)
    c_df = rmat_imat(preconditioner, df_box)
    residual = tuple(
        tuple(Interval(eye[i][j], eye[i][j]) - c_df[i][j] for j in range(n)) for i in range(n)
    )
    newton_step = rmat_ivec(preconditioner, f_center)
    contraction = tuple(_dot(residual[i], deviation) for i in range(n))
    image = tuple(c - s + t for c, s, t in zip(c_box, newton_step, contraction))
    checks = (
        Check("center_in_box", all(b.contains(q) for b, q in zip(box, center)), "c in X"),
        Check("krawczyk_in_interior", subset_interior(image, box), f"K={image}"),
    )
    return KrawczykVerdict(all(ch.passed for ch in checks), image, checks)


def _dot(row: Sequence[Interval], vec: Sequence[Interval]) -> Interval:
    total = Interval(Fraction(0), Fraction(0))
    for a, b in zip(row, vec):
        total = total + a * b
    return total
