"""계(질량·에너지·J), 좌표 단면, 틀(affine frame), 적분 설정. 모든 수치 입력은 정확 유리수로 받는다(함정 ①)."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from .exact import Interval, RationalLike, RationalMatrix, identity, matrix_inverse, rational

# 축소 좌표 y = (r, x, y, pr, px, py) 의 성분 번호 — docs/T3-cap/REPORT.md 「축소 좌표」
R, X, Y, PR, PX, PY = range(6)
COORD_NAMES = ("r", "x", "y", "pr", "px", "py")


def as_interval(value: Interval | RationalLike) -> Interval:
    return value if isinstance(value, Interval) else Interval.point(value)


@dataclass(frozen=True)
class SystemParams:
    """질량·에너지·각운동량. T7 은 각 항목에 폭 있는 Interval 을 넣어 모든 값에 대해 동시에 감싼다."""

    masses: tuple[Interval, Interval, Interval]
    energy: Interval
    angular_momentum: Interval

    @staticmethod
    def make(
        masses: Sequence[Interval | RationalLike] = (1, 1, 1),
        energy: Interval | RationalLike = Fraction(-1, 2),
        angular_momentum: Interval | RationalLike = 0,
    ) -> "SystemParams":
        m = tuple(as_interval(v) for v in masses)
        if len(m) != 3 or any(mi.lo <= 0 for mi in m):
            raise ValueError("three positive masses required")
        return SystemParams(m, as_interval(energy), as_interval(angular_momentum))  # type: ignore[arg-type]


@dataclass(frozen=True)
class CoordinateSection:
    """단면 y_coord = value 를 direction 방향으로 통과. 에너지 식은 solved_momentum 에 대해 풀고 branch 로 근을 고른다.

    branch 는 dH/dy_m 의 부호다. 차트 좌표 u 는 나머지 4성분(오름차순)이다.
    """

    coord: int
    value: Fraction
    direction: int
    solved_momentum: int
    branch: int

    def __post_init__(self) -> None:
        if not 0 <= self.coord < 6 or self.solved_momentum not in (PR, PX, PY) or self.coord == self.solved_momentum:
            raise ValueError("invalid section coordinates")
        if self.direction not in (1, -1) or self.branch not in (1, -1):
            raise ValueError("direction and branch must be +1 or -1")

    @staticmethod
    def syzygy(direction: int = 1) -> "CoordinateSection":
        """일직선 배치 y = (rho1 x rho2)/|rho1| = 0. 근 선택은 dH/dpy = dy/dt 이므로 통과 방향과 같다."""
        return CoordinateSection(Y, Fraction(0), direction, PY, direction)

    @property
    def free_coords(self) -> tuple[int, ...]:
        return tuple(i for i in range(6) if i not in (self.coord, self.solved_momentum))

    def engine_tokens(self) -> str:
        v = Interval.point(self.value)
        return f"{self.coord} {self.direction} {self.solved_momentum} {self.branch} {v.engine_tokens()}"


@dataclass(frozen=True)
class Frame:
    """차트 좌표의 틀 u = center + matrix @ w. matrix 는 정확 유리수(보통 double 값 그대로)로 둔다."""

    center: tuple[Fraction, Fraction, Fraction, Fraction]
    matrix: RationalMatrix

    @staticmethod
    def identity() -> "Frame":
        return Frame((Fraction(0),) * 4, identity(4))  # type: ignore[arg-type]

    @staticmethod
    def make(center: Sequence[RationalLike], matrix: Sequence[Sequence[RationalLike]]) -> "Frame":
        c = tuple(rational(v) for v in center)
        m = tuple(tuple(rational(v) for v in row) for row in matrix)
        if len(c) != 4 or len(m) != 4 or any(len(row) != 4 for row in m):
            raise ValueError("frame must be 4-dimensional")
        return Frame(c, m)  # type: ignore[arg-type]

    def inverse(self) -> RationalMatrix:
        return matrix_inverse(self.matrix)

    def to_frame(self, u: Sequence[Fraction]) -> tuple[Fraction, ...]:
        inv = self.inverse()
        d = [ui - ci for ui, ci in zip(u, self.center, strict=True)]
        return tuple(sum(inv[i][j] * d[j] for j in range(4)) for i in range(4))

    def from_frame(self, w: Sequence[Fraction]) -> tuple[Fraction, ...]:
        return tuple(self.center[i] + sum(self.matrix[i][j] * w[j] for j in range(4)) for i in range(4))

    def engine_tokens_in(self) -> str:
        parts = [Interval.point(c).engine_tokens() for c in self.center]
        parts += [Interval.point(v).engine_tokens() for row in self.matrix for v in row]
        return " ".join(parts)

    def engine_tokens_out(self) -> str:
        """도착 틀은 역행렬로 좌표를 잰다. 정확 역행렬을 엔진 산술로 감싸도록 유리수 그대로 넘긴다."""
        parts = [Interval.point(c).engine_tokens() for c in self.center]
        parts += [Interval.point(v).engine_tokens() for row in self.inverse() for v in row]
        return " ".join(parts)


@dataclass(frozen=True)
class Settings:
    """적분 설정. arithmetic='double' 또는 'mp:<bits>'(MPFR, bits≥53), build='native-strict'(arm64) 또는
    'x86filib-strict'(Rosetta 2). 두 빌드 모두 CAPD·엔진을 -ffp-contract=off 로 만든다(build_capd.sh)."""

    order: int = 20
    step: Fraction | None = None
    arithmetic: str = "double"
    build: str = "native-strict"
    workers: int = 1
    timeout_seconds: int = 3600

    def engine_header(self) -> str:
        step = "adaptive" if self.step is None else f"fixed {Interval.point(self.step).engine_tokens().split()[0]}"
        return f"order {self.order}\nstep {step}"
