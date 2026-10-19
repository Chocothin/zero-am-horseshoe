"""T6·T7 용 공개 API: 축소 Poincaré 사상의 C0·C1 감싸기, 차트, 벡터장. 결과 성공·실패는 반환 타입이 운반한다."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence, Union

from . import engine
from .exact import Box, Interval, IntervalMatrix
from .model import CoordinateSection, Frame, Settings, SystemParams


@dataclass(frozen=True)
class Failure:
    """엔진이 감싸기를 내지 못함(교차 미증명·판별식 미증명·적분 실패 등). 증명 근거로 쓸 수 없다."""

    reason: str
    seconds: float


@dataclass(frozen=True)
class Crossings:
    """흐름 관 단계로 센 출력 단면 통과(함정 ②)의 결과. 엔진이 성공한 감싸기라면 창 묶음은 1, 앞선 통과는 returns−1 이다."""

    window_runs: int       # 반환 창과 겹친 닿는 묶음 수
    counted_before: int    # 그 묶음 앞에서 끝난 지정 방향 통과 수
    runs_before: int       # 그 묶음 앞에서 닫힌 닿는 묶음 수(출발 묶음 포함)
    window_run: Interval   # 창 묶음 단계들의 시각 외피


@dataclass(frozen=True)
class TubeBounds:
    """출발 집합에서 반환 시간 상한까지 흐름 관 전체의 상호 거리 감싸기(함정 ④)와 같은 단계로 센 통과(함정 ②)."""

    r12: Interval
    r13: Interval
    r23: Interval
    steps: int
    t_end: Interval
    crossings: Crossings

    @property
    def min_distance_lower(self) -> Fraction:
        return min(self.r12.lo, self.r13.lo, self.r23.lo)

    @property
    def max_distance_upper(self) -> Fraction:
        return max(self.r12.hi, self.r13.hi, self.r23.hi)


@dataclass(frozen=True)
class C0Enclosure:
    image: Box                 # 도착 틀 좌표 w' 의 감싸기
    hull: tuple[Interval, ...]  # 축소 좌표 y 의 감싸기(회전각 포함 시 7성분)
    return_time: Interval
    discriminant: Interval     # 출발 차트의 판별식 D (함정 ⑤: D.lo > 0 이어야 차트가 정의된다)
    branch_out: Interval       # branch * dH/dy_m 을 상 위에서 감싼 값(> 0 이어야 도착 차트 근이 맞다)
    tube: TubeBounds
    theta: Interval | None     # rotation=True 일 때 반환까지 rho1 회전각(함정 ⑦)
    seconds: float


@dataclass(frozen=True)
class C1Enclosure:
    image: Box
    hull: tuple[Interval, ...]
    return_time: Interval
    discriminant: Interval
    branch_out: Interval
    tube: TubeBounds
    derivative: IntervalMatrix  # 틀 좌표 미분 Minv_out . Pi_out . DP6 . DPhi . M_in 를 상자 전체에서 감싼 것
    dp6: IntervalMatrix         # 축소 6차원 Poincaré 사상 미분 감싸기
    seconds: float


@dataclass(frozen=True)
class ChartEnclosure:
    hull: tuple[Interval, ...]
    dpsi: IntervalMatrix
    discriminant: Interval
    energy: Interval
    seconds: float


@dataclass(frozen=True)
class VectorFieldEnclosure:
    value: tuple[Interval, ...]
    energy: Interval
    r12: Interval
    r13: Interval
    r23: Interval
    seconds: float


C0Result = Union[C0Enclosure, Failure]
C1Result = Union[C1Enclosure, Failure]


def poincare_c0(
    boxes: Sequence[Box],
    *,
    system: SystemParams,
    section_in: CoordinateSection,
    section_out: CoordinateSection | None = None,
    returns: int = 1,
    frame_in: Frame | None = None,
    frame_out: Frame | None = None,
    settings: Settings = Settings(),
    rotation: bool = False,
) -> list[C0Result]:
    """상자(출발 틀 좌표 w) 마다 P^returns 상의 C0 감싸기. 상자 전체를 집합으로 적분한다(꼭짓점 표본 아님, 함정 ⑥)."""
    records = engine.run(
        "c0", boxes, **_common(system, section_in, section_out, returns, frame_in, frame_out, settings, rotation)
    )
    return [_c0(r) for r in records]


def poincare_c1(
    boxes: Sequence[Box],
    *,
    system: SystemParams,
    section_in: CoordinateSection,
    section_out: CoordinateSection | None = None,
    returns: int = 1,
    frame_in: Frame | None = None,
    frame_out: Frame | None = None,
    settings: Settings = Settings(),
) -> list[C1Result]:
    """상자마다 P^returns 의 틀 좌표 미분을 상자 전체에서 감싼다(C1 Lohner)."""
    records = engine.run(
        "c1", boxes, **_common(system, section_in, section_out, returns, frame_in, frame_out, settings, False)
    )
    return [_c1(r) for r in records]


def chart(
    boxes: Sequence[Box],
    *,
    system: SystemParams,
    section: CoordinateSection,
    frame: Frame | None = None,
    settings: Settings = Settings(),
) -> list[Union[ChartEnclosure, Failure]]:
    records = engine.run("chart", boxes, **_common(system, section, None, 1, frame, None, settings, False))
    return [
        Failure(r.failure or "unknown", r.seconds)
        if not r.ok
        else ChartEnclosure(
            _vec(r.fields["chart_hull"]), _mat(r.fields["chart_dpsi"]), _iv(r.fields["disc"]), _iv(r.fields["energy"]),
            r.seconds,
        )
        for r in records
    ]


def vector_field(
    points: Sequence[Box], *, system: SystemParams, settings: Settings = Settings()
) -> list[Union[VectorFieldEnclosure, Failure]]:
    """축소 좌표 6차원 점(구간) 에서 벡터장·에너지·상호 거리 감싸기 — 시험·대조용."""
    section = CoordinateSection.syzygy()
    records = engine.run("vf", points, **_common(system, section, None, 1, None, None, settings, False))
    return [
        Failure(r.failure or "unknown", r.seconds)
        if not r.ok
        else VectorFieldEnclosure(
            _vec(r.fields["vf"]), _iv(r.fields["energy"]), _iv(r.fields["r12"]), _iv(r.fields["r13"]),
            _iv(r.fields["r23"]), r.seconds,
        )
        for r in records
    ]


def face_boxes(b: Box, coord: int, side: int, pieces: int) -> list[Box]:
    """상자의 면 {w_coord = 끝점} 전체를 나머지 좌표 방향으로 pieces^(d-1) 조각 내 덮는다(함정 ⑥: 꼭짓점 표본 금지)."""
    fixed = b[coord].hi if side > 0 else b[coord].lo
    axes = [i for i in range(len(b)) if i != coord]
    pieces_per_axis = [_split(b[i], pieces) for i in axes]
    out: list[Box] = []

    def rec(level: int, acc: dict[int, Interval]) -> None:
        if level == len(axes):
            out.append(tuple(Interval(fixed, fixed) if i == coord else acc[i] for i in range(len(b))))
            return
        for piece in pieces_per_axis[level]:
            rec(level + 1, {**acc, axes[level]: piece})

    rec(0, {})
    return out


def _split(iv: Interval, pieces: int) -> list[Interval]:
    edges = [iv.lo + iv.width * Fraction(k, pieces) for k in range(pieces + 1)]
    return [Interval(edges[k], edges[k + 1]) for k in range(pieces)]


def _common(system, section_in, section_out, returns, frame_in, frame_out, settings, rotation) -> dict:
    return dict(
        system=system,
        section_in=section_in,
        section_out=section_out or section_in,
        returns=returns,
        frame_in=frame_in or Frame.identity(),
        frame_out=frame_out or Frame.identity(),
        settings=settings,
        rotation=rotation,
    )


def _iv(tokens: list[str]) -> Interval:
    return Interval.from_engine(tokens[0], tokens[1])


def _vec(tokens: list[str]) -> tuple[Interval, ...]:
    return tuple(Interval.from_engine(tokens[k], tokens[k + 1]) for k in range(0, len(tokens), 2))


def _mat(tokens: list[str]) -> IntervalMatrix:
    rows, cols = int(tokens[0]), int(tokens[1])
    flat = _vec(tokens[2:])
    return tuple(tuple(flat[i * cols + j] for j in range(cols)) for i in range(rows))


def _tube(f: dict[str, list[str]]) -> TubeBounds:
    window_runs, counted_before, runs_before = (int(v) for v in f["crossings"])
    return TubeBounds(_iv(f["tube_r12"]), _iv(f["tube_r13"]), _iv(f["tube_r23"]), int(f["tube_steps"][0]),
                      _iv(f["tube_tend"]), Crossings(window_runs, counted_before, runs_before, _iv(f["crossing_run"])))


def _c0(r: engine.BoxRecord) -> C0Result:
    if not r.ok:
        return Failure(r.failure or "unknown", r.seconds)
    f = r.fields
    return C0Enclosure(
        _vec(f["image"]), _vec(f["hull"]), _iv(f["return_time"]), _iv(f["disc"]), _iv(f["branch_out"]), _tube(f),
        _iv(f["theta"]) if "theta" in f else None, r.seconds,
    )


def _c1(r: engine.BoxRecord) -> C1Result:
    if not r.ok:
        return Failure(r.failure or "unknown", r.seconds)
    f = r.fields
    return C1Enclosure(
        _vec(f["image"]), _vec(f["hull"]), _iv(f["return_time"]), _iv(f["disc"]), _iv(f["branch_out"]), _tube(f),
        _mat(f["deriv"]), _mat(f["dp6"]), r.seconds,
    )
