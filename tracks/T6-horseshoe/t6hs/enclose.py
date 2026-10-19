"""조각 상 감싸기 조립: 평균값 형식 P(조각) ⊂ P(w_c) + DP(조각)·(조각 − w_c) [CAP 입력은 T3 엔진].

T3 c0 의 상 감싸기는 귀환 시각을 점과 무관한 구간으로 넣어(CAPD 아핀 C⁰ 통과식) 이 궤도에서 1차로 과대평가된다
(enclosure_diagnosis.json). 그래서 상은 점 상(MP)과 c1 도함수로 감싸고, 조각 전체의 통과 세기·흐름 관·차트 판별식은
c0 로 인증한다. c1 이 첫 통과의 사상을 미분했다는 것은 c1 귀환 창이 c0 로 잰 둘째 영점 창보다 앞선다는 것으로 확인한다
((0, T_lo) 에 영점이 없음은 c0 통과 세기가 이미 보인다).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from t3cap import CoordinateSection, Failure, Frame, Interval, Settings, SystemParams, poincare_c0, poincare_c1
from t3cap.api import C0Enclosure, C1Enclosure, TubeBounds
from t3cap.exact import Box
from t3cap.verdict import certify_return

SECTION_IN = CoordinateSection.syzygy(+1)
SECTION_OUT = CoordinateSection.syzygy(-1)
# 출발 단면과 같은 방향으로 다시 통과하는 시각 = 다음 syzygy 다음의 영점(둘째 영점)
SECOND_ZERO = CoordinateSection.syzygy(+1)


@dataclass(frozen=True)
class HSet:
    """h-set c + M·[−1,1]^4. M = (단위 방향 열)·diag(반경) 을 정확 유리수로 든다."""

    name: str
    center: tuple[Fraction, ...]
    matrix: tuple[tuple[Fraction, ...], ...]

    def frame(self) -> Frame:
        return Frame.make(self.center, self.matrix)


def hset_from_design(name: str, vertex: dict) -> HSet:
    """graph_design.json 정점(중심·단위 방향 16진, 반경) → 정확 유리수 h-set. 반경 double 값을 그대로 유리수로 쓴다."""
    c = tuple(Fraction(float.fromhex(h)) for h in vertex["center_hex"])
    f = [[Fraction(float.fromhex(h)) for h in row] for row in vertex["frame_hex"]]
    r = [Fraction(float(v)) for v in vertex["radii"]]
    return HSet(name, c, tuple(tuple(f[i][j] * r[j] for j in range(4)) for i in range(4)))


@dataclass(frozen=True)
class EngineSettings:
    c0: Settings = Settings(order=20, workers=3)
    c1: Settings = Settings(order=20, workers=3)
    point: Settings = Settings(order=30, arithmetic="mp:128", workers=3)


@dataclass(frozen=True)
class EngineRuns:
    """조각 하나를 받치는 엔진 실행 넷. 덮개 기록에 그대로 남겨 저장값만으로 사용 조건을 다시 보게 한다."""

    c0: C0Enclosure
    c1: C1Enclosure
    second_zero: C0Enclosure
    point: C0Enclosure


@dataclass(frozen=True)
class PieceResult:
    piece: Box
    image: Box | None              # 평균값 형식 감싸기(실패면 None)
    certified: bool
    tube: TubeBounds | None
    return_time: Interval | None   # c0 첫 통과 창
    c1_return_time: Interval | None
    second_zero_time: Interval | None
    discriminant: Interval | None
    branch_out: Interval | None
    failures: tuple[str, ...]
    seconds: float
    runs: EngineRuns | None        # 실행 넷이 모두 성공했을 때만 있다


def center_of(piece: Box) -> tuple[Fraction, ...]:
    return tuple(iv.mid for iv in piece)


def mean_value_image(center_image: Box, derivative, piece: Box, center: Sequence[Fraction]) -> Box:
    """I_c + D·(P − w_c), 정확 유리수 구간 산술."""
    dev = [Interval(iv.lo - c, iv.hi - c) for iv, c in zip(piece, center)]
    out = []
    for k in range(4):
        acc = center_image[k]
        for l in range(4):
            acc = acc + derivative[k][l] * dev[l]
        out.append(acc)
    return tuple(out)


def usage_failures(r0, r1, r2, rp, min_distance: Fraction) -> list[str]:
    """조각을 증명 근거로 쓸 자격(한 곳에서만 선언): 네 엔진 실행 성공, c0·c1·중심 점의 certify_return,
    c1 귀환 창 위끝 < 둘째 영점 창 아래끝."""
    named = (("c0", r0), ("c1", r1), ("second-zero c0", r2), ("point", rp))
    failures = [f"{tag}: {r.reason}" for tag, r in named if isinstance(r, Failure)]
    if failures:
        return failures
    for tag, r in (("c0", r0), ("c1", r1), ("point", rp)):
        failures += [f"{tag}: {c.name} {c.detail}" for c in certify_return(r, min_distance=min_distance).failed]
    if not r1.return_time.hi < r2.return_time.lo:
        failures.append(f"c1 return window {r1.return_time} reaches the second zero {r2.return_time}")
    return failures


def _result(piece, image, runs: EngineRuns | None, failures, seconds) -> PieceResult:
    if image is None or runs is None:
        return PieceResult(piece, None, False, None, None, None, None, None, None, tuple(failures), seconds, None)
    r0, r1, r2 = runs.c0, runs.c1, runs.second_zero
    return PieceResult(piece, image, not failures, r0.tube, r0.return_time, r1.return_time, r2.return_time,
                       r0.discriminant, r0.branch_out, tuple(failures), seconds, runs)


def _kw(src, dst, system, section_out):
    return dict(system=system or SystemParams.make(), section_in=SECTION_IN, section_out=section_out,
                frame_in=src.frame(), frame_out=dst.frame())


@dataclass(frozen=True)
class HSetImage:
    """단일 중심 평균값 형식: 상자 B 와 출구 면 F⊂B 모두 P(0)+D(B)·(조각) 으로 감싼다(B 는 0 을 품은 볼록 집합)."""

    whole: PieceResult
    faces: dict[tuple[int, int], PieceResult]
    center_image: Box | None       # P(0) 의 MP 점 감싸기(재판정용)
    derivative: tuple | None       # D(B) c1 감싸기(재판정용)


def enclose_hset(src: HSet, dst: HSet, faces: Sequence[tuple[int, int]], *, min_distance: Fraction,
                 settings: EngineSettings = EngineSettings(), system: SystemParams | None = None,
                 section_out: CoordinateSection = SECTION_OUT) -> HSetImage:
    """section_out 은 G6 음성 대조(반대 방향 단면)에서만 바꾼다."""
    started = time.time()
    kw = _kw(src, dst, system, section_out)
    box = tuple(Interval(Fraction(-1), Fraction(1)) for _ in range(4))
    zero = (Fraction(0),) * 4
    r0 = poincare_c0([box], settings=settings.c0, **kw)[0]
    r1 = poincare_c1([box], settings=settings.c1, **kw)[0]
    r2 = poincare_c0([box], settings=settings.c0, **{**kw, "section_out": SECOND_ZERO})[0]
    rp = poincare_c0([tuple(Interval(c, c) for c in zero)], settings=settings.point, **kw)[0]
    failures = usage_failures(r0, r1, r2, rp, min_distance)
    seconds = time.time() - started
    usable = not any(isinstance(r, Failure) for r in (r0, r1, r2, rp))
    runs = EngineRuns(r0, r1, r2, rp) if usable else None
    face_boxes = {(j, s): tuple(Interval(Fraction(s), Fraction(s)) if i == j else Interval(Fraction(-1), Fraction(1))
                                for i in range(4)) for j, s in faces}

    def result(piece: Box) -> PieceResult:
        image = mean_value_image(rp.image, r1.derivative, piece, zero) if usable else None
        return _result(piece, image, runs, failures, seconds)

    return HSetImage(result(box), {key: result(b) for key, b in face_boxes.items()},
                     rp.image if usable else None, r1.derivative if usable else None)


def enclose_pieces(src: HSet, dst: HSet, pieces: Sequence[Box], *, min_distance: Fraction,
                   settings: EngineSettings = EngineSettings(), system: SystemParams | None = None) -> list[PieceResult]:
    """조각 분할 판정용: 조각마다 c0·c1·둘째 영점 c0 와 조각 중심의 MP 점 상."""
    started = time.time()
    kw = _kw(src, dst, system, SECTION_OUT)
    c0 = poincare_c0(list(pieces), settings=settings.c0, **kw)
    c1 = poincare_c1(list(pieces), settings=settings.c1, **kw)
    c2 = poincare_c0(list(pieces), settings=settings.c0, **{**kw, "section_out": SECOND_ZERO})
    centers = [center_of(p) for p in pieces]
    points = poincare_c0([tuple(Interval(c, c) for c in cen) for cen in centers], settings=settings.point, **kw)
    seconds = (time.time() - started) / max(1, len(pieces))
    out = []
    for piece, cen, r0, r1, r2, rp in zip(pieces, centers, c0, c1, c2, points):
        failures = usage_failures(r0, r1, r2, rp, min_distance)
        usable = not any(isinstance(r, Failure) for r in (r0, r1, r2, rp))
        image = mean_value_image(rp.image, r1.derivative, piece, cen) if usable else None
        out.append(_result(piece, image, EngineRuns(r0, r1, r2, rp) if usable else None, failures, seconds))
    return out
