"""덮개 관계 판정층(G7 감사 대상). 감싸기를 정확 유리수 구간으로 받아 ZG2004 Theorem 15 의 충분조건을 판정한다.

h-set N, M 은 정규 좌표에서 [−1,1]^4 이고 앞 두 성분이 불안정(u=2), 뒤 두 성분이 안정(s=2)이다(sup 노름의 단위 공).
판정하는 조건(조각들의 합집합이 상자·출구 면을 정확히 덮는다는 전제 위에서):
  whole: 상자 조각마다 상이 「어떤 불안정 성분이 통째로 |w'_j|>1」 이거나 「두 안정 성분이 통째로 (−1,1) 안」
         → (64)·(66) [q0=0, S(M)⁻={‖p‖>1} 은 열린집합]
  faces: 출구 면 {w_j=σ}(j 불안정, σ=±1) 조각마다 s_j·σ·w'_j > 1 → (65)·(68), 그리고
         직선 호모토피 (1−t)A+t·diag(2s_0, 2s_1) 가 ∂B_u 에서 B_u 와 만나지 않으므로 (69) deg = s_0·s_1
결과(증명 여부)는 반환 타입이 운반한다. 조건 하나라도 못 보이면 실패다.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from t3cap.exact import Box, Interval

UNSTABLE = (0, 1)
STABLE = (2, 3)
ONE = Fraction(1)


@dataclass(frozen=True)
class PieceImage:
    piece: Box                # 출발 정규 좌표의 조각
    image: Box                # 도착 정규 좌표의 상 감싸기
    certified: bool           # 이 조각을 받치는 엔진 결과가 모두 사용 자격 검사를 통과했는가
    note: str = ""


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class CoveringVerdict:
    proven: bool
    degree: int               # s_0·s_1 (proven 일 때만 의미)
    checks: tuple[Check, ...]

    @property
    def failed(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if not c.passed)


def unit_box(dim: int = 4) -> Box:
    return tuple(Interval(-ONE, ONE) for _ in range(dim))


def face_box(coord: int, side: int, dim: int = 4) -> Box:
    return tuple(Interval(Fraction(side), Fraction(side)) if i == coord else Interval(-ONE, ONE) for i in range(dim))


def bisect_box(piece: Box) -> list[Box]:
    """퇴화하지 않은 모든 축을 정확한 유리수 중점에서 반으로 — 조각 합집합은 원래 상자와 같다."""
    out = [piece]
    for a in (i for i, iv in enumerate(piece) if iv.width > 0):
        nxt = []
        for p in out:
            m = p[a].mid
            nxt.append(tuple(Interval(p[a].lo, m) if i == a else p[i] for i in range(len(p))))
            nxt.append(tuple(Interval(m, p[a].hi) if i == a else p[i] for i in range(len(p))))
        out = nxt
    return out


def exactly_covers(pieces: Sequence[Box], target: Box) -> bool:
    """조각들이 target 안에 있고, 내부가 서로 겹치지 않으며, 부피(퇴화 축 제외) 합이 target 과 같다 → 합집합 = target."""
    if not pieces:
        return False
    axes = [i for i, iv in enumerate(target) if iv.width > 0]
    if any(len(p) != len(target) for p in pieces):
        return False
    for p in pieces:
        # 퇴화 축(면의 고정 좌표)에서 대상에 포함되면 조각 폭도 0 이므로 따로 검사하지 않는다
        if not all(t.contains_interval(q) for q, t in zip(p, target)):
            return False

    def vol(b: Box) -> Fraction:
        v = ONE
        for i in axes:
            v *= b[i].width
        return v

    if sum(vol(p) for p in pieces) != vol(target):
        return False
    for a in range(len(pieces)):
        for b in range(a + 1, len(pieces)):
            if all(max(pieces[a][i].lo, pieces[b][i].lo) < min(pieces[a][i].hi, pieces[b][i].hi) for i in axes):
                return False
    return True


def outside_unstable(image: Box, threshold: Fraction = ONE) -> bool:
    return any(image[j].lo > threshold or image[j].hi < -threshold for j in UNSTABLE)


def inside_stable(image: Box, threshold: Fraction = ONE) -> bool:
    return all(-threshold < image[k].lo and image[k].hi < threshold for k in STABLE)


def exit_face_out(image: Box, coord: int, side: int, sign: int, threshold: Fraction = ONE) -> bool:
    """s_j·σ·w'_j > threshold 를 감싸기 전체에서."""
    orientation = sign * side
    return image[coord].lo > threshold if orientation > 0 else image[coord].hi < -threshold


def covering_verdict(whole: Sequence[PieceImage], faces: dict[tuple[int, int], Sequence[PieceImage]],
                     signs: tuple[int, int], *, unstable_threshold: Fraction = ONE,
                     stable_threshold: Fraction = ONE) -> CoveringVerdict:
    """문턱은 정리의 조건에서 1 이다. 다른 값은 음성 대조(문턱 강화)에서만 쓴다."""
    if set(faces) != {(j, s) for j in UNSTABLE for s in (-1, 1)} or any(s not in (-1, 1) for s in signs):
        raise ValueError("covering_verdict: need the four exit faces and signs ±1")
    ut, st = unstable_threshold, stable_threshold
    checks = [
        Check("whole_box_covered", exactly_covers([p.piece for p in whole], unit_box()), f"{len(whole)} pieces"),
        Check("whole_box_certified", all(p.certified for p in whole), "engine usage checks"),
        Check("whole_box_images_ok", all(outside_unstable(p.image, ut) or inside_stable(p.image, st) for p in whole),
              "each piece: some unstable |w'|>1 or stable w' in (-1,1)"),
    ]
    for (coord, side), pieces in sorted(faces.items()):
        tag = f"face_w{coord}={side:+d}"
        checks.append(Check(f"{tag}_covered", exactly_covers([p.piece for p in pieces], face_box(coord, side)),
                            f"{len(pieces)} pieces"))
        checks.append(Check(f"{tag}_certified", all(p.certified for p in pieces), "engine usage checks"))
        checks.append(Check(f"{tag}_out", all(exit_face_out(p.image, coord, side, signs[coord], ut) for p in pieces),
                            f"sign s_{coord}={signs[coord]:+d}"))
    proven = all(c.passed for c in checks)
    return CoveringVerdict(proven, signs[0] * signs[1], tuple(checks))


def certified_margin(whole: Sequence[PieceImage], faces: dict[tuple[int, int], Sequence[PieceImage]],
                     signs: tuple[int, int]) -> Fraction:
    """문턱 1 을 바깥(불안정)·안쪽(안정)으로 함께 이만큼 조여도 판정 부등식이 유지되는 최대 폭(조각별 최솟값)."""
    def whole_slack(img: Box) -> Fraction:
        out = max(max(img[j].lo - ONE, -ONE - img[j].hi) for j in UNSTABLE)
        inside = min(min(img[k].lo + ONE, ONE - img[k].hi) for k in STABLE)
        return max(out, inside)

    def face_slack(img: Box, coord: int, side: int) -> Fraction:
        return img[coord].lo - ONE if signs[coord] * side > 0 else -ONE - img[coord].hi

    slacks = [whole_slack(p.image) for p in whole]
    slacks += [face_slack(p.image, c, s) for (c, s), ps in faces.items() for p in ps]
    return min(slacks)
