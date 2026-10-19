"""저장된 T6 덮개 기록 → 원뿔 판정 입력(조립층, G7 감사 대상). 재판정·대조·정리 조립이 이 한 경로를 쓴다.

원뿔 판정은 기록의 D(B)(상자 전체 c1 도함수 감싸기를 정규 좌표로 옮긴 구간 행렬)만 쓴다. 그 D(B) 를 믿을 자격은
T6 의 저장값 재판정(records.covering_input 의 사용 조건, covering_verdict 의 덮개 증명)과 설계 해시가 정한다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from t3cap.exact import Interval
from t3cap.export import interval_from_json
from t6hs import covering, records

Matrix = tuple[tuple[Interval, ...], ...]


@dataclass(frozen=True)
class ConeInput:
    edge: str
    derivative: Matrix | None
    problems: tuple[str, ...]      # 이 기록의 D(B) 를 원뿔 판정에 쓸 수 없는 이유(없으면 사용 가능)

    @property
    def usable(self) -> bool:
        return self.derivative is not None and not self.problems


def derivative_of(rec: dict) -> Matrix | None:
    der = rec.get("derivative")
    if der is None:
        return None
    return tuple(tuple(interval_from_json(x) for x in row) for row in der)


def cone_input(rec: dict, design_hash: str) -> ConeInput:
    c = records.covering_input(rec)
    problems = list(c.problems)
    if rec["design_sha256"] != design_hash:
        problems.append("record was made for another graph design")
    if rec["method"] != "single-center":
        problems.append("no single-center derivative D(B) in the record")
    if not covering.covering_verdict(c.whole, c.faces, c.signs).proven:
        problems.append("covering relation not proven on re-judgement")
    return ConeInput(rec["edge"], derivative_of(rec), tuple(problems))


def compact_input(row: dict) -> ConeInput:
    """G8 압축 기록(원 기록에서 D(B)·사용 판정만 옮긴 것) → 판정 입력."""
    der = tuple(tuple(interval_from_json(x) for x in r) for r in row["derivative"]) if row["derivative"] else None
    return ConeInput(row["edge"], der, tuple(row["problems"]))


def load(path: Path) -> dict:
    return json.loads(path.read_text())
