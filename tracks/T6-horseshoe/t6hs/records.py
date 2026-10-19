"""저장된 덮개 기록(coverings/<변>.json) → 판정 입력. 재판정(check_results)·음성 대조·정리 조립이 이 한 경로를 쓴다."""
from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from t3cap.export import interval_from_json

from .covering import PieceImage

FACE_KEYS = ((0, -1), (0, 1), (1, -1), (1, 1))
MIN_DISTANCE = Fraction(1, 5)
RETURNS = 1  # 덮개 사상은 모두 출발 단면에서 다음 syzygy(첫 지정 방향 통과)로 간다
RUN_KEYS = ("c0", "c1", "second_zero", "point")
CERTIFIED_RUNS = ("c0", "c1", "point")  # enclose.usage_failures 가 certify_return 을 요구하는 실행
TUBE_KEYS = ("r12", "r13", "r23", "t_end")


@dataclass(frozen=True)
class CoveringInput:
    edge: str
    whole: list[PieceImage]
    faces: dict[tuple[int, int], list[PieceImage]]
    signs: tuple[int, int]
    tube_lower: Fraction        # 전체 상자 조각 흐름 관 r_ij 하한의 최솟값
    tube_upper: Fraction
    return_time_lower: Fraction
    problems: tuple[str, ...]   # 저장값에서 다시 본 사용 조건 위반


def _box(js):
    return tuple(interval_from_json(x) for x in js)


def run_certified(run: dict) -> bool:
    """실행 하나가 T3 certify_return 의 가설을 만족한다: 판별식>0, 도착 근>0, 귀환>0, 흐름 관이 반환을 덮음, r_ij>1/5."""
    tube = {k: interval_from_json(run["tube"][k]) for k in TUBE_KEYS}
    rt = interval_from_json(run["return_time"])
    return (rt.lo > 0 and tube["t_end"].lo >= rt.hi and all(tube[k].lo > MIN_DISTANCE for k in ("r12", "r13", "r23"))
            and interval_from_json(run["discriminant"]).lo > 0 and interval_from_json(run["branch_out"]).lo > 0)


def first_crossing(run: dict) -> bool:
    """엔진 통과 세기 결과가 「반환은 t>0 의 RETURNS 번째 지정 방향 통과」를 말한다: 반환 창과 겹친 닿는 묶음이 하나이고,
    그 앞에서 끝난 지정 방향 통과가 RETURNS−1 개이며, 그 묶음의 시각이 반환 창과 겹친다."""
    c = run["crossings"]
    window, rt = interval_from_json(c["window_run"]), interval_from_json(run["return_time"])
    return (c["window_runs"] == 1 and c["counted_before"] == RETURNS - 1
            and window.lo <= rt.hi and rt.lo <= window.hi)


def piece_matches_runs(p: dict, runs: dict) -> bool:
    """조각에 옮겨 적은 c0 흐름 관·귀환 창·판별식·도착 근, c1·둘째 영점 창이 실행 기록과 같다(조립·대조는 조각 값을 읽는다)."""
    c0 = runs["c0"]
    same = lambda a, b: interval_from_json(a) == interval_from_json(b)
    return (all(same(p["tube"][k], c0["tube"][k]) for k in TUBE_KEYS)
            and all(same(p[k], c0[k]) for k in ("return_time", "discriminant", "branch_out"))
            and same(p["c1_return_time"], runs["c1"]["return_time"])
            and same(p["second_zero_time"], runs["second_zero"]["return_time"]))


def usage_ok(p: dict, runs: list[dict]) -> bool:
    """저장값으로 다시 보는 사용 조건(enclose.usage_failures 와 같은 조건): 조각 상과 실행 넷의 기록이 있고, c0·c1·점 실행이
    certify_return 의 가설을, 네 실행이 첫 통과 세기 결과를 만족하며, c1 창 < 둘째 영점 창이다."""
    index = p.get("runs_index")
    if p["image"] is None or p["tube"] is None or index is None or not 0 <= index < len(runs):
        return False
    r = runs[index]
    return (p["certified"] and piece_matches_runs(p, r)
            and all(run_certified(r[k]) for k in CERTIFIED_RUNS) and all(first_crossing(r[k]) for k in RUN_KEYS)
            and interval_from_json(r["c1"]["return_time"]).hi < interval_from_json(r["second_zero"]["return_time"]).lo)


def covering_input(rec: dict) -> CoveringInput:
    detail = rec["pieces_detail"]
    runs = rec.get("runs", [])
    problems = []

    def images(plist, key):
        out = []
        for p in plist:
            ok = usage_ok(p, runs)
            if not ok:
                problems.append(f"{key}: usage condition fails on stored data")
            if p["image"] is not None:
                out.append(PieceImage(_box(p["piece"]), _box(p["image"]), ok))
        return out

    whole = images(detail.get("whole", []), "whole")
    faces = {k: images(detail.get(f"face_w{k[0]}{k[1]:+d}", []), f"face{k}") for k in FACE_KEYS}
    tubes = [{k: interval_from_json(v) for k, v in p["tube"].items()} for p in detail.get("whole", []) if p["tube"]]
    lower = min((t[k].lo for t in tubes for k in ("r12", "r13", "r23")), default=Fraction(0))
    upper = max((t[k].hi for t in tubes for k in ("r12", "r13", "r23")), default=Fraction(0))
    rts = [interval_from_json(p["return_time"]).lo for p in detail.get("whole", []) if p["return_time"]]
    return CoveringInput(rec["edge"], whole, faces, tuple(rec["signs"]), lower, upper, min(rts, default=Fraction(0)),
                         tuple(problems))


def load(path: Path) -> dict:
    return json.loads(path.read_text())
