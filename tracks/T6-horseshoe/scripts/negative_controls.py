"""G6 음성 대조: 덮개 판정이 반드시 실패해야 하는 변형 4종을 대표 변에 적용한다(하나라도 증명되면 Critical).

  shrink_expanding : 출발 h-set 불안정 반경을 줄여 선형 배율 |D_jj|r_j(v)/r_j(w) 를 0.5 로 낮춘다(펴짐 방향 축소)
  swap_target_roles: 도착 틀의 약불안정·약안정 열을 맞바꾼다(틀 회전)
  opposite_section : 출력 단면을 반대 방향 syzygy(+1) 로 둔다(다음 syzygy 가 아니라 그다음 통과를 센다)
  shift_target     : 도착 중심을 약안정 방향으로 그 반경의 2.5배 옮긴다
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from t6hs.results import DESIGN_DIR, T3_DIR, write_result  # noqa: E402

sys.path.insert(0, str(T3_DIR))
from t6hs import covering, enclose  # noqa: E402

MIN_DISTANCE = Fraction(1, 5)
FACES = [(j, s) for j in covering.UNSTABLE for s in (-1, 1)]
CHAIN_SAMPLE = (-148, -120, -90, -60, -30, -7, 0, 7, 60, 147)
FAR = tuple(enclose.Interval(Fraction(-10**9), Fraction(10**9)) for _ in range(4))
CONTROL_SETTINGS = enclose.EngineSettings(c0=enclose.Settings(order=20, workers=1), c1=enclose.Settings(order=20, workers=1),
                                          point=enclose.Settings(order=30, arithmetic="mp:128", workers=1))


def scaled(h: enclose.HSet, col_scale) -> enclose.HSet:
    return enclose.HSet(h.name, h.center, tuple(tuple(h.matrix[i][j] * col_scale[j] for j in range(4)) for i in range(4)))


def variants(edge, design):
    src = enclose.hset_from_design(edge["source"], design["vertices"][edge["source"]])
    dst = enclose.hset_from_design(edge["target"], design["vertices"][edge["target"]])
    rv = np.array(design["vertices"][edge["source"]]["radii"])
    rw = np.array(design["vertices"][edge["target"]]["radii"])
    d = np.array(edge["linear"])
    ratio = [abs(d[j][j]) * rv[j] / rw[j] for j in range(4)]
    shrink = [Fraction(0.5 / ratio[j]).limit_denominator(10**6) if j in covering.UNSTABLE else Fraction(1)
              for j in range(4)]
    swapped = enclose.HSet(dst.name, dst.center, tuple((row[0], row[2], row[1], row[3]) for row in dst.matrix))
    shift = tuple(dst.center[i] + Fraction(5, 2) * dst.matrix[i][2] for i in range(4))
    return {
        "shrink_expanding": (scaled(src, shrink), dst, enclose.SECTION_OUT),
        "swap_target_roles": (src, swapped, enclose.SECTION_OUT),
        "opposite_section": (src, dst, enclose.CoordinateSection.syzygy(+1)),
        "shift_target": (src, enclose.HSet(dst.name, shift, dst.matrix), enclose.SECTION_OUT),
    }


def run_control(task):
    edge, name, (src, dst, section), signs = task
    t0 = time.time()
    hs = enclose.enclose_hset(src, dst, FACES, min_distance=MIN_DISTANCE, section_out=section, settings=CONTROL_SETTINGS)
    pi = lambda r: covering.PieceImage(r.piece, r.image if r.image is not None else FAR, r.certified)
    v = covering.covering_verdict([pi(hs.whole)], {k: [pi(hs.faces[k])] for k in FACES}, signs)
    return {"edge": f"{edge['source']}>{edge['target']}", "control": name, "proven": v.proven,
            "failed_checks": [c.name for c in v.failed], "engine_failures": list(hs.whole.failures)[:3],
            "seconds": round(time.time() - t0, 1)}


def control_edges(design: dict, quick: bool) -> list[dict]:
    """대조를 거는 대표 변: cycle 변 16개, 사슬 표본 CHAIN_SAMPLE 에서 나가는 변, 도착 접합(빠른 모드는 N0·M+0 에서 나가는 변)."""
    k = design["k_junction"]
    names = {f"N{i}" for i in range(16)} | {f"M{j:+d}" for j in CHAIN_SAMPLE}
    edges = [e for e in design["edges"] if e["source"] in names or e["target"] == f"N{design['summary']['arrival']}"
             and e["source"] == f"M{k - 1:+d}"]
    return [e for e in edges if e["source"] in ("N0", "M+0")] if quick else edges


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="verify.sh 빠른 모드: 변 2개")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    design = json.loads((DESIGN_DIR / "graph_design.json").read_text())
    edges = control_edges(design, args.quick)
    tasks = []
    for e in edges:
        signs = tuple(int(np.sign(e["linear"][j][j])) for j in covering.UNSTABLE)
        for name, spec in variants(e, design).items():
            tasks.append((e, name, spec, signs))
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        rows = list(pool.map(run_control, tasks))
    leaked = [r for r in rows if r["proven"]]
    for r in rows:
        print(json.dumps(r), flush=True)
    print(f"controls {len(rows)}, wrongly proven {len(leaked)}")
    if not args.no_write:
        write_result("negative_controls", {"tag": "G6 음성 대조(모두 실패해야 함)", "rows": rows, "leaked": len(leaked)},
                     script=Path(__file__).resolve(), params=vars(args), started=started)
    return 1 if leaked else 0


if __name__ == "__main__":
    raise SystemExit(main())
