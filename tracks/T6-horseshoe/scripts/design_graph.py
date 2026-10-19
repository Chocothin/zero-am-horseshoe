"""[수치] 덮개 그래프 설계(이정표 ①): 접합 깊이 K 후보마다 반경·여유·순환 길이·엔트로피 하한·감싸기 초과 예측을 낸다.

감싸기 초과 예측은 enclosure_diagnosis.json 의 c1 도함수 폭 계수 c(정규 폭/r)로 3·c·r_max 를 쓴다(조각 1개 기준).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from t6hs import design  # noqa: E402
from t6hs.results import DESIGN_DIR, write_result  # noqa: E402


def width_coefficients(diag: dict, refit: dict) -> dict[str, float]:
    """변 이름 → c1 폭 계수. |k|>진단 범위인 사슬 변은 따라가는 cycle 단계의 값을 쓴다."""
    rows = {r["label"]: r.get("c1_width_per_r") for r in diag["rows"]}
    shadow = dict(zip(refit["homoclinic"]["k"], refit["homoclinic"]["shadow_cycle_index"]))
    coef = {}
    for i in range(16):
        coef[f"N{i}"] = rows[f"cycle {i}"]
    for k, j in shadow.items():
        coef[design.hom_name(k)] = rows.get(f"hom {k}") or rows[f"cycle {j}"]
    return coef


def summarize(graph: design.Graph, coef: dict[str, float]) -> dict:
    sl = design.slacks(graph)
    worst = min(sl, key=lambda s: s.value)
    radii = np.array([v.radii for v in graph.vertices.values()])
    excess = {e.source: 3 * coef[e.source] * float(np.max(graph.vertices[e.source].radii)) for e in graph.edges}
    l1, l2 = design.loop_lengths(graph)
    return {
        "k_junction": graph.k_junction, "departure": graph.departure, "arrival": graph.arrival,
        "margin": graph.margin, "vertices": len(graph.vertices), "edges": len(graph.edges),
        "loops": [l1, l2], "entropy_lower_bound_per_step": design.entropy_lower_bound(l1, l2),
        "radius_min": float(radii.min()), "radius_max": float(radii.max()),
        "cycle_radius_max": float(max(np.max(graph.vertices[f"N{i}"].radii) for i in range(16))),
        "slack_min": worst.value, "slack_min_at": f"{worst.edge} k={worst.component} {worst.kind}",
        "slacks_negative": sum(1 for s in sl if s.value < 0),
        "excess_max": max(excess.values()), "excess_max_at": max(excess, key=excess.get),
        "edges_excess_over_0.03": sum(1 for v in excess.values() if v > 0.03),
        "meta": graph.meta,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default="119,135,143,144,149,150,160,161")
    parser.add_argument("--choose", type=int, default=149, help="이정표 ① 로 확정하는 접합 깊이")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    refit = design.load_refit()
    diag = json.loads((DESIGN_DIR / "enclosure_diagnosis.json").read_text())
    coef = width_coefficients(diag, refit)
    out = []
    for k in [int(v) for v in args.k.split(",")]:
        if k > refit["homoclinic"]["k_max"] - 1:
            print(f"K={k}: refit orbit too short (k_max={refit['homoclinic']['k_max']})")
            continue
        g = design.design(k, refit=refit)
        s = summarize(g, coef)
        out.append(s)
        print(json.dumps({key: s[key] for key in ("k_junction", "loops", "entropy_lower_bound_per_step", "radius_min",
                                                   "radius_max", "slack_min", "slack_min_at", "slacks_negative",
                                                   "excess_max", "excess_max_at", "edges_excess_over_0.03")}))
    chosen = design.design(args.choose, refit=refit)
    detail = {
        "summary": summarize(chosen, coef),
        "vertices": {n: {"center_hex": [float(c).hex() for c in v.center], "frame_hex": [[float(x).hex() for x in row]
                                                                                       for row in v.frame],
                         "radii": v.radii.tolist()} for n, v in chosen.vertices.items()},
        "edges": [{"source": e.source, "target": e.target, "linear": e.linear.tolist(), "offset": e.offset.tolist()}
                  for e in chosen.edges],
        "slacks": [s.__dict__ for s in design.slacks(chosen)],
    }
    if not args.no_write:
        write_result("graph_design_scan", {"tag": "[수치] 선형 설계", "candidates": out},
                     script=Path(__file__).resolve(), params={"k": args.k}, started=started, results_dir=DESIGN_DIR)
        write_result("graph_design", {"tag": "[수치] 선형 설계(이정표 ①)", "k_junction": args.choose, **detail},
                     script=Path(__file__).resolve(), params={"choose": args.choose}, started=started,
                     results_dir=DESIGN_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
