"""T11 조립: 314변의 흐름 관(tubes/)·상자 감싸기(stage2/) 기록을 정확 판정(exact_judge.judge_v5)에 넣고, 변 판정과
정리 수준 측정값을 만들어 judge_claims 로 판정한다. 외피·그래프((A3)(A4)(A6))는 감사 V3 사본(graph_and_hulls)의 함수로 잰다.

사용: t11cli.py assemble [--tubes DIR] [--boxes DIR] [--out PATH] [--merged-tubes PATH]
exit 0 은 주장이 모두 성립하고 변별 기록 목록이 manifest 와 맞을 때만이다(manifest 대조는 manifest.py).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import exact_judge as EJ
import graph_and_hulls as GH
import v7exact as X

from . import claims as C
from . import design as dz
from .paths import BOXES, RESULTS, TUBES, edge_stem

OUT = RESULTS / "theorem_claims.json"
MERGED_TUBES = RESULTS / "rigorous_tubes.json"
PAIRS = ("r12", "r13", "r23")


@dataclass(frozen=True)
class EdgeVerdict:
    edge: str
    cover_proven: bool
    cover_margin: Fraction | None
    cone_proven_t7: bool
    a_star: Fraction | None
    b_star: Fraction | None
    degree: int | None
    structure_failures: tuple


def structure_failures(tube: dict | None, box: dict | None) -> tuple:
    """(A2) 를 받치는 변 조건(한 곳에서 선언). 빈 튜플이면 성립.
    V2 흐름 관: 출발 집합이 단면 위, 첫 영점 전 A>0, 교차 창에서 Ȧ<0(정확히 한 번 횡단), 관 전체 r_ij²>0, 판별식>0.
    V5 외피 비행: 출발 방향 위로, 첫 영점 아래로(통과 단계 사전 감싸기에서 ds/dt≠0), 판별식>0, 도착 근>0.
    두 창은 같은 τ(B) 를 감싸므로 서로소이면 모순이다."""
    out = []
    if tube is None:
        out.append("tube record missing")
    elif not tube.get("certified"):
        out.append(f"tube not certified: {tube.get('reason')}")
    if box is None:
        out.append("box record missing")
    if out:
        return tuple(out)
    if not Fraction(tube["disc_lower"]) > 0:
        out.append("V2 discriminant lower bound not positive")
    w0, w1 = (Fraction(x) for x in tube["window"])
    if not 0 < w0 <= w1:
        out.append("V2 return window not positive")
    t0, t1 = X.iv(box["tau_hull_exact"])
    if not 0 < t0:
        out.append("V5 return window not positive")
    if w1 < t0 or t1 < w0:
        out.append("V2 and V5 return windows disjoint")
    if not X.iv(box["disc_hull_exact"])[0] > 0:
        out.append("V5 discriminant lower bound not positive")
    if not X.iv(box["branch_out_hull_exact"])[0] > 0:
        out.append("V5 arrival root -dA/dt/r not positive")
    if not all(Fraction(tube["r_lower"][p]) > 0 for p in PAIRS):
        out.append("V2 tube r_ij lower bound not positive")
    return tuple(out)


def edge_verdict(edge: str, tube: dict | None, box: dict | None, judged: dict | None, signs: list) -> EdgeVerdict:
    if judged is None:
        return EdgeVerdict(edge, False, None, False, None, None, None, structure_failures(tube, box))
    a, b = Fraction(judged["a_star_exact"]), Fraction(judged["b_star_exact"])
    cov = Fraction(judged["cover_margin_mv_exact"])
    return EdgeVerdict(edge, cov > 0, cov, a < C.THEOREM["T7.mu_s"] and b > C.THEOREM["T7.mu_u"], a, b,
                       signs[0] * signs[1], structure_failures(tube, box))


def expected_edges() -> set:
    """T6 REPORT §3.3 의 그래프: cycle N_i→N_{i+1 mod 16}, N8→M−148, M_k→M_{k+1}(k=−148…147), M+148→N8."""
    out = {(f"N{i}", f"N{(i + 1) % 16}") for i in range(16)} | {("N8", "M-148"), ("M+148", "N8")}
    return out | {(f"M{k:+d}", f"M{k + 1:+d}") for k in range(-148, 148)}


def hull_gaps(design: dict) -> dict:
    """(A3) M+0 외피와 다른 정점 외피의 최소 분리 간격, (A6) cycle 16개 외피의 쌍별 최소 분리 간격(V3 사본 함수)."""
    V = design["vertices"]
    hulls = {v: GH.hull(*GH.hset(V[v])) for v in V}
    a3 = min(((GH.separation(hulls["M+0"], hulls[v]), v) for v in hulls if v != "M+0"))
    cyc = [f"N{i}" for i in range(16)]
    a6 = min(((GH.separation(hulls[p], hulls[q]), f"{p}·{q}") for p, q in itertools.combinations(cyc, 2)))
    return {"A3_gap": a3, "A6_gap": a6}


def assemble(records: dict, design: dict) -> dict:
    """records[edge] = {"tube": row|None, "box_path": Path|None}. 반환: 변 판정, 측정값, 주장 판정."""
    signs = EJ.design_signs(design)
    edges = [f"{e['source']}>{e['target']}" for e in design["edges"]]
    verdicts, judged_rows = {}, {}
    for e in edges:
        rec = records.get(e, {})
        tube, path = rec.get("tube"), rec.get("box_path")
        box = json.loads(Path(path).read_text()) if path is not None else None
        judged = EJ.judge_v5(path, signs) if path is not None else None
        judged_rows[e] = judged
        verdicts[e] = edge_verdict(e, tube, box, judged, signs[e])
    present = [e for e in edges if verdicts[e].a_star is not None and records.get(e, {}).get("tube") is not None]
    missing = tuple(e for e in edges if e not in present)
    tubes = {e: records[e]["tube"] for e in present}
    values, at = {n: Fraction(0) for n in C.MEASURED}, {}

    def pick(name, items, best):
        v, e = best(items, key=lambda t: t[0])
        values[name], at[name] = v, e

    if present:
        pick("cover_margin", [(verdicts[e].cover_margin, e) for e in present], min)
        pick("a_max", [(verdicts[e].a_star, e) for e in present], max)
        pick("b_min", [(verdicts[e].b_star, e) for e in present], min)
        pick("g1", [(min(verdicts[e].b_star - 1, 1 - verdicts[e].a_star), e) for e in present], min)
        pick("delta", [(Fraction(tubes[e]["r_lower"][p]), f"{e}:{p}") for e in present for p in PAIRS], min)
        pick("R", [(Fraction(tubes[e]["r_upper"][p]), f"{e}:{p}") for e in present for p in PAIRS], max)
        pick("t_min", [(Fraction(tubes[e]["window"][0]), e) for e in present], min)
        pick("tau_max", [(Fraction(tubes[e]["window"][1]), e) for e in present], max)
    gaps = hull_gaps(design)
    for name, (v, where) in gaps.items():
        values[name], at[name] = v, where
    degrees = {}
    for e in present:
        degrees[verdicts[e].degree] = degrees.get(verdicts[e].degree, 0) + 1
    measured = C.Measured(
        edges=len(present), design_edges=len(edges), missing=missing,
        cover_unproven=tuple(e for e in present if not verdicts[e].cover_proven),
        degrees=degrees, structure_failed=tuple(e for e in edges if verdicts[e].structure_failures),
        g1_nonpositive=tuple(e for e in present if not min(verdicts[e].b_star - 1, 1 - verdicts[e].a_star) > 0),
        graph_matches_expected={(e["source"], e["target"]) for e in design["edges"]} == expected_edges()
        and len(design["edges"]) == len(expected_edges()),
        values=values, at=at)
    checks = C.judge_claims(measured)
    return {"verdicts": verdicts, "judged": judged_rows, "measured": measured, "checks": checks}


def load_records(tube_dir: Path, box_dir: Path, design: dict) -> dict:
    out = {}
    for e in design["edges"]:
        edge = f"{e['source']}>{e['target']}"
        tp = tube_dir / f"{edge_stem(edge)}.json"
        bp = box_dir / f"{edge_stem(edge)}.json"
        out[edge] = {"tube": json.loads(tp.read_text())["rows"][0] if tp.exists() else None,
                     "box_path": bp if bp.exists() else None}
    return out


def directed(q: Fraction, mode: str) -> str:
    return X.fstr(q, 12, mode)


def report(result: dict) -> dict:
    m = result["measured"]
    checks = {c.name: {"holds": c.holds, "relation": c.relation, "margin": str(c.margin), "margin_float": float(c.margin),
                       "measured": None if c.measured is None else str(c.measured),
                       "constant": None if c.constant is None else str(c.constant), "detail": c.detail}
              for c in result["checks"]}
    values = {n: {"exact": str(v), "directed_12": directed(v, "up" if n in ("a_max", "R", "tau_max") else "down"),
                  "at": m.at.get(n)} for n, v in m.values.items()}
    return {
        "tag": "[CAP] T11 정리 수준 주장(Arb 감싸기 + 정확 유리수 판정, CAPD 무관)",
        "claimed": {k: (str(v) if isinstance(v, Fraction) else v) for k, v in C.CLAIMED.items() if k != "degrees"},
        "claimed_degrees": {str(k): v for k, v in C.CLAIMED["degrees"].items()},
        "theorem_constants": {k: str(v) for k, v in C.THEOREM.items()},
        "measured": values,
        "edges": m.edges, "design_edges": m.design_edges, "missing": list(m.missing),
        "cover_unproven": list(m.cover_unproven), "degrees": {str(k): v for k, v in m.degrees.items()},
        "structure_failed": {e: list(result["verdicts"][e].structure_failures) for e in m.structure_failed},
        "g1_nonpositive": list(m.g1_nonpositive),
        "graph_matches_expected": m.graph_matches_expected,
        "cone_t7_unproven": [e for e, v in result["verdicts"].items() if not v.cone_proven_t7],
        "checks": checks,
        "all_claims_hold": all(c.holds for c in result["checks"]),
    }


def merged_tubes(records: dict, design: dict) -> dict:
    """V2 형식(rigorous_controls 입력)으로 변별 흐름 관 기록을 합친다."""
    rows = [records[f"{e['source']}>{e['target']}"]["tube"] for e in design["edges"]]
    good = [r for r in rows if r is not None and r["certified"]]
    dmin = min(good, key=lambda r: Fraction(r["min_lower"]))
    rmax = max(good, key=lambda r: Fraction(r["max_upper"]))
    tmin = min(good, key=lambda r: Fraction(r["window"][0]))
    tmax = max(good, key=lambda r: Fraction(r["window"][1]))
    summary = {"edges": len(rows), "certified": len(good),
               "delta_mine": dmin["min_lower"], "delta_edge": dmin["edge"],
               "R_mine": rmax["max_upper"], "R_edge": rmax["edge"],
               "tmin_mine": tmin["window"][0], "tmin_edge": tmin["edge"],
               "tmax_mine": tmax["window"][1], "tmax_edge": tmax["edge"]}
    return {"tag": "[CAP] T11 변별 흐름 관 기록(tubes/) 합본", "summary": summary, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tubes", default=str(TUBES))
    ap.add_argument("--boxes", default=str(BOXES))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--merged-tubes", default=str(MERGED_TUBES))
    args = ap.parse_args(argv)
    started = time.time()
    design = dz.design()
    records = load_records(Path(args.tubes), Path(args.boxes), design)
    result = assemble(records, design)
    out = report(result)
    out["edge_verdicts"] = [{**{k: (str(v) if isinstance(v, Fraction) else v) for k, v in asdict(ev).items()},
                             "structure_failures": list(ev.structure_failures)} for ev in result["verdicts"].values()]
    out["seconds"] = round(time.time() - started, 1)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    if not out["missing"]:
        Path(args.merged_tubes).write_text(json.dumps(merged_tubes(records, design), ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("edges", "missing", "cover_unproven", "degrees", "structure_failed",
                                          "all_claims_hold")}, ensure_ascii=False))
    for name, c in out["checks"].items():
        print(f"  {name}: holds={c['holds']} margin={c['margin_float']:.6g} | {c['detail']}")
    return 0 if out["all_claims_hold"] else 1


if __name__ == "__main__":
    sys.exit(main())
