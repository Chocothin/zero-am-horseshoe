"""[CAP] 정리 A′ 조립: T6 정리 A 의 가정·주장을 T6 코드로 다시 확인한 뒤, 저장 D(B) 로 원뿔 판정을 모아 정리 수준 주장 8개
(A5 전 변 원뿔, μ_u, μ_s, A6 cycle h-set 서로소, 쌍곡 상수 λ_s·λ_u·C_s·C_u)를 판정하고 주장마다 전용 음성 대조 2개를 돌린다(G6).
쌍곡 상수는 hyperbolic.rate_constants 의 방향 반올림 값과 CLAIMED 가 같아야 한다.

판정 논리는 t7hb/hyperbolic.py 한 곳에 있다(G7 감사 대상). 대조는 엔진을 다시 돌리지 않고 실데이터에 심는다.
  강화 대조 : 주장한 상수를 다시 잰 값의 (1±1e-12) 바깥으로 옮긴다(같은 judge_claims)
  심은 입력 : 저장 기록의 D(B) 성분·사용 조건, 외피를 바꿔 stored.cone_input → hyperbolic.assemble → judge_claims 를 지난다
모든 주장이 성립하고, 주장마다 거절된 대조가 2개 이상이며(controls_cover), 새어 나간 대조가 없고, 강화의 안쪽(1∓1e-12)
에서는 여전히 성립해야 exit 0 이다.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from t7hb.results import T3_DIR, T6_DIR, T6_DESIGN, T6_RESULTS, sha256_file, t6_record_problems, write_result  # noqa: E402

sys.path.insert(0, str(T3_DIR))
sys.path.insert(0, str(T6_DIR))
from t3cap.exact import Interval  # noqa: E402
from t3cap.export import interval_from_json, to_json  # noqa: E402
from t6hs import assembly as t6assembly, enclose  # noqa: E402

from t7hb import hyperbolic, stored  # noqa: E402
from t7hb.cone import STABLE, UNSTABLE, cone_bounds  # noqa: E402

# 보고서 정리 A′ 가 적은 상수. 데이터와 독립으로 고정하고, 다시 잰 값이 이를 받치는지 판정한다
CLAIMED = hyperbolic.Claimed(mu_s=Fraction("0.8944"), mu_u=Fraction("1.1167"), lambda_s=Fraction("0.9458"),
                             lambda_u=Fraction("1.0567"), c_s=Fraction("3.078"), c_u=Fraction("3.252"))
THRESHOLD_DIGITS = 30
RELATIVE = Fraction(1, 10**12)
PLANT = Fraction(1, 10**3)
MIN_CONTROLS = 2


@dataclass(frozen=True)
class Sources:
    records: dict[str, dict]
    design_hash: str
    edges: tuple[str, ...]
    hulls: dict[str, list]


def load_sources(cover_dir: Path) -> Sources:
    design_path = T6_DESIGN / "graph_design.json"
    design = json.loads(design_path.read_text())
    recs = {}
    for e in design["edges"]:
        path = cover_dir / f"{e['source']}__{e['target']}.json"
        if path.exists():
            rec = stored.load(path)
            recs[rec["edge"]] = rec
    hulls = {n: t6assembly.hull(h.center, h.matrix)
             for n, h in ((n, enclose.hset_from_design(n, v)) for n, v in design["vertices"].items())}
    return Sources(recs, sha256_file(design_path), tuple(f"{e['source']}>{e['target']}" for e in design["edges"]),
                   hulls)


def assemble_from(src: Sources, recs: dict | None = None, hulls: dict | None = None) -> hyperbolic.Assembly:
    inputs = [stored.cone_input(r, src.design_hash) for r in (src.records if recs is None else recs).values()]
    return hyperbolic.assemble(inputs, src.edges, src.hulls if hulls is None else hulls)


def t6_theorem_a() -> dict:
    """T6 정리 A 의 가정과 주장 7개를 T6 코드 그대로 다시 판정한다(정리 A′ 의 전제)."""
    spec = importlib.util.spec_from_file_location("t6_assemble_theorem", T6_DIR / "scripts" / "assemble_theorem.py")
    mod = importlib.util.module_from_spec(spec)
    # dataclass 가 모듈 이름공간을 sys.modules 에서 찾으므로 실행 전에 등록한다
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    a, _ = mod.build()
    claims = t6assembly.judge_claims(a, mod.CLAIMED)
    return {"hypotheses_certified": a.certified, "claims_hold": all(c.holds for c in claims),
            "claims": {c.name: c.holds for c in claims}}


def check_of(a: hyperbolic.Assembly, claimed: hyperbolic.Claimed, name: str) -> hyperbolic.ClaimCheck:
    return next(c for c in hyperbolic.judge_claims(a, claimed) if c.name == name)


def as_json(x: Fraction | None) -> dict | None:
    return None if x is None else {"exact": str(x), "float": float(x)}


def run_control(claim: str, kind: str, control: str, planted: str, run) -> dict:
    row = {"claim": claim, "kind": kind, "control": control, "planted": planted}
    try:
        a, claimed = run()
        c = check_of(a, claimed, claim)
        return {**row, "computed": as_json(c.margin), "judge_output": c.holds,
                "reason": "judge false" if not c.holds else "LEAKED", "rejected": not c.holds}
    except ValueError as exc:
        return {**row, "computed": None, "judge_output": None, "reason": f"guard exception: {exc}", "rejected": True}


def scaled_entry(rec: dict, i: int, j: int, factor: Fraction) -> dict:
    """기록 D(B) 의 (i,j) 구간 끝점을 factor 배 한 사본."""
    out = copy.deepcopy(rec)
    iv = interval_from_json(rec["derivative"][i][j])
    lo, hi = sorted((iv.lo * factor, iv.hi * factor))
    out["derivative"][i][j] = to_json(Interval(lo, hi))
    return out


def broken_usage(rec: dict) -> dict:
    """기록 상자 조각의 c1 귀환 창 위끝을 둘째 영점 창 아래끝으로 올린 사본(사용 조건 「c1 창 < 둘째 영점 창」 위반).
    c1 실행 기록과 그 실행을 쓰는 조각 사본을 함께 바꾸고 c1 흐름 관 끝을 그 위끝까지 늘려 그 조건 하나만 깨지게 한다."""
    out = copy.deepcopy(rec)
    index = out["pieces_detail"]["whole"][0]["runs_index"]
    run = out["runs"][index]["c1"]
    c1 = interval_from_json(run["return_time"])
    z2 = interval_from_json(out["runs"][index]["second_zero"]["return_time"])
    raised = to_json(Interval(c1.lo, max(c1.hi, z2.lo)))
    t_end = interval_from_json(run["tube"]["t_end"])
    run["return_time"] = raised
    run["tube"]["t_end"] = to_json(Interval(max(t_end.lo, z2.lo), max(t_end.hi, z2.lo)))
    for p in (q for ps in out["pieces_detail"].values() for q in ps if q.get("runs_index") == index):
        p["c1_return_time"] = raised
    return out


def binding_row(rec: dict, rows: tuple[int, ...], pick) -> int:
    b = cone_bounds(stored.derivative_of(rec))
    value = {i: (b.diag_lower[i] - b.radii[i]) if i in UNSTABLE else (b.radii[i] - b.diag_lower[i]) for i in rows}
    return pick(rows, key=lambda i: value[i])


def controls(src: Sources, a: hyperbolic.Assembly) -> tuple[list[dict], list[dict]]:
    def with_record(edge, rec):
        return {**src.records, edge: rec}

    up_row = binding_row(src.records[a.upper_edge], UNSTABLE, min)
    lo_row = binding_row(src.records[a.lower_edge], STABLE, max)
    first, second = a.cycle_pair
    dropped = "M+0>M+1"

    def inflated(e):
        return {**src.hulls, first: [(lo - e, hi + e) for lo, hi in src.hulls[first]]}

    planted_u = assemble_from(src, recs=with_record(a.upper_edge, scaled_entry(
        src.records[a.upper_edge], up_row, up_row, 1 - PLANT)))
    planted_s = assemble_from(src, recs=with_record(a.lower_edge, scaled_entry(
        src.records[a.lower_edge], lo_row, lo_row, 1 + PLANT)))
    eps = hyperbolic.derived_eps(CLAIMED)

    def threshold(q, upper):
        return hyperbolic.sqrt_bound(q, THRESHOLD_DIGITS, upper)

    beyond = {"lambda_s": threshold(CLAIMED.mu_s, False) * (1 - RELATIVE),
              "lambda_u": threshold(CLAIMED.mu_u, True) * (1 + RELATIVE),
              "c_s": threshold(1 / eps, False) * (1 - RELATIVE), "c_u": threshold(CLAIMED.mu_u / eps, False) * (1 - RELATIVE)}
    within = {"lambda_s": threshold(CLAIMED.mu_s, True) * (1 + RELATIVE),
              "lambda_u": threshold(CLAIMED.mu_u, False) * (1 - RELATIVE),
              "c_s": threshold(1 / eps, True) * (1 + RELATIVE), "c_u": threshold(CLAIMED.mu_u / eps, True) * (1 + RELATIVE)}
    rate_names = {"lambda_s": "lambda_s", "lambda_u": "lambda_u", "c_s": "C_s", "c_u": "C_u"}
    rate_rows = [row for field, claim in rate_names.items() for row in (
        run_control(claim, "tightened", f"{field} moved to (1-+1e-12) x its exact threshold, on the wrong side",
                    f"{field} {float(beyond[field]):.17g}", lambda f=field: (a, replace(CLAIMED, **{f: beyond[f]}))),
        run_control(claim, "planted", (f"record {a.lower_edge}: D(B)[{lo_row}][{lo_row}] scaled by (1+1e-3), so mu_s fails"
                                       if field in ("lambda_s", "c_s") else
                                       f"record {a.upper_edge}: D(B)[{up_row}][{up_row}] scaled by (1-1e-3), so mu_u fails"),
                    "mu claim it relies on is broken in the data",
                    lambda f=field: ((planted_s if f in ("lambda_s", "c_s") else planted_u), CLAIMED)))]

    rows = [
        run_control("A5_all_edges_cone", "planted", f"record {dropped} removed from the inputs", f"records - {dropped}",
                    lambda: (assemble_from(src, recs={k: v for k, v in src.records.items() if k != dropped}), CLAIMED)),
        run_control("A5_all_edges_cone", "planted", "record N0>N1: c1 return window raised onto the second-zero window",
                    "c1_return_time.hi = second_zero_time.lo",
                    lambda: (assemble_from(src, recs=with_record("N0>N1", broken_usage(src.records["N0>N1"]))), CLAIMED)),
        run_control("mu_u", "tightened", "claim raised to (1+1e-12) x re-measured min b*",
                    f"mu_u {float(a.upper_min * (1 + RELATIVE)):.17g}",
                    lambda: (a, replace(CLAIMED, mu_u=a.upper_min * (1 + RELATIVE)))),
        run_control("mu_u", "planted", f"record {a.upper_edge}: D(B)[{up_row}][{up_row}] scaled by (1-1e-3)",
                    f"D[{up_row}][{up_row}] x {1 - PLANT}", lambda: (planted_u, CLAIMED)),
        run_control("mu_s", "tightened", "claim lowered to (1-1e-12) x re-measured max a*",
                    f"mu_s {float(a.lower_max * (1 - RELATIVE)):.17g}",
                    lambda: (a, replace(CLAIMED, mu_s=a.lower_max * (1 - RELATIVE)))),
        run_control("mu_s", "planted", f"record {a.lower_edge}: D(B)[{lo_row}][{lo_row}] scaled by (1+1e-3)",
                    f"D[{lo_row}][{lo_row}] x {1 + PLANT}", lambda: (planted_s, CLAIMED)),
        run_control("A6_cycle_disjoint", "tightened", f"hull of {first} inflated by (1+1e-12) x re-measured min gap",
                    f"inflate {float(a.cycle_gap * (1 + RELATIVE)):.17g}",
                    lambda: (assemble_from(src, hulls=inflated(a.cycle_gap * (1 + RELATIVE))), CLAIMED)),
        run_control("A6_cycle_disjoint", "planted", f"hull of {second} replaced by the hull of {first}",
                    f"hull[{second}] = hull[{first}]",
                    lambda: (assemble_from(src, hulls={**src.hulls, second: src.hulls[first]}), CLAIMED)),
    ] + rate_rows
    inside = [
        ("mu_u", replace(CLAIMED, mu_u=a.upper_min * (1 - RELATIVE)), a),
        ("mu_s", replace(CLAIMED, mu_s=a.lower_max * (1 + RELATIVE)), a),
        ("A6_cycle_disjoint", CLAIMED, assemble_from(src, hulls=inflated(a.cycle_gap * (1 - RELATIVE)))),
    ] + [(claim, replace(CLAIMED, **{field: within[field]}), a) for field, claim in rate_names.items()]
    sharp = [{"claim": n, "inside_tightening_still_holds": check_of(aa, cl, n).holds} for n, cl, aa in inside]
    return rows, sharp


def derived(claimed: hyperbolic.Claimed) -> dict:
    """정리 A′ 의 쌍곡 상수: μ 에서 방향 반올림으로 만든 값과 CLAIMED 를 나란히 적는다(float 는 표시용)."""
    eps = hyperbolic.derived_eps(claimed)
    rounded = hyperbolic.rate_constants(claimed.mu_s, claimed.mu_u)
    return {"eps": str(eps), "eps_float": float(eps),
            "directed_rounding": {k: str(v) for k, v in rounded.items()},
            "matches_claimed": all(rounded[k] == getattr(claimed, k) for k in rounded),
            "exact_floats": {"lambda_s": math.sqrt(claimed.mu_s), "lambda_u": math.sqrt(claimed.mu_u),
                             "c_s": 1 / math.sqrt(eps), "c_u": math.sqrt(claimed.mu_u / eps)}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    premise = t6_theorem_a()
    src = load_sources(T6_RESULTS / "coverings")
    a = assemble_from(src)
    claims = hyperbolic.judge_claims(a, CLAIMED)
    rows, sharp = controls(src, a)
    per_claim = {c.name: sum(1 for r in rows if r["claim"] == c.name and r["rejected"]) for c in claims}
    summary = {
        "t6_premise_holds": premise["hypotheses_certified"] and premise["claims_hold"],
        "claims_hold": all(c.holds for c in claims), "claims": len(claims), "controls": len(rows),
        "controls_per_claim": per_claim, "controls_cover": all(v >= MIN_CONTROLS for v in per_claim.values()),
        "leaked": sum(1 for r in rows if not r["rejected"]),
        "sharpness_all": all(s["inside_tightening_still_holds"] for s in sharp),
        "rate_constants_match_directed_rounding": derived(CLAIMED)["matches_claimed"],
        "t6_record_integrity_problems": t6_record_problems(
            [T6_RESULTS / "coverings" / f"{e.replace('>', '__')}.json" for e in src.edges]),
    }
    out = {
        "theorem_Aprime_hypotheses_certified": a.certified, "t6_premise": premise,
        "edges": len(a.edges), "missing": list(a.missing), "uncertified": list(a.uncertified),
        "max_lower_a": as_json(a.lower_max), "max_lower_edge": a.lower_edge,
        "min_upper_b": as_json(a.upper_min), "min_upper_edge": a.upper_edge,
        "cycle_min_separation": as_json(a.cycle_gap), "cycle_closest_pair": list(a.cycle_pair or ()),
        "claimed": {k: str(v) for k, v in vars(CLAIMED).items()}, "derived_constants": derived(CLAIMED),
        "claims": [{"claim": c.name, "holds": c.holds, "margin": as_json(c.margin)} for c in claims],
        "claim_controls_summary": summary, "claim_controls": rows, "claim_controls_sharpness": sharp,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "claim_controls"}, indent=1, ensure_ascii=False))
    if not args.no_write:
        write_result("theorem_Aprime", {"tag": "[CAP] 정리 A′ 가정·주장 판정, G6 주장별 전용 대조", **out},
                     script=Path(__file__).resolve(), params=vars(args), started=started)
    ok = (not summary["t6_record_integrity_problems"] and summary["t6_premise_holds"] and a.certified
          and summary["claims_hold"] and summary["controls_cover"]
          and not summary["leaked"] and summary["sharpness_all"] and summary["rate_constants_match_directed_rounding"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
