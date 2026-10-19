"""[CAP] 정리 A 조립: 저장된 덮개 감싸기를 다시 판정하고 그래프 구조·M₀ 분리·흐름 관 상수·엔트로피 하한을 모은 뒤,
보고서가 적은 정리 수준 주장 7개(δ, R, t_min, (A3), (A4), ρ, ln ρ)를 판정하고 주장마다 전용 음성 대조 2개를 돌린다(G6).

판정 논리는 t6hs/assembly.py 한 곳에 있다(G7 감사 대상). 대조는 엔진을 다시 돌리지 않고 실데이터에 심는다.
  강화 대조 : 주장한 상수를 다시 잰 값의 (1±1e-12) 바깥으로 옮긴다(같은 judge_claims)
  심은 입력 : 저장 기록 하나의 흐름 관·귀환 하한, 외피, 설계 변·접합 첨자, ρ 를 바꿔 records.covering_input →
              assembly.assemble → judge_claims 경로를 그대로 지난다
모든 주장이 성립하고, 주장마다 거절된 대조가 2개 이상이며(controls_cover), 새어 나간 대조가 없고, 강화 대조의 안쪽
(1∓1e-12)에서는 여전히 성립해야 exit 0 이다.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from t6hs.results import DESIGN_DIR, RESULTS_DIR, T3_DIR, sha256_file, write_result  # noqa: E402

sys.path.insert(0, str(T3_DIR))
from t3cap.exact import Interval  # noqa: E402
from t3cap.export import interval_from_json, to_json  # noqa: E402

from t6hs import assembly, covering, enclose, records  # noqa: E402

# 보고서 정리 A 가 적은 상수. 데이터와 독립으로 고정하고, 다시 잰 값이 이를 받치는지 판정한다
CLAIMED = assembly.Claimed(delta=Fraction("0.217857228763"), rmax=Fraction("7.419986043806"),
                           tmin=Fraction("0.276707990268"), rho=Fraction(1217868143156043955884963, 2**80),
                           ln_rho=Fraction("0.0073696936384425343859428"))
RELATIVE = Fraction(1, 10**12)
PLANT = Fraction(1, 10**6)
MIN_CONTROLS = 2
MARKER = "M+0"


@dataclass(frozen=True)
class Sources:
    records: dict[str, dict]           # 변 → 저장 기록(설계 해시가 맞는 것만)
    edges: frozenset[tuple[str, str]]
    k: int
    departure: int
    arrival: int
    hulls: dict[str, list]


def load_sources(cover_dir: Path) -> Sources:
    design_path = DESIGN_DIR / "graph_design.json"
    design = json.loads(design_path.read_text())
    design_hash = sha256_file(design_path)
    recs = {}
    for e in design["edges"]:
        path = cover_dir / f"{e['source']}__{e['target']}.json"
        if path.exists():
            rec = records.load(path)
            # 다른 설계로 낸 기록은 없는 것으로 친다
            if rec["design_sha256"] == design_hash:
                recs[rec["edge"]] = rec
    s = design["summary"]
    hulls = {n: assembly.hull(h.center, h.matrix)
             for n, h in ((n, enclose.hset_from_design(n, v)) for n, v in design["vertices"].items())}
    edges = frozenset((e["source"], e["target"]) for e in design["edges"])
    return Sources(recs, edges, design["k_junction"], s["departure"], s["arrival"], hulls)


def assemble_from(src: Sources, recs=None, edges=None, departure=None, hulls=None) -> assembly.Assembly:
    inputs = [records.covering_input(r) for r in (src.records if recs is None else recs).values()]
    return assembly.assemble(inputs, set(src.edges if edges is None else edges), src.k,
                             src.departure if departure is None else departure, src.arrival,
                             src.hulls if hulls is None else hulls, marker=MARKER)


def default_dir(tag: str) -> Path:
    return RESULTS_DIR / "coverings" / tag if tag else RESULTS_DIR / "coverings"


def degrees_of(src: Sources) -> dict[str, int]:
    return {c.edge: covering.covering_verdict(c.whole, c.faces, c.signs).degree
            for c in (records.covering_input(r) for r in src.records.values())}


def build(tag: str = "", cover_dir: Path | None = None):
    src = load_sources(cover_dir or default_dir(tag))
    return assemble_from(src), degrees_of(src)


def check_of(a: assembly.Assembly, claimed: assembly.Claimed, name: str) -> assembly.ClaimCheck:
    return next(c for c in assembly.judge_claims(a, claimed) if c.name == name)


def as_json(x: Fraction) -> dict:
    """짧은 정확값은 그대로 적고, ρ·ln ρ 여유처럼 수천 자리인 값은 아래·위로 자른 30자리 소수 구간으로 적는다."""
    if max(x.numerator.bit_length(), x.denominator.bit_length()) <= 1024:
        return {"exact": str(x), "float": float(x)}
    k = 30 - math.floor(math.log10(abs(float(x)))) if x else 30
    return {"enclosure": [str(Fraction(math.floor(x * 10**k), 10**k)), str(Fraction(math.ceil(x * 10**k), 10**k))],
            "float": float(x)}


def short(x: Fraction) -> str:
    return json.dumps(as_json(x))


def run_control(claim: str, kind: str, control: str, planted: str, run) -> dict:
    """run() → (Assembly, Claimed). 판정은 judge_claims 가 낸 그 주장 하나의 결과다."""
    row = {"claim": claim, "kind": kind, "control": control, "planted": planted}
    try:
        a, claimed = run()
        c = check_of(a, claimed, claim)
        return {**row, "computed": None if c.margin is None else as_json(c.margin), "judge_output": c.holds,
                "reason": "judge false" if not c.holds else "LEAKED", "rejected": not c.holds}
    except ValueError as exc:
        return {**row, "computed": None, "judge_output": None, "reason": f"guard exception: {exc}", "rejected": True}


def planted_record(rec: dict, path: tuple[str, ...], lo: Fraction | None = None, hi: Fraction | None = None) -> dict:
    """저장 기록 상자 조각이 쓰는 c0 실행의 구간 하나(흐름 관 r12 또는 귀환 시각)를 심은 끝점으로 바꾼 사본. 그 실행 기록과
    그 실행을 쓰는 조각 사본(상자·출구 면)을 함께 바꿔 사용 조건(사본 = 실행 기록)은 그대로 두고 그 값만 바꾼다."""
    out = copy.deepcopy(rec)
    index = out["pieces_detail"]["whole"][0]["runs_index"]
    sharing = [p for ps in out["pieces_detail"].values() for p in ps if p.get("runs_index") == index]
    for target in (out["runs"][index]["c0"], *sharing):
        for key in path[:-1]:
            target = target[key]
        old = interval_from_json(target[path[-1]])
        target[path[-1]] = to_json(Interval(old.lo if lo is None else lo, old.hi if hi is None else hi))
    return out


def controls(src: Sources, a: assembly.Assembly) -> tuple[list[dict], list[dict]]:
    inputs = {e: records.covering_input(r) for e, r in src.records.items()}
    low_tube = min(inputs, key=lambda e: inputs[e].tube_lower)
    high_tube = max(inputs, key=lambda e: inputs[e].tube_upper)
    low_time = min(inputs, key=lambda e: inputs[e].return_time_lower)

    def with_record(edge, rec):
        return {**src.records, edge: rec}

    delta_plant = CLAIMED.delta * (1 - PLANT)
    rmax_plant = CLAIMED.rmax * (1 + PLANT)
    tmin_plant = CLAIMED.tmin * (1 - PLANT)
    m0 = src.hulls[MARKER]
    nearest = min((n for n in src.hulls if n != MARKER), key=lambda n: assembly.separation(m0, src.hulls[n]))

    def inflated(e):
        return {**src.hulls, MARKER: [(lo - e, hi + e) for lo, hi in m0]}

    ln_bound = assembly.ln_lower_bound(CLAIMED.rho)
    dropped = ("N15", "N0")
    rows = [
        run_control("delta", "tightened", "claim raised to (1+1e-12) x re-measured min tube r_ij lower",
                    f"delta {short(a.delta * (1 + RELATIVE))}", lambda: (a, replace(CLAIMED, delta=a.delta * (1 + RELATIVE)))),
        run_control("delta", "planted", f"record {low_tube}: whole-box tube r12 lower set below the claim",
                    f"r12.lo {short(delta_plant)}",
                    lambda: (assemble_from(src, recs=with_record(low_tube, planted_record(
                        src.records[low_tube], ("tube", "r12"), lo=delta_plant))), CLAIMED)),
        run_control("R", "tightened", "claim lowered to (1-1e-12) x re-measured max tube r_ij upper",
                    f"R {short(a.rmax * (1 - RELATIVE))}", lambda: (a, replace(CLAIMED, rmax=a.rmax * (1 - RELATIVE)))),
        run_control("R", "planted", f"record {high_tube}: whole-box tube r12 upper set above the claim",
                    f"r12.hi {short(rmax_plant)}",
                    lambda: (assemble_from(src, recs=with_record(high_tube, planted_record(
                        src.records[high_tube], ("tube", "r12"), hi=rmax_plant))), CLAIMED)),
        run_control("t_min", "tightened", "claim raised to (1+1e-12) x re-measured min return-time lower",
                    f"t_min {short(a.tmin * (1 + RELATIVE))}", lambda: (a, replace(CLAIMED, tmin=a.tmin * (1 + RELATIVE)))),
        run_control("t_min", "planted", f"record {low_time}: whole-box c0 return window lower set below the claim",
                    f"return_time.lo {short(tmin_plant)}",
                    lambda: (assemble_from(src, recs=with_record(low_time, planted_record(
                        src.records[low_time], ("return_time",), lo=tmin_plant))), CLAIMED)),
        run_control("A3_m0_disjoint", "tightened", "M+0 hull inflated by (1+1e-12) x re-measured min separation",
                    f"inflate {short(a.m0_gap * (1 + RELATIVE))}",
                    lambda: (assemble_from(src, hulls=inflated(a.m0_gap * (1 + RELATIVE))), CLAIMED)),
        run_control("A3_m0_disjoint", "planted", f"hull of the nearest vertex {nearest} replaced by the M+0 hull",
                    f"hull[{nearest}] = hull[M+0]",
                    lambda: (assemble_from(src, hulls={**src.hulls, nearest: m0}), CLAIMED)),
        run_control("A4_graph", "planted", f"design edge {dropped[0]}>{dropped[1]} removed", f"edges - {dropped}",
                    lambda: (assemble_from(src, edges=src.edges - {dropped}), CLAIMED)),
        run_control("A4_graph", "planted", "departure index shifted by +1", f"departure {src.departure + 1}",
                    lambda: (assemble_from(src, departure=src.departure + 1), CLAIMED)),
        run_control("rho", "tightened", "claim raised to (1+1e-12) x claimed rho", f"rho {short(CLAIMED.rho * (1 + RELATIVE))}",
                    lambda: (a, replace(CLAIMED, rho=CLAIMED.rho * (1 + RELATIVE)))),
        run_control("rho", "planted", "loop lengths (16,299) via departure +1, claimed rho kept",
                    "loops (16, 299)", lambda: (assemble_from(src, departure=src.departure + 1), CLAIMED)),
        run_control("ln_rho", "tightened", "claim raised to (1+1e-12) x certified ln lower bound",
                    f"ln_rho {short(ln_bound * (1 + RELATIVE))}",
                    lambda: (a, replace(CLAIMED, ln_rho=ln_bound * (1 + RELATIVE)))),
        run_control("ln_rho", "planted", "rho lowered to (1-1e-6) x claimed rho, ln claim kept",
                    f"rho {short(CLAIMED.rho * (1 - PLANT))}", lambda: (a, replace(CLAIMED, rho=CLAIMED.rho * (1 - PLANT)))),
    ]
    inside = [
        ("delta", replace(CLAIMED, delta=a.delta * (1 - RELATIVE)), a),
        ("R", replace(CLAIMED, rmax=a.rmax * (1 + RELATIVE)), a),
        ("t_min", replace(CLAIMED, tmin=a.tmin * (1 - RELATIVE)), a),
        ("A3_m0_disjoint", CLAIMED, assemble_from(src, hulls=inflated(a.m0_gap * (1 - RELATIVE)))),
        ("rho", replace(CLAIMED, rho=CLAIMED.rho * (1 - RELATIVE)), a),
        ("ln_rho", replace(CLAIMED, ln_rho=ln_bound * (1 - RELATIVE)), a),
    ]
    sharp = [{"claim": n, "inside_tightening_still_holds": check_of(aa, cl, n).holds} for n, cl, aa in inside]
    return rows, sharp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="")
    parser.add_argument("--cover-dir", default="", help="기록 폴더(R6 기준 상대 경로). 기본은 coverings/<tag>")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    src = load_sources(RESULTS_DIR / args.cover_dir if args.cover_dir else default_dir(args.tag))
    a, degrees = assemble_from(src), degrees_of(src)
    claims = assembly.judge_claims(a, CLAIMED)
    rows, sharp = controls(src, a)
    per_claim = {c.name: sum(1 for r in rows if r["claim"] == c.name and r["rejected"]) for c in claims}
    summary = {
        "claims_hold": all(c.holds for c in claims), "claims": len(claims), "controls": len(rows),
        "controls_per_claim": per_claim, "controls_cover": all(v >= MIN_CONTROLS for v in per_claim.values()),
        "leaked": sum(1 for r in rows if not r["rejected"]),
        "sharpness_all": all(s["inside_tightening_still_holds"] for s in sharp),
    }
    out = {
        "theorem_A_hypotheses_certified": a.certified,
        "unproven": list(a.unproven), "missing": list(a.missing), "graph_matches_design": a.graph_matches,
        "m0_overlaps": list(a.m0_overlaps), "m0_min_separation": as_json(a.m0_gap) if a.m0_gap is not None else None,
        "collision_distance_lower_bound_delta": str(a.delta), "delta_float": float(a.delta) if a.delta else None,
        "size_upper_bound_R": str(a.rmax), "R_float": float(a.rmax) if a.rmax else None,
        # 귀환 시각 하한이 양수면 syzygy 사이 시간의 합이 발산해 흐름 궤도가 모든 t 에서 정의된다
        "return_time_lower_bound": str(a.tmin), "tmin_float": float(a.tmin) if a.tmin else None,
        "loops": list(a.loops), "rho_lower_bound": str(a.rho_lower), "rho_float": float(a.rho_lower),
        "entropy_lower_bound_ln_rho_ge": assembly.floor_decimal(assembly.ln_lower_bound(a.rho_lower)),
        "claimed": {k: str(v) for k, v in vars(CLAIMED).items()},
        "claims": [{"claim": c.name, "holds": c.holds, "margin": None if c.margin is None else as_json(c.margin)}
                   for c in claims],
        "claim_controls_summary": summary, "claim_controls": rows, "claim_controls_sharpness": sharp,
        "degrees": degrees, "degree_counts": {str(d): sum(1 for v in degrees.values() if v == d) for d in (-1, 1)},
    }
    print(json.dumps({k: v for k, v in out.items() if k not in ("degrees", "claim_controls")}, indent=1,
                     ensure_ascii=False))
    if not args.no_write:
        write_result(f"theorem_A{('_' + args.tag) if args.tag else ''}",
                     {"tag": "[CAP] 정리 A 가정·주장 판정, G6 주장별 전용 대조", **out},
                     script=Path(__file__).resolve(), params=vars(args), started=started)
    ok = (a.certified and summary["claims_hold"] and summary["controls_cover"] and not summary["leaked"]
          and summary["sharpness_all"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
