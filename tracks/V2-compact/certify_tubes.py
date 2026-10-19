"""[재계산·엄밀] V2-compact: 변마다 상자 전체의 흐름 관 r_ij 상·하한, 귀환 창, 첫 통과 구조를 독립 인증한다.

엄밀 계산은 rigorous_tube.fly(python-flint arb)다. 계획용 중심 귀환 시각만 nbody_mp(mpmath, 비엄밀)로 구하고,
그 값은 단계 배치에만 쓰므로 틀려도 거짓 인증이 나오지 않고 실패로 끝난다.
T6 기록(CAPD 값)은 비교용으로만 읽는다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nbody_mp as nb  # noqa: E402
import rigorous_tube as rt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
R6 = REPO / "results" / "2026-10-15-T6-horseshoe"
OUT = REPO / "results" / "2026-10-16-V2-compact"
CLAIMED = {"delta": Fraction("0.217857228763"), "R": Fraction("7.419986043806"), "tmin": Fraction("0.276707990268")}


def q(tok: str) -> Fraction:
    t = tok.strip()
    if t.lower().lstrip("+-").startswith("0x"):
        return Fraction(float.fromhex(t))
    return Fraction(t)


def exact(x) -> Fraction:
    m, e = x.man_exp()
    m, e = int(m), int(e)
    return Fraction(m) * (Fraction(2) ** e)


def hset(vertex):
    c = [Fraction(float.fromhex(h)) for h in vertex["center_hex"]]
    f = [[Fraction(float.fromhex(h)) for h in row] for row in vertex["frame_hex"]]
    r = [Fraction(float(x)) for x in vertex["radii"]]
    return c, [[f[i][j] * r[j] for j in range(4)] for i in range(4)]


def plan_tau(c, m) -> float:
    """중심 점(w=0)의 첫 syzygy 시각(비엄밀, 단계 배치용)."""
    with mp.workdps(24):
        state, _ = nb.lift_sigma_plus([nb.to_mpf(x) for x in c])
        qq, vv = nb.reduced_to_inertial(state)
        return float(nb.fly_to_syzygy(qq, vv, order=22)["tau"])


def certify_edge(edge: str, design: dict, cover_dir: Path, order: int, kappa: float, sub: int) -> dict:
    s_name, t_name = edge.split(">")
    c, m = hset(design["vertices"][s_name])
    started = time.time()
    tau_plan = plan_tau(c, m)
    row = {"edge": edge, "tau_plan": tau_plan}
    try:
        res = rt.fly(c, m, tau_plan, order=order, kappa=kappa, sub=sub)
    except (rt.FlightFailure, ArithmeticError) as exc:
        row.update({"certified": False, "reason": str(exc), "seconds": round(time.time() - started, 1)})
        return row
    names = ("r12", "r13", "r23")
    r_lo = {n: res["r2_lower"][k].sqrt().lower() for k, n in enumerate(names)}
    r_hi = {n: res["r2_upper"][k].sqrt().upper() for k, n in enumerate(names)}
    lo = {n: exact(r_lo[n]) for n in names}
    hi = {n: exact(r_hi[n]) for n in names}
    w0, w1 = res["window"]
    rec = json.loads((cover_dir / f"{s_name}__{t_name}.json").read_text())
    whole = rec["pieces_detail"]["whole"][0]
    capd_lo = {n: q(whole["tube"][n]["lo"]) for n in names}
    capd_hi = {n: q(whole["tube"][n]["hi"]) for n in names}
    capd_win = (q(whole["return_time"]["lo"]), q(whole["return_time"]["hi"]))
    row.update({
        "certified": True, "steps": res["steps"], "seconds": round(time.time() - started, 1),
        "disc_lower": str(exact(res["disc"].lower())),
        "r_lower": {n: str(lo[n]) for n in names}, "r_upper": {n: str(hi[n]) for n in names},
        "r_lower_float": {n: float(lo[n]) for n in names}, "r_upper_float": {n: float(hi[n]) for n in names},
        "window": [str(w0), str(w1)], "window_float": [float(w0), float(w1)],
        "min_lower": str(min(lo.values())), "max_upper": str(max(hi.values())),
        # CAPD 값과의 관계(비교용): 내 하한이 CAPD 하한보다 크면 양수
        "vs_capd_lower": {n: float(lo[n] - capd_lo[n]) for n in names},
        "vs_capd_upper": {n: float(capd_hi[n] - hi[n]) for n in names},
        "window_vs_capd": [float(w0 - capd_win[0]), float(capd_win[1] - w1)],
        "windows_overlap": not (w1 < capd_win[0] or capd_win[1] < w0),
    })
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default="all")
    ap.add_argument("--shard", default="0/1", help="i/n: 변 목록의 i 번째 몫만")
    ap.add_argument("--order", type=int, default=18)
    ap.add_argument("--kappa", type=float, default=0.08)
    ap.add_argument("--sub", type=int, default=32)
    ap.add_argument("--cover-dir", default=str(R6 / "coverings"))
    ap.add_argument("--out", default=str(OUT / "rigorous_tubes.json"))
    args = ap.parse_args()
    started = time.time()
    design = json.loads((R6 / "graph_design.json").read_text())
    all_edges = [f"{e['source']}>{e['target']}" for e in design["edges"]]
    edges = all_edges if args.edges == "all" else [e for e in args.edges.split(",") if e]
    i, n = (int(x) for x in args.shard.split("/"))
    edges = edges[i::n]
    rows = []
    for e in edges:
        row = certify_edge(e, design, Path(args.cover_dir), args.order, args.kappa, args.sub)
        rows.append(row)
        print(json.dumps({k: row[k] for k in row if k in ("edge", "certified", "reason", "steps", "seconds",
                                                          "min_lower", "max_upper", "window_float")}),
              flush=True)
    good = [r for r in rows if r["certified"]]
    summary = {"edges": len(rows), "certified": len(good)}
    if good:
        dmin = min(good, key=lambda r: Fraction(r["min_lower"]))
        rmax = max(good, key=lambda r: Fraction(r["max_upper"]))
        tmin = min(good, key=lambda r: Fraction(r["window"][0]))
        tmax = max(good, key=lambda r: Fraction(r["window"][1]))
        summary.update({
            "delta_mine": dmin["min_lower"], "delta_edge": dmin["edge"],
            "R_mine": rmax["max_upper"], "R_edge": rmax["edge"],
            "tmin_mine": tmin["window"][0], "tmin_edge": tmin["edge"],
            "tmax_mine": tmax["window"][1], "tmax_edge": tmax["edge"],
            "delta_claim_supported": Fraction(dmin["min_lower"]) >= CLAIMED["delta"],
            "R_claim_supported": Fraction(rmax["max_upper"]) <= CLAIMED["R"],
            "tmin_claim_supported": Fraction(tmin["window"][0]) >= CLAIMED["tmin"],
            "all_windows_overlap_capd": all(r["windows_overlap"] for r in good),
        })
    payload = {"tag": "[재계산·엄밀] python-flint arb 독립 흐름 관·귀환 창 인증", "prec_bits": rt.ctx.prec,
               "order": args.order, "kappa": args.kappa, "sub": args.sub, "shard": args.shard,
               "summary": summary, "rows": rows, "seconds": round(time.time() - started, 1)}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["certified"] == summary["edges"] else 1


if __name__ == "__main__":
    sys.exit(main())
