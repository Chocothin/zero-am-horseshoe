"""2단계: 변마다 h-set 전체의 C¹ 감싸기를 제 엔진으로 만들고, 덮개 (W)(F)와 Gershgorin 원뿔 부등식을 제 감싸기만으로
다시 판정한다. CAPD D(B) 와 성분별로 겹침(서로소면 모순)과 폭 비를 잰다.
사용: python stage2.py [변 ...] (없으면 selected_edges.json 의 selected)
"""
from __future__ import annotations

import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flint import arb, ctx  # noqa: E402

from v5 import boxmap, compare, data, margins  # noqa: E402

RES = data.REPO / "results" / "2026-10-16-V5-indep"
OUTDIR = Path(os.environ.get("V5_STAGE2_OUT", RES / "stage2"))
ONE = Fraction(1)


def frac_iv(a: arb):
    return compare.bounds(a)


def verdicts(res, signs):
    """덮개 (W)(F) 와 원뿔 a*·b* 를 정확 유리수로 판정한다."""
    unit = [arb(0, 1)] * 4
    whole = [frac_iv(v) for v in boxmap.image_of(res, unit)]
    out = max(max(whole[j][0] - ONE, -ONE - whole[j][1]) for j in (0, 1))
    inside = min(min(whole[k][0] + ONE, ONE - whole[k][1]) for k in (2, 3))
    items = {"whole": max(out, inside)}
    for j in (0, 1):
        for s in (-1, 1):
            piece = [arb(s) if i == j else arb(0, 1) for i in range(4)]
            img = [frac_iv(v) for v in boxmap.image_of(res, piece)]
            items[f"face_w{j}{s:+d}"] = img[j][0] - ONE if signs[j] * s > 0 else -ONE - img[j][1]
    cov = min(items.values())
    Dm = [[frac_iv(res.deriv[i][j]) for j in range(4)] for i in range(4)]
    a, b, sdiag, R = margins.gershgorin(Dm)
    return {"covering_margin": float(cov), "covering_items": {k: float(v) for k, v in items.items()},
            "covering_proven": bool(cov > 0), "a_star": a, "b_star": b,
            "mu_s_margin": margins.MU_S - a, "mu_u_margin": b - margins.MU_U,
            "g1": min(b - 1, 1 - a), "cone_proven": bool(a < margins.MU_S and b > margins.MU_U)}


def run_edge(e: str, g8: dict) -> dict:
    rec = data.record(e)
    res = boxmap.enclose_box(e)
    ctx.prec = 128
    signs_capd = rec["signs"]
    signs_mine = [1 if res.center_deriv[j][j] > 0 else -1 for j in (0, 1)]
    v = verdicts(res, signs_mine)
    D = [[data.interval(rec["derivative"][i][j]) for j in range(4)] for i in range(4)]
    DG8 = [[data.interval(g8[e][i][j]) for j in range(4)] for i in range(4)]
    a_c, b_c, _, _ = margins.gershgorin(D)
    comp, comp_g8, ratio, ratio_hull, disjoint = [], [], [], [], []
    for i in range(4):
        for j in range(4):
            mine = res.deriv[i][j]
            ml, mh = frac_iv(mine)
            for tag, ref, store in (("main", D, comp), ("g8", DG8, comp_g8)):
                lo, hi = ref[i][j]
                rel = compare.relation(mine, lo, hi)
                store.append(rel)
                if rel["disjoint"]:
                    disjoint.append(f"{tag} D[{i}][{j}]")
            ratio.append(float((D[i][j][1] - D[i][j][0]) / (mh - ml)))
            hl, hh = frac_iv(res.hull_deriv[i][j])
            ratio_hull.append(float((D[i][j][1] - D[i][j][0]) / (hh - hl)))
    whole = rec["pieces_detail"]["whole"][0]
    rt = data.interval(whole["return_time"])
    rel_tau = compare.relation(res.tau_hull, *rt)
    if rel_tau["disjoint"]:
        disjoint.append("return_time")
    bo = data.interval(whole["branch_out"])
    rel_bo = compare.relation(res.branch_out_hull, *bo)
    if rel_bo["disjoint"]:
        disjoint.append("branch_out")
    di = data.interval(whole["discriminant"])
    rel_disc = compare.relation(res.disc_hull, *di)
    if rel_disc["disjoint"]:
        disjoint.append("discriminant")
    tube_capd = {k: [float(x) for x in data.interval(whole["tube"][k])] for k in ("r12", "r13", "r23")}
    out = {
        "edge": e,
        "signs_capd": signs_capd, "signs_mine": signs_mine, "signs_agree": signs_capd == signs_mine,
        "covering_proven": v["covering_proven"], "covering_margin": v["covering_margin"],
        "covering_items": v["covering_items"],
        "cone_proven": v["cone_proven"],
        "a_star": f"{float(v['a_star']):.12f}", "b_star": f"{float(v['b_star']):.12f}",
        "mu_s_margin": f"{float(v['mu_s_margin']):.6e}", "mu_u_margin": f"{float(v['mu_u_margin']):.6e}",
        "g1": f"{float(v['g1']):.8f}",
        "a_star_exact": str(v["a_star"]), "b_star_exact": str(v["b_star"]),
        "capd_a_star": f"{float(a_c):.12f}", "capd_b_star": f"{float(b_c):.12f}",
        "capd_mu_s_margin": f"{float(margins.MU_S - a_c):.6e}", "capd_mu_u_margin": f"{float(b_c - margins.MU_U):.6e}",
        "width_ratio_capd_over_mine": {"min": min(ratio), "median": sorted(ratio)[8], "max": max(ratio)},
        "width_ratio_capd_over_mine_hull_only": {"min": min(ratio_hull), "median": sorted(ratio_hull)[8],
                                                 "max": max(ratio_hull)},
        "disjoint": disjoint,
        "deriv_vs_DB_contained": all(c["contained"] for c in comp),
        "DB_contains_mine_count": sum(1 for c in comp if c["contained"]),
        "deriv_mine": [[compare.fmt(res.deriv[i][j], 20) for j in range(4)] for i in range(4)],
        "deriv_mine_exact": [[[str(x) for x in frac_iv(res.deriv[i][j])] for j in range(4)] for i in range(4)],
        "center_image_exact": [[str(x) for x in frac_iv(v)] for v in res.center_image],
        "image_whole_exact": [[str(x) for x in frac_iv(v)] for v in boxmap.image_of(res, [arb(0, 1)] * 4)],
        "d2_rad_sum_max": max(float(sum((abs(res.d2[i][j][b]) for b in range(4)), arb(0)).upper())
                              for i in range(4) for j in range(4)),
        "d2_max_abs": max(float(abs(res.d2[i][j][b]).upper()) for i in range(4) for j in range(4) for b in range(4)),
        "tau_hull": compare.fmt(res.tau_hull), "tau_vs_capd": rel_tau,
        "branch_out_vs_capd": rel_bo, "disc_vs_capd": rel_disc,
        "tube_mine_outer": {"lo": [float(x) for x in res.tube_lo], "hi": [float(x) for x in res.tube_hi]},
        "tube_capd": tube_capd,
        "seconds_center": round(res.seconds_center, 1), "seconds_hull": round(res.seconds_hull, 1),
        "steps_hull": res.steps_hull, "halvings_hull": res.halvings_hull, "max_rem_hull": res.max_rem_hull,
    }
    return out


def main(argv):
    argv = [e for a in argv for e in a.split()]          # zsh 는 따옴표 없는 변수를 단어로 나누지 않는다
    edges = argv or json.loads((RES / "selected_edges.json").read_text())["selected"]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    g8 = {r["edge"]: r["derivative"] for r in json.loads((data.R7 / "g8_derivatives.json").read_text())["rows"]}
    for e in edges:
        t0 = time.time()
        out = run_edge(e, g8)
        (OUTDIR / f"{e.replace('>', '__')}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
        print(f"{e}: {time.time() - t0:.1f}s cover={out['covering_proven']}({out['covering_margin']:.4f}) "
              f"cone={out['cone_proven']} a*={out['a_star']} b*={out['b_star']} "
              f"(capd a*={out['capd_a_star']} b*={out['capd_b_star']}) "
              f"width capd/mine median={out['width_ratio_capd_over_mine']['median']:.3g} "
              f"disjoint={out['disjoint']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
