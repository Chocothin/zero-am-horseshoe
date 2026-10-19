"""2단계: 변마다 h-set 전체의 C¹ 감싸기를 제 엔진으로 만들고, 덮개 (W)(F)와 Gershgorin 원뿔 부등식을 제 감싸기만으로
다시 판정한다. T11 사본: CAPD 기록을 읽지 않는다(CAPD 와의 교차 대조는 t11/g8.py).
사용: python stage2.py [변 ...] (없으면 graph_design 의 314변)
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

RES = data.REPO / "results" / "2026-10-17-T11-arbcert"
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


def run_edge(e: str) -> dict:
    res = boxmap.enclose_box(e)
    ctx.prec = 128
    signs_mine = [1 if res.center_deriv[j][j] > 0 else -1 for j in (0, 1)]
    v = verdicts(res, signs_mine)
    out = {
        "edge": e,
        "signs_mine": signs_mine,
        "covering_proven": v["covering_proven"], "covering_margin": v["covering_margin"],
        "covering_items": v["covering_items"],
        "cone_proven": v["cone_proven"],
        "a_star": f"{float(v['a_star']):.12f}", "b_star": f"{float(v['b_star']):.12f}",
        "mu_s_margin": f"{float(v['mu_s_margin']):.6e}", "mu_u_margin": f"{float(v['mu_u_margin']):.6e}",
        "g1": f"{float(v['g1']):.8f}",
        "a_star_exact": str(v["a_star"]), "b_star_exact": str(v["b_star"]),
        "deriv_mine": [[compare.fmt(res.deriv[i][j], 20) for j in range(4)] for i in range(4)],
        "deriv_mine_exact": [[[str(x) for x in frac_iv(res.deriv[i][j])] for j in range(4)] for i in range(4)],
        "center_image_exact": [[str(x) for x in frac_iv(v)] for v in res.center_image],
        "image_whole_exact": [[str(x) for x in frac_iv(v)] for v in boxmap.image_of(res, [arb(0, 1)] * 4)],
        "d2_rad_sum_max": max(float(sum((abs(res.d2[i][j][b]) for b in range(4)), arb(0)).upper())
                              for i in range(4) for j in range(4)),
        "d2_max_abs": max(float(abs(res.d2[i][j][b]).upper()) for i in range(4) for j in range(4) for b in range(4)),
        "tau_hull": compare.fmt(res.tau_hull),
        "tube_mine_outer": {"lo": [float(x) for x in res.tube_lo], "hi": [float(x) for x in res.tube_hi]},
        # T11: CAPD 교차 대조(t11/g8.py)와 첫 통과 구조 판정이 읽는 정확 끝점
        "center_deriv_exact": [[[str(x) for x in frac_iv(res.center_deriv[i][j])] for j in range(4)] for i in range(4)],
        "hull_image_exact": [[str(x) for x in frac_iv(v)] for v in res.hull_image],
        "hull_deriv_exact": [[[str(x) for x in frac_iv(res.hull_deriv[i][j])] for j in range(4)] for i in range(4)],
        "tau_hull_exact": [str(x) for x in frac_iv(res.tau_hull)],
        "branch_out_hull_exact": [str(x) for x in frac_iv(res.branch_out_hull)],
        "disc_hull_exact": [str(x) for x in frac_iv(res.disc_hull)],
        "tube_outer_exact": {"lo": [str(frac_iv(x)[0]) for x in res.tube_lo],
                             "hi": [str(frac_iv(x)[1]) for x in res.tube_hi]},
        "seconds_center": round(res.seconds_center, 1), "seconds_hull": round(res.seconds_hull, 1),
        "steps_hull": res.steps_hull, "halvings_hull": res.halvings_hull, "max_rem_hull": res.max_rem_hull,
    }
    return out


def main(argv):
    argv = [e for a in argv for e in a.split()]          # zsh 는 따옴표 없는 변수를 단어로 나누지 않는다
    edges = argv or data.edges_from_design(data.design())
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for e in edges:
        t0 = time.time()
        out = run_edge(e)
        (OUTDIR / f"{e.replace('>', '__')}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
        print(f"{e}: {time.time() - t0:.1f}s cover={out['covering_proven']}({out['covering_margin']:.4f}) "
              f"cone={out['cone_proven']} a*={out['a_star']} b*={out['b_star']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
