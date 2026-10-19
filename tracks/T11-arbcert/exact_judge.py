"""T11 정확 판정: stage2 저장 감싸기(314변)의 a*·b*·(W)(F). 상은 P̂(0) + [DP̂(B)]·Q 만으로 다시 계산한다(외피 교집합 없음).

T11 사본: 감사 V7 판정의 CAPD 기록 재판정(judge_capd)과 CAPD 포함 비교를 뺐다(교차 대조는 t11/g8.py).
덮개 부호 s_j 는 CAPD 기록 대신 graph_design 의 linear 대각 부호에서 받는다(T6 설계가 부호를 정한 규칙과 같다).
사용: python -I -B exact_judge.py [--out PATH] [--v5dir DIR ...]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import v7exact as X  # noqa: E402

REPO = HERE.parents[1]
R6 = REPO / "results" / "2026-10-15-T6-horseshoe"
R11 = REPO / "results" / "2026-10-17-T11-arbcert"
MU_S = Fraction("0.8944")
MU_U = Fraction("1.1167")
FACES = [(j, s) for j in (0, 1) for s in (-1, 1)]


def design_signs(design) -> dict:
    """변마다 (s_0, s_1) = sign(linear[j][j]). 0·비유한 대각은 부호가 없어 덮개 판정을 할 수 없으므로 거부한다."""
    out = {}
    for e in design["edges"]:
        diag = [e["linear"][j][j] for j in (0, 1)]
        if not all(math.isfinite(v) and v != 0 for v in diag):
            raise ValueError(f"design edge {e['source']}>{e['target']}: unstable diagonal without sign {diag}")
        out[f"{e['source']}>{e['target']}"] = [1 if v > 0 else -1 for v in diag]
    return out


def judge_v5(path, signs):
    r = json.loads(Path(path).read_text())
    e = r["edge"]
    D5 = [[X.iv(r["deriv_mine_exact"][i][j]) for j in range(4)] for i in range(4)]
    c5 = [X.iv(v) for v in r["center_image_exact"]]
    whole5 = [X.iv(v) for v in r["image_whole_exact"]]
    signs_e = signs[e]
    # V5 대각 부호: 대각 구간이 0 을 품지 않으면 그 부호
    sg = []
    for j in (0, 1):
        lo, hi = D5[j][j]
        sg.append(1 if lo > 0 else (-1 if hi < 0 else 0))
    box = X.affine_image(c5, D5, [X.UNIT] * 4)
    faces_img = {f: X.affine_image(c5, D5, X.face_piece(*f)) for f in FACES}
    w, w_in, faces = X.cover_margins(box, faces_img, signs_e)
    w_stored, _, _ = X.cover_margins(whole5, {}, signs_e)
    a, b = X.gershgorin(D5)
    return {
        "edge": e,
        "signs_from_D5_diag": sg, "signs_design": signs_e, "signs_agree": sg == list(signs_e),
        "cover_margin_mv": X.fstr(min([w] + list(faces.values())), 12, "down"),
        "W_mv": X.fstr(w, 12, "down"), "W_stored_whole": X.fstr(w_stored, 12, "down"),
        "faces_mv": {f"w{j}{s:+d}": X.fstr(v, 12, "down") for (j, s), v in faces.items()},
        "cover_proven_mv": bool(min([w] + list(faces.values())) > 0),
        "a_star": X.fstr(a, 12, "up"), "b_star": X.fstr(b, 12, "down"),
        "a_star_equals_v5": a == Fraction(r["a_star_exact"]), "b_star_equals_v5": b == Fraction(r["b_star_exact"]),
        "mu_s_margin": X.fstr(MU_S - a, 12, "down"), "mu_u_margin": X.fstr(b - MU_U, 12, "down"),
        "cone_proven": bool(a < MU_S and b > MU_U),
        "a_star_exact": str(a), "b_star_exact": str(b), "cover_margin_mv_exact": str(min([w] + list(faces.values()))),
        "v5_reported": {"covering_margin": r["covering_margin"], "a_star": r["a_star"], "b_star": r["b_star"]},
        "_a": a, "_b": b, "_cov": min([w] + list(faces.values())),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R11 / "exact_judge.json"))
    ap.add_argument("--v5dir", action="append", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    design = json.loads((R6 / "graph_design.json").read_text())
    signs = design_signs(design)
    out = {}
    v5dirs = args.v5dir or [str(R11 / "stage2")]
    rows = []
    for d in v5dirs:
        for p in sorted(Path(d).glob("*.json")):
            rows.append(judge_v5(p, signs))
    if rows:
        out["v5_stage2"] = {
            "dirs": [X.relpath(d, REPO) for d in v5dirs],
            "edges": len(rows),
            "all_cover_proven_mv": all(r["cover_proven_mv"] for r in rows),
            "all_cone_proven": all(r["cone_proven"] for r in rows),
            "all_signs_agree": all(r["signs_agree"] for r in rows),
            "all_ab_equal_v5": all(r["a_star_equals_v5"] and r["b_star_equals_v5"] for r in rows),
            "min_cover_mv": min(((r["edge"], r["_cov"]) for r in rows), key=lambda t: t[1])[0],
            "min_cover_mv_value_down": X.fstr(min(r["_cov"] for r in rows), 12, "down"),
            "max_a_edge": max(rows, key=lambda r: r["_a"])["edge"],
            "max_a_up": X.fstr(max(r["_a"] for r in rows), 12, "up"),
            "min_b_edge": min(rows, key=lambda r: r["_b"])["edge"],
            "min_b_down": X.fstr(min(r["_b"] for r in rows), 12, "down"),
            "min_mu_s_margin_down": X.fstr(min(MU_S - r["_a"] for r in rows), 12, "down"),
            "min_mu_u_margin_down": X.fstr(min(r["_b"] - MU_U for r in rows), 12, "down"),
            "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        }
    out["seconds"] = round(time.time() - t0, 2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    if rows:
        print(json.dumps({k: v for k, v in out["v5_stage2"].items() if k != "rows"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
