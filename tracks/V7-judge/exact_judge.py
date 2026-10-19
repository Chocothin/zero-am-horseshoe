"""V7 정확 판정: (1) CAPD 덮개 기록 314변의 a*·b*·(W)(F), (2) V5 stage2 저장 감싸기 28변의 a*·b*·(W)(F)와 포함 관계.

원 트랙 코드와 감사자 코드를 import 하지 않는다. 입력은 원자료 JSON(덮개 기록, T7 cones.json)과 V5 저장 JSON 이다.
V5 의 상은 외피 직접 감싸기와의 교집합을 쓰지 않고 P̂(0) + [DP̂(B)]·Q 만으로 다시 계산한다(더 보수적인 판정).
사용: python -I -B exact_judge.py [--out PATH] [--v5dir DIR ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import v7exact as X  # noqa: E402

REPO = HERE.parents[1]
R6 = REPO / "results" / "2026-10-15-T6-horseshoe"
R7 = REPO / "results" / "2026-10-16-T7-hyperbolic"
R5 = REPO / "results" / "2026-10-16-V5-indep"
MU_S = Fraction("0.8944")
MU_U = Fraction("1.1167")
FACES = [(j, s) for j in (0, 1) for s in (-1, 1)]


def rec_D(rec):
    return [[X.iv(rec["derivative"][i][j]) for j in range(4)] for i in range(4)]


def judge_capd(rec):
    D = rec_D(rec)
    c = [X.iv(v) for v in rec["center_image"]]
    det = rec["pieces_detail"]
    signs = rec["signs"]
    stored_box = [X.iv(v) for v in det["whole"][0]["image"]]
    stored_faces = {(j, s): [X.iv(v) for v in det[f"face_w{j}{s:+d}"][0]["image"]] for (j, s) in FACES}
    # 저장 조각이 상자 전체·면 전체인지
    box_piece = [X.iv(v) for v in det["whole"][0]["piece"]]
    pieces_ok = all(p == X.UNIT for p in box_piece) and len(det["whole"]) == 1
    for (j, s) in FACES:
        pc = det[f"face_w{j}{s:+d}"]
        pieces_ok &= len(pc) == 1 and [X.iv(v) for v in pc[0]["piece"]] == X.face_piece(j, s)
    mine_box = X.affine_image(c, D, [X.UNIT] * 4)
    mine_faces = {f: X.affine_image(c, D, X.face_piece(*f)) for f in FACES}
    images_equal = mine_box == stored_box and all(mine_faces[f] == stored_faces[f] for f in FACES)
    w, w_in, faces = X.cover_margins(mine_box, mine_faces, signs)
    a, b = X.gershgorin(D)
    return {"D": D, "c": c, "signs": signs, "pieces_full": pieces_ok, "images_equal_stored": images_equal,
            "W": w, "W_stable_inside": w_in, "faces": faces, "cover": min([w] + list(faces.values())),
            "a": a, "b": b, "degree": rec["degree"]}


def judge_v5(path, capd):
    r = json.loads(Path(path).read_text())
    e = r["edge"]
    D5 = [[X.iv(r["deriv_mine_exact"][i][j]) for j in range(4)] for i in range(4)]
    c5 = [X.iv(v) for v in r["center_image_exact"]]
    whole5 = [X.iv(v) for v in r["image_whole_exact"]]
    k = capd[e]
    # V5 대각 부호: 대각 구간이 0 을 품지 않으면 그 부호
    sg = []
    for j in (0, 1):
        lo, hi = D5[j][j]
        sg.append(1 if lo > 0 else (-1 if hi < 0 else 0))
    box = X.affine_image(c5, D5, [X.UNIT] * 4)
    faces_img = {f: X.affine_image(c5, D5, X.face_piece(*f)) for f in FACES}
    w, w_in, faces = X.cover_margins(box, faces_img, k["signs"])
    w_stored, _, _ = X.cover_margins(whole5, {}, k["signs"])
    a, b = X.gershgorin(D5)
    contained = sum(1 for i in range(4) for j in range(4) if X.contains(k["D"][i][j], D5[i][j]))
    center_in = all(X.contains(k["c"][i], c5[i]) for i in range(4))
    return {
        "edge": e,
        "signs_from_D5_diag": sg, "signs_record": k["signs"], "signs_agree": sg == list(k["signs"]),
        "cover_margin_mv": X.fstr(min([w] + list(faces.values())), 12, "down"),
        "W_mv": X.fstr(w, 12, "down"), "W_stored_whole": X.fstr(w_stored, 12, "down"),
        "faces_mv": {f"w{j}{s:+d}": X.fstr(v, 12, "down") for (j, s), v in faces.items()},
        "cover_proven_mv": bool(min([w] + list(faces.values())) > 0),
        "a_star": X.fstr(a, 12, "up"), "b_star": X.fstr(b, 12, "down"),
        "a_star_equals_v5": a == Fraction(r["a_star_exact"]), "b_star_equals_v5": b == Fraction(r["b_star_exact"]),
        "mu_s_margin": X.fstr(MU_S - a, 12, "down"), "mu_u_margin": X.fstr(b - MU_U, 12, "down"),
        "cone_proven": bool(a < MU_S and b > MU_U),
        "a_star_exact": str(a), "b_star_exact": str(b), "cover_margin_mv_exact": str(min([w] + list(faces.values()))),
        "capd_D_contains_v5_D": contained, "capd_center_contains_v5_center": center_in,
        "v5_reported": {"covering_margin": r["covering_margin"], "a_star": r["a_star"], "b_star": r["b_star"],
                        "DB_contains_mine_count": r["DB_contains_mine_count"]},
        "_a": a, "_b": b, "_cov": min([w] + list(faces.values())),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results" / "2026-10-16-V7-judge" / "exact_judge.json"))
    ap.add_argument("--v5dir", action="append", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    design = json.loads((R6 / "graph_design.json").read_text())
    edges = [f"{e['source']}>{e['target']}" for e in design["edges"]]
    capd = {}
    for e in edges:
        s, t = e.split(">")
        capd[e] = judge_capd(json.loads((R6 / "coverings" / f"{s}__{t}.json").read_text()))
    cones = {row["edge"]: row for row in json.loads((R7 / "cones.json").read_text())["rows"]}
    t7_equal_a = sum(1 for e in edges if capd[e]["a"] == Fraction(cones[e]["lower_a"]))
    t7_equal_b = sum(1 for e in edges if capd[e]["b"] == Fraction(cones[e]["upper_b"]))
    amax_e = max(edges, key=lambda e: capd[e]["a"])
    bmin_e = min(edges, key=lambda e: capd[e]["b"])
    cmin_e = min(edges, key=lambda e: capd[e]["cover"])
    ranked_cover = sorted(edges, key=lambda e: capd[e]["cover"])[:10]
    ranked_cone = sorted(edges, key=lambda e: min(MU_S - capd[e]["a"], capd[e]["b"] - MU_U))[:10]
    out = {
        "capd_records": {
            "edges": len(edges),
            "pieces_full_all": all(capd[e]["pieces_full"] for e in edges),
            "images_equal_stored_all": all(capd[e]["images_equal_stored"] for e in edges),
            "cover_proven_all": all(capd[e]["cover"] > 0 for e in edges),
            "W_by_stable_inside_all": all(capd[e]["W_stable_inside"] > 0 for e in edges),
            "degrees": {str(d): sum(1 for e in edges if capd[e]["degree"] == d) for d in (1, -1)},
            "degree_equals_signs_product": all(capd[e]["degree"] == capd[e]["signs"][0] * capd[e]["signs"][1]
                                               for e in edges),
            "min_cover": {"edge": cmin_e, "value_down": X.fstr(capd[cmin_e]["cover"], 12, "down")},
            "cover_rank10": [[e, X.fstr(capd[e]["cover"], 9, "down")] for e in ranked_cover],
            "max_a": {"edge": amax_e, "value_up": X.fstr(capd[amax_e]["a"], 15, "up"),
                      "mu_s_margin_down": X.fstr(MU_S - capd[amax_e]["a"], 15, "down")},
            "min_b": {"edge": bmin_e, "value_down": X.fstr(capd[bmin_e]["b"], 15, "down"),
                      "mu_u_margin_down": X.fstr(capd[bmin_e]["b"] - MU_U, 15, "down")},
            "cone_proven_all": all(capd[e]["a"] < MU_S and capd[e]["b"] > MU_U for e in edges),
            "min_g1_down": X.fstr(min(min(capd[e]["b"] - 1, 1 - capd[e]["a"]) for e in edges), 15, "down"),
            "cone_rank10": [[e, X.fstr(MU_S - capd[e]["a"], 9, "down"), X.fstr(capd[e]["b"] - MU_U, 9, "down")]
                            for e in ranked_cone],
            "t7_cones_equal": {"a": t7_equal_a, "b": t7_equal_b},
        },
    }
    v5dirs = args.v5dir or [str(R5 / "stage2")]
    rows = []
    for d in v5dirs:
        for p in sorted(Path(d).glob("*.json")):
            rows.append(judge_v5(p, capd))
    if rows:
        out["v5_stage2"] = {
            "dirs": [X.relpath(d, REPO) for d in v5dirs],
            "edges": len(rows),
            "all_cover_proven_mv": all(r["cover_proven_mv"] for r in rows),
            "all_cone_proven": all(r["cone_proven"] for r in rows),
            "all_signs_agree": all(r["signs_agree"] for r in rows),
            "all_ab_equal_v5": all(r["a_star_equals_v5"] and r["b_star_equals_v5"] for r in rows),
            "capd_D_contains_v5_D_all16": all(r["capd_D_contains_v5_D"] == 16 for r in rows),
            "capd_center_contains_v5_center_all": all(r["capd_center_contains_v5_center"] for r in rows),
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
    s = {k: v for k, v in out["capd_records"].items() if k not in ("cover_rank10", "cone_rank10")}
    print(json.dumps(s, ensure_ascii=False, indent=1))
    if rows:
        print(json.dumps({k: v for k, v in out["v5_stage2"].items() if k != "rows"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
