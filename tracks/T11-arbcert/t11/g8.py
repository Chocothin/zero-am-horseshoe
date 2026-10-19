"""G8 차등 재검증: CAPD 덮개 기록(독립 구현)과 T11 감싸기의 314변 교차 대조. 인증 입력이 아니라 대조일 뿐이다.

- 포함: CAPD center_image ⊇ T11 중심 상(4성분), CAPD derivative D(B) ⊇ T11 D(B)(16성분)
- 모순 검사: 같은 집합을 감싸는 두 감싸기는 서로소일 수 없다. 귀환 창(CAPD·V2·V5), 흐름 관 r_ij 범위(CAPD·V2·V5),
  판별식·도착 근(CAPD·V5)이 서로소이면 둘 중 하나가 틀렸다는 뜻이라 실패로 끝낸다.
- 덮개 부호: T11 이 설계 linear 대각에서 받은 부호가 CAPD 기록의 signs 와 같고, 차수 집계가 기록의 degree 와 같은지 본다.
사용: t11cli.py g8 [--capd-dir results/2026-10-15-T6-horseshoe/coverings] [--out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import exact_judge as EJ
import v7exact as X

from . import design as dz
from .paths import BOXES, REPO, RESULTS, TUBES, edge_stem

DEFAULT_CAPD = REPO / "results" / "2026-10-15-T6-horseshoe" / "coverings"
PAIRS = ("r12", "r13", "r23")
FACES = [(j, s) for j in (0, 1) for s in (-1, 1)]
MU_S, MU_U = Fraction("0.8944"), Fraction("1.1167")


def disjoint(a, b) -> bool:
    return a[1] < b[0] or b[1] < a[0]


def compare_edge(e: str, capd: dict, box: dict, tube: dict, signs: list) -> dict:
    whole = capd["pieces_detail"]["whole"][0]
    c_capd = [X.iv(v) for v in capd["center_image"]]
    D_capd = [[X.iv(capd["derivative"][i][j]) for j in range(4)] for i in range(4)]
    c_t11 = [X.iv(v) for v in box["center_image_exact"]]
    D_t11 = [[X.iv(box["deriv_mine_exact"][i][j]) for j in range(4)] for i in range(4)]
    ratios = sorted((D_capd[i][j][1] - D_capd[i][j][0]) / (D_t11[i][j][1] - D_t11[i][j][0])
                    for i in range(4) for j in range(4))
    contradictions = []
    pairs = {"return_time capd~V2": (X.iv(whole["return_time"]), tuple(Fraction(x) for x in tube["window"])),
             "return_time capd~V5": (X.iv(whole["return_time"]), X.iv(box["tau_hull_exact"])),
             "discriminant capd~V5": (X.iv(whole["discriminant"]), X.iv(box["disc_hull_exact"])),
             "branch_out capd~V5": (X.iv(whole["branch_out"]), X.iv(box["branch_out_hull_exact"]))}
    for k, p in enumerate(PAIRS):
        cap = X.iv(whole["tube"][p])
        pairs[f"tube {p} capd~V2"] = (cap, (Fraction(tube["r_lower"][p]), Fraction(tube["r_upper"][p])))
        pairs[f"tube {p} capd~V5"] = (cap, (X.parse(box["tube_outer_exact"]["lo"][k]),
                                            X.parse(box["tube_outer_exact"]["hi"][k])))
    for i in range(4):
        pairs[f"center_image[{i}] capd~T11"] = (c_capd[i], c_t11[i])
        for j in range(4):
            pairs[f"derivative[{i}][{j}] capd~T11"] = (D_capd[i][j], D_t11[i][j])
    for name, (a, b) in pairs.items():
        if disjoint(a, b):
            contradictions.append(name)
    # 여유 비교용: CAPD 감싸기에 같은 판정 함수를 적용한 값(교차 대조 정보일 뿐 T11 판정에는 쓰지 않는다)
    box_img = X.affine_image(c_capd, D_capd, [X.UNIT] * 4)
    faces = {f: X.affine_image(c_capd, D_capd, X.face_piece(*f)) for f in FACES}
    w, _, fm = X.cover_margins(box_img, faces, signs)
    a_c, b_c = X.gershgorin(D_capd)
    return {
        "edge": e,
        "capd_cover_margin": str(min([w] + list(fm.values()))), "capd_a_star": str(a_c), "capd_b_star": str(b_c),
        "capd_r_lower": str(min(X.iv(whole["tube"][p])[0] for p in PAIRS)),
        "capd_r_upper": str(max(X.iv(whole["tube"][p])[1] for p in PAIRS)),
        "capd_window": [str(x) for x in X.iv(whole["return_time"])],
        "center_contained": sum(1 for i in range(4) if X.contains(c_capd[i], c_t11[i])),
        "derivative_contained": sum(1 for i in range(4) for j in range(4) if X.contains(D_capd[i][j], D_t11[i][j])),
        "width_ratio_capd_over_t11": {"min": float(ratios[0]), "median": float(ratios[8]), "max": float(ratios[-1])},
        "window_v2_inside_capd": X.contains(X.iv(whole["return_time"]), tuple(Fraction(x) for x in tube["window"])),
        "signs_design": signs, "signs_capd": list(capd["signs"]), "signs_agree": list(capd["signs"]) == list(signs),
        "degree_t11": signs[0] * signs[1], "degree_capd": capd["degree"],
        "contradictions": contradictions,
    }


def capd_extremes(rows: list) -> dict:
    """CAPD 감싸기로 잰 변 수준 최솟값·최댓값과 그 변(T11 여유와 비교하는 표용)."""
    if not rows:
        return {}

    def ext(key, best, conv=Fraction):
        r = best(rows, key=lambda r: conv(r[key]))
        return {"exact": r[key], "edge": r["edge"]}

    out = {"cover_margin_min": ext("capd_cover_margin", min), "a_star_max": ext("capd_a_star", max),
           "b_star_min": ext("capd_b_star", min), "r_lower_min": ext("capd_r_lower", min),
           "r_upper_max": ext("capd_r_upper", max)}
    lo = min(rows, key=lambda r: Fraction(r["capd_window"][0]))
    hi = max(rows, key=lambda r: Fraction(r["capd_window"][1]))
    out["window_lower_min"] = {"exact": lo["capd_window"][0], "edge": lo["edge"]}
    out["window_upper_max"] = {"exact": hi["capd_window"][1], "edge": hi["edge"]}
    g = min(rows, key=lambda r: min(Fraction(r["capd_b_star"]) - 1, 1 - Fraction(r["capd_a_star"])))
    out["g1_min"] = {"exact": str(min(Fraction(g["capd_b_star"]) - 1, 1 - Fraction(g["capd_a_star"]))), "edge": g["edge"]}
    for v in out.values():
        v["float"] = float(Fraction(v["exact"]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capd-dir", default=str(DEFAULT_CAPD))
    ap.add_argument("--tubes", default=str(TUBES))
    ap.add_argument("--boxes", default=str(BOXES))
    ap.add_argument("--out", default=str(RESULTS / "g8_capd_crosscheck.json"))
    args = ap.parse_args(argv)
    started = time.time()
    capd_dir = Path(args.capd_dir)
    rows, missing = [], []
    signs = EJ.design_signs(dz.design())
    digest = hashlib.sha256()
    for e in dz.edges():
        cp = capd_dir / f"{edge_stem(e)}.json"
        bp = Path(args.boxes) / f"{edge_stem(e)}.json"
        tp = Path(args.tubes) / f"{edge_stem(e)}.json"
        if not (cp.exists() and bp.exists() and tp.exists()):
            missing.append(e)
            continue
        raw = cp.read_bytes()
        digest.update(hashlib.sha256(raw).hexdigest().encode())
        rows.append(compare_edge(e, json.loads(raw), json.loads(bp.read_text()),
                                 json.loads(tp.read_text())["rows"][0], signs[e]))
    contradictions = [(r["edge"], c) for r in rows for c in r["contradictions"]]
    out = {
        "tag": "G8 CAPD 기록 교차 대조(인증 입력 아님)",
        "capd_dir": X.relpath(capd_dir, REPO), "capd_records_digest_of_sha256": digest.hexdigest(),
        "edges": len(rows), "missing": missing,
        "center_contained_all": all(r["center_contained"] == 4 for r in rows),
        "derivative_contained_all16": all(r["derivative_contained"] == 16 for r in rows),
        "center_not_contained": [r["edge"] for r in rows if r["center_contained"] != 4],
        "derivative_not_contained": [r["edge"] for r in rows if r["derivative_contained"] != 16],
        "width_ratio_min": min(r["width_ratio_capd_over_t11"]["min"] for r in rows) if rows else None,
        "width_ratio_max": max(r["width_ratio_capd_over_t11"]["max"] for r in rows) if rows else None,
        "width_ratio_median_range": [min(r["width_ratio_capd_over_t11"]["median"] for r in rows),
                                     max(r["width_ratio_capd_over_t11"]["median"] for r in rows)] if rows else None,
        "v2_window_inside_capd_window": sum(1 for r in rows if r["window_v2_inside_capd"]),
        "signs_agree_all": all(r["signs_agree"] for r in rows),
        "signs_disagree": [r["edge"] for r in rows if not r["signs_agree"]],
        "degrees_t11": {str(d): sum(1 for r in rows if r["degree_t11"] == d) for d in (1, -1)},
        "degrees_capd": {str(d): sum(1 for r in rows if r["degree_capd"] == d) for d in (1, -1)},
        "degree_agree_all": all(r["degree_t11"] == r["degree_capd"] for r in rows),
        "contradictions": contradictions,
        "capd_extremes": capd_extremes(rows),
        "rows": rows,
        "seconds": round(time.time() - started, 1),
    }
    out["pass"] = (not missing and not contradictions and out["center_contained_all"]
                   and out["derivative_contained_all16"] and out["signs_agree_all"] and out["degree_agree_all"])
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=1))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
