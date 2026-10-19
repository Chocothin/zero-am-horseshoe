"""[CAP] 옛 덮개 기록(커밋 93a1692, results/2026-10-15-T6-horseshoe)과 융합 금지 빌드로 다시 낸 기록(RESULTS_DIR)을 314변
전수로 맞댄다(엔진 없음). 변마다 판정(증명·차수·방법·조각 수), 덮개 여유, 감싸기 값이 비트 단위로 같은지와 폭 비를 보고,
정리 A 의 주장 판정·여유와 다시 잰 상수를 옛·새 theorem_A.json 에서 비교한다.

옛 기록은 v2(실행 기록 없음)라 판정은 기록된 proven 과, 저장된 조각 표시(certified)로 covering_verdict 를 다시 돌린 값을 함께
적는다. 새 기록은 records.covering_input(저장값 사용 조건 포함)으로 판정한다. 판정이 하나라도 다르면 exit 1 이다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from t6hs.results import DESIGN_DIR, REPO_DIR, RESULTS_DIR, T3_DIR, write_result  # noqa: E402

sys.path.insert(0, str(T3_DIR))
from t3cap.export import interval_from_json  # noqa: E402

from t6hs import covering, records  # noqa: E402

OLD_DIR = REPO_DIR / "results" / "2026-10-15-T6-horseshoe"
PIECE_INTERVALS = ("return_time", "c1_return_time", "second_zero_time", "discriminant", "branch_out")


def old_input(rec: dict) -> records.CoveringInput:
    """v2 기록: 조각 상과 기록된 사용 표시로 판정 입력을 만든다(그때의 저장값 사용 조건은 표시가 운반했다)."""
    def box(js):
        return tuple(interval_from_json(x) for x in js)

    def images(plist):
        return [covering.PieceImage(box(p["piece"]), box(p["image"]), p["certified"]) for p in plist if p["image"] is not None]

    detail = rec["pieces_detail"]
    return records.CoveringInput(rec["edge"], images(detail["whole"]),
                                 {k: images(detail[f"face_w{k[0]}{k[1]:+d}"]) for k in records.FACE_KEYS},
                                 tuple(rec["signs"]), Fraction(0), Fraction(0), Fraction(0), ())


def enclosure_values(rec: dict) -> list:
    """비트 단위 대조 대상: 중심 상·D(B)와 조각마다 평균값 상·흐름 관·귀환 창·판별식·도착 근(정확 유리수)."""
    out = []
    if rec.get("center_image"):
        out += [interval_from_json(x) for x in rec["center_image"]]
        out += [interval_from_json(x) for row in rec["derivative"] for x in row]
    for key in sorted(rec["pieces_detail"]):
        for p in rec["pieces_detail"][key]:
            out += [interval_from_json(x) for x in p["image"]]
            out += [interval_from_json(p["tube"][k]) for k in records.TUBE_KEYS]
            out += [interval_from_json(p[k]) for k in PIECE_INTERVALS]
    return out


def width_ratios(new: list, old: list) -> list[Fraction]:
    return [a.width / b.width for a, b in zip(new, old, strict=True) if b.width > 0]


def edge_row(old_rec: dict, new_rec: dict) -> dict:
    o, n = old_input(old_rec), records.covering_input(new_rec)
    vo, vn = covering.covering_verdict(o.whole, o.faces, o.signs), covering.covering_verdict(n.whole, n.faces, n.signs)
    mo, mn = covering.certified_margin(o.whole, o.faces, o.signs), covering.certified_margin(n.whole, n.faces, n.signs)
    pieces = lambda r: sum(len(v) for v in r["pieces_detail"].values())
    eo, en = enclosure_values(old_rec), enclosure_values(new_rec)
    ratios = width_ratios(en, eo) if len(en) == len(eo) else []
    return {"edge": new_rec["edge"],
            "old": {"proven_recorded": old_rec["proven"], "proven_rejudged": vo.proven, "degree": vo.degree,
                    "method": old_rec["method"], "pieces": pieces(old_rec), "margin": str(mo)},
            "new": {"proven_recorded": new_rec["proven"], "proven_rejudged": vn.proven and not n.problems,
                    "degree": vn.degree, "method": new_rec["method"], "pieces": pieces(new_rec), "margin": str(mn),
                    "problems": list(n.problems)},
            "verdict_same": (old_rec["proven"] == vo.proven == new_rec["proven"] == (vn.proven and not n.problems)
                             and vo.degree == vn.degree and old_rec["method"] == new_rec["method"]
                             and pieces(old_rec) == pieces(new_rec)),
            "enclosures_identical": eo == en, "intervals_compared": len(en),
            "intervals_differing": sum(a != b for a, b in zip(en, eo)) if len(en) == len(eo) else None,
            "margin_change": str(mn - mo),
            "width_ratio_min": float(min(ratios)) if ratios else None,
            "width_ratio_max": float(max(ratios)) if ratios else None}


def claims_of(path: Path) -> dict:
    t = json.loads(path.read_text())
    return {"claims": {c["claim"]: {"holds": c["holds"], "margin": c["margin"]} for c in t["claims"]},
            "claimed": t["claimed"], "delta": t["collision_distance_lower_bound_delta"], "R": t["size_upper_bound_R"],
            "t_min": t["return_time_lower_bound"], "rho": t["rho_lower_bound"],
            "ln_rho": t["entropy_lower_bound_ln_rho_ge"], "m0_min_separation": t["m0_min_separation"],
            "degree_counts": t["degree_counts"], "hypotheses_certified": t["theorem_A_hypotheses_certified"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    design = json.loads((DESIGN_DIR / "graph_design.json").read_text())
    names = [f"{e['source']}__{e['target']}.json" for e in design["edges"]]
    rows = [edge_row(records.load(OLD_DIR / "coverings" / name), records.load(RESULTS_DIR / "coverings" / name))
            for name in names]
    old_min = min(rows, key=lambda r: Fraction(r["old"]["margin"]))
    new_min = min(rows, key=lambda r: Fraction(r["new"]["margin"]))
    ratios = [r for r in rows if r["width_ratio_min"] is not None]
    theorem = {"old": claims_of(OLD_DIR / "theorem_A.json"), "new": claims_of(RESULTS_DIR / "theorem_A.json")}
    summary = {
        "edges": len(rows), "verdicts_differing": [r["edge"] for r in rows if not r["verdict_same"]],
        "new_problems": [r["edge"] for r in rows if r["new"]["problems"]],
        "enclosures_identical_edges": sum(r["enclosures_identical"] for r in rows),
        "intervals_compared": sum(r["intervals_compared"] for r in rows),
        "intervals_differing": sum(r["intervals_differing"] or 0 for r in rows),
        "min_margin_old": {"edge": old_min["edge"], "margin": old_min["old"]["margin"]},
        "min_margin_new": {"edge": new_min["edge"], "margin": new_min["new"]["margin"]},
        "max_abs_margin_change": str(max(abs(Fraction(r["margin_change"])) for r in rows)),
        "width_ratio_min": min((r["width_ratio_min"] for r in ratios), default=None),
        "width_ratio_max": max((r["width_ratio_max"] for r in ratios), default=None),
        "theorem_claims_same": {k: (theorem["old"]["claims"][k]["holds"], theorem["new"]["claims"][k]["holds"])
                                for k in theorem["new"]["claims"]},
        "claimed_constants_same": theorem["old"]["claimed"] == theorem["new"]["claimed"],
        "remeasured_same": {k: theorem["old"][k] == theorem["new"][k] for k in ("delta", "R", "t_min", "rho", "ln_rho",
                                                                                "m0_min_separation", "degree_counts")},
    }
    print(json.dumps(summary, indent=1))
    if not args.no_write:
        write_result("compare_strict_rebuild", {"tag": "[CAP] 옛(93a1692)·새(융합 금지 빌드) 덮개 기록 314변 대조(엔진 없음)",
                                                "old_dir": str(OLD_DIR.relative_to(REPO_DIR)), "summary": summary,
                                                "theorem_A": theorem, "rows": rows},
                     script=Path(__file__).resolve(), params=vars(args), started=started)
    ok = not summary["verdicts_differing"] and not summary["new_problems"] and summary["claimed_constants_same"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
