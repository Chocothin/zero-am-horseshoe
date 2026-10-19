"""G6 덮개별 전용 음성 대조(엔진 재계산 없음): 저장된 감싸기를 records.covering_input 으로 꺼내 covering_verdict 로 판정한다.

덮개마다 판정이 반드시 거짓이어야 하는 대조 3종을 심는다.
  tighten_threshold : 문턱 1 을 인증 여유 μ 바로 너머(μ·(1+1e-12))로 조인다(불안정 문턱 1+μ′, 안정 문턱 1−μ′)
  swap_weak_roles   : 상 좌표의 약불안정(1)·약안정(2) 역할을 맞바꾼다
  swap_strong_roles : 상 좌표의 강불안정(0)·강안정(3) 역할을 맞바꾼다
대조마다 computed 는 그 구성에서 다시 잰 부호 있는 여유다(음수 = 판정 부등식 하나가 깨짐). 문턱 강화가 날카로운지
(μ·(1−1e-12) 로 조이면 여전히 증명됨)도 함께 본다. 요약은 모든 덮개에 실패한 대조가 2개 이상인지(controls_cover),
새어 나간 증명 수(leaked), 판정과 여유 부호가 일치하는지(margin_sign_agrees)를 운반한다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from t6hs.results import DESIGN_DIR, RESULTS_DIR, T3_DIR, write_result  # noqa: E402

sys.path.insert(0, str(T3_DIR))
from t6hs import covering, records  # noqa: E402

RELATIVE = Fraction(1, 10**12)
MIN_CONTROLS = 2


def swapped(c: records.CoveringInput, a: int, b: int):
    def swap(img):
        out = list(img)
        out[a], out[b] = img[b], img[a]
        return tuple(out)

    whole = [covering.PieceImage(p.piece, swap(p.image), p.certified) for p in c.whole]
    faces = {k: [covering.PieceImage(p.piece, swap(p.image), p.certified) for p in ps] for k, ps in c.faces.items()}
    return whole, faces


def run_control(name: str, planted: str, call, computed: Fraction) -> dict:
    row = {"control": name, "planted": planted, "computed_margin": str(computed), "computed_float": float(computed)}
    try:
        v = call()
        return {**row, "proven": v.proven, "failed_checks": [x.name for x in v.failed],
                "reason": "verdict false" if not v.proven else "LEAKED"}
    except ValueError as exc:
        return {**row, "proven": False, "failed_checks": [], "reason": f"guard exception: {exc}"}


def controls_for(rec: dict) -> dict:
    c = records.covering_input(rec)
    base = covering.covering_verdict(c.whole, c.faces, c.signs)
    margin = covering.certified_margin(c.whole, c.faces, c.signs)
    beyond = margin * (1 + RELATIVE)
    inside = margin * (1 - RELATIVE)
    weak, strong = swapped(c, 1, 2), swapped(c, 0, 3)
    out = [
        run_control("tighten_threshold", f"thresholds 1±{float(beyond):.17g} (margin·(1+1e-12))",
                    lambda: covering.covering_verdict(c.whole, c.faces, c.signs, unstable_threshold=1 + beyond,
                                                      stable_threshold=1 - beyond), margin - beyond),
        run_control("swap_weak_roles", "image coords 1<->2",
                    lambda: covering.covering_verdict(*weak, c.signs), covering.certified_margin(*weak, c.signs)),
        run_control("swap_strong_roles", "image coords 0<->3",
                    lambda: covering.covering_verdict(*strong, c.signs), covering.certified_margin(*strong, c.signs)),
    ]
    sharp = covering.covering_verdict(c.whole, c.faces, c.signs, unstable_threshold=1 + inside,
                                      stable_threshold=1 - inside).proven
    return {"edge": c.edge, "baseline_proven": base.proven and not c.problems, "computed_margin": str(margin),
            "margin_float": float(margin), "sharpness_inside_margin_still_proven": sharp, "controls": out,
            "failed_controls": sum(1 for x in out if not x["proven"])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    design = json.loads((DESIGN_DIR / "graph_design.json").read_text())
    cover_dir = RESULTS_DIR / "coverings" / args.tag if args.tag else RESULTS_DIR / "coverings"
    rows = [controls_for(records.load(cover_dir / f"{e['source']}__{e['target']}.json")) for e in design["edges"]]
    leaked = sum(1 for r in rows for x in r["controls"] if x["proven"])
    summary = {
        "coverings": len(rows), "controls_total": sum(len(r["controls"]) for r in rows), "leaked": leaked,
        "baseline_all_proven": all(r["baseline_proven"] for r in rows),
        "controls_cover": all(r["failed_controls"] >= MIN_CONTROLS for r in rows) and len(rows) == len(design["edges"]),
        "sharpness_all": all(r["sharpness_inside_margin_still_proven"] for r in rows),
        "margin_sign_agrees": all(x["proven"] == (Fraction(x["computed_margin"]) > 0) for r in rows for x in r["controls"]),
        "computed_max": max(x["computed_float"] for r in rows for x in r["controls"]),
        "min_margin": min(r["margin_float"] for r in rows),
    }
    print(json.dumps(summary))
    if not args.no_write:
        write_result(f"stored_controls{('_' + args.tag) if args.tag else ''}",
                     {"tag": "G6 덮개별 전용 음성 대조(저장 감싸기, 모두 판정 거짓이어야 함)", "summary": summary, "rows": rows},
                     script=Path(__file__).resolve(), params=vars(args), started=started)
    ok = (summary["controls_cover"] and not leaked and summary["baseline_all_proven"] and summary["sharpness_all"]
          and summary["margin_sign_agrees"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
