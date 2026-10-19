"""G8 차등 대조: 본 실행과 다른 설정(--tag) 실행의 덮개 기록을 변마다 맞대어 본다(엔진 없음).

두 실행 모두 같은 집합(조각의 P̃ 상)을 엄밀히 감쌌다면 두 감싸기는 성분마다 겹쳐야 한다. 겹치지 않으면 어느 한쪽이
틀렸다는 뜻이다. 판정(증명·차수)이 같은지, 여유가 얼마나 달라졌는지, 감싸기 폭 비가 얼마인지도 함께 남긴다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from t6hs.results import DESIGN_DIR, RESULTS_DIR, T3_DIR, write_result  # noqa: E402

sys.path.insert(0, str(T3_DIR))
from t6hs import covering, records  # noqa: E402


def overlaps(a, b) -> bool:
    return all(max(x.lo, y.lo) <= min(x.hi, y.hi) for x, y in zip(a, b))


def compare(main: records.CoveringInput, other: records.CoveringInput) -> dict:
    pairs = list(zip(main.whole, other.whole))
    pairs += [(p, q) for k in records.FACE_KEYS for p, q in zip(main.faces[k], other.faces[k])]
    same_pieces = all(p.piece == q.piece for p, q in pairs) and len(main.whole) == len(other.whole)
    widths = [float(q.image[k].width / p.image[k].width) for p, q in pairs for k in range(4) if p.image[k].width > 0]
    vm = covering.covering_verdict(main.whole, main.faces, main.signs)
    vo = covering.covering_verdict(other.whole, other.faces, other.signs)
    mm = covering.certified_margin(main.whole, main.faces, main.signs)
    mo = covering.certified_margin(other.whole, other.faces, other.signs)
    return {"edge": main.edge, "same_pieces": same_pieces,
            "images_overlap": same_pieces and all(overlaps(p.image, q.image) for p, q in pairs),
            "both_proven": vm.proven and vo.proven and not main.problems and not other.problems,
            "same_degree": vm.degree == vo.degree, "margin_main": float(mm), "margin_other": float(mo),
            "width_ratio_min": min(widths), "width_ratio_max": max(widths)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    design = json.loads((DESIGN_DIR / "graph_design.json").read_text())
    rows = []
    for e in design["edges"]:
        name = f"{e['source']}__{e['target']}.json"
        rows.append(compare(records.covering_input(records.load(RESULTS_DIR / "coverings" / name)),
                            records.covering_input(records.load(RESULTS_DIR / "coverings" / args.tag / name))))
    summary = {
        "edges": len(rows), "all_overlap": all(r["images_overlap"] for r in rows),
        "all_proven_both": all(r["both_proven"] for r in rows), "all_same_degree": all(r["same_degree"] for r in rows),
        "margin_min_main": min(r["margin_main"] for r in rows), "margin_min_other": min(r["margin_other"] for r in rows),
        "max_abs_margin_change": max(abs(r["margin_main"] - r["margin_other"]) for r in rows),
        "width_ratio_min": min(r["width_ratio_min"] for r in rows), "width_ratio_max": max(r["width_ratio_max"] for r in rows),
    }
    print(json.dumps(summary))
    if not args.no_write:
        write_result(f"compare_{args.tag}", {"tag": "[CAP] G8 차등 대조(두 실행의 감싸기 겹침·판정 일치)",
                                             "summary": summary, "rows": rows},
                     script=Path(__file__).resolve(), params=vars(args), started=started)
    ok = summary["all_overlap"] and summary["all_proven_both"] and summary["all_same_degree"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
