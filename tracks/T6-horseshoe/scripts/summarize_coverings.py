"""[CAP] 덮개 인증 결과 요약: 구간별 증명 수, 판정 문턱(±1)까지의 최소 여유, 흐름 관 하한, 차수 분포, 둘째 영점 간격."""
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
from t3cap.export import interval_from_json  # noqa: E402


def segment(src: str, dst: str) -> str:
    if src.startswith("N") and dst.startswith("N"):
        return "cycle"
    if src.startswith("N"):
        return "departure"
    if dst.startswith("N"):
        return "arrival"
    k = int(src[1:])
    return "excursion |k|<20" if abs(k) < 20 else ("backward tail" if k < 0 else "forward tail")


def margins(rec: dict) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    signs = rec["signs"]
    d = rec["pieces_detail"]
    stable = min(min(1 - interval_from_json(p["image"][k]).hi, interval_from_json(p["image"][k]).lo + 1)
                 for p in d["whole"] for k in (2, 3))
    face = Fraction(10**9)
    for key, plist in d.items():
        if key == "whole":
            continue
        j, side = int(key[6]), int(key[7:])
        for p in plist:
            iv = interval_from_json(p["image"][j])
            face = min(face, (iv.lo - 1) if signs[j] * side > 0 else (-1 - iv.hi))
    tube = min(interval_from_json(p["tube"][k]).lo for p in d["whole"] for k in ("r12", "r13", "r23"))
    gap = min(interval_from_json(p["second_zero_time"]).lo - interval_from_json(p["c1_return_time"]).hi
              for p in d["whole"])
    return face, stable, tube, gap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    design = json.loads((DESIGN_DIR / "graph_design.json").read_text())
    cover_dir = RESULTS_DIR / "coverings" / args.tag if args.tag else RESULTS_DIR / "coverings"
    rows, by_seg = [], {}
    for e in design["edges"]:
        rec = json.loads((cover_dir / f"{e['source']}__{e['target']}.json").read_text())
        face, stable, tube, gap = margins(rec)
        seg = segment(e["source"], e["target"])
        rows.append({"edge": rec["edge"], "segment": seg, "proven": rec["proven"], "degree": rec["degree"],
                     "method": rec["method"], "exit_face_margin": float(face), "stable_margin": float(stable),
                     "tube_min": float(tube), "second_zero_gap": float(gap), "seconds": rec["seconds"]})
        s = by_seg.setdefault(seg, {"edges": 0, "proven": 0, "degrees": {}, "exit_face_margin_min": None,
                                    "stable_margin_min": None, "tube_min": None, "second_zero_gap_min": None})
        s["edges"] += 1
        s["proven"] += int(rec["proven"])
        s["degrees"][str(rec["degree"])] = s["degrees"].get(str(rec["degree"]), 0) + 1
        for key, val in (("exit_face_margin_min", face), ("stable_margin_min", stable), ("tube_min", tube),
                         ("second_zero_gap_min", gap)):
            s[key] = float(val) if s[key] is None else min(s[key], float(val))
    for seg, s in by_seg.items():
        print(seg, json.dumps(s))
    total = {"edges": len(rows), "proven": sum(r["proven"] for r in rows),
             "pieces_method": sum(r["method"] == "pieces" for r in rows),
             "exit_face_margin_min": min(r["exit_face_margin"] for r in rows),
             "stable_margin_min": min(r["stable_margin"] for r in rows),
             "tube_min": min(r["tube_min"] for r in rows), "second_zero_gap_min": min(r["second_zero_gap"] for r in rows),
             "engine_seconds_total": round(sum(r["seconds"] for r in rows), 1)}
    print("total", json.dumps(total))
    if not args.no_write:
        write_result(f"coverings_summary{('_' + args.tag) if args.tag else ''}",
                     {"tag": "[CAP] 덮개 인증 요약", "total": total, "segments": by_seg, "rows": rows},
                     script=Path(__file__).resolve(), params=vars(args), started=started)
    return 0 if total["proven"] == total["edges"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
