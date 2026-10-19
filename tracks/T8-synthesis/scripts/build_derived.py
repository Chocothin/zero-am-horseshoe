"""원자료 여러 파일에 걸친 최솟값·최댓값·합을 정확 유리수로 모아 derived_stats.json 을 만든다(등록부 원자료 raw 가 이 파일을 읽는다).
--check 는 쓰지 않고 저장본과 같은지만 본다(verify.sh 용).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results/2026-10-18-T8-synthesis/derived_stats.json"
A15 = "results/2026-10-15-"
A17 = "results/2026-10-17-"


def rat(x) -> str:
    f = Fraction(x)
    return f"{f.numerator}/{f.denominator}"


def hexfrac(s: str) -> Fraction:
    return Fraction(float.fromhex(s))


def load(path):
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def compute() -> dict:
    out = {}
    pieces = []
    files = sorted(glob.glob(str(REPO / (A17 + "T6-horseshoe/coverings/*.json"))))
    for f in files:
        for group in json.loads(Path(f).read_text(encoding="utf-8"))["pieces_detail"].values():
            pieces += group
    out["covering_files"] = len(files)
    out["covering_pieces"] = len(pieces)
    out["discmin"] = rat(min(hexfrac(p["discriminant"]["lo"]) for p in pieces))
    out["branchmin"] = rat(min(hexfrac(p["branch_out"]["lo"]) for p in pieces))
    out["tendmargin"] = rat(min(hexfrac(p["tube"]["t_end"]["lo"]) - hexfrac(p["return_time"]["hi"]) for p in pieces))
    out["retwidth"] = rat(max(hexfrac(p["return_time"]["hi"]) - hexfrac(p["return_time"]["lo"]) for p in pieces))

    rows = load(A15 + "T6-horseshoe/enclosure_diagnosis.json")["rows"]
    ratios = [max(Fraction(v) for v in r["c0_half_over_linear"]) for r in rows]
    widths = [Fraction(r["c1_width_per_r"]) for r in rows]
    out["diagnosis_rows"] = len(rows)
    out["cozratmin"], out["cozratmax"] = rat(min(ratios)), rat(max(ratios))
    out["cowidmin"], out["cowidmax"] = rat(min(widths)), rat(max(widths))

    labels = [r["label"] for r in rows]
    cycle_rows = [l for l in labels if l.startswith("cycle ")]
    hom_k = [int(l.split()[1]) for l in labels if l.startswith("hom ")]
    assert len(cycle_rows) + len(hom_k) == len(labels)
    out["diagcyc"], out["diaghom"] = len(cycle_rows), len(hom_k)
    out["diagkmin"], out["diagkmax"] = min(hom_k), max(hom_k)

    plan = [q["label"] for q in load(A17 + "T11-arbcert/g5_plan.json")["points"]]
    centres = sum(re.fullmatch(r"center\d*", l) is not None for l in plan)
    randoms = sum(re.fullmatch(r"random\d*", l) is not None for l in plan)
    out["g5centres"], out["g5other"] = centres, len(plan) - centres - randoms

    segments = load(A17 + "T6-horseshoe/coverings_summary.json")["segments"]
    (excursion,) = [k for k in segments if k.startswith("excursion")]
    out["excurk"] = int(re.fullmatch(r"excursion \|k\|<(\d+)", excursion).group(1))

    scan = load(A15 + "T6-horseshoe/graph_design_scan.json")["candidates"]
    (threshold,) = {m.group(1) for c in scan for k in c if (m := re.fullmatch(r"edges_excess_over_(.+)", k))}
    out["excessthr"] = threshold

    out["sampsix"] = sum(r["points"] for r in load(A17 + "T6-horseshoe/g5_soundness.json")["rows"])

    last = {}
    for line in (REPO / (A17 + "T11-arbcert/run_log.jsonl")).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        last[(r["job"], r["kind"], r["edge"])] = r
    seconds = collections.defaultdict(Fraction)
    for r in last.values():
        if r["output_written"]:
            seconds[r["kind"]] += Fraction(r["seconds"])
    out["arbjobs"] = len(last)
    out["arbtubesec"], out["arbboxsec"] = rat(seconds["tube"]), rat(seconds["box"])
    cone_rows = load(A17 + "T7-hyperbolic/cones.json")["rows"]
    upper_b = [Fraction(r["upper_b"]) for r in cone_rows]
    out["cone_rows"] = len(cone_rows)
    out["bcutcount"] = sum(b > Fraction(8, 5) for b in upper_b)
    out["bmax"] = rat(max(upper_b))
    return {"tag": "[수치: 기계 집계] 원자료 여러 파일의 최솟값·최댓값·합(정확 유리수). 조각별 하한·상한은 16진 부동소수 끝점 그대로, 작업 시간은 작업별 마지막 기록",
            "script": "tracks/T8-synthesis/scripts/build_derived.py", **out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = json.dumps(compute(), ensure_ascii=False, indent=1) + "\n"
    if args.check:
        same = OUT.read_text(encoding="utf-8") == text
        print(f"derived_stats.json {'matches' if same else 'DIFFERS FROM'} the raw files")
        return 0 if same else 1
    OUT.write_text(text, encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
