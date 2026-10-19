"""R1-05 단언을 넣은 뒤의 흐름 관 재실행이 감사 V2 원출력과 같은지 본다(재현 대조, 인증 입력 아님).

T11 tubes/ 의 변별 행을 results/2026-10-16-V2-compact/rigorous_tubes.json 의 같은 변 행과 비교한다. V2 행에만 있는 CAPD 비교 칸과
실행 시간 칸은 뺀다. --before 를 주면 단언 전 T11 출력 폴더와도 비교한다.
사용: t11cli.py tube-rerun [--before DIR] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import design as dz
from .paths import REPO, RESULTS, TUBES, edge_stem

OUT = RESULTS / "tube_rerun_r105.json"
V2 = REPO / "results" / "2026-10-16-V2-compact" / "rigorous_tubes.json"
SKIP = {"seconds", "vs_capd_lower", "vs_capd_upper", "window_vs_capd", "windows_overlap"}


def same(a: dict, b: dict) -> list:
    keys = (set(a) | set(b)) - SKIP
    return sorted(k for k in keys if a.get(k) != b.get(k))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    v2 = {r["edge"]: r for r in json.loads(V2.read_text())["rows"]}
    diffs, diffs_before, identical, identical_before = [], [], 0, 0
    for e in dz.edges():
        row = json.loads((TUBES / f"{edge_stem(e)}.json").read_text())["rows"][0]
        d = same(row, v2[e])
        identical += not d
        if d:
            diffs.append({"edge": e, "fields": d})
        if args.before:
            old = json.loads((Path(args.before) / f"{edge_stem(e)}.json").read_text())["rows"][0]
            db = same(row, old)
            identical_before += not db
            if db:
                diffs_before.append({"edge": e, "fields": db})
    out = {"tag": "R1-05 단언 뒤 흐름 관 314변 재실행의 재현 대조", "compared_with": str(V2.relative_to(REPO)),
           "fields": "실행 시간·CAPD 비교 칸을 뺀 모든 칸", "identical": identical, "differences": diffs}
    if args.before:
        out.update({"identical_with_pre_assert_t11": identical_before, "differences_pre_assert": diffs_before})
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("differences")}, ensure_ascii=False))
    return 0 if identical == len(v2) == 314 and not diffs else 1


if __name__ == "__main__":
    sys.exit(main())
