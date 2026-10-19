"""복사 출처 기록: 감사 코드 원본과 T11 사본의 sha256, 사본이 고친 곳(unified diff)과 이유를 남기고 다시 대조한다.

사용: t11cli.py provenance write | check
check 는 원본 sha256·사본 sha256·원본→사본 diff 가 기록과 모두 같을 때만 exit 0 이다(사본을 몰래 고치거나 원본이 바뀌면 실패).
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path

from .paths import REPO, RESULTS

OUT = RESULTS / "provenance.json"
T = "tracks/T11-arbcert"
IO_ONLY = "입출력(계약 1 (가))"
LEAD_FIX = "리드 지시 수정(계약 1 (나), R1 발견)"
JUDGE_INPUT = "판정 입력 출처 변경(계약 1 의 독립성 이행: CAPD 기록을 판정 입력으로 쓰지 않기 위해) + 입출력"

COPIES = [
    ("tracks/V2-compact/rigorous_tube.py", f"{T}/rigorous_tube.py", LEAD_FIX,
     "R1-05: aff_func 의 2차 나머지는 나머지 공 r 이 0 을 담을 때만 감싸므로 r ∋ 0 단언(어기면 ArithmeticError)을 넣음. "
     "314변 흐름 관 재실행 출력이 단언 전과 같음(tube_rerun_r105.json)"),
    ("tracks/V2-compact/nbody_mp.py", f"{T}/nbody_mp.py", None),
    ("tracks/V2-compact/certify_tubes.py", f"{T}/certify_tubes.py",
     "출력 경로를 T11 로 바꿈. certify_edge 가 CAPD 덮개 기록을 열어 비교하던 칸(vs_capd·window_vs_capd·windows_overlap)과 "
     "--cover-dir·q()·요약의 all_windows_overlap_capd 를 뺌(CAPD 교차 대조는 t11/g8.py). 그로 쓰이지 않게 된 t_name 을 _ 로"),
    ("tracks/V2-compact/rigorous_controls.py", f"{T}/rigorous_controls.py",
     "출력 경로를 T11 로 바꿈. 표본 포함 대조의 입력을 V2 표본 파일에서 T11 G5 출력(g5_tube_samples.json, 같은 형식)으로, "
     "상자 이동 대조의 기준 창을 CAPD 창에서 T11 의 이동 전 창으로 바꿈"),
    *[(f"tracks/V5-indep/v5/{n}.py", f"{T}/v5/{n}.py", None)
      for n in ("boxmap", "charts", "compare", "data", "edge", "flow", "jets", "margins", "taylor")],
    ("tracks/V5-indep/scripts/stage2.py", f"{T}/scripts/stage2.py",
     "출력 경로를 T11 로 바꿈. run_edge 의 CAPD 기록·T7 g8_derivatives 읽기와 비교 칸을 빼고, 교차 대조·첫 통과 구조 판정이 "
     "읽는 정확 끝점(center_deriv·hull_image·hull_deriv·tau_hull·branch_out_hull·disc_hull·외피 관)을 출력에 더함. "
     "변 목록 기본값을 selected_edges.json 대신 graph_design 의 314변으로"),
    ("tracks/V5-indep/scripts/validate_engine.py", f"{T}/scripts/validate_engine.py",
     "기본 출력 경로를 V5 결과 폴더에서 T11 으로 바꿈(환경 변수 없이 돌려도 감사 결과를 덮어쓰지 않게)"),
    ("tracks/V5-indep/tests/test_engine.py", f"{T}/tests/test_engine.py", None),
    ("tracks/V7-judge/v7exact.py", f"{T}/v7exact.py", None),
    ("tracks/V7-judge/exact_judge.py", f"{T}/exact_judge.py", JUDGE_INPUT,
     "CAPD 기록 314개 재판정(judge_capd·rec_D)과 CAPD 포함 비교 칸을 뺌(t11/g8.py 로). judge_v5 의 덮개 부호를 CAPD 기록의 "
     "signs 대신 graph_design 의 linear 대각 부호(design_signs, T6 certify_graph 의 규칙)에서 받음. 기본 입출력 경로를 T11 로"),
    ("tracks/V7-judge/v5_symmetry_test.py", f"{T}/v5_symmetry_test.py",
     "sys.path 를 tracks/V5-indep 대신 T11 사본으로, 기본 출력 경로를 T11 로 바꿈"),
    ("tracks/V3-cone/graph_and_hulls.py", f"{T}/graph_and_hulls.py",
     "변 집합과 비교하는 파일 이름 폴더를 CAPD coverings/ 대신 T11 stage2/ 로, 기본 출력 경로를 T11 로 바꿈"),
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def diff_text(orig: Path, copy: Path, orig_rel: str, copy_rel: str) -> str:
    a = orig.read_text().splitlines(keepends=True)
    b = copy.read_text().splitlines(keepends=True)
    return "".join(difflib.unified_diff(a, b, fromfile=orig_rel, tofile=copy_rel, n=1))


def snapshot() -> list[dict]:
    rows = []
    for entry in COPIES:
        orig_rel, copy_rel, *rest = entry
        klass, reason = (rest if len(rest) == 2 else (IO_ONLY if rest[0] else "바이트 동일", rest[0]))
        orig, copy = REPO / orig_rel, REPO / copy_rel
        d = diff_text(orig, copy, orig_rel, copy_rel)
        changed = [line for line in d.splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
        rows.append({"original": orig_rel, "copy": copy_rel, "original_sha256": sha(orig), "copy_sha256": sha(copy),
                     "identical": sha(orig) == sha(copy), "reason": reason or "바이트 동일",
                     "class": klass,
                     "changed_lines": len(changed), "diff": d, "diff_sha256": hashlib.sha256(d.encode()).hexdigest()})
    return rows


def check(stored: dict) -> list[str]:
    problems = []
    now = {r["copy"]: r for r in snapshot()}
    for r in stored["copies"]:
        n = now.get(r["copy"])
        if n is None:
            problems.append(f"{r['copy']}: no longer listed")
            continue
        for k in ("original_sha256", "copy_sha256", "diff_sha256"):
            if n[k] != r[k]:
                problems.append(f"{r['copy']}: {k} differs")
        if (n["reason"] == "바이트 동일") != n["identical"]:
            problems.append(f"{r['copy']}: declared byte-identical but differs (or the reverse)")
    if len(now) != len(stored["copies"]):
        problems.append("copy list length differs")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("write", "check"))
    args = ap.parse_args(argv)
    if args.cmd == "write":
        rows = snapshot()
        bad = [r["copy"] for r in rows if (r["reason"] == "바이트 동일") != r["identical"]]
        if bad:
            print("declared byte-identical but differ:", bad, file=sys.stderr)
            return 1
        OUT.write_text(json.dumps({"tag": "감사 코드 복사 출처(원본 sha256·사본 sha256·diff)", "base_commit": "93d2801",
                                   "copies": rows}, ensure_ascii=False, indent=1) + "\n")
        print(json.dumps([{k: r[k] for k in ("copy", "identical", "changed_lines")} for r in rows], ensure_ascii=False))
        return 0
    problems = check(json.loads(OUT.read_text()))
    for p in problems:
        print("PROVENANCE:", p)
    print("provenance:", "PASS" if not problems else "FAIL", f"({len(COPIES)} files)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
