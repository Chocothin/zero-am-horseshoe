"""시험 공백 확인(R1 발견): 2계 통과 사상의 f·τ_ab 항을 지운 사본에서 어떤 시험·대조가 실패하는지 기록한다.

사본은 레포 밖 임시 폴더에 두고 results 는 읽기 전용 심볼릭 링크로 둔다. 원본 계산 코드는 바꾸지 않는다.
확인 항목: 새 G9 시험(test_crossing_jets_spec), V5 단위 시험, validate_engine --quick, V5 대칭 항등식 시험,
변이 사본 stage2 로 다시 낸 D(B)·판정, 저장된 G5 기준 해 점이 변이 D(B)·상 안에 드는지.
사용: t11cli.py crossing-mutant [--edges ...] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

from . import design as dz
from . import g5
from .paths import BLAS_ONE, PYTHON, RESULTS, TUBES, edge_stem
from .mutation import staged_copy

OUT = RESULTS / "crossing_jets_mutant.json"
VERTEX_POINTS = RESULTS / "g5_vertex_points.jsonl"
TARGET = "v5/boxmap.py"
ORIGINAL = "        D2y[(a, b)] = [rest[c] + f[c] * tab for c in range(12)]\n"
MUTATED = "        D2y[(a, b)] = [rest[c] for c in range(12)]\n"
EDGES = ["N0>N1", "N1>N2", "N8>M-148", "M-148>M-147", "M-100>M-99", "M-1>M+0", "M+0>M+1", "M+100>M+101",
         "M+146>M+147", "M+148>N8"]


def run(cmd, cwd, extra=None, timeout=1800) -> dict:
    started = time.time()
    proc = subprocess.run(["nice", "-n", "10", str(PYTHON), *cmd], cwd=cwd, capture_output=True, text=True,
                          env={**os.environ, **BLAS_ONE, **(extra or {})}, timeout=timeout)
    return {"cmd": " ".join(cmd), "exit": proc.returncode, "tail": proc.stdout[-400:] + proc.stderr[-300:],
            "seconds": round(time.time() - started, 1)}


def caught_by(points: list, tube: dict, box: dict) -> tuple[int, dict]:
    """감싸기 밖으로 나간 점 수와 실패 항목별 개수."""
    caught, kinds = 0, {}
    for p in points:
        fails = [k for k, (mg, _) in g5.judge_point(p, tube, box).items() if mg < 0]
        caught += bool(fails)
        for k in fails:
            kinds[k.split("[")[0]] = kinds.get(k.split("[")[0], 0) + 1
    return caught, kinds


def vertex_references(edges: list[str]) -> list[dict]:
    """꼭짓점 기준 해: 저장본이 있으면 읽고, 없으면 3프로세스로 계산해 저장한다."""
    if VERTEX_POINTS.exists():
        rows = [json.loads(x) for x in VERTEX_POINTS.read_text().splitlines() if x.strip()]
        if {r["edge"] for r in rows} >= set(edges):
            return [r for r in rows if r["edge"] in edges]
    pts = g5.vertex_points(edges)
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(g5.reference, pts))
    VERTEX_POINTS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default=",".join(EDGES))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--vertices", action="store_true", help="확인 변의 꼭짓점 16개씩도 기준 해로 대조한다(3프로세스)")
    args = ap.parse_args(argv)
    edges = [e for e in args.edges.split(",") if e]
    started = time.time()
    out = {"tag": "R1 시험 공백: crossing_jets 의 f·τ_ab 항을 지운 변이 사본", "mutation": {"file": TARGET,
           "original": ORIGINAL.strip(), "mutated": MUTATED.strip()}}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        track = staged_copy(tmp)
        src = (track / TARGET).read_text()
        if src.count(ORIGINAL) != 1:
            print("mutation site not found exactly once", file=sys.stderr)
            return 1
        (track / TARGET).write_text(src.replace(ORIGINAL, MUTATED))
        work = tmp / "work"
        (work / "stage2").mkdir(parents=True)
        out["mutant_runs"] = {
            "new_G9_test_crossing_jets": run(["-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                              "tests/test_crossing_jets_spec.py"], track),
            "V5_unit_tests": run(["-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_engine.py"], track),
            "validate_engine_quick": run(["-I", "-B", "scripts/validate_engine.py", "--quick"], track,
                                         {"V5_VALIDATE_OUT": str(work / "validate.json")}),
            "V5_symmetry_test": run(["-I", "-B", "v5_symmetry_test.py", "--out", str(work / "sym.json")], track),
        }
        boxes = {}
        for e in edges:
            r = run(["-I", "-B", "scripts/stage2.py", e], track, {"V5_STAGE2_OUT": str(work / "stage2")})
            p = work / "stage2" / f"{edge_stem(e)}.json"
            boxes[e] = json.loads(p.read_text()) if p.exists() else {"error": r["tail"]}
        judged = run(["-I", "-B", "exact_judge.py", "--v5dir", str(work / "stage2"), "--out", str(work / "judge.json")],
                     track)
        jrows = {r["edge"]: r for r in json.loads((work / "judge.json").read_text())["v5_stage2"]["rows"]} \
            if judged["exit"] == 0 else {}
    stored_points = g5.load_rows(not g5.POINTS.exists())
    vertex_rows = vertex_references(edges) if args.vertices else []
    per_edge = []
    total_points, caught_points = 0, 0
    for e in edges:
        box = boxes[e]
        if "error" in box:
            per_edge.append({"edge": e, "error": box["error"]})
            continue
        orig = json.loads((RESULTS / "stage2" / f"{edge_stem(e)}.json").read_text())
        shift = max(abs((Fraction(a[0]) + Fraction(a[1])) - (Fraction(b[0]) + Fraction(b[1]))) / 2
                    for ra, rb in zip(box["deriv_mine_exact"], orig["deriv_mine_exact"]) for a, b in zip(ra, rb))
        tube = json.loads((TUBES / f"{edge_stem(e)}.json").read_text())["rows"][0]
        pts = [r for r in stored_points.values() if r["edge"] == e and "error" not in r]
        caught, bad_kinds = caught_by(pts, tube, box)
        verts = [r for r in vertex_rows if r["edge"] == e]
        v_caught, v_kinds = caught_by(verts, tube, box)
        v_orig, _ = caught_by(verts, tube, orig)
        total_points += len(pts)
        caught_points += caught
        j = jrows.get(e, {})
        width = max(((Fraction(a[1]) - Fraction(a[0])) / (Fraction(b[1]) - Fraction(b[0])))
                    for ra, rb in zip(box["deriv_mine_exact"], orig["deriv_mine_exact"]) for a, b in zip(ra, rb))
        width_min = min(((Fraction(a[1]) - Fraction(a[0])) / (Fraction(b[1]) - Fraction(b[0])))
                        for ra, rb in zip(box["deriv_mine_exact"], orig["deriv_mine_exact"]) for a, b in zip(ra, rb))
        per_edge.append({"edge": e, "type": dz.edge_type(e), "mutant_cover_proven": j.get("cover_proven_mv"),
                         "mutant_cone_proven": j.get("cone_proven"), "max_center_shift_of_D(B)": float(shift),
                         "D(B)_width_ratio_mutant_over_original": [float(width_min), float(width)],
                         "g5_points": len(pts), "g5_points_outside_mutant_enclosures": caught, "failed_checks": bad_kinds,
                         "vertex_points": len(verts), "vertex_points_outside_mutant": v_caught,
                         "vertex_failed_checks": v_kinds, "vertex_points_outside_original": v_orig})
    out.update({"per_edge": per_edge, "g5_points_total": total_points, "g5_points_catching_mutant": caught_points,
                "vertex_points_total": len(vertex_rows),
                "vertex_points_catching_mutant": sum(r.get("vertex_points_outside_mutant", 0) for r in per_edge),
                "vertex_points_outside_original": sum(r.get("vertex_points_outside_original", 0) for r in per_edge),
                "types_caught_by_g5_random": sorted({r["type"] for r in per_edge if r.get("g5_points_outside_mutant_enclosures")}),
                "types_caught_by_vertices": sorted({r["type"] for r in per_edge if r.get("vertex_points_outside_mutant")}),
                "seconds": round(time.time() - started, 1)})
    out["summary"] = {k: v["exit"] for k, v in out["mutant_runs"].items()}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("summary", "g5_points_total", "g5_points_catching_mutant", "vertex_points_total",
                                          "vertex_points_catching_mutant", "vertex_points_outside_original",
                                          "types_caught_by_g5_random", "types_caught_by_vertices")}, ensure_ascii=False))
    for r in per_edge:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
