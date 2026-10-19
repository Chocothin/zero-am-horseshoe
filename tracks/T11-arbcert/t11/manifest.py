"""manifest.json(CLAUDE.md 규칙 5): 기준 커밋, 스크립트 sha256, 변별 기록 sha256 목록·묶음 digest, 출력 sha256, 매개변수,
정밀도, 실행 시간, 환경. check 는 기록 목록·digest·개수·출력·스크립트 해시를 다시 대조한다(판정이 믿는 기록의 고정).

사용: t11cli.py manifest write | check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from .paths import BOXES, REPO, RESULTS, TRACK, TUBES

OUT = RESULTS / "manifest.json"
OUTPUTS = ["provenance.json", "exact_judge.json", "theorem_claims.json", "rigorous_tubes.json", "controls.json",
           "graph_and_hulls.json", "g5_plan.json", "g5_points.jsonl", "g5_soundness.json", "g5_tube_samples.json",
           "rigorous_controls.json", "g8_capd_crosscheck.json", "mutation_audit.json", "v5_symmetry_test.json",
           "validate_engine.json", "independence.json", "spec_tests.json", "crossing_jets_mutant.json",
           "tube_rigor_mutants.json", "tube_rerun_r105.json", "g5_vertex_plan.json", "g5_vertex_points.jsonl",
           "g5_vertex_soundness.json", "run_log.jsonl", "g8_capd_crosscheck_2026-10-17.json"]
RECORD_DIRS = {"tubes": TUBES, "stage2": BOXES}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def records(d: Path) -> dict:
    files = sorted(d.glob("*.json"))
    listing = {p.name: sha(p) for p in files}
    digest = hashlib.sha256("".join(listing[n] for n in sorted(listing)).encode()).hexdigest()
    return {"count": len(files), "digest_of_sorted_sha256": digest, "files": listing}


def scripts() -> dict:
    files = sorted(p for p in TRACK.rglob("*") if p.is_file() and p.suffix in (".py", ".sh")
                   and "__pycache__" not in p.parts)
    return {str(p.relative_to(TRACK)): sha(p) for p in files}


def seconds() -> dict:
    out = {}
    log = RESULTS / "run_log.jsonl"
    if log.exists():
        # 같은 작업을 다시 돌렸으면(예: 흐름 관 재실행) 마지막 실행만 센다
        last = {}
        for x in log.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                last[r["job"]] = r
        for kind in ("box", "tube"):
            s = [r["seconds"] for r in last.values() if r["kind"] == kind]
            out[f"{kind}_jobs"] = {"n": len(s), "sum": round(sum(s), 1), "max": max(s, default=None)}
    for name, key in (("g5_soundness.json", "seconds_reference_total"), ("g5_vertex_soundness.json", "seconds_reference_total"),
                      ("controls.json", "seconds"),
                      ("mutation_audit.json", "seconds"), ("v5_symmetry_test.json", "seconds"),
                      ("theorem_claims.json", "seconds"), ("g8_capd_crosscheck.json", "seconds")):
        p = RESULTS / name
        if p.exists():
            out[name] = json.loads(p.read_text()).get(key)
    return out


def write() -> int:
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = {
        "track": "T11-arbcert (DAG tbp01-p1, 둘째 인증 경로)",
        "base_commit": head,
        "inputs_sha256": {"results/2026-10-15-T6-horseshoe/graph_design.json":
                          sha(REPO / "results/2026-10-15-T6-horseshoe/graph_design.json")},
        "records": {k: records(d) for k, d in RECORD_DIRS.items()},
        "outputs_sha256": {n: sha(RESULTS / n) for n in OUTPUTS if (RESULTS / n).exists()},
        "scripts_sha256": scripts(),
        "parameters": {
            "tube": {"library": "python-flint arb", "prec_bits": 128, "order": 18, "kappa": 0.08, "sub": 32,
                     "set": "w 에 대한 아핀 꼴 + 나머지 공", "a_priori": "1차 Picard X0+[0,h]f(E)⊆E"},
            "box": {"library": "python-flint arb", "center": {"prec_bits": 256, "order": 30, "tol": 1e-40},
                    "hull": {"prec_bits": 128, "order": 24, "tol": 1e-28}, "set": "공 + 2계 제트, 평균값 형식"},
            "judge": "fractions.Fraction 정확 유리수(v7exact)",
            "g5": {"reference": "mpmath Jacobi 좌표 Taylor(차수 32)", "dps_random": 45, "dps_center": 60},
        },
        "seconds": seconds(),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "machine": platform.machine(), "python-flint": metadata.version("python-flint"),
                        "mpmath": metadata.version("mpmath"), "sympy": metadata.version("sympy"),
                        "numpy": metadata.version("numpy"),
                        "budget": "nice -n 10, BLAS 1스레드, 동시 계산 3개 이하"},
    }
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: (v["count"], v["digest_of_sorted_sha256"][:16]) for k, v in manifest["records"].items()}))
    return 0


def check() -> int:
    m = json.loads(OUT.read_text())
    problems = []
    for k, d in RECORD_DIRS.items():
        now = records(d)
        if now["count"] != m["records"][k]["count"] or now["digest_of_sorted_sha256"] != m["records"][k]["digest_of_sorted_sha256"]:
            problems.append(f"{k}: record list or digest differs")
        if now["files"] != m["records"][k]["files"]:
            problems.append(f"{k}: per-file sha256 differs")
    for n, h in m["outputs_sha256"].items():
        if not (RESULTS / n).exists() or sha(RESULTS / n) != h:
            problems.append(f"output {n} differs or missing")
    now_scripts = scripts()
    if now_scripts != m["scripts_sha256"]:
        diff = sorted(set(now_scripts.items()) ^ set(m["scripts_sha256"].items()))
        problems.append(f"scripts differ: {[d[0] for d in diff][:10]}")
    for n, h in m["inputs_sha256"].items():
        if sha(REPO / n) != h:
            problems.append(f"input {n} differs")
    for p in problems:
        print("MANIFEST:", p)
    print("manifest:", "PASS" if not problems else "FAIL")
    return 0 if not problems else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("write", "check"))
    return write() if ap.parse_args(argv).cmd == "write" else check()


if __name__ == "__main__":
    sys.exit(main())
