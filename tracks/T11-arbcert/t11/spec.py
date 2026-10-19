"""G9 명세 시험 실행 기록: pytest 를 돌려 시험마다 통과·실패를 JSON 으로 남긴다.

사용: t11cli.py spec [--out PATH] [--select tests/a.py,tests/b.py]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .paths import BLAS_ONE, PYTHON, RESULTS, TRACK

OUT = RESULTS / "spec_tests.json"
DEFAULT = ["tests/test_engine.py", "tests/test_judge_spec.py", "tests/test_claims_spec.py", "tests/test_crossing_jets_spec.py",
           "tests/test_assemble_spec.py", "tests/test_guards_spec.py", "tests/test_tube_spec.py",
           "tests/test_tube_rigor_spec.py", "tests/test_g5ref_spec.py", "tests/test_gate_guards_spec.py"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--select", default=",".join(DEFAULT))
    args = ap.parse_args(argv)
    started = time.time()
    files = [f for f in args.select.split(",") if f]
    proc = subprocess.run(["nice", "-n", "10", str(PYTHON), "-B", "-m", "pytest", "-q", "-rA", "-p",
                           "no:cacheprovider", *files], cwd=TRACK, capture_output=True, text=True,
                          env={**os.environ, **BLAS_ONE}, timeout=1800)
    results = {}
    for line in proc.stdout.splitlines():
        m = re.match(r"^(PASSED|FAILED|ERROR|SKIPPED)\s+(\S+)", line)
        if m:
            results[m.group(2)] = m.group(1)
    out = {"tag": "G9 명세 시험(pytest)", "files": files, "tests": len(results),
           "passed": sum(v == "PASSED" for v in results.values()),
           "failed": [k for k, v in results.items() if v in ("FAILED", "ERROR")],
           "skipped": [k for k, v in results.items() if v == "SKIPPED"],
           "results": results, "exit": proc.returncode, "seconds": round(time.time() - started, 1)}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(proc.stdout[-1500:])
    print(json.dumps({k: out[k] for k in ("tests", "passed", "failed", "skipped", "exit")}, ensure_ascii=False))
    return 0 if proc.returncode == 0 and not out["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
