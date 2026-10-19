"""CAPD 자체 시험 스위트(ctest)를 고정 커밋·NATIVE 구간·MPFR·융합 금지(-ffp-contract=off) 구성으로 빌드해 돌리고 결과를 기록한다.

빌드는 gitignore 된 build/capd-native-strict-tests 에만 쓴다. 병렬도는 트랙 CPU 예산에 맞춰 2로 둔다(nice 는 호출자가 건다).
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from common import TRACK_DIR, write_result

JOBS = "2"


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 7200) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def main() -> int:
    started = time.time()
    src = TRACK_DIR / "third_party" / "capd"
    build = TRACK_DIR / "build" / "capd-native-strict-tests"
    configure = run(["cmake", "-S", str(src), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                     "-DCMAKE_C_COMPILER=/usr/bin/clang", "-DCMAKE_CXX_COMPILER=/usr/bin/clang++",
                     "-DCAPD_INTERVAL_TYPE=NATIVE", "-DCAPD_ENABLE_MULTIPRECISION=ON", "-DCAPD_BUILD_TESTS=ON",
                     "-DCMAKE_PREFIX_PATH=/opt/homebrew", "-DCMAKE_CXX_FLAGS=-I/opt/homebrew/include -ffp-contract=off",
                     "-DCMAKE_C_FLAGS=-ffp-contract=off",
                     "-DCMAKE_EXE_LINKER_FLAGS=-L/opt/homebrew/lib"])
    if configure.returncode != 0:
        raise SystemExit("configure failed:\n" + configure.stdout[-3000:] + configure.stderr[-3000:])
    compile_ = run(["cmake", "--build", str(build), "-j", JOBS])
    compile_tail = (compile_.stdout + compile_.stderr)[-4000:]
    ctest = run(["ctest", "--test-dir", str(build), "-j", JOBS, "--output-on-failure", "--timeout", "1800"])
    out = ctest.stdout + ctest.stderr
    summary = re.search(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)", out)
    failed = re.findall(r"^\s*\d+ - (\S+) \((\w+)\)", out, flags=re.M)
    payload = {
        "tag": "CAPD 자체 시험 스위트(ctest) 결과",
        "configuration": "commit 2f06098, CAPD_INTERVAL_TYPE=NATIVE, MPFR on, arm64, clang, -ffp-contract=off",
        "build_returncode": compile_.returncode,
        "ctest_returncode": ctest.returncode,
        "summary": summary.group(0) if summary else None,
        "failed_tests": [{"name": n, "reason": r} for n, r in failed],
        "build_log_tail": compile_tail if compile_.returncode != 0 else "",
        "ctest_log_tail": out[-6000:],
    }
    write_result("capd_selftests", payload, script=Path(__file__).resolve(),
                 params={"jobs": JOBS, "build_dir": str(build.relative_to(TRACK_DIR))}, started=started)
    print(payload["summary"], payload["failed_tests"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
