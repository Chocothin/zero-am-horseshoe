"""엔진 빌드가 곱셈·덧셈 융합 없이(-ffp-contract=off) 만들어졌는지 본다.

- CAPD strict 설치의 컴파일 명령(compile_commands.json) 전부와 엔진 실행 파일의 컴파일 명령(build_engine.sh 가 남긴 .cmd)에
  -ffp-contract=off 와 -frounding-math 가 있어야 한다.
- strict 엔진 바이너리와 libcapd.a(x86 은 libfilib.a 도)의 융합 명령 수가 0 이어야 한다. arm64 는 fmadd·fmsub·fnmadd·fnmsub,
  x86_64 는 vfmadd·vfmsub·vfnmadd·vfnmsub 계열을 센다. 기존 빌드(감사가 대조한 원 바이너리)의 개수는 비교용으로만 적는다.
--write 면 results/<날짜>-T3-cap/build_flags.json 과 manifest 항목을 쓴다. 검사가 하나라도 깨지면 exit 1.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from common import TRACK_DIR, write_result

BUILD = TRACK_DIR / "build"
REQUIRED_FLAGS = ("-ffp-contract=off", "-frounding-math")
FUSED = {"arm64": re.compile(r"\s(fmadd|fmsub|fnmadd|fnmsub)\s"), "x86_64": re.compile(r"\svfn?m(add|sub)\d{3}")}
STRICT = {"native-strict": "arm64", "x86filib-strict": "x86_64"}
LEGACY = {"native": "arm64", "x86filib": "x86_64"}
ENGINE_PROGRAMS = ("t3cap_engine", "arith_selftest", "compiler_probe", "c1_mitigation")


def fused_count(path: Path, arch: str) -> int:
    listing = subprocess.run(["otool", "-tV", str(path)], capture_output=True, text=True, check=True).stdout
    return sum(1 for line in listing.splitlines() if FUSED[arch].search(line))


def compile_commands(config: str) -> dict:
    entries = json.loads((BUILD / f"capd-{config}" / "compile_commands.json").read_text())
    args = [e["arguments"] if "arguments" in e else shlex.split(e["command"]) for e in entries]
    missing = [e["file"] for e, a in zip(entries, args) if not all(flag in a for flag in REQUIRED_FLAGS)]
    return {"entries": len(entries), "missing_flags": missing}


def engine_commands(config: str) -> dict:
    out = {}
    for name in ENGINE_PROGRAMS:
        cmd = (BUILD / f"engine-{config}" / f"{name}.cmd").read_text().split()
        out[name] = all(flag in cmd for flag in REQUIRED_FLAGS)
    return out


def binaries(config: str, arch: str) -> dict:
    libs = ["libcapd.a"] + (["libfilib.a"] if arch == "x86_64" else [])
    return {"t3cap_engine": fused_count(BUILD / f"engine-{config}" / "t3cap_engine", arch),
            **{lib: fused_count(BUILD / f"capd-{config}-install" / "lib" / lib, arch) for lib in libs}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    strict = {config: {"arch": arch, "capd_compile_commands": compile_commands(config),
                       "engine_compile_commands": engine_commands(config), "fused_instructions": binaries(config, arch)}
              for config, arch in STRICT.items()}
    legacy = {config: binaries(config, arch) for config, arch in LEGACY.items()
              if (BUILD / f"engine-{config}" / "t3cap_engine").exists()}
    problems = []
    for config, r in strict.items():
        cc = r["capd_compile_commands"]
        if cc["entries"] == 0 or cc["missing_flags"]:
            problems.append(f"{config}: CAPD compile commands without the flags: {cc['missing_flags'][:5]}")
        problems += [f"{config}: {name} compiled without the flags" for name, ok in r["engine_compile_commands"].items()
                     if not ok]
        problems += [f"{config}: {name} has {n} fused instructions" for name, n in r["fused_instructions"].items() if n]
    payload = {"tag": "빌드 확인: 융합 금지 플래그와 바이너리의 융합 명령 수", "required_flags": list(REQUIRED_FLAGS),
               "strict": strict, "legacy_for_comparison": legacy, "problems": problems}
    print(json.dumps(payload, indent=1))
    if args.write:
        write_result("build_flags", payload, script=Path(__file__).resolve(), params={}, started=started)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
