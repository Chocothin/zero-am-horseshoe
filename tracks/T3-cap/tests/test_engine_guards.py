"""요청 끝점 가드의 C++ 시험(engine_tests/endpoint_guard_test.cpp)을 빌드해 돌리고, 같은 가드를 실제 엔진 요청으로 확인한다.

명세: 16진 끝점은 그 실행의 산술(double 은 53비트·double 범위, MP 는 설정 비트)로 정확히 나타날 때만 그 값 그대로 받고,
반올림이 필요한 토큰은 거부한다. MP 산술은 53비트 미만을 거부한다(그보다 짧으면 2^53 이하 p/q 를 점 구간으로 정확히 담지 못한다).
빌드는 본 엔진 빌드와 섞이지 않게 build/engine-tests-native-strict 에만 한다.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from t3cap import CoordinateSection, Frame, Interval, SystemParams
from t3cap.engine import build_request, engine_binary
from t3cap.model import Settings

TRACK = Path(__file__).resolve().parent.parent


def test_endpoint_guards_hold_for_spec_derived_tokens():
    out = TRACK / "build" / "engine-tests-native-strict"
    env = dict(os.environ, ENGINE_SRC_DIR=str(TRACK / "engine_tests"), ENGINE_OUT_DIR=str(out))
    subprocess.run(["bash", str(TRACK / "build_engine.sh"), "native-strict"], env=env, check=True, capture_output=True,
                   text=True)
    proc = subprocess.run([str(out / "endpoint_guard_test")], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def chart_request(arithmetic: str) -> str:
    box = (Interval.point("5.14856798"), Interval.point(0), Interval.point(0), Interval.point("0.43269065"))
    return build_request("chart", [box], system=SystemParams.make(), section_in=CoordinateSection.syzygy(+1),
                         section_out=CoordinateSection.syzygy(+1), returns=1, frame_in=Frame.identity(),
                         frame_out=Frame.identity(), settings=Settings(arithmetic=arithmetic), rotation=False)


def run_engine(request: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(engine_binary("native-strict"))], input=request, capture_output=True, text=True,
                          timeout=120)


def test_engine_refuses_mp_precision_below_53_bits():
    """mp:52 요청 - 엔진이 요청 오류(exit 2)로 끝냄. mp:53 은 같은 요청을 받아 차트를 감쌈"""
    short = run_engine(chart_request("mp:52"))
    assert short.returncode == 2 and "at least 53 bits" in short.stdout
    enough = run_engine(chart_request("mp:53"))
    assert enough.returncode == 0 and "box 0 ok" in enough.stdout


def with_first_endpoints(request: str, token: str) -> str:
    """상자 첫 성분의 두 끝점을 token 으로 바꾼 요청."""
    lines = request.splitlines()
    k = lines.index("boxes 1 4") + 1
    lines[k] = " ".join([token, token] + lines[k].split()[2:])
    return "\n".join(lines) + "\n"


def test_mp_engine_judges_hex_endpoints_at_its_own_precision():
    """MP 실행의 16진 끝점 - 설정 비트로 정확히 나타나면 받고(57비트 토큰을 mp:57·mp:128 에서), 넘치면 거부(mp:56)"""
    token = "0x1.00000000000001p+2"
    for arithmetic, accepted in (("mp:56", False), ("mp:57", True), ("mp:128", True)):
        proc = run_engine(with_first_endpoints(chart_request(arithmetic), token))
        if accepted:
            assert proc.returncode == 0, proc.stdout
        else:
            assert proc.returncode == 2 and f"not exactly an {arithmetic} number" in proc.stdout


def test_engine_refuses_hex_endpoint_that_needs_rounding():
    """57비트 가수의 16진 끝점 - 최근접으로 반올림해 받지 않고 요청 오류로 끝냄(53비트 토큰은 받음)"""
    request = chart_request("double")
    accepted = run_engine(with_first_endpoints(request, "0x1.0000000000001p+2"))
    refused = run_engine(with_first_endpoints(request, "0x1.00000000000001p+2"))
    assert accepted.returncode == 0, accepted.stdout
    assert refused.returncode == 2 and "not exactly a double" in refused.stdout
