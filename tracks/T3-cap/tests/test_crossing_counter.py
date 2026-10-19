"""엔진 출력 단면 통과 세기(CrossingCounter, 함정 ②)의 C++ 시험을 빌드해 돌린다.

사례는 「t>0 의 returns 번째 지정 방향 통과」라는 정의에서 정했다(engine_tests/crossing_counter_test.cpp).
빌드는 본 엔진 빌드와 섞이지 않게 build/engine-tests-native-strict 에만 한다.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

TRACK = Path(__file__).resolve().parent.parent


def test_crossing_counter_rules_hold_for_synthetic_step_sequences():
    out = TRACK / "build" / "engine-tests-native-strict"
    env = dict(os.environ, ENGINE_SRC_DIR=str(TRACK / "engine_tests"), ENGINE_OUT_DIR=str(out))
    subprocess.run(["bash", str(TRACK / "build_engine.sh"), "native-strict"], env=env, check=True, capture_output=True,
                   text=True)
    proc = subprocess.run([str(out / "crossing_counter_test")], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
