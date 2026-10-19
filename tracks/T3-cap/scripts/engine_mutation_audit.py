"""G7 엔진 판정부 mutation 감사: 출력 단면 통과 세기(엔진 CrossingCounter, 함정 ②)와 요청·상 가드(16진 끝점의 정확성,
MP 정밀도 하한, c0 두 반환 창의 일치, 세기 결과 출력)의 조건을 하나씩 바꿔 빌드하고, 명세에서 정한 사례의 판정(감싸기
또는 함정 ② 실패, 반환 시간 범위)이 달라지는지 본다. 바꾼 엔진 소스로 C++ 시험(engine_tests/crossing_counter_test.cpp:
합성 단계열과 각본 적분기, engine_tests/endpoint_guard_test.cpp: 끝점·정밀도 가드)도 함께 빌드해 돌린다.
빌드·실행은 임시 디렉터리에서만 한다.

사례는 「t>0 의 n번째 지정 방향 통과」라는 정의와 8자 궤도의 대칭에서 정했다:
주기 T≈26.128, syzygy 간격 T/6, r12 최대 간격 T/2, 최대 뒤 최소까지 T/4.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from common import TRACK_DIR, sha256_file, write_result
from t3cap import CoordinateSection, Failure, Frame, Interval, Settings, SystemParams
from t3cap.api import _c0
from t3cap.engine import EngineError, build_request, parse_response

F = Fraction
ENGINE_SRC = TRACK_DIR / "engine" / "t3cap_engine.cpp"
UNIT_TESTS = (TRACK_DIR / "engine_tests" / "crossing_counter_test.cpp", TRACK_DIR / "engine_tests" / "endpoint_guard_test.cpp")
# 8자 궤도 Σ+ 통과점 근처: 오일러 배치라 r12 가 최대(p_r≈0)
U = (F(5148567982527431, 10**15), F(0), F(0), F(432690650760826, 10**15))
EYE = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
SP, SM = CoordinateSection.syzygy(+1), CoordinateSection.syzygy(-1)
R12_MAX = CoordinateSection(3, F(0), -1, 5, -1)
R12_MIN = CoordinateSection(3, F(0), +1, 5, +1)


@dataclass(frozen=True)
class Case:
    name: str
    section_out: CoordinateSection
    returns: int
    pr_offset: Fraction
    rho: Fraction
    window: tuple[Fraction, Fraction] | None  # None: 함정 ② 로 실패해야 한다
    step: Fraction | None = None
    order: int = 20


CASES = [
    Case("straddle-to-r12-max", R12_MAX, 1, F(0), F(1, 10**6), None),
    Case("off-section-to-r12-max", R12_MAX, 1, F(-3, 10**6), F(1, 10**6), (F(129, 10), F(132, 10))),
    Case("straddle-to-r12-min", R12_MIN, 1, F(0), F(1, 10**6), (F(64, 10), F(67, 10))),
    Case("before-section-to-r12-max", R12_MAX, 1, F(3, 10**6), F(1, 10**6), (F(0), F(1, 1000))),
    Case("before-section-second-r12-max", R12_MAX, 2, F(3, 10**6), F(1, 10**6), (F(129, 10), F(132, 10))),
    Case("half-segment", SM, 1, F(0), F(1, 10**6), (F(43, 10), F(44, 10))),
    # 큰 상자·고정 소보폭(낮은 차수로 비용을 줄임): 통과 하나가 여러 단계에 걸쳐 닿는 묶음이 된다
    Case("half-segment-multi-step-run", SM, 1, F(0), F(1, 10**4), (F(43, 10), F(44, 10)), F(1, 1000), 8),
    Case("second-S-minus", SM, 2, F(0), F(1, 10**8), (F(129, 10), F(132, 10))),
    Case("period-P3", SP, 3, F(0), F(1, 10**7), (F(260, 10), F(263, 10))),
]

# (원문, 바꾼 글, 동치 근거 또는 None). 원문은 엔진 소스에 정확히 한 번 나와야 한다
MUTANTS = [
    ("if (runSign_ != dir_) throw std::runtime_error(\"crossing: the run at the return time",
     "if (runSign_ == dir_) throw std::runtime_error(\"crossing: the run at the return time", None),
    ("      if (runFromStart_ && onSection_)\n        throw std::runtime_error(\"crossing: return-time window",
     "      if (runFromStart_ || onSection_)\n        throw std::runtime_error(\"crossing: return-time window", None),
    ("if (++windowRuns_ == 1) countedBefore_ = counted_;", "if (++windowRuns_ >= 1) countedBefore_ = counted_;",
     "windowRuns_>=2 이면 finish() 가 countedBefore_ 와 무관하게 실패한다"),
    ("if (++windowRuns_ == 1) countedBefore_ = counted_;", "if (++windowRuns_ == 1) countedBefore_ = 0;", None),
    ("if (runSign_ == dir_) ++counted_;", "if (runSign_ != dir_) ++counted_;", None),
    ("} else if (before_ != after) {", "} else if (before_ == after) {", None),
    ("        if (runSign_ == dir_)\n          throw std::runtime_error(\"crossing: initial set straddles",
     "        if (runSign_ != dir_)\n          throw std::runtime_error(\"crossing: initial set straddles", None),
    ("if (windowRuns_ != 1)", "if (windowRuns_ > 1)",
     "windowRuns_=0 이면 countedBefore_=-1 이 남아 바로 다음 비교(returns-1>=0)가 같은 실패를 낸다"),
    ("if (countedBefore_ != returns_ - 1)", "if (countedBefore_ != returns_)", None),
    ("    const bool onSection = req_.in.k == k &&", "    const bool onSection = false && req_.in.k == k &&", None),
    ("req_.cIn.leftBound() == req_.cOut.leftBound()", "req_.cIn.leftBound() != req_.cOut.leftBound()", None),
    ("      runFromStart_ = first_;", "      runFromStart_ = false;", None),
    ("    } else if (sign != runSign_) {", "    } else if (sign == runSign_) {", None),
    # 엔진 경로에서는 닿을 수 없는 분기지만(이웃 단계 감싸기가 경계 시각의 같은 상태를 담음) 합성 단계열 시험은 닿는다
    ("    } else if (sign != runSign_) {\n"
     "      throw std::runtime_error(\"crossing: ds/dt changes sign while touching the output section (trap 2)\");\n"
     "    }\n", "    }\n", None),
    ("    const bool inWindow = !(runStart_.leftBound() > window_.rightBound()) &&",
     "    const bool inWindow = !(runStart_.leftBound() > window_.rightBound()) ||", None),
    ("        if (after != runSign_) throw std::runtime_error(\"crossing: sign change disagrees",
     "        if (after == runSign_) throw std::runtime_error(\"crossing: sign change disagrees", None),
    ("        if (after != runSign_) throw std::runtime_error(\"crossing: departure from the section",
     "        if (after == runSign_) throw std::runtime_error(\"crossing: departure from the section", None),
    ("    crossings.finish();\n", "", None),
    ("    if (runSign_ != 0) closeRun(0);\n", "", None),
    ("      if (runSign_ != 0) closeRun(sign);", "      if (runSign_ == 0) closeRun(sign);", None),
    ("      before_ = sign;\n      first_ = false;\n      return;", "      first_ = false;\n      return;", None),
    ("    if (!(dsdt.leftBound() > 0 || dsdt.rightBound() < 0))", "    if (false)", None),
    ("s0.leftBound() > 0 ? 1 : (s0.rightBound() < 0 ? -1 : 0), onSection);", "0, onSection);", None),
    ("crossings.step(enclosure[k] - req_.cOut, f(enclosure)[k],", "crossings.step(enclosure[k] - req_.cOut, -f(enclosure)[k],",
     None),
    ("    runEnd_ = t1;", "    runEnd_ = t0;", None),
    ("      runStart_ = t0;", "      runStart_ = t1;", None),
    # 요청·상 가드와 세기 결과 출력
    ("    if (precisionBits < 53)\n", "    if (precisionBits < 52)\n", None),
    ("    if (h.oddBits > 53 || h.lowest < -1074", "    if (h.oddBits > 54 || h.lowest < -1074", None),
    ("h.lowest < -1074 ||", "h.lowest < -1075 ||", None),
    ("h.lowest + h.oddBits - 1 > 1023)", "h.lowest + h.oddBits - 1 > 1024)", None),
    ("    if (h.oddBits > bits())\n", "    if (h.oddBits > bits() + 1)\n", None),
    ("    if (h.oddBits > bits())\n", "    if (h.oddBits >= bits())\n", None),
    ("    if (!(returnTime2.leftBound() == returnTime.leftBound() && returnTime2.rightBound()",
     "    if (!(returnTime2.leftBound() == returnTime.leftBound() || returnTime2.rightBound()", None),
    ("      throw std::runtime_error(\"c0: affine and hull return times differ",
     "      (void)std::runtime_error(\"c0: affine and hull return times differ", None),
    ("      if (windowRuns_ == 1) windowRun_ = A::hull(runStart_, runEnd_);",
     "      if (windowRuns_ == 1) windowRun_ = A::hull(runEnd_, runEnd_);", None),
    ("      ++runsBefore_;\n", "", None),
    ("<< crossings.countedBefore() <<", "<< 0 <<", None),
]


def request(case: Case) -> str:
    u = (U[0], U[1], U[2] + case.pr_offset, U[3])
    out_center = (u[0], u[1], F(0), u[3]) if case.section_out.coord == 3 else u
    box = tuple(Interval(-case.rho, case.rho) for _ in range(4))
    return build_request("c0", [box], system=SystemParams.make(), section_in=SP, section_out=case.section_out,
                         returns=case.returns, frame_in=Frame.make(u, EYE), frame_out=Frame.make(out_center, EYE),
                         settings=Settings(order=case.order, step=case.step), rotation=False)


def as_expected(binary: Path, case: Case) -> bool:
    proc = subprocess.run([str(binary)], input=request(case), capture_output=True, text=True, timeout=900)
    try:
        result = _c0(parse_response(proc.stdout, 1)[0])
    except EngineError:
        return False  # 바이너리가 깨진 응답을 내면 그 자체로 검출이다
    if case.window is None:
        return isinstance(result, Failure) and "(trap 2)" in result.reason
    lo, hi = case.window
    return not isinstance(result, Failure) and lo < result.return_time.lo and result.return_time.hi < hi


def build_dir(src: Path, out: Path) -> None:
    env = dict(os.environ, ENGINE_SRC_DIR=str(src), ENGINE_OUT_DIR=str(out))
    subprocess.run(["bash", str(TRACK_DIR / "build_engine.sh"), "native-strict"], env=env, check=True, capture_output=True,
                   text=True)


def build(source: str, workdir: Path) -> Path:
    """시험 소스가 ../engine/t3cap_engine.cpp 로 포함하도록 바꾼 소스를 workdir/engine 에 둔다."""
    src = workdir / "engine"
    src.mkdir()
    (src / ENGINE_SRC.name).write_text(source)
    build_dir(src, workdir / "out")
    return workdir / "out" / "t3cap_engine"


def unit_failures(workdir: Path) -> list[str]:
    """workdir/engine 의 (바꾼) 엔진 소스로 C++ 시험들을 빌드해 돌리고, 기대와 다르게 나온 시험 사례 이름을 돌려준다."""
    tests = workdir / "engine_tests"
    tests.mkdir()
    for test in UNIT_TESTS:
        (tests / test.name).write_text(test.read_text())
    build_dir(tests, workdir / "out_tests")
    failed = []
    for test in UNIT_TESTS:
        proc = subprocess.run([str(workdir / "out_tests" / test.stem)], capture_output=True, text=True, timeout=900)
        names = [line.removeprefix("FAIL ").split(" | ")[0] for line in proc.stdout.splitlines() if line.startswith("FAIL ")]
        failed += names if names or proc.returncode == 0 else [f"<{test.stem} crashed>"]
    return ["unit: " + name for name in failed]


def evaluate(source: str) -> list[str]:
    """기대와 다르게 나온 사례 이름들(빈 목록 = 모든 사례가 명세대로)."""
    with tempfile.TemporaryDirectory(prefix="t3cap-engine-mutant-") as tmp:
        try:
            binary = build(source, Path(tmp))
            unit = unit_failures(Path(tmp))
        except subprocess.CalledProcessError:
            return ["<build failed>"]
        return [c.name for c in CASES if not as_expected(binary, c)] + unit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)  # 트랙 CPU 예산(D36: worker 2 이하)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    original = ENGINE_SRC.read_text()
    for old, _, _ in MUTANTS:
        if original.count(old) != 1:
            raise SystemExit(f"mutant anchor must occur exactly once: {old[:60]!r}")
    baseline = evaluate(original)
    if baseline:
        print("baseline engine does not meet the specification:", baseline)
        return 1
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = list(pool.map(lambda m: evaluate(original.replace(m[0], m[1])), MUTANTS))
    rows = [{"original": old, "mutant": new, "equivalent": why, "killed_by": failed}
            for (old, new, why), failed in zip(MUTANTS, outcomes)]
    scored = [r for r in rows if r["equivalent"] is None]
    killed = sum(1 for r in scored if r["killed_by"])
    # 동치라고 적은 mutant 가 잡히면 그 근거가 틀린 것이다
    wrong_equivalence = [r["mutant"] for r in rows if r["equivalent"] is not None and r["killed_by"]]
    for r in rows:
        status = "KILLED" if r["killed_by"] else ("EQUIVALENT" if r["equivalent"] else "SURVIVED")
        print(status, repr(r["mutant"][:70]), r["killed_by"][:3])
    print(f"engine mutation: killed {killed}/{len(scored)} non-equivalent ({len(rows)} mutants, "
          f"{len(rows) - len(scored)} declared equivalent), wrong equivalence claims {len(wrong_equivalence)}")
    if not args.no_write:
        write_result("engine_mutation_audit",
                     {"tag": "G7 엔진 판정부(출력 단면 통과 세기·요청과 상 가드) mutation 감사", "cases": [c.name for c in CASES],
                      "unit_tests": [str(test.relative_to(TRACK_DIR)) for test in UNIT_TESTS],
                      "mutants": len(rows), "equivalent": len(rows) - len(scored), "scored": len(scored),
                      "killed": killed, "wrong_equivalence": wrong_equivalence, "rows": rows},
                     script=Path(__file__).resolve(), started=started,
                     params={"workers": args.workers,
                             "extra_code_sha256": {rel: sha256_file(TRACK_DIR / rel) for rel in
                                                   ("build_engine.sh", *(str(t.relative_to(TRACK_DIR)) for t in UNIT_TESTS))}})
    return 0 if killed / len(scored) >= 0.9 and not wrong_equivalence else 1


if __name__ == "__main__":
    sys.exit(main())
