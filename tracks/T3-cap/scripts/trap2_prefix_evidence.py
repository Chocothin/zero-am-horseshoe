"""수정 전 엔진의 함정 ② 구멍 재현(G5 Critical 증거). 출력 단면 통과 세기만 끈 엔진은 수정 전 엔진과 출력이 같다.
그 엔진으로 출력 단면에 걸친 상자의 C1 감싸기를 만들고, G5 와 같은 시드·기준 해로 점마다 대조한다.
수정 후 엔진은 같은 요청을 Failure(함정 ②)로 거부해야 한다. 빌드는 임시 디렉터리에서만 한다.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import mpmath as mp

import g5_soundness as g5
from common import RESULTS_DIR, TRACK_DIR, sha256_file, write_result
from engine_mutation_audit import ENGINE_SRC, build
from t3cap import CoordinateSection, Failure, Frame, Interval, Settings, SystemParams, poincare_c1
from t3cap.api import _c1
from t3cap.engine import build_request, parse_response

# 흐름 관 단계의 통과 세기 호출 두 줄만 지운다(CrossingCounter 는 예외만 던지므로 나머지 출력은 그대로다)
COUNTER_OFF = [("      crossings.step(enclosure[k] - req_.cOut, f(enclosure)[k], t0, set.getCurrentTime());\n", ""),
               ("    crossings.finish();\n", "")]
R12_MAX = CoordinateSection(3, Fraction(0), -1, 5, -1)
REPRESENTATIVES = 5


def prefix_config() -> g5.Config:
    """수정 전 G5 의 f8-to-pr-max-section-c1 과 같은 설정(Σ+ 위 상자가 p_r=0 에 걸침)."""
    data = json.loads((RESULTS_DIR / "figure8_newton.json").read_text())
    c0 = tuple(Fraction(float(v)) for v in data["shooting_points"][0])
    eye = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    box = tuple(Interval(Fraction(-1, 10**6), Fraction(1, 10**6)) for _ in range(4))
    return g5.Config("f8-to-pr-max-section-c1", "c1", SystemParams.make(), CoordinateSection.syzygy(+1), R12_MAX, 1,
                     Frame.make(c0, eye), Frame.make((c0[0], c0[1], Fraction(0), c0[3]), eye), box, Settings())


def counter_off_enclosure(cfg: g5.Config) -> dict:
    source = ENGINE_SRC.read_text()
    for old, new in COUNTER_OFF:
        if source.count(old) != 1:
            raise SystemExit(f"anchor must occur exactly once: {old[:60]!r}")
        source = source.replace(old, new)
    request = build_request("c1", [cfg.box], system=cfg.system, section_in=cfg.section_in, section_out=cfg.section_out,
                            returns=cfg.returns, frame_in=cfg.frame_in, frame_out=cfg.frame_out, settings=cfg.settings,
                            rotation=False)
    with tempfile.TemporaryDirectory(prefix="t3cap-prefix-") as tmp:
        binary = build(source, Path(tmp))
        proc = subprocess.run([str(binary)], input=request, capture_output=True, text=True, timeout=900)
    result = _c1(parse_response(proc.stdout, 1)[0])
    if isinstance(result, Failure):
        raise SystemExit(f"counter-off engine failed: {result.reason}")
    return {"image": result.image, "return_time": result.return_time, "tube": result.tube,
            "derivative": result.derivative}


def sample(cfg: g5.Config, seed: int):
    """check_point 과 같은 순서로 표본(w, 질량, h, J)을 뽑는다."""
    rng = random.Random(seed)
    w = [g5.random_in(iv, rng) for iv in cfg.box]
    masses = [g5.random_in(m, rng) for m in cfg.system.masses]
    return w, masses, g5.random_in(cfg.system.energy, rng), g5.random_in(cfg.system.angular_momentum, rng)


def start_pr(cfg: g5.Config, seed: int) -> Fraction:
    return cfg.frame_in.center[2] + sample(cfg, seed)[0][2]


def reference_time(cfg: g5.Config, seed: int) -> mp.mpf:
    _, crossing = g5.reference_map(cfg, *sample(cfg, seed))
    return crossing.time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=3)  # 트랙 CPU 예산(BRIEF rev3)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    cfg = prefix_config()
    fixed = poincare_c1([cfg.box], system=cfg.system, section_in=cfg.section_in, section_out=cfg.section_out,
                        returns=cfg.returns, frame_in=cfg.frame_in, frame_out=cfg.frame_out, settings=cfg.settings)[0]
    refused = isinstance(fixed, Failure) and "(trap 2)" in fixed.reason
    enc = counter_off_enclosure(cfg)
    seeds = [1000 * k + 7 for k in range(args.points)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(g5.check_point, [(cfg, enc, s) for s in seeds]))
    bad = [r for r in results if r["violations"]]
    good = [r for r in results if not r["violations"]]
    picks = [r["seed"] for r in bad[:REPRESENTATIVES]] + [r["seed"] for r in good[:2]]
    representatives = [{"seed": seed, "start_pr": float(start_pr(cfg, seed)),
                        "reference_return_time": mp.nstr(reference_time(cfg, seed), 20),
                        "violations": next(r["violations"] for r in results if r["seed"] == seed)[:3]}
                       for seed in picks]
    summary = {"config": cfg.name, "points": len(results), "violations": len(bad),
               "violating_with_start_pr_positive": sum(1 for r in bad if start_pr(cfg, r["seed"]) > 0),
               "clean_with_start_pr_negative": sum(1 for r in good if start_pr(cfg, r["seed"]) < 0),
               "enclosure_return_time": [float(enc["return_time"].lo), float(enc["return_time"].hi)],
               "fixed_engine_refuses": refused, "fixed_engine_reason": fixed.reason if refused else None}
    print(json.dumps(summary), flush=True)
    for rep in representatives:
        print(json.dumps(rep), flush=True)
    if not args.no_write:
        write_result("g5_prefix_trap2",
                     {"tag": "수정 전 함정 ② 구멍 재현 [CAP 시험: 통과 세기를 끈 엔진 + mpmath 기준 해]",
                      "summary": summary, "representatives": representatives,
                      "original_run": "수정 전 최종 G5(PID 15803) 로그: raw-local/g5_prefix_run.log — 1000점 중 491점 위반"},
                     script=Path(__file__).resolve(), started=started,
                     params={"points": args.points, "workers": args.workers,
                             "extra_code_sha256": {rel: sha256_file(TRACK_DIR / rel) for rel in
                                                   ("build_engine.sh", "scripts/g5_soundness.py",
                                                    "scripts/engine_mutation_audit.py")}})
    return 0 if refused and bad else 1


if __name__ == "__main__":
    sys.exit(main())
