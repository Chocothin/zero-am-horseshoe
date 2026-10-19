"""G5 CAP 건전성(PBT): 입력 상자 안 무작위 점을 mpmath(32자리) 표준 8차원 Jacobi 적분으로 보내 출력 감싸기 안에 드는지 본다.

하나라도 벗어나면 Critical 이다(종료 코드 1). 질량·J·h 가 구간이면 그 값도 무작위로 뽑는다(T7 용도).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import mpmath as mp

from common import RESULTS_DIR, write_result
from reference import mp_reference as ref
from t3cap import CoordinateSection, Failure, Frame, Interval, Settings, SystemParams, poincare_c0, poincare_c1
from t3cap.exact import Box

DPS = 32
FD_STEP = mp.mpf("1e-10")


@dataclass(frozen=True)
class Config:
    name: str
    mode: str                  # "c0" | "c1"
    system: SystemParams
    section_in: CoordinateSection
    section_out: CoordinateSection
    returns: int
    frame_in: Frame
    frame_out: Frame
    box: Box
    settings: Settings


def figure8_configs() -> list[Config]:
    data = json.loads((RESULTS_DIR / "figure8_newton.json").read_text())
    c0, c1 = (tuple(Fraction(float(v)) for v in p) for p in data["shooting_points"][:2])
    eye = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    tilt = ((1, Fraction(1, 3), 0, Fraction(-1, 7)), (Fraction(1, 5), 1, Fraction(1, 11), 0),
            (0, Fraction(-1, 4), 1, Fraction(1, 9)), (Fraction(1, 6), 0, Fraction(-1, 8), 1))
    sp, sm = CoordinateSection.syzygy(+1), CoordinateSection.syzygy(-1)
    system = SystemParams.make()
    loose = SystemParams.make(
        masses=[Interval(Fraction(1) - Fraction(1, 10**6), Fraction(1) + Fraction(1, 10**6))] * 3,
        energy=Interval(Fraction(-1, 2) - Fraction(1, 10**7), Fraction(-1, 2) + Fraction(1, 10**7)),
        angular_momentum=Interval(Fraction(-1, 10**6), Fraction(1, 10**6)))

    def cube(r: Fraction) -> Box:
        return tuple(Interval(-r, r) for _ in range(4))

    return [
        Config("f8-period-c0", "c0", system, sp, sp, 3, Frame.make(c0, eye), Frame.make(c0, eye),
               cube(Fraction(1, 10**7)), Settings()),
        Config("f8-half-c1", "c1", system, sp, sm, 1, Frame.make(c0, eye), Frame.make(c1, eye),
               cube(Fraction(1, 10**6)), Settings()),
        Config("f8-tilted-frames-c1", "c1", system, sp, sm, 1, Frame.make(c0, tilt), Frame.make(c1, tilt),
               cube(Fraction(1, 10**6)), Settings()),
        Config("f8-interval-params-c0", "c0", loose, sp, sm, 1, Frame.make(c0, eye), Frame.make(c1, eye),
               cube(Fraction(1, 10**6)), Settings()),
        Config("f8-half-c0-x86filib", "c0", system, sp, sm, 1, Frame.make(c0, eye), Frame.make(c1, eye),
               cube(Fraction(1, 10**5)), Settings(build="x86filib-strict")),
        # 일반 좌표 단면: syzygy 에서 r12 최대(pr=0, 감소 방향)로, 도착 차트는 py 를 branch -1 로 푼다.
        # 8자 사격점은 pr≈0 이라 상자를 pr<0 쪽으로 옮긴다 — 단면에 걸친 출발은 엔진이 거부한다(함정 ②, test_api_contract)
        Config("f8-off-section-to-pr-max-c1", "c1", system, sp, CoordinateSection(3, Fraction(0), -1, 5, -1), 1,
               Frame.make((c0[0], c0[1], c0[2] - Fraction(3, 10**6), c0[3]), eye),
               Frame.make((c0[0], c0[1], Fraction(0), c0[3]), eye), cube(Fraction(1, 10**6)), Settings()),
    ] + butterfly_configs(system, cube)


def butterfly_configs(system: SystemParams, cube) -> list[Config]:
    """HE 궤도(butterfly I) 첫 반환 — 흐름 관 하한이 8자보다 작은 구간(약 0.135)에서 시험한다.

    근접 조우(r 약 0.047)를 지나는 한 주기는 MPFR 96비트·rho=1e-9 에서도 엔진이 비횡단 반환으로 실패해 넣지 못했다.
    """
    path = RESULTS_DIR / "orbit_butterfly1.json"
    if not path.exists():
        return []
    orb = json.loads(path.read_text())
    u = tuple(Fraction(float(v)) for v in orb["fixed_point_u"])
    sec = CoordinateSection.syzygy(orb["direction"])
    eye = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    frame = Frame.make(u, eye)
    return [Config("bf1-first-return-c0", "c0", system, sec, sec, 1, frame, frame, cube(Fraction(1, 10**8)),
                   Settings(order=30))]


def random_in(iv: Interval, rng: random.Random) -> Fraction:
    return iv.lo + iv.width * Fraction(rng.getrandbits(60), 1 << 60)


def frame_map(frame: Frame, w):
    return [frame.center[i] + sum(frame.matrix[i][j] * w[j] for j in range(4)) for i in range(4)]


def reference_map(cfg: Config, w, masses, h, J):
    """기준 해로 w -> w' (도착 틀 좌표). 정확 유리수 입력, mp 출력."""
    mp.mp.dps = DPS
    md = ref.MassData.of(masses)
    u = frame_map(cfg.frame_in, w)
    s_in, s_out = cfg.section_in, cfg.section_out
    sec = lambda s: ref.SectionData(s.coord, s.value, s.direction, s.solved_momentum, s.branch)
    out, crossing = ref.reduced_map(u, section_in=sec(s_in), section_out=sec(s_out), returns=cfg.returns, md=md,
                                    h=h, J=J, order=30)
    inv = cfg.frame_out.inverse()
    d = [out[i] - ref.to_mpf(cfg.frame_out.center[i]) for i in range(4)]
    return [sum(ref.to_mpf(inv[i][j]) * d[j] for j in range(4)) for i in range(4)], crossing


def mp_to_fraction(x: mp.mpf) -> Fraction:
    """mpf 의 정확한 이진 값(부호 포함)."""
    sign, man, exp, _ = mp.mpf(x)._mpf_
    value = Fraction(int(man)) * (Fraction(2) ** int(exp))
    return -value if sign else value


def check_point(task) -> dict:
    cfg, enclosure, seed = task
    rng = random.Random(seed)
    w = [random_in(iv, rng) for iv in cfg.box]
    masses = [random_in(m, rng) for m in cfg.system.masses]
    h = random_in(cfg.system.energy, rng)
    J = random_in(cfg.system.angular_momentum, rng)
    image, crossing = reference_map(cfg, w, masses, h, J)
    violations = []
    margins = []
    for i, (val, iv) in enumerate(zip(image, enclosure["image"])):
        q = mp_to_fraction(val)
        if not iv.contains(q):
            violations.append(f"image[{i}]={mp.nstr(val, 20)} outside {iv}")
        margins.append(float(min(q - iv.lo, iv.hi - q) / iv.width) if iv.width else 0.0)
    t = mp_to_fraction(crossing.time)
    if not enclosure["return_time"].contains(t):
        violations.append(f"return time {mp.nstr(crossing.time, 20)} outside {enclosure['return_time']}")
    tube = enclosure["tube"]
    for name, sampled_min, sampled_max, iv in zip(("r12", "r13", "r23"), crossing.min_distances,
                                                  crossing.max_distances, (tube.r12, tube.r13, tube.r23)):
        if mp_to_fraction(sampled_min) < iv.lo or mp_to_fraction(sampled_max) > iv.hi:
            violations.append(f"tube {name} sampled [{mp.nstr(sampled_min, 12)}, {mp.nstr(sampled_max, 12)}] vs {iv}")
    if cfg.mode == "c1":
        for j in range(4):
            plus = [w[k] + (mp_to_fraction(FD_STEP) if k == j else 0) for k in range(4)]
            minus = [w[k] - (mp_to_fraction(FD_STEP) if k == j else 0) for k in range(4)]
            ip, _ = reference_map(cfg, plus, masses, h, J)
            im, _ = reference_map(cfg, minus, masses, h, J)
            for i in range(4):
                dval = (ip[i] - im[i]) / (2 * FD_STEP)
                iv = enclosure["derivative"][i][j]
                if not iv.contains(mp_to_fraction(dval)):
                    violations.append(f"D[{i}][{j}]={mp.nstr(dval, 15)} outside {iv}")
    return {"seed": seed, "violations": violations, "min_margin": min(margins),
            "image": [str(mp_to_fraction(v)) for v in image]}


def shrunk(iv: Interval, keep: Fraction) -> Interval:
    """음성 대조용: 감싸기를 가운데 keep 비율만 남긴다. 건전한 시험기는 여기서 위반을 잡아야 한다."""
    half = iv.width * keep / 2
    return Interval(iv.mid - half, iv.mid + half)


def enclosure_of(cfg: Config) -> dict:
    fn = poincare_c1 if cfg.mode == "c1" else poincare_c0
    res = fn([cfg.box], system=cfg.system, section_in=cfg.section_in, section_out=cfg.section_out,
             returns=cfg.returns, frame_in=cfg.frame_in, frame_out=cfg.frame_out, settings=cfg.settings)[0]
    if isinstance(res, Failure):
        raise SystemExit(f"{cfg.name}: engine failure {res.reason}")
    out = {"image": res.image, "return_time": res.return_time, "tube": res.tube}
    if cfg.mode == "c1":
        out["derivative"] = res.derivative
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=3)  # 트랙 CPU 예산(BRIEF rev3)
    parser.add_argument("--only", default="")
    parser.add_argument("--no-write", action="store_true", help="results/ 에 쓰지 않는다")
    parser.add_argument("--quick", action="store_true", help="verify.sh 빠른 모드: 설정당 12점, 쓰지 않음")
    args = parser.parse_args()
    if args.quick:
        args.points, args.no_write = 12, True
    started = time.time()
    summary = []
    configs = [c for c in figure8_configs() if args.only in c.name]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for cfg in configs:
            t0 = time.time()
            enc = enclosure_of(cfg)
            results = list(pool.map(check_point, [(cfg, enc, 1000 * k + 7) for k in range(args.points)]))
            bad = [r for r in results if r["violations"]]
            narrow = [shrunk(iv, Fraction(1, 5)) for iv in enc["image"]]
            tightness = sum(1 for r in results
                            if any(not iv.contains(Fraction(v)) for iv, v in zip(narrow, r["image"])))
            moved = [Interval(iv.lo + iv.width, iv.hi + iv.width) for iv in enc["image"]]
            detected = sum(1 for r in results
                           if any(not iv.contains(Fraction(v)) for iv, v in zip(moved, r["image"])))
            # 제 폭만큼 옮긴(틀린) 감싸기를 통과시키는 시험기는 믿을 수 없다 — 그 경우도 Critical 로 센다
            blind = detected != len(results)
            entry = {"config": cfg.name, "mode": cfg.mode, "points": len(results), "violations": len(bad),
                     "min_relative_margin": min(r["min_margin"] for r in results),
                     "max_image_width": max(float(iv.width) for iv in enc["image"]),
                     "negative_control_shifted_by_width_detected": detected,
                     "tightness_points_outside_central_20pct": tightness,
                     "tester_blind": blind, "examples": bad[:5], "seconds": round(time.time() - t0, 1)}
            summary.append(entry)
            print(json.dumps({k: entry[k] for k in ("config", "points", "violations", "min_relative_margin",
                                                    "negative_control_shifted_by_width_detected",
                                                    "tightness_points_outside_central_20pct", "seconds")}),
                  flush=True)
    critical = sum(entry["violations"] + int(entry["tester_blind"]) for entry in summary)
    if not args.no_write:
        write_result("g5_soundness", {"tag": "G5 [CAP 건전성 시험]", "dps": DPS, "fd_step": str(FD_STEP),
                                      "configs": summary, "critical": critical},
                     script=Path(__file__).resolve(), params={"points": args.points, "dps": DPS}, started=started)
    return 1 if critical else 0


if __name__ == "__main__":
    sys.exit(main())
