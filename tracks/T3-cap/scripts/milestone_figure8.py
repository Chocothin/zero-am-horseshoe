"""[CAP] 이정표: h=-1/2, J=0, 등질량 평면 3체의 축소 Poincaré 사상(syzygy 단면)에 8자 궤도 고정점이 유일하게 있음을 증명한다.

방법(Kapela–Simó 2011 §4.5, Remark 1): syzygy 통과마다 끊은 반 구간 6개로 다중 사격하고, 24차원 Krawczyk 연산자를
정확 유리수 구간 산술로 판정한다. F(c)·DF(X) 감싸기는 CAPD 엔진이 내고 certify_return 으로 사용 자격을 확인한다.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np

from common import RESULTS_DIR, write_result
from t3cap import CoordinateSection, Failure, Frame, Interval, Settings, SystemParams, poincare_c0, poincare_c1
from t3cap.exact import rational_matrix_from_floats
from t3cap.export import to_json
from t3cap.verdict import certify_return, krawczyk

SEGMENTS = 6
MIN_DISTANCE = Fraction(1, 10)  # 흐름 관 r_ij 하한이 넘어야 하는 값(충돌 배제의 여유)
MAX_ENGINE_PROCESSES = 3
SECTIONS = [CoordinateSection.syzygy(+1) if k % 2 == 0 else CoordinateSection.syzygy(-1) for k in range(SEGMENTS)]


def load_centers() -> list[tuple[Fraction, ...]]:
    """[수치] Newton 사격점을 가장 가까운 double 로 반올림해 정확 유리수 중심으로 쓴다(중심은 아무 점이어도 결론이 엄밀)."""
    data = json.loads((RESULTS_DIR / "figure8_newton.json").read_text())
    return [tuple(Fraction(float(v)) for v in point) for point in data["shooting_points"]]


def run_case(name: str, centers, rho: Fraction, settings: Settings, system: SystemParams,
             shift: Fraction = Fraction(0), target_scale: Fraction = Fraction(1),
             min_distance: Fraction = MIN_DISTANCE) -> dict:
    """한 설정으로 Krawczyk 를 판정한다. shift 는 중심 0 의 첫 성분을 옮긴 음성 대조, target_scale<1 은 부등식 강화 대조."""
    started = time.time()
    centers = [tuple(c[i] + (shift if (k == 0 and i == 0) else 0) for i in range(4)) for k, c in enumerate(centers)]
    frames = [Frame.make(c, ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))) for c in centers]
    point = tuple(Interval.point(0) for _ in range(4))
    cube = tuple(Interval(-rho, rho) for _ in range(4))
    def segment_kw(k: int) -> dict:
        return dict(system=system, section_in=SECTIONS[k], section_out=SECTIONS[(k + 1) % SEGMENTS], returns=1,
                    frame_in=frames[k], frame_out=frames[(k + 1) % SEGMENTS], settings=settings)

    # 트랙 동시 계산 예산(BRIEF rev3: worker 3 이하)
    with ThreadPoolExecutor(max_workers=MAX_ENGINE_PROCESSES) as pool:
        f0 = [pool.submit(poincare_c0, [point], **segment_kw(k)) for k in range(SEGMENTS)]
        f1 = [pool.submit(poincare_c1, [cube], **segment_kw(k)) for k in range(SEGMENTS)]
        c0 = [f.result()[0] for f in f0]
        c1 = [f.result()[0] for f in f1]
    failures = [f"segment {k} {kind}: {r.reason}" for k, (a, b) in enumerate(zip(c0, c1))
                for kind, r in (("c0", a), ("c1", b)) if isinstance(r, Failure)]
    if failures:
        return {"case": name, "proven": False, "reason": "engine failure", "failures": failures,
                "seconds": round(time.time() - started, 2)}
    certificates = [certify_return(r, min_distance=min_distance) for r in c0 + c1]
    n = 4 * SEGMENTS
    f_center = tuple(iv for r in c0 for iv in r.image)
    df = [[Interval.point(0) for _ in range(n)] for _ in range(n)]
    for k in range(SEGMENTS):
        nxt = (k + 1) % SEGMENTS
        for i in range(4):
            for j in range(4):
                df[4 * k + i][4 * k + j] = c1[k].derivative[i][j]
            df[4 * k + i][4 * nxt + i] = df[4 * k + i][4 * nxt + i] - Interval.point(1)
    df_t = tuple(tuple(row) for row in df)
    mid = np.array([[float(e.mid) for e in row] for row in df_t])
    preconditioner = rational_matrix_from_floats(np.linalg.inv(mid))
    box = tuple(Interval(-rho, rho) for _ in range(n))
    target = tuple(Interval(-rho * target_scale, rho * target_scale) for _ in range(n))
    kv = krawczyk(tuple(Fraction(0) for _ in range(n)), box, f_center, df_t, preconditioner)
    inside_target = all(t.lo < k_.lo and k_.hi < t.hi for k_, t in zip(kv.image, target))
    certified = all(c.proven for c in certificates)
    proven = kv.proven and certified and inside_target
    k_radius = max(max(abs(i.lo), abs(i.hi)) for i in kv.image)
    deriv_width = max(float(e.width) for r in c1 for row in r.derivative for e in row)
    return {
        "case": name,
        "proven": proven,
        "krawczyk_in_interior": kv.proven,
        "inside_target": inside_target,
        "all_returns_certified": certified,
        "failed_checks": [f"{k}:{c.name}" for k, cert in enumerate(certificates) for c in cert.failed],
        "rho": str(rho),
        "k_radius_over_rho": float(k_radius / rho),
        "max_derivative_width": deriv_width,
        "max_f_center_width": max(float(i.width) for i in f_center),
        "max_f_center_magnitude": max(float(i.magnitude()) for i in f_center),
        "tube_min_distance_lower": min(float(r.tube.min_distance_lower) for r in c0 + c1),
        "return_times": [to_json(r.return_time) for r in c0],
        "fixed_point_enclosure_u0": [to_json(Interval(c + i.lo, c + i.hi)) for c, i in zip(centers[0], kv.image[:4])],
        "engine_seconds": round(sum(r.seconds for r in c0 + c1), 2),
        # arb 독립 재계산(scripts/arb_krawczyk.py)의 입력: CAPD 가 낸 감싸기를 정확 끝점으로 그대로 남긴다
        "krawczyk_inputs": {"rho": str(rho), "f_center": to_json(f_center),
                            "derivative_blocks": [to_json(r.derivative) for r in c1],
                            "certify": [{"discriminant": to_json(r.discriminant), "branch_out": to_json(r.branch_out),
                                         "return_time": to_json(r.return_time), "tube_r12": to_json(r.tube.r12),
                                         "tube_r13": to_json(r.tube.r13), "tube_r23": to_json(r.tube.r23),
                                         "tube_t_end": to_json(r.tube.t_end)} for r in c0 + c1],
                            "min_distance": str(min_distance), "target_scale": str(target_scale),
                            "centers": [[str(q) for q in c] for c in centers],
                            "masses": [str(m.lo) for m in system.masses], "energy": str(system.energy.lo),
                            "angular_momentum": str(system.angular_momentum.lo),
                            "preconditioner_used": [[str(q) for q in row] for row in preconditioner]},
        "seconds": round(time.time() - started, 2),
        "settings": {"order": settings.order, "step": str(settings.step), "arithmetic": settings.arithmetic,
                     "build": settings.build},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="본 증명 1건과 음성 대조 2건만")
    parser.add_argument("--no-write", action="store_true", help="results/ 에 쓰지 않는다(verify.sh 빠른 모드)")
    args = parser.parse_args()
    started = time.time()
    centers = load_centers()
    system = SystemParams.make()
    rho = Fraction(1, 10**9)
    base = Settings(order=20, workers=1)
    cases = [("main: order20 double native rho=1e-9", centers, rho, base, system, Fraction(0), Fraction(1)),
             ("neg: center shifted by 5 rho", centers, rho, base, system, 5 * rho, Fraction(1)),
             ("neg: target shrunk to rho/100", centers, rho, base, system, Fraction(0), Fraction(1, 100))]
    if not args.quick:
        cases += [
            ("neg: energy -1/2+1/1000", centers, rho, base, SystemParams.make(energy=Fraction(-499, 1000)),
             Fraction(0), Fraction(1)),
            ("neg: rho too small 1e-14", centers, Fraction(1, 10**14), base, system, Fraction(0), Fraction(1)),
            ("neg: collision margin 2 > tube min", centers, rho, base, system, Fraction(0), Fraction(1),
             Fraction(2)),
            ("g8: order 16", centers, rho, Settings(order=16), system, Fraction(0), Fraction(1)),
            ("g8: order 28", centers, rho, Settings(order=28), system, Fraction(0), Fraction(1)),
            ("g8: fixed step 1/20", centers, rho, Settings(order=20, step=Fraction(1, 20)), system, Fraction(0),
             Fraction(1)),
            ("g8: x86_64 filib (Rosetta 2)", centers, rho, Settings(order=20, build="x86filib-strict"), system, Fraction(0),
             Fraction(1)),
            ("g8: rho=1e-8", centers, Fraction(1, 10**8), base, system, Fraction(0), Fraction(1)),
            ("g8: mp:128 order 30", centers, rho, Settings(order=30, arithmetic="mp:128"), system, Fraction(0),
             Fraction(1)),
        ]
    results = []
    for case in cases:
        res = run_case(*case)
        print(json.dumps({k: res[k] for k in res if k in ("case", "proven", "k_radius_over_rho",
                                                            "max_derivative_width", "seconds", "reason")}),
              flush=True)
        results.append(res)
    payload = {"tag": "[CAP] 정확 유리수 Krawczyk + CAPD 구간 감싸기", "segments": SEGMENTS,
               "min_distance": str(MIN_DISTANCE), "cases": results}
    if not args.no_write:
        write_result("milestone_figure8" + ("_quick" if args.quick else ""), payload, script=Path(__file__).resolve(),
                     params={"quick": args.quick, "rho": str(rho), "energy": "-1/2", "J": "0", "masses": "1,1,1"},
                     started=started)
    # 기대: main·g8 는 증명, neg 는 실패 — 하나라도 어긋나면 실패로 끝낸다
    wrong = [r["case"] for r in results if r["proven"] != (not r["case"].startswith("neg"))]
    if wrong:
        print("UNEXPECTED:", wrong)
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
