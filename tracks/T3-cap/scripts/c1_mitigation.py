"""C1 과대평가 완화책 실측(8자 궤도, h=-1/2): CAPD C1 집합 표현, 상자 분할, MPFR, 다중 사격 합성을 같은 상자에서 비교한다.

과대평가 배수 = 도함수 감싸기 최대 폭 / (2 rho * 2계 미분 척도). 척도는 benchmark.json 의 [수치] 값을 쓴다.
엔진 동시 호출은 3개 이하(트랙 CPU 예산).
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from common import RESULTS_DIR, TRACK_DIR, sha256_file, write_result
from t3cap import CoordinateSection, Failure, Frame, Interval, Settings, SystemParams, poincare_c0, poincare_c1
from t3cap.api import _c1
from t3cap.engine import build_request, engine_binary, parse_response

MAX_PROCESSES = 3  # 트랙 CPU 예산 상한. 다른 작업과 겹칠 때는 --processes 로 줄인다
EYE = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
SP, SM = CoordinateSection.syzygy(+1), CoordinateSection.syzygy(-1)
SYSTEM = SystemParams.make()


def shooting_points() -> list[tuple[Fraction, ...]]:
    data = json.loads((RESULTS_DIR / "figure8_newton.json").read_text())
    return [tuple(Fraction(float(v)) for v in p) for p in data["shooting_points"]]


def segment_kw(points, k: int, settings: Settings) -> dict:
    sec = [SP, SM] * 3
    return dict(system=SYSTEM, section_in=sec[k], section_out=sec[(k + 1) % 6], returns=1,
                frame_in=Frame.make(points[k], EYE), frame_out=Frame.make(points[(k + 1) % 6], EYE), settings=settings)


def cube(rho: Fraction) -> tuple[Interval, ...]:
    return tuple(Interval(-rho, rho) for _ in range(4))


def width(m) -> float:
    return max(float(e.width) for row in m for e in row)


def policy_c1(policy: str, box, kw: dict):
    """같은 요청을 집합 표현만 바꾼 c1_mitigation 바이너리로 돌린다."""
    engine_binary("native-strict")  # 증분 빌드(c1_mitigation 도 함께 만든다)
    request = build_request("c1", [box], rotation=False, **kw)
    t0 = time.time()
    proc = subprocess.run([str(TRACK_DIR / "build" / "engine-native-strict" / "c1_mitigation"), policy], input=request,
                          capture_output=True, text=True, timeout=1800)
    record = parse_response(proc.stdout, 1)[0]
    return _c1(record), time.time() - t0


def imat_mul(a, b):
    n, m, p = len(a), len(b), len(b[0])
    out = []
    for i in range(n):
        row = []
        for j in range(p):
            acc = Interval(Fraction(0), Fraction(0))
            for k in range(m):
                acc = acc + a[i][k] * b[k][j]
            row.append(acc)
        out.append(tuple(row))
    return tuple(out)


def composed_period_derivative(points, rho: Fraction) -> tuple[float | str, float]:
    """다중 사격 합성: X_{k+1} 은 P_k(X_k) 의 C0 상 상자, DP 는 반 구간 도함수 감싸기 6개의 구간 행렬 곱."""
    t0 = time.time()
    box = cube(rho)
    product = None
    for k in range(6):
        kw = segment_kw(points, k, Settings())
        c0 = poincare_c0([box], **kw)[0]
        c1 = poincare_c1([box], **kw)[0]
        if isinstance(c0, Failure) or isinstance(c1, Failure):
            return f"FAIL segment {k}", time.time() - t0
        product = c1.derivative if product is None else imat_mul(c1.derivative, product)
        box = c0.image
    return width(product), time.time() - t0


def subdivided(points, rho: Fraction, pieces: int) -> tuple[float, float, int]:
    """상자를 축마다 pieces 조각으로 나눠 조각별 C1 감싸기의 외피를 잡는다(같은 상자 전체의 도함수 감싸기)."""
    edges = [-rho + 2 * rho * Fraction(i, pieces) for i in range(pieces + 1)]
    sub = [tuple(Interval(edges[a], edges[a + 1]) for a in combo)
           for combo in itertools.product(range(pieces), repeat=4)]
    t0 = time.time()
    kw = segment_kw(points, 0, Settings(workers=MAX_PROCESSES))
    results = poincare_c1(sub, **kw)
    if any(isinstance(r, Failure) for r in results):
        return float("nan"), time.time() - t0, len(sub)
    hull = [[None] * 4 for _ in range(4)]
    for r in results:
        for i in range(4):
            for j in range(4):
                e = r.derivative[i][j]
                hull[i][j] = e if hull[i][j] is None else hull[i][j].hull(e)
    return width(hull), time.time() - t0, len(sub)


def main() -> None:
    global MAX_PROCESSES
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=MAX_PROCESSES)
    MAX_PROCESSES = min(MAX_PROCESSES, parser.parse_args().processes)
    started = time.time()
    points = shooting_points()
    bench = json.loads((RESULTS_DIR / "benchmark.json").read_text())
    scale = {t["map"]: t["second_derivative_scale"] for t in bench["tables"]}
    d2_half = scale["figure-eight half segment (S+ -> S-)"]
    d2_period = scale["figure-eight full period P^3"]
    rows = []

    def record(method: str, rho: Fraction, w, seconds: float, d2: float, note: str = "") -> None:
        over = (w / (2 * float(rho) * d2)) if isinstance(w, float) else None
        rows.append({"method": method, "rho": str(rho), "derivative_width": w, "overestimation": over,
                     "seconds": round(seconds, 2), "note": note})
        print(json.dumps(rows[-1]), flush=True)

    jobs = []
    with ThreadPoolExecutor(max_workers=MAX_PROCESSES) as pool:
        for rho in (Fraction(1, 10**8), Fraction(1, 10**6)):
            for policy in ("rect2", "pped2", "horect2"):
                jobs.append((f"set policy C1{policy.capitalize()}Set (half segment)", rho, d2_half,
                             pool.submit(policy_c1, policy, cube(rho), segment_kw(points, 0, Settings()))))
        for method, rho, d2, fut in jobs:
            res, secs = fut.result()
            record(method, rho, width(res.derivative) if not isinstance(res, Failure) else f"FAIL {res.reason[:60]}",
                   secs, d2)
    rho = Fraction(1, 10**6)
    for pieces in (2, 3):
        w, secs, n = subdivided(points, rho, pieces)
        record(f"subdivision {pieces}^4={n} pieces (half segment)", rho, w, secs, d2_half,
               note="같은 상자 전체의 도함수 감싸기(조각 외피)")
    for arith, order in (("double", 20), ("mp:128", 30)):
        t0 = time.time()
        r = poincare_c1([cube(Fraction(1, 10**8))], **segment_kw(points, 0, Settings(order=order, arithmetic=arith)))[0]
        record(f"arithmetic {arith} order {order} (half segment)", Fraction(1, 10**8),
               width(r.derivative) if not isinstance(r, Failure) else "FAIL", time.time() - t0, d2_half)
    for rho in (Fraction(1, 10**10), Fraction(1, 10**8)):
        w, secs = composed_period_derivative(points, rho)
        record("multiple shooting: product of 6 half-segment enclosures (full period)", rho, w, secs, d2_period)
    direct = {r["rho"]: r for t in bench["tables"] if t["map"] == "figure-eight full period P^3" for r in t["rows"]}
    write_result("c1_mitigation", {"tag": "[CAP 감싸기 폭] + [수치 2계 척도]", "second_derivative_scale":
                                   {"half_segment": d2_half, "full_period": d2_period},
                                   "direct_full_period_from_benchmark": direct, "rows": rows},
                 script=Path(__file__).resolve(), started=started,
                 params={"max_processes": MAX_PROCESSES,
                         "extra_code_sha256": {"engine/c1_mitigation.cpp": sha256_file(TRACK_DIR / "engine" / "c1_mitigation.cpp")}})


if __name__ == "__main__":
    main()
