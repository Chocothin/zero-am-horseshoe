"""벤치마크: 주기점 둘레 상자 반경별 C0 상 폭·래핑 과대평가·C1 도함수 폭·실행 시간 (CAPD 엔진, double, 차수 20).

래핑 지표 = (C0 상 폭) / (선형 예측 폭 sum_j |DP_ij| 2rho). 1 에 가까울수록 덮개 관계에 쓸 만하다.
DP 는 점 상자 C1 감싸기의 중점(비엄밀 [수치])으로 잡는다.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from common import RESULTS_DIR, write_result
from t3cap import CoordinateSection, Failure, Frame, Interval, Settings, SystemParams, poincare_c0, poincare_c1
from t3cap.engine import EngineError

RADII = [Fraction(1, 10**k) for k in (12, 10, 8, 6, 5, 4, 3, 2)]
TIMEOUT_SECONDS = 300
MAX_PROCESSES = 3  # 트랙 CPU 예산(BRIEF rev3)
EYE = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))


def maps(orbits: list[str]) -> list[dict]:
    f8 = json.loads((RESULTS_DIR / "figure8_newton.json").read_text())
    p0, p1 = (tuple(Fraction(float(v)) for v in p) for p in f8["shooting_points"][:2])
    sp, sm = CoordinateSection.syzygy(+1), CoordinateSection.syzygy(-1)
    out = [
        {"name": "figure-eight full period P^3", "center_in": p0, "center_out": p0, "sec_in": sp, "sec_out": sp,
         "returns": 3},
        {"name": "figure-eight half segment (S+ -> S-)", "center_in": p0, "center_out": p1, "sec_in": sp,
         "sec_out": sm, "returns": 1},
    ]
    for name in orbits:
        orb = json.loads((RESULTS_DIR / f"orbit_{name}.json").read_text())
        u = tuple(Fraction(float(v)) for v in orb["fixed_point_u"])
        sec = CoordinateSection.syzygy(orb["direction"])
        for returns in (1, orb["returns"]):
            out.append({"name": f"{orb['name']} P^{returns} of {orb['returns']} ({orb['classification']})",
                        "center_in": u, "center_out": u, "sec_in": sec, "sec_out": sec, "returns": returns})
    return out


def measure(m: dict, rho: Fraction, dp_mid: list[list[float]] | None, d2: float | None) -> dict:
    kw = dict(system=SystemParams.make(), section_in=m["sec_in"], section_out=m["sec_out"], returns=m["returns"],
              frame_in=Frame.make(m["center_in"], EYE), frame_out=Frame.make(m["center_out"], EYE),
              settings=Settings(order=20, timeout_seconds=TIMEOUT_SECONDS))
    box = tuple(Interval(-rho, rho) for _ in range(4))
    t0 = time.time()
    r0 = _guarded(poincare_c0, box, kw)
    r1 = _guarded(poincare_c1, box, kw)
    row = {"rho": f"1e-{len(str(rho.denominator)) - 1}", "wall_seconds": round(time.time() - t0, 2)}
    if isinstance(r0, Failure):
        row["c0"] = f"FAIL: {r0.reason[:90]}"
    else:
        width = max(float(iv.width) for iv in r0.image)
        row.update(c0_image_width=width, c0_growth=width / float(2 * rho), c0_seconds=round(r0.seconds, 2),
                   tube_min=float(r0.tube.min_distance_lower), return_time_width=float(r0.return_time.width))
        if dp_mid is not None:
            linear = max(sum(abs(v) for v in rowd) for rowd in dp_mid) * float(2 * rho)
            row["wrapping_excess"] = width / linear
    if isinstance(r1, Failure):
        row["c1"] = f"FAIL: {r1.reason[:90]}"
    else:
        dwidth = max(float(e.width) for rr in r1.derivative for e in rr)
        row.update(c1_derivative_width=dwidth, c1_seconds=round(r1.seconds, 2))
        if d2:
            row["c1_overestimation"] = dwidth / (2 * float(rho) * d2)
    return row


def _guarded(fn, box, kw):
    """시간 초과도 실패 결과로 남긴다 — 표에서 「어디서부터 못 쓰는가」가 관측 대상이다."""
    try:
        return fn([box], **kw)[0]
    except EngineError as exc:
        return Failure(str(exc), TIMEOUT_SECONDS)


def derivative_at(m: dict, w: tuple[Fraction, ...]) -> list[list[float]] | None:
    kw = dict(system=SystemParams.make(), section_in=m["sec_in"], section_out=m["sec_out"], returns=m["returns"],
              frame_in=Frame.make(m["center_in"], EYE), frame_out=Frame.make(m["center_out"], EYE),
              settings=Settings(order=20, timeout_seconds=TIMEOUT_SECONDS))
    r = _guarded(poincare_c1, tuple(Interval.point(q) for q in w), kw)
    return None if isinstance(r, Failure) else [[float(e.mid) for e in row] for row in r.derivative]


def second_derivative_scale(m: dict) -> float | None:
    """[수치] max_ij sum_k |d DP_ij / d u_k| — 점 상자 C1 중점의 중심 차분(delta=1e-5). 과대평가 배수의 분모."""
    delta = Fraction(1, 10**5)
    shifted = []
    for k in range(4):
        pair = []
        for sign in (1, -1):
            w = tuple(delta * sign if i == k else Fraction(0) for i in range(4))
            pair.append(derivative_at(m, w))
        shifted.append(pair)
    if any(d is None for pair in shifted for d in pair):
        return None
    return max(sum(abs(shifted[k][0][i][j] - shifted[k][1][i][j]) / (2 * float(delta)) for k in range(4))
               for i in range(4) for j in range(4))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orbits", default="", help="results/ 의 orbit_<name>.json 이름들(쉼표 구분)")
    args = parser.parse_args()
    started = time.time()
    tables = []
    for m in maps([o for o in args.orbits.split(",") if o]):
        dp = derivative_at(m, (Fraction(0),) * 4)
        # 중심 C1 이 실패하는 사상에서는 유한차분도 실패하므로 건너뛴다
        d2 = second_derivative_scale(m) if dp is not None else None
        with ThreadPoolExecutor(max_workers=MAX_PROCESSES) as pool:
            rows = list(pool.map(lambda r: measure(m, r, dp, d2), RADII))
        norm = max(sum(abs(v) for v in row) for row in dp) if dp else None
        tables.append({"map": m["name"], "returns": m["returns"], "dp_row_sum_norm": norm,
                       "second_derivative_scale": d2, "rows": rows})
        print(m["name"], "|DP|_inf =", norm, "D2 scale =", d2, flush=True)
        for row in rows:
            print("  ", json.dumps(row), flush=True)
    write_result("benchmark", {"tag": "[CAP 감싸기 폭] + [수치 DP 중점]", "order": 20, "arithmetic": "double native",
                               "tables": tables},
                 script=Path(__file__).resolve(), params={"radii": [str(r) for r in RADII], "orbits": args.orbits},
                 started=started)


if __name__ == "__main__":
    main()
