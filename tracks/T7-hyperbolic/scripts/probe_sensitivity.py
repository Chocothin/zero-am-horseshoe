"""[수치] 정리 B 설계 입력: 질량·J 를 점 값으로 조금 바꿨을 때 h-set 중심의 상 P̃(c_v) 가 도착 정규 좌표에서 움직이는
양(민감도)과, 같은 점 실행에 매개변수 구간을 넣었을 때 감싸기 폭을 엔진 MP 점 실행(T6 점 상과 같은 설정)으로 잰다.

민감도 s_p = (w′(p₀+δ·e_p) − w′(p₀))/δ 의 단위는 「정규 좌표 / 매개변수」다(δ=1e-10, 정확 유리수). 성분별 덮개 여유 μ_k 를
Σ_p|s_kp| 로 나눈 값이 같은 h-set 을 그대로 쓸 때 그 변이 견디는 매개변수 반폭의 선형 예측이다. 구간 매개변수 실행의
감싸기 폭을 선형 예측 2·Σ_p|s_kp|·ε 로 나눈 값이 매개변수 감싸기의 과대평가 배수다. 설계 판단용이고 증명 근거가 아니다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from t7hb.results import T3_DIR, T6_DIR, T6_DESIGN, T6_RESULTS, write_result  # noqa: E402

sys.path.insert(0, str(T3_DIR))
sys.path.insert(0, str(T6_DIR))
from t3cap import CoordinateSection, Failure, Interval, Settings, SystemParams, poincare_c0  # noqa: E402
from t3cap.export import interval_from_json  # noqa: E402
from t6hs import enclose  # noqa: E402

EDGES = ("N0>N1", "N1>N2", "N5>N6", "N8>M-148", "M-146>M-145", "M-1>M+0", "M+0>M+1", "M+145>M+146", "M+148>N8")
DELTA = Fraction(1, 10**10)
PARAMS = ("m1", "m2", "m3", "J")
POINT = Settings(order=30, arithmetic="mp:128", workers=1)


def system_for(name: str, delta: Fraction | Interval | None) -> SystemParams:
    if delta is None:
        return SystemParams.make()
    if isinstance(delta, Interval):
        one = Interval(1 + delta.lo, 1 + delta.hi)
        return SystemParams.make(masses=(one, one, one), angular_momentum=delta)
    masses = [Fraction(1) + (delta if name == f"m{i + 1}" else 0) for i in range(3)]
    return SystemParams.make(masses=masses, angular_momentum=delta if name == "J" else 0)


def point_image(src, dst, system):
    zero = tuple(Interval(Fraction(0), Fraction(0)) for _ in range(4))
    r = poincare_c0([zero], system=system, section_in=CoordinateSection.syzygy(+1),
                    section_out=CoordinateSection.syzygy(-1), frame_in=src.frame(), frame_out=dst.frame(),
                    settings=POINT)[0]
    return r


def margins(rec: dict) -> list[Fraction]:
    """성분별 덮개 여유: 안정 성분은 상자 상이 (−1,1) 안쪽에서 떨어진 거리, 불안정 성분은 출구 면 상이 문턱 1 을 넘은 거리."""
    detail = rec["pieces_detail"]
    whole = [interval_from_json(x) for x in detail["whole"][0]["image"]]
    out = [Fraction(10**9)] * 4
    for k in (2, 3):
        out[k] = min(1 - whole[k].hi, whole[k].lo + 1)
    for j in (0, 1):
        for side in (-1, 1):
            img = [interval_from_json(x) for x in detail[f"face_w{j}{side:+d}"][0]["image"]]
            orient = rec["signs"][j] * side
            out[j] = min(out[j], img[j].lo - 1 if orient > 0 else -1 - img[j].hi)
    return out


def probe(edge: str, design: dict, interval_eps: Fraction) -> dict:
    s, t = edge.split(">")
    src = enclose.hset_from_design(s, design["vertices"][s])
    dst = enclose.hset_from_design(t, design["vertices"][t])
    rec = json.loads((T6_RESULTS / "coverings" / f"{s}__{t}.json").read_text())
    started = time.time()
    base = point_image(src, dst, system_for("", None))
    if isinstance(base, Failure):
        return {"edge": edge, "failure": base.reason}
    w0 = [iv.mid for iv in base.image]
    sens = {}
    for p in PARAMS:
        r = point_image(src, dst, system_for(p, DELTA))
        if isinstance(r, Failure):
            return {"edge": edge, "failure": f"{p}: {r.reason}"}
        sens[p] = [(iv.mid - c) / DELTA for iv, c in zip(r.image, w0)]
    marg = margins(rec)
    total = [sum(abs(sens[p][k]) for p in PARAMS[:3]) for k in range(4)]
    eps_mass = min(marg[k] / total[k] for k in range(4) if total[k] > 0)
    eps_j = min(marg[k] / abs(sens["J"][k]) for k in range(4) if sens["J"][k] != 0)
    box = point_image(src, dst, system_for("", Interval(-interval_eps, interval_eps)))
    wrap = None
    if not isinstance(box, Failure):
        linear = [2 * sum(abs(sens[p][k]) for p in PARAMS) * interval_eps for k in range(4)]
        wrap = [float(iv.width / lin) if lin > 0 else None for iv, lin in zip(box.image, linear)]
    return {
        "edge": edge, "margins": [float(m) for m in marg],
        "sensitivity_per_unit": {p: [float(v) for v in sens[p]] for p in PARAMS},
        "linear_eps_mass_all_three": float(eps_mass), "linear_eps_J": float(eps_j),
        "interval_eps": str(interval_eps),
        "interval_run": None if wrap is None else {"widths": [float(iv.width) for iv in box.image],
                                                    "overestimate_vs_linear": wrap},
        "interval_failure": box.reason if isinstance(box, Failure) else None,
        "return_time": [float(base.return_time.lo), float(base.return_time.hi)],
        "seconds": round(time.time() - started, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--interval-eps", default="1/10000000000000")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    design = json.loads((T6_DESIGN / "graph_design.json").read_text())
    edges = [e for e in EDGES if not args.only or e in args.only.split(",")]
    eps = Fraction(args.interval_eps)
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        rows = list(pool.map(lambda e: probe(e, design, eps), edges))
    for r in rows:
        print(json.dumps({k: r.get(k) for k in ("edge", "linear_eps_mass_all_three", "linear_eps_J", "seconds",
                                               "interval_failure")}), flush=True)
    ok = [r for r in rows if "failure" not in r]
    summary = {"edges": len(rows), "failed": len(rows) - len(ok),
               "min_linear_eps_mass": min((r["linear_eps_mass_all_three"] for r in ok), default=None),
               "min_linear_eps_J": min((r["linear_eps_J"] for r in ok), default=None)}
    print("summary", json.dumps(summary))
    if not args.no_write:
        write_result("sensitivity_probe", {"tag": "[수치] 정리 B 설계 입력(민감도·매개변수 감싸기 폭)", "delta": str(DELTA),
                                           "summary": summary, "rows": rows},
                     script=Path(__file__).resolve(), params=vars(args), started=started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
