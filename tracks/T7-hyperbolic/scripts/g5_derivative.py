"""G5 건전성 [수치: G5 표본 대조]: 원뿔 판정이 믿는 도함수 감싸기 D(B) 가 mpmath 40자리 기준 도함수를 담는지 본다(D39 모형 방식).

덮개 v→w 마다
  ① h-set 중심의 P̃ u 좌표 중심차분 야코비안(보폭 1e-15, T6 reference: CAPD·T3 와 코드를 공유하지 않는 mpmath Taylor 적분)을
     정규 좌표 A_c = M_w⁻¹ J_c M_v 로 옮긴다. 덮개 16개마다 보폭 1e-13 과 비교해 차분 오차를 잰다.
  ② 상자 안 점 w ≥1000개의 기준 도함수를 1차 모형 DP̂(w) ≈ A_c + Σ_b H̃_{··b} w_b 로 낸다. H̃ 는 정규 좌표 2계 도함수의
     float64 중심차분 추정(T6 flow)이다. 모형 허용치는 τ = 0.1·max_{k,a} Σ_b|H̃_kab| + 1e-15 이다.
  ③ 덮개마다 무작위 점에서 mpmath 야코비안을 직접 구해 모형 편차가 τ 안인지, 감싸기 안인지 본다.
감싸기를 제 폭만큼 옮긴 대조에서 모든 점을 잡지 못하면 시험기 결함(Critical)이다.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from t7hb.results import T3_DIR, T6_DIR, T6_DESIGN, T6_RESULTS, write_result  # noqa: E402

sys.path.insert(0, str(T3_DIR))
sys.path.insert(0, str(T6_DIR))
from t3cap.exact import matrix_inverse  # noqa: E402
from t3cap.export import interval_from_json  # noqa: E402
from t6hs import enclose, flow, reference  # noqa: E402

FD_STEP = Fraction(1, 10**15)
FD_STEP_CHECK = Fraction(1, 10**13)
HESS_STEP = 1e-4  # 단위 틀 좌표 보폭(T6 g5 와 같음): float64 P̃ 오차 1e-13/h² ~ 1e-5, 절단 h²|D⁴| ~ 1e-3
MODEL_SHARE = Fraction(1, 10)
MODEL_FLOOR = Fraction(1, 10**15)


def mpf_to_fraction(x) -> Fraction:
    sign, man, exp, _ = mp.mpf(x)._mpf_
    v = Fraction(int(man)) * Fraction(2) ** int(exp)
    return -v if sign else v


def rand_unit(rng) -> Fraction:
    return Fraction(rng.getrandbits(60), 1 << 59) - 1


def normalized_hessian(src_vertex: dict, dst_vertex: dict) -> np.ndarray:
    """H̃[k,a,b] = ∂²P̂_k/∂w_a∂w_b. 단위 틀 좌표 ξ=diag(r_v)w 에서 차분한 뒤 r_a r_b / r_k(w) 를 곱한다."""
    c = np.array([float.fromhex(h) for h in src_vertex["center_hex"]])
    fv = np.array([[float.fromhex(h) for h in row] for row in src_vertex["frame_hex"]])
    fw_inv = np.linalg.inv(np.array([[float.fromhex(h) for h in row] for row in dst_vertex["frame_hex"]]))
    rv, rw = np.array(src_vertex["radii"]), np.array(dst_vertex["radii"])
    f = lambda xi: fw_inv @ flow.next_syzygy(c + fv @ xi).u
    e = np.eye(4) * HESS_STEP
    f0 = f(np.zeros(4))
    hess = np.zeros((4, 4, 4))
    for a in range(4):
        hess[:, a, a] = (f(e[a]) - 2 * f0 + f(-e[a])) / HESS_STEP ** 2
        for b in range(a + 1, 4):
            hess[:, a, b] = hess[:, b, a] = (f(e[a] + e[b]) - f(e[a] - e[b]) - f(-e[a] + e[b])
                                             + f(-e[a] - e[b])) / (4 * HESS_STEP ** 2)
    return np.einsum("kab,a,b,k->kab", hess, rv, rv, 1 / rw)


def normalized_jacobian(jac_u, src, minv):
    q = reference.to_mpf
    mv = [[q(x) for x in row] for row in src.matrix]
    mi = [[q(x) for x in row] for row in minv]
    jm = [[mp.fsum(jac_u[i][l] * mv[l][j] for l in range(4)) for j in range(4)] for i in range(4)]
    return [[mp.fsum(mi[k][i] * jm[i][j] for i in range(4)) for j in range(4)] for k in range(4)]


def check_edge(task) -> dict:
    name, src_vertex, dst_vertex, rec, n_points, n_direct, seed, check_fd = task
    s, t = name.split(">")
    src = enclose.hset_from_design(s, src_vertex)
    dst = enclose.hset_from_design(t, dst_vertex)
    started = time.time()
    der = [[interval_from_json(x) for x in row] for row in rec["derivative"]]
    minv = matrix_inverse(dst.matrix)
    mp.mp.dps = reference.DPS
    _, jac, _ = reference.jacobian(src.center, step=FD_STEP)
    a_c = [[mpf_to_fraction(x) for x in row] for row in normalized_jacobian(jac, src, minv)]
    fd_diff = None
    if check_fd:
        _, jac2, _ = reference.jacobian(src.center, step=FD_STEP_CHECK)
        a2 = normalized_jacobian(jac2, src, minv)
        fd_diff = float(max(abs(mpf_to_fraction(a2[k][j]) - a_c[k][j]) for k in range(4) for j in range(4)))
    hn = normalized_hessian(src_vertex, dst_vertex)
    hvar = float(np.max(np.sum(np.abs(hn), axis=2)))
    tol = MODEL_SHARE * Fraction(hvar) + MODEL_FLOOR
    hq = [[[Fraction(float(hn[k, a, b])) for b in range(4)] for a in range(4)] for k in range(4)]

    def model(w):
        return [[a_c[k][a] + sum(hq[k][a][b] * w[b] for b in range(4)) for a in range(4)] for k in range(4)]

    rng = random.Random(seed)
    violations, closest, shifted_hits = [], Fraction(1), 0
    for i in range(n_points):
        m = model([rand_unit(rng) for _ in range(4)])
        hit = False
        for k in range(4):
            for a in range(4):
                iv, v = der[k][a], m[k][a]
                if v < iv.lo - tol or v > iv.hi + tol:
                    violations.append(f"point#{i} D[{k}][{a}]={float(v):.17g} outside {iv}")
                if iv.width > 0:
                    closest = min(closest, min(v - iv.lo, iv.hi - v) / iv.width)
                    hit = hit or not (iv.lo + iv.width <= v <= iv.hi + iv.width)
        shifted_hits += hit
    direct, deviation = [], Fraction(0)
    for _ in range(n_direct):
        w = [rand_unit(rng) for _ in range(4)]
        u = [src.center[i] + sum(src.matrix[i][j] * w[j] for j in range(4)) for i in range(4)]
        _, jac_w, _ = reference.jacobian(u, step=FD_STEP)
        a_w = [[mpf_to_fraction(x) for x in row] for row in normalized_jacobian(jac_w, src, minv)]
        m = model(w)
        dev = max(abs(a_w[k][a] - m[k][a]) for k in range(4) for a in range(4))
        deviation = max(deviation, dev)
        inside = all(der[k][a].contains(a_w[k][a]) for k in range(4) for a in range(4))
        direct.append({"deviation": float(dev), "inside_enclosure": inside})
        if not inside:
            violations.append("direct derivative outside the c1 enclosure D(B)")
    if deviation > tol:
        violations.append(f"model deviation {float(deviation):.3g} exceeds tolerance {float(tol):.3g}")
    return {"edge": name, "points": n_points, "violations": violations[:10], "n_violations": len(violations),
            "tolerance": float(tol), "first_order_variation": hvar, "model_deviation": float(deviation),
            "deviation_over_tolerance": float(deviation / tol), "closest_relative_margin": float(closest),
            "shifted_control_detected": shifted_hits, "tester_blind": shifted_hits != n_points,
            "fd_two_step_difference": fd_diff, "min_enclosure_width": float(min(iv.width for row in der for iv in row)),
            "direct": direct, "seconds": round(time.time() - started, 1)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--direct", type=int, default=1)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--fd-check-every", type=int, default=16)
    parser.add_argument("--only", default="")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    design = json.loads((T6_DESIGN / "graph_design.json").read_text())
    wanted = set(s for s in args.only.split(",") if s)
    tasks = []
    for i, e in enumerate(design["edges"]):
        name = f"{e['source']}>{e['target']}"
        if wanted and name not in wanted:
            continue
        rec = json.loads((T6_RESULTS / "coverings" / f"{e['source']}__{e['target']}.json").read_text())
        tasks.append((name, design["vertices"][e["source"]], design["vertices"][e["target"]], rec, args.points,
                      args.direct, 7000 + i, i % args.fd_check_every == 0))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(check_edge, tasks))
    critical = sum(r["n_violations"] + int(r["tester_blind"]) for r in rows)
    for r in rows:
        print(json.dumps({k: r[k] for k in ("edge", "n_violations", "tolerance", "model_deviation",
                                            "closest_relative_margin", "tester_blind", "seconds")}), flush=True)
    summary = {"edges": len(rows), "points": sum(r["points"] for r in rows),
               "direct": sum(len(r["direct"]) for r in rows), "critical": critical,
               "max_deviation_over_tolerance": max((r["deviation_over_tolerance"] for r in rows), default=None),
               "max_fd_two_step_difference": max((r["fd_two_step_difference"] for r in rows
                                                  if r["fd_two_step_difference"] is not None), default=None),
               "min_closest_relative_margin": min((r["closest_relative_margin"] for r in rows), default=None),
               "max_tolerance": max((r["tolerance"] for r in rows), default=None)}
    print("summary", json.dumps(summary))
    if not args.no_write:
        write_result("g5_derivative", {"tag": "[수치: G5 표본 대조] D(B) 도함수 감싸기, D39 모형 방식", "dps": reference.DPS,
                                       "summary": summary, "rows": rows},
                     script=Path(__file__).resolve(), params=vars(args), started=started)
    return 1 if critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
