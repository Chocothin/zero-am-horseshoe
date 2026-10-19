"""G5 건전성 [수치: G5 표본 대조]: 저장된 덮개 감싸기가 mpmath 40자리 기준 해를 담는지 본다(D39 모형 방식).

덮개 v→w 마다
  ① h-set 중심 c_v 의 P̃·u 좌표 중심차분 야코비안(보폭 h=1e-15)·귀환 시각 기울기를 mpmath 40자리로 직접 적분해 얻는다.
     절단 오차는 이론상 h²|D³P̃|/6 ~ 1e-25, 반올림 오차는 10^{-40}/h ~ 1e-25 이고, 덮개 16개마다 보폭 1e-13 과 비교해 잰다.
  ② 상자 안 점 ≥1000개와 출구 면 점의 기준 상을 정규 좌표 1차 모형 w′=b+A w 로 낸다. 모형 오차 상계는
     e₂ = ½·max_k Σ_ab |H_kab| r_a(v) r_b(v) / r_k(w) 이고, H 는 단위 틀 좌표 2계 도함수의 float64 중심차분 추정이다
     (H 는 [수치] 라서 상계도 수치 추정이다). 덮개마다 무작위 3점을 mpmath 로 직접 적분해 「모형 대 직접」 편차가
     2e₂+1e-12 안인지 본다. 1000점 판정의 허용치도 같은 값이다.
  ③ 점마다 감싸기 포함, 귀환 시각(1차 모형) 포함을 본다. 기준 해 귀환은 t>0 의 첫 g 하강 영점이므로 그 시각이
     엔진 귀환 창 안이면 반환이 「다음 syzygy」 다. 직접 적분 3점은 반환 전 g 표본이 모두 양수(통과 횟수 1)인지도 본다.
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
from t6hs.results import DESIGN_DIR, RESULTS_DIR, T3_DIR, write_result  # noqa: E402

sys.path.insert(0, str(T3_DIR))
from t3cap.exact import matrix_inverse  # noqa: E402
from t3cap.export import interval_from_json  # noqa: E402

from t6hs import enclose, flow, reference  # noqa: E402

FACES = [(0, -1), (0, 1), (1, -1), (1, 1)]
DERIV_TOL = Fraction(1, 10**4)
TIME_TOL = Fraction(1, 10**12)  # 1차 모형의 귀환 시각 오차(2차 항 ~1e-16)보다 크고 c0 귀환 창 폭(~1e-9)보다 작다
FD_STEP = Fraction(1, 10**15)
FD_STEP_CHECK = Fraction(1, 10**13)
HESS_STEP = 1e-4  # 단위 틀 좌표 보폭: float64 P̃ 오차 1e-13/h² ~ 1e-5, 절단 h²|D⁴| ~ 1e-3


def mpf_to_fraction(x) -> Fraction:
    sign, man, exp, _ = mp.mpf(x)._mpf_
    v = Fraction(int(man)) * Fraction(2) ** int(exp)
    return -v if sign else v


def box_of(js):
    return tuple(interval_from_json(x) for x in js)


def rand_unit(rng) -> Fraction:
    return Fraction(rng.getrandbits(60), 1 << 59) - 1


def second_order_bound(src_vertex: dict, dst_vertex: dict) -> tuple[float, float]:
    """(e₂, max|H|): 단위 틀 좌표 2계 도함수를 float64 P̃ 중심차분으로 추정해 정규 좌표 2차 항 상계를 낸다."""
    c = np.array([float.fromhex(h) for h in src_vertex["center_hex"]])
    fv = np.array([[float.fromhex(h) for h in row] for row in src_vertex["frame_hex"]])
    fw = np.array([[float.fromhex(h) for h in row] for row in dst_vertex["frame_hex"]])
    rv, rw = np.array(src_vertex["radii"]), np.array(dst_vertex["radii"])
    fw_inv = np.linalg.inv(fw)
    f = lambda xi: fw_inv @ flow.next_syzygy(c + fv @ xi).u
    e = np.eye(4) * HESS_STEP
    f0 = f(np.zeros(4))
    hess = np.zeros((4, 4, 4))
    for a in range(4):
        hess[:, a, a] = (f(e[a]) - 2 * f0 + f(-e[a])) / HESS_STEP ** 2
        for b in range(a + 1, 4):
            hess[:, a, b] = hess[:, b, a] = (f(e[a] + e[b]) - f(e[a] - e[b]) - f(-e[a] + e[b])
                                             + f(-e[a] - e[b])) / (4 * HESS_STEP ** 2)
    quad = np.einsum("kab,a,b->k", np.abs(hess), rv, rv)
    return float(np.max(0.5 * quad / rw)), float(np.abs(hess).max())


def check_edge(task) -> dict:
    name, src_vertex, dst_vertex, rec, n_box, n_face, n_direct, seed, check_fd = task
    s, t = name.split(">")
    src = enclose.hset_from_design(s, src_vertex)
    dst = enclose.hset_from_design(t, dst_vertex)
    started = time.time()
    e2, hmax = second_order_bound(src_vertex, dst_vertex)
    base, jac, grad = reference.jacobian(src.center, step=FD_STEP)
    fd_diff = None
    if check_fd:
        _, jac2, _ = reference.jacobian(src.center, step=FD_STEP_CHECK)
        fd_diff = float(max(abs(a - b) for ra, rb in zip(jac, jac2) for a, b in zip(ra, rb)))
    mp.mp.dps = reference.DPS
    minv = matrix_inverse(dst.matrix)
    q = reference.to_mpf
    mv = [[q(x) for x in row] for row in src.matrix]
    mi = [[q(x) for x in row] for row in minv]
    du = [base.u_out[i] - q(dst.center[i]) for i in range(4)]
    b = [mp.fsum(mi[k][l] * du[l] for l in range(4)) for k in range(4)]
    jm = [[mp.fsum(jac[i][l] * mv[l][j] for l in range(4)) for j in range(4)] for i in range(4)]
    a = [[mp.fsum(mi[k][i] * jm[i][j] for i in range(4)) for j in range(4)] for k in range(4)]
    gt = [mp.fsum(grad[l] * mv[l][j] for l in range(4)) for j in range(4)]

    def model(w):
        return [b[k] + mp.fsum(a[k][j] * q(w[j]) for j in range(4)) for k in range(4)]

    rng = random.Random(seed)
    detail = rec["pieces_detail"]
    whole = box_of(detail["whole"][0]["image"])
    rt = interval_from_json(detail["whole"][0]["return_time"])
    tube = {k: interval_from_json(v) for k, v in detail["whole"][0]["tube"].items()}
    tol = Fraction(2 * e2) + Fraction(1, 10**12)
    violations = []
    deviation = Fraction(0)
    direct = []
    for _ in range(n_direct):
        w = [rand_unit(rng) for _ in range(4)]
        u = [src.center[i] + sum(src.matrix[i][j] * w[j] for j in range(4)) for i in range(4)]
        fl = reference.next_syzygy(u)
        wd = [mp.fsum(mi[k][l] * (fl.u_out[l] - q(dst.center[l])) for l in range(4)) for k in range(4)]
        dev = max(abs(mpf_to_fraction(x - y)) for x, y in zip(wd, model(w)))
        deviation = max(deviation, dev)
        crossings_ok = fl.g_min_before is not None and fl.g_min_before > 0
        direct.append({"deviation": float(dev), "g_min_before_return": mp.nstr(fl.g_min_before, 6),
                       "samples_before_return": fl.samples_before, "single_crossing": crossings_ok})
        if not crossings_ok:
            violations.append("direct flight: g touched zero before the returned crossing")
        if not all(whole[k].contains(mpf_to_fraction(wd[k])) for k in range(4)):
            violations.append("direct image outside the whole-box enclosure")
        if not rt.contains(mpf_to_fraction(fl.time)):
            violations.append("direct return time outside the c0 window")
    if deviation > tol:
        violations.append(f"model deviation {float(deviation):.3g} exceeds 2e2+1e-12={float(tol):.3g}")
    closest, shifted_hits = Fraction(1), 0

    def check(img, w, label):
        nonlocal closest, shifted_hits
        val = model(w)
        hit = False
        for k in range(4):
            v = mpf_to_fraction(val[k])
            iv = img[k]
            if v < iv.lo - tol or v > iv.hi + tol:
                violations.append(f"{label} w'[{k}]={float(v):.6g} outside [{float(iv.lo):.6g},{float(iv.hi):.6g}]")
            if iv.width > 0:
                closest = min(closest, min(v - iv.lo, iv.hi - v) / iv.width)
                hit = hit or not (iv.lo + iv.width <= v <= iv.hi + iv.width)
        shifted_hits += hit
        tf = mpf_to_fraction(base.time + mp.fsum(gt[j] * q(w[j]) for j in range(4)))
        if not (rt.lo - TIME_TOL <= tf <= rt.hi + TIME_TOL):
            violations.append(f"{label} return time {float(tf):.12g} outside {rt}")

    for i in range(n_box):
        check(whole, [rand_unit(rng) for _ in range(4)], f"box#{i}")
    for (j, side) in FACES:
        img = box_of(detail[f"face_w{j}{side:+d}"][0]["image"])
        for i in range(n_face):
            w = [rand_unit(rng) for _ in range(4)]
            w[j] = Fraction(side)
            check(img, w, f"face{j}{side:+d}#{i}")
    for idx, key in enumerate(("r12", "r13", "r23")):
        lo, hi = mpf_to_fraction(base.min_distance[idx]), mpf_to_fraction(base.max_distance[idx])
        if lo < tube[key].lo or hi > tube[key].hi:
            violations.append(f"tube {key} sampled [{float(lo):.6g},{float(hi):.6g}] vs {tube[key]}")
    deriv_ok = None
    if rec.get("derivative"):
        der = [[interval_from_json(x) for x in row] for row in rec["derivative"]]
        deriv_ok = all(der[k][j].lo - DERIV_TOL <= mpf_to_fraction(a[k][j]) <= der[k][j].hi + DERIV_TOL
                       for k in range(4) for j in range(4))
        if not deriv_ok:
            violations.append("center derivative outside c1 enclosure")
    points = n_box + 4 * n_face
    return {"edge": name, "points": points, "violations": violations[:10], "n_violations": len(violations),
            "second_order_bound_e2": e2, "hessian_max_unit_frame": hmax, "model_deviation": float(deviation),
            "tolerance": float(tol), "fd_step": str(FD_STEP), "fd_step_check": str(FD_STEP_CHECK) if check_fd else None,
            "fd_two_step_difference": fd_diff, "closest_relative_margin": float(closest),
            "shifted_control_detected": shifted_hits, "tester_blind": shifted_hits != points,
            "derivative_contains_center_jacobian": deriv_ok, "reference_return_time": mp.nstr(base.time, 20),
            "direct": direct, "seconds": round(time.time() - started, 1)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--face-points", type=int, default=50)
    parser.add_argument("--direct", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--fd-check-every", type=int, default=16)
    parser.add_argument("--only", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    design = json.loads((DESIGN_DIR / "graph_design.json").read_text())
    cover_dir = RESULTS_DIR / "coverings" / args.tag if args.tag else RESULTS_DIR / "coverings"
    wanted = set(s for s in args.only.split(",") if s)
    tasks = []
    for i, e in enumerate(design["edges"]):
        name = f"{e['source']}>{e['target']}"
        if wanted and name not in wanted:
            continue
        rec = json.loads((cover_dir / f"{e['source']}__{e['target']}.json").read_text())
        tasks.append((name, design["vertices"][e["source"]], design["vertices"][e["target"]], rec, args.points,
                      args.face_points, args.direct, 1000 + i, i % args.fd_check_every == 0))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(check_edge, tasks))
    critical = sum(r["n_violations"] + int(r["tester_blind"]) for r in rows)
    for r in rows:
        print(json.dumps({k: r[k] for k in ("edge", "points", "n_violations", "second_order_bound_e2", "model_deviation",
                                            "closest_relative_margin", "tester_blind", "seconds")}), flush=True)
    summary = {"edges": len(rows), "critical": critical,
               "max_model_deviation": max(r["model_deviation"] for r in rows) if rows else None,
               "max_e2": max(r["second_order_bound_e2"] for r in rows) if rows else None,
               "max_fd_two_step_difference": max((r["fd_two_step_difference"] for r in rows
                                                  if r["fd_two_step_difference"] is not None), default=None),
               "min_closest_relative_margin": min(r["closest_relative_margin"] for r in rows) if rows else None}
    print("summary", json.dumps(summary))
    if not args.no_write:
        write_result(f"g5_soundness{('_' + args.tag) if args.tag else ''}",
                     {"tag": "[수치: G5 표본 대조] D39 모형 방식", "dps": reference.DPS, "summary": summary, "rows": rows},
                     script=Path(__file__).resolve(), params=vars(args), started=started)
    return 1 if critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
