"""엔진 검증: 정확해가 있는 두 경우와 레포의 8자 궤도 이정표.

1. Kepler 타원(m=(1,1,0), 물체 3 은 질량 0): 상대 벡터 ρ=q2−q1 은 GM=2 의 Kepler 운동이다. Kepler 방정식
   E − e sin E = n t 를 Arb 구간 Newton 으로 엄밀하게 풀어 정확해를 감싸고, 제 적분 감싸기와 겹치는지 본다.
2. Lagrange 정삼각 회전해(등질량): 한 변 a, ω² = 3/a³. 정확해와 겹치는지 본다.
3. 8자 궤도: T3 이정표 상자 중심에서 6-syzygy 귀환 사상 P⁶ 의 Newton 고정점을 제 엔진으로 구하고, 반 주기 시각,
   매 syzygy 의 오일러 배치(가운데 물체에서 두 거리 같음), 한 주기 승수, 흐름 관 최소 거리, 이정표 상자 포함을 본다.
4. 음성 대조: 정확해를 제 감싸기 반경의 3배만큼 옮기면 서로소가 되어야 한다.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from flint import arb, ctx  # noqa: E402

from v5 import compare, data  # noqa: E402
from v5.charts import D, chart_out, lift, poincare_derivative  # noqa: E402
from v5.flow import FirstOrder, FlightLog, Traj, energy, flow_to_time, integrate, safe_distances  # noqa: E402

OUT = Path(os.environ.get("V5_VALIDATE_OUT", data.REPO / "results" / "2026-10-17-T11-arbcert" / "validate_engine.json"))


def overlap_report(mine, exact, shift_factor=3):
    """제 공들과 정확해 공들이 모두 겹치는지, 그리고 정확해를 제 반경의 shift_factor 배 옮기면 모두 서로소인지."""
    ok = all(m.overlaps(e) for m, e in zip(mine, exact))
    max_rad = max(float(m.rad()) for m in mine)
    max_err = max(float(abs(m.mid() - e.mid())) for m, e in zip(mine, exact))
    neg = []
    for m, e in zip(mine, exact):
        shift = arb(m.rad()) * shift_factor + arb(e.rad()) * shift_factor + arb(10) ** -300
        moved = e + shift
        neg.append(not m.overlaps(moved))
    return {"overlap_all": ok, "max_rad_mine": max_rad, "max_mid_error": max_err,
            "negative_control_shift_rejected_all": all(neg)}


def kepler_case():
    ctx.prec = 192
    a, e = arb(1), arb(3) / 5
    GM = arb(2)
    n = (GM / a ** 3).sqrt()
    rp = a * (1 - e)
    vp = (GM * (1 + e) / (a * (1 - e))).sqrt()
    rho, drho = (rp, arb(0)), (arb(0), vp)
    x0 = [-rho[0] / 2, -rho[1] / 2, rho[0] / 2, rho[1] / 2, arb(20), arb(0),
          -drho[0] / 2, -drho[1] / 2, drho[0] / 2, drho[1] / 2, arb(0), arb(0)]
    t_end = arb(29) / 4
    t0 = time.time()
    z, steps = flow_to_time(x0, FirstOrder(0, masses=(1, 1, 0)), t_end, order=30, tol=1e-45)
    sec = time.time() - t0
    M = n * t_end
    E = M
    for _ in range(60):                      # 근삿값(점 Newton)
        E = arb((E - (E - e * E.sin() - M) / (1 - e * E.cos())).mid())
    box = E + arb(0, 1e-40)
    for _ in range(5):                       # 구간 Newton(포함이면 유일한 근)
        Nb = E - (E - e * E.sin() - M) / (1 - e * box.cos())
        if not box.contains(Nb):
            raise ArithmeticError("Kepler interval Newton did not contract")
        box = Nb
    E = box
    rx, ry = a * (E.cos() - e), a * (1 - e * e).sqrt() * E.sin()
    den = 1 - e * E.cos()
    vx, vy = -a * n * E.sin() / den, a * n * (1 - e * e).sqrt() * E.cos() / den
    mine = [z[2] - z[0], z[3] - z[1], z[8] - z[6], z[9] - z[7]]
    rep = overlap_report(mine, [rx, ry, vx, vy])
    # 상대 운동 에너지 μ|ρ'|²/2 − m1m2/|ρ| (μ=1/2) 의 정확값은 −m1m2/(2a) = −1/2
    vv = (z[8] - z[6]) * (z[8] - z[6]) + (z[9] - z[7]) * (z[9] - z[7])
    rr = ((z[2] - z[0]) * (z[2] - z[0]) + (z[3] - z[1]) * (z[3] - z[1])).sqrt()
    e_rel = vv / 4 - 1 / rr
    rep.update({"case": "Kepler e=0.6 a=1 GM=2, t=7.25", "steps": steps, "seconds": round(sec, 1),
                "rel_energy_contains_exact": bool(e_rel.contains(arb(-1) / 2)), "rel_energy_rad": float(e_rel.rad())})
    return rep


def lagrange_case():
    ctx.prec = 192
    a = arb(3) / 2
    R = a / arb(3).sqrt()
    w = (3 / a ** 3).sqrt()
    th = [arb(1) / 10 + 2 * arb.pi() * k / 3 for k in range(3)]
    x0 = []
    for k in range(3):
        x0 += [R * th[k].cos(), R * th[k].sin()]
    for k in range(3):
        x0 += [-w * R * th[k].sin(), w * R * th[k].cos()]
    t_end = arb(7) / 2
    t0 = time.time()
    z, steps = flow_to_time(x0, FirstOrder(0), t_end, order=30, tol=1e-45)
    sec = time.time() - t0
    ex = []
    for k in range(3):
        ph = w * t_end + th[k]
        ex += [R * ph.cos(), R * ph.sin()]
    for k in range(3):
        ph = w * t_end + th[k]
        ex += [-w * R * ph.sin(), w * R * ph.cos()]
    rep = overlap_report(z[:12], ex)
    rep.update({"case": "Lagrange equilateral a=1.5, t=3.5", "steps": steps, "seconds": round(sec, 1)})
    return rep


def p6(u, prec=256, order=30, tol=1e-40):
    """syzygy 차트 u(Σ⁺) 에서 6번째 영점(Σ⁺) 까지. 반환 (u6, DP⁶, 통과 시각 6개, 상태 6개, 최소 거리 표본)."""
    ctx.prec = prec
    ud = [D(v, [arb(1) if a == i else arb(0) for a in range(4)]) for i, v in enumerate(u)]
    state, disc = lift(ud, arb(-1) / 2)
    z0 = [s.v for s in state] + [s.g[a] for a in range(4) for s in state]
    tr = Traj(z0)
    log = FlightLog()
    integrate([tr], FirstOrder(4), order=order, tol=tol, crossings_wanted=6, directions=[-1, 1, -1, 1, -1, 1],
              samples_per_step=4, track_until=6, log=log)
    c6 = tr.crossings[5]
    x6, T6 = c6.state[:12], [c6.state[12 + 12 * a: 24 + 12 * a] for a in range(4)]
    u6, Dm = poincare_derivative(x6, T6)
    return u6, Dm, [c.time for c in tr.crossings], [c.state[:12] for c in tr.crossings], tr, log


def figure8_case():
    lo = [Fraction("5.1485679824408320"), Fraction("-2.5892e-10"), Fraction("-4.9268e-11"),
          Fraction("0.43269065071325791")]
    hi = [Fraction("5.1485679826138113"), Fraction("2.5803e-10"), Fraction("4.9437e-11"),
          Fraction("0.43269065080851421")]
    t0 = time.time()
    u = [data.q2arb((l + h) / 2) for l, h in zip(lo, hi)]
    history = []
    for it in range(3):
        u6, Dm, times, states, tr, log = p6(u)
        history.append(max(float(abs((u6[i] - u[i]).mid())) for i in range(4)))
        # 비엄밀 Newton 보정(고정점 근삿값을 찾을 뿐이다. 판정은 마지막 엄밀 감싸기로 한다)
        from flint import arb_mat
        Jarb = [[Dm[i][j] - (1 if i == j else 0) for j in range(4)] for i in range(4)]
        sol = arb_mat(Jarb).solve(arb_mat([[u6[i] - u[i]] for i in range(4)]))
        u = [arb((u[i] - sol[i, 0]).mid()) for i in range(4)]
    u6, Dm, times, states, tr, log = p6(u)
    resid = max(float(abs(u6[i] - u[i]).upper()) for i in range(4))
    halves = [times[0]] + [times[k] - times[k - 1] for k in range(1, 6)]
    t3_value = arb("4.354684703218104547906804")
    euler = []
    for x in states:
        r12, r13, r23 = safe_distances(x)
        euler.append(min(float(abs(r12 - r13).upper()), float(abs(r12 - r23).upper()), float(abs(r13 - r23).upper())))
    Jm = np.array([[float(Dm[i][j].mid()) for j in range(4)] for i in range(4)])
    eig = sorted(np.linalg.eigvals(Jm), key=lambda z: (round(z.real, 6), z.imag))
    in_box = all(data.q2arb(l) < u[i] < data.q2arb(h) for i, (l, h) in enumerate(zip(lo, hi)))
    min_dist_upper = min(float(v.upper()) for v in tr.sample_min)
    tube_lo = min(float(v) for v in log.tube_lo)
    return {
        "case": "figure-eight 6-syzygy fixed point (T3 milestone box centre, 3 Newton steps)",
        "newton_residual_history": history,
        "final_residual_upper": resid,
        "fixed_point_in_T3_box": in_box,
        "fixed_point": [compare.fmt(v, 30) for v in u],
        "half_period_times": [compare.fmt(v, 30) for v in halves],
        "half_times_minus_T3_mpmath_max": max(float(abs(hv - t3_value).upper()) for hv in halves),
        "half_times_in_T3_window": all(arb("4.35468470321768") < hv < arb("4.35468470321852") for hv in halves),
        "euler_configuration_defect_max": max(euler),
        "multipliers_float": [[round(z.real, 15), round(z.imag, 15)] for z in eig],
        "multipliers_T3_quoted_KS": [[0.998599982092038, -0.0528968407920606], [0.998599982092038, 0.0528968407920606],
                                     [-0.297596667518713, -0.954691690275848], [-0.297596667518713, 0.954691690275848]],
        "min_distance_sample_upper": min_dist_upper,
        "min_distance_rigorous_lower": tube_lo,
        "T3_tube_claim": 1.73168936,
        "energy_at_return": compare.fmt(energy(states[-1]), 15),
        "seconds": round(time.time() - t0, 1),
    }


def main():
    quick = "--quick" in sys.argv[1:]
    out = {"kepler": kepler_case(), "lagrange": lagrange_case()}
    if not quick:
        out["figure8"] = figure8_case()
    k, l = out["kepler"], out["lagrange"]
    out["verdict"] = {
        "kepler_ok": k["overlap_all"] and k["negative_control_shift_rejected_all"] and k["rel_energy_contains_exact"],
        "lagrange_ok": l["overlap_all"] and l["negative_control_shift_rejected_all"],
    }
    if not quick:
        f = out["figure8"]
        out["verdict"]["figure8_ok"] = (f["fixed_point_in_T3_box"] and f["half_times_in_T3_window"]
                                        and f["min_distance_rigorous_lower"] > 1.73
                                        and f["final_residual_upper"] < 1e-25)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    sys.exit(0 if all(out["verdict"].values()) else 1)


if __name__ == "__main__":
    main()
