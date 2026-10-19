"""[수치] h=-1/2 8자 궤도의 축소 Poincaré 사상 고정점을 mpmath 기준 해로 Newton 보정하고, 반 구간 6개의 사격점을 만든다.

출력은 Krawczyk 상자의 중심(근사)일 뿐이다. 존재 증명은 milestone_figure8.py 의 [CAP] 가 한다.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction

import mpmath as mp

from common import write_result
from reference import mp_reference as ref
from t3cap import coords

DPS = 40
SIMO_Q = [("0.97000436", "-0.24308753"), ("-0.97000436", "0.24308753"), ("0", "0")]
SIMO_V = [("0.4662036850", "0.4323657300"), ("0.4662036850", "0.4323657300"), ("-0.93240737", "-0.86473146")]
PLUS = ref.SectionData(2, Fraction(0), 1, 5, 1)
MINUS = ref.SectionData(2, Fraction(0), -1, 5, -1)
H = Fraction(-1, 2)
MD = ref.MassData.of((1, 1, 1))


def full_period(u: list) -> list:
    mp.mp.dps = DPS
    out, _ = ref.reduced_map(u, section_in=PLUS, section_out=PLUS, returns=3, md=MD, h=H, J=0)
    return out


def simo_start() -> list:
    mp.mp.dps = DPS
    state = coords.inertial_to_jacobi(SIMO_Q, SIMO_V)
    lam = coords.scale_factor_to_energy(coords.jacobi_energy(*state))
    y, _, _ = coords.jacobi_to_reduced(*coords.scale_jacobi(state, lam))
    return [y[0], y[1], y[3], y[4]]


def newton(u: list, pool: ProcessPoolExecutor, delta=mp.mpf("1e-14")) -> tuple[list, list, mp.matrix]:
    mp.mp.dps = DPS
    shifted = [[u[j] + (delta if j == i else 0) * s for j in range(4)] for i in range(4) for s in (1, -1)]
    results = list(pool.map(full_period, [u] + shifted))
    g = mp.matrix([results[0][i] - u[i] for i in range(4)])
    jac = mp.matrix(4, 4)
    for i in range(4):
        plus, minus = results[1 + 2 * i], results[2 + 2 * i]
        for r in range(4):
            jac[r, i] = (plus[r] - minus[r]) / (2 * delta)
    step = mp.lu_solve(jac - mp.eye(4), -g)
    return [u[i] + step[i] for i in range(4)], [g[i] for i in range(4)], jac


def main() -> None:
    started = time.time()
    mp.mp.dps = DPS
    u = simo_start()
    history = []
    with ProcessPoolExecutor(max_workers=3) as pool:  # 트랙 CPU 예산(BRIEF rev3): worker 3 이하
        for it in range(8):
            u_next, residual, monodromy = newton(u, pool)
            norm = max(abs(v) for v in residual)
            history.append({"iteration": it, "residual_max": mp.nstr(norm, 5)})
            print(it, mp.nstr(norm, 5), flush=True)
            u = u_next
            if norm < mp.mpf("1e-28"):
                break
    eig = mp.eig(monodromy)[0]
    points = [u]
    sections = [PLUS, MINUS] * 3
    times = []
    for k in range(6):
        nxt, crossing = ref.reduced_map(points[-1], section_in=sections[k], section_out=sections[(k + 1) % 6],
                                        returns=1, md=MD, h=H, J=0)
        points.append(nxt)
        times.append(crossing.time)
    closure = max(abs(points[6][i] - points[0][i]) for i in range(4))
    payload = {
        "tag": "[수치] mpmath 기준 해 Newton(비엄밀). 존재 증명 아님",
        "dps": DPS,
        "energy": "-1/2",
        "fixed_point_u": [mp.nstr(v, 36) for v in u],
        "newton_history": history,
        "monodromy_eigenvalues": [mp.nstr(e, 15) for e in eig],
        "monodromy_eigenvalue_moduli": [mp.nstr(abs(e), 15) for e in eig],
        "shooting_points": [[mp.nstr(v, 36) for v in p] for p in points[:6]],
        "shooting_sections": ["plus" if s is PLUS else "minus" for s in sections],
        "segment_times": [mp.nstr(t, 25) for t in times],
        "period": mp.nstr(sum(times), 25),
        "closure_error": mp.nstr(closure, 5),
    }
    write_result("figure8_newton", payload, script=__import__("pathlib").Path(__file__).resolve(),
                 params={"dps": DPS, "energy": "-1/2", "start": "Simo IC scaled to h=-1/2"}, started=started)
    print(payload)


if __name__ == "__main__":
    main()
