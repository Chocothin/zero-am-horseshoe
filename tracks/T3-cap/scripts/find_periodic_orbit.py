"""[수치] Šuvakov–Dmitrašinović 형식 초기값(q1=(-1,0), q2=(1,0), q3=0, v1=v2=(vx,vy), v3=-2v1)을 h=-1/2 축소 좌표로
옮기고, syzygy 단면 축소 사상 P^n 의 고정점을 mpmath 기준 해 Newton 으로 보정한다. 승수로 쌍곡성을 분류한다(비엄밀)."""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import mpmath as mp

from common import write_result
from reference import mp_reference as ref
from t3cap import coords

DPS = 40
MD = ref.MassData.of((1, 1, 1))
H = Fraction(-1, 2)


def start_point(vx: str, vy: str) -> tuple[list, int, mp.mpf]:
    """(u, 방향, 크기 조절 인자 lambda). q3 가 가운데인 오일러 배치라 y=0 단면 위에서 출발한다."""
    mp.mp.dps = DPS
    q = [("-1", "0"), ("1", "0"), ("0", "0")]
    v = [(vx, vy), (vx, vy), (str(-2 * mp.mpf(vx)), str(-2 * mp.mpf(vy)))]
    state = coords.inertial_to_jacobi(q, v)
    lam = coords.scale_factor_to_energy(coords.jacobi_energy(*state))
    y, J, _ = coords.jacobi_to_reduced(*coords.scale_jacobi(state, lam))
    direction = 1 if y[5] > 0 else -1
    return [y[0], y[1], y[3], y[4]], direction, lam


def section(direction: int) -> ref.SectionData:
    return ref.SectionData(2, Fraction(0), direction, 5, direction)


def count_returns(u: list, direction: int, period: mp.mpf) -> int:
    """대략 한 주기 동안 같은 방향 통과 횟수 — 반환 횟수 n 을 정한다."""
    mp.mp.dps = 30
    n, total = 0, mp.mpf(0)
    while True:
        out, c = ref.reduced_map(u, section_in=section(direction), section_out=section(direction), returns=1,
                                 md=MD, h=H, J=0, order=28)
        total += c.time
        n += 1
        if total > period * mp.mpf("0.97"):
            return n
        u = out


def period_map(args) -> list:
    u, direction, n = args
    mp.mp.dps = DPS
    out, _ = ref.reduced_map(u, section_in=section(direction), section_out=section(direction), returns=n, md=MD,
                             h=H, J=0, order=32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("vx")
    parser.add_argument("vy")
    parser.add_argument("period", help="문헌 주기(원래 에너지 기준)")
    args = parser.parse_args()
    started = time.time()
    mp.mp.dps = DPS
    u, direction, lam = start_point(args.vx, args.vy)
    T_scaled = mp.mpf(args.period) * lam ** mp.mpf(1.5)
    n = count_returns(u, direction, T_scaled)
    print("returns", n, "scaled period ~", mp.nstr(T_scaled, 10), flush=True)
    history = []
    delta = mp.mpf("1e-14")
    with ProcessPoolExecutor(max_workers=3) as pool:  # 트랙 CPU 예산(BRIEF rev3): worker 3 이하
        for it in range(12):
            shifted = [[u[j] + (delta if j == i else 0) * s for j in range(4)] for i in range(4) for s in (1, -1)]
            res = list(pool.map(period_map, [(p, direction, n) for p in [u] + shifted]))
            g = mp.matrix([res[0][i] - u[i] for i in range(4)])
            jac = mp.matrix(4, 4)
            for i in range(4):
                for r in range(4):
                    jac[r, i] = (res[1 + 2 * i][r] - res[2 + 2 * i][r]) / (2 * delta)
            norm = max(abs(v) for v in g)
            history.append(mp.nstr(norm, 5))
            print(it, mp.nstr(norm, 5), flush=True)
            if norm < mp.mpf("1e-26"):
                break
            step = mp.lu_solve(jac - mp.eye(4), -g)
            u = [u[i] + step[i] for i in range(4)]
    eig = mp.eig(jac)[0]
    _, crossing = ref.reduced_map(u, section_in=section(direction), section_out=section(direction), returns=n,
                                  md=MD, h=H, J=0, order=32)
    moduli = sorted((abs(e) for e in eig), reverse=True)
    real = [abs(mp.im(e)) < mp.mpf("1e-12") for e in eig]
    kind = ("HH" if all(real) and moduli[0] > 1 and moduli[1] > 1 else
            "loxodromic" if not any(real) and moduli[0] > 1 + mp.mpf("1e-9") else
            "EE" if all(abs(m - 1) < mp.mpf("1e-9") for m in moduli) else "HE/other")
    payload = {
        "tag": "[수치] mpmath 기준 해 Newton(비엄밀)",
        "source": f"Suvakov-Dmitrasinovic initial condition vx={args.vx}, vy={args.vy}, T={args.period}",
        "name": args.name, "dps": DPS, "energy": "-1/2", "direction": direction, "returns": n,
        "fixed_point_u": [mp.nstr(v, 34) for v in u], "newton_residuals": history,
        "period": mp.nstr(crossing.time, 25), "multipliers": [mp.nstr(e, 15) for e in eig],
        "multiplier_moduli": [mp.nstr(m, 15) for m in moduli], "classification": kind,
        "min_distances_sampled": [mp.nstr(v, 10) for v in crossing.min_distances],
    }
    write_result(f"orbit_{args.name}", payload, script=Path(__file__).resolve(),
                 params={"name": args.name, "vx": args.vx, "vy": args.vy, "period": args.period, "dps": DPS},
                 started=started)
    print(payload)


if __name__ == "__main__":
    main()
