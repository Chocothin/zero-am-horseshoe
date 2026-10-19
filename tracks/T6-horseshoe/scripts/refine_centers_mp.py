"""[수치] h-set 중심 고정밀 재고정: cycle·호모클리닉 다중 사격 잔차를 mpmath 40자리 P̃ 로 재고, 야코비안은 float64 값을 쓴다.

float64 P̃(DOP853 rtol 2.3e-14)는 p₀→p₁ 같은 긴 근접 비행에서 참값과 약 1e-12 어긋난다. 사슬 h-set 반경이 1.6e-12 까지
작으므로 중심 사이 오프셋 δ=P̃(c_v)−c_w 를 1e-15 아래로 맞춰야 한다. 결과는 double 로 반올림한 중심과, 그 중심의 mp 상
P̃(c_v)(설계 오프셋 계산용)를 orbit_refit_mp.json 에 남긴다. 호모클리닉 뒤쪽 절반은 u_{−k}=R u_k 이다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from t6hs import orbits, reference  # noqa: E402
from t6hs.results import DESIGN_DIR, write_result  # noqa: E402

DIGITS = 45


def mp_image(u_str):
    """문자열 좌표(45자리) → mp P̃ 상(문자열)과 비행 시간."""
    mp.mp.dps = reference.DPS
    u = [mp.mpf(s) for s in u_str]
    fl = reference.next_syzygy(u)
    return [mp.nstr(x, DIGITS) for x in fl.u_out], mp.nstr(fl.time, 20)


def to_str(vec):
    return [mp.nstr(mp.mpf(x), DIGITS) for x in vec]


def images(pool, nodes):
    return list(pool.map(mp_image, [to_str(n) for n in nodes]))


def newton_cycle(pool, pts, jac, iterations=4, log=print):
    m = len(pts)
    for it in range(iterations):
        imgs = images(pool, pts)
        res = [[mp.mpf(a) - b for a, b in zip(imgs[i][0], pts[(i + 1) % m])] for i in range(m)]
        worst = max(abs(x) for r in res for x in r)
        log(f"cycle iteration {it}: max residual {mp.nstr(worst, 5)}")
        if worst < mp.mpf("1e-30"):
            break
        big = np.zeros((4 * m, 4 * m))
        for i in range(m):
            big[4 * i:4 * i + 4, 4 * i:4 * i + 4] = jac[i]
            j = (i + 1) % m
            big[4 * i:4 * i + 4, 4 * j:4 * j + 4] -= np.eye(4)
        delta = np.linalg.solve(big, np.array([float(x) for r in res for x in r]))
        pts = [[p[c] - mp.mpf(float(delta[4 * i + c])) for c in range(4)] for i, p in enumerate(pts)]
    return pts


def newton_homoclinic(pool, nodes, jac, cycle_pt, frame_inv, iterations=4, log=print):
    """미지수 (a, e, u_1..u_K): P̃(u_k)=u_{k+1}, 착지 cycle 점 고유좌표 불안정 성분 2개 = 0."""
    k_land = len(nodes) - 1
    n = 2 + 4 * k_land
    for it in range(iterations):
        imgs = images(pool, nodes[:-1])
        res = [[mp.mpf(a) - b for a, b in zip(imgs[k][0], nodes[k + 1])] for k in range(k_land)]
        diff = [nodes[-1][c] - cycle_pt[c] for c in range(4)]
        landing = [mp.fsum(mp.mpf(float(frame_inv[r][c])) * diff[c] for c in range(4)) for r in range(2)]
        worst = max(max(abs(x) for r in res for x in r), max(abs(x) for x in landing))
        log(f"homoclinic iteration {it}: max residual {mp.nstr(worst, 5)}")
        if worst < mp.mpf("1e-30"):
            break
        big = np.zeros((n, n))
        big[0:4, 0:2] = jac[0][:, [0, 3]]
        for k in range(k_land):
            row = 4 * k
            if k >= 1:
                big[row:row + 4, 2 + 4 * (k - 1):2 + 4 * k] = jac[k]
            big[row:row + 4, 2 + 4 * k:2 + 4 * (k + 1)] = -np.eye(4)
        big[4 * k_land:, 2 + 4 * (k_land - 1):] = frame_inv[:2]
        rhs = np.array([float(x) for r in res for x in r] + [float(x) for x in landing])
        delta = np.linalg.solve(big, rhs)
        nodes[0] = [nodes[0][0] - mp.mpf(float(delta[0])), mp.mpf(0), mp.mpf(0), nodes[0][3] - mp.mpf(float(delta[1]))]
        for k in range(1, k_land + 1):
            nodes[k] = [nodes[k][c] - mp.mpf(float(delta[2 + 4 * (k - 1) + c])) for c in range(4)]
    return nodes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    mp.mp.dps = reference.DPS
    refit = json.loads((DESIGN_DIR / "orbit_refit.json").read_text())
    cyc, hom = refit["cycle"], refit["homoclinic"]
    k_max = hom["k_max"]
    full_pts = [[mp.mpf(float.fromhex(h)) for h in row] for row in hom["points_hex"]]
    full_jac = np.array(hom["jacobians"])
    fwd = full_pts[k_max:]
    fwd_jac = full_jac[k_max:]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cpts = newton_cycle(pool, [[mp.mpf(float.fromhex(h)) for h in row] for row in cyc["points_hex"]],
                            np.array(cyc["jacobians"]))
        j_land = orbits.cycle_index_forward(k_max)
        frame_inv = np.linalg.inv(np.array(cyc["eigenframes"])[j_land])
        fwd = newton_homoclinic(pool, fwd, fwd_jac, cpts[j_land], frame_inv)
        cycle_d = [[float(x) for x in p] for p in cpts]
        fwd_d = [[float(x) for x in u] for u in fwd]
        # 뒤쪽 절반 R u_k: b, c 부호만 바뀌어 double 에서도 정확하다
        full_d = [[u[0], -u[1], -u[2], u[3]] for u in fwd_d[:0:-1]] + fwd_d
        cyc_img = images(pool, cycle_d)
        hom_img = images(pool, full_d)
    defect = lambda img, nxt: max(abs(mp.mpf(a) - mp.mpf(b)) for a, b in zip(img, nxt))
    cyc_def = [defect(cyc_img[i][0], cycle_d[(i + 1) % 16]) for i in range(16)]
    hom_def = [defect(hom_img[i][0], full_d[i + 1]) for i in range(len(full_d) - 1)]
    print("double-rounded centres: cycle max defect", mp.nstr(max(cyc_def), 4), "| homoclinic max defect",
          mp.nstr(max(hom_def), 4))
    print("shift from float refit: cycle", max(abs(a - float.fromhex(b)) for p, q in zip(cycle_d, cyc["points_hex"])
                                               for a, b in zip(p, q)),
          "| homoclinic", max(abs(a - float.fromhex(b)) for p, q in zip(full_d, hom["points_hex"]) for a, b in zip(p, q)))
    payload = {
        "tag": "[수치] mpmath 40자리 다중 사격 재고정(야코비안은 float64)",
        "cycle": {"points_hex": [[float(x).hex() for x in p] for p in cycle_d], "images_mp": [i[0] for i in cyc_img],
                  "flight_times_mp": [i[1] for i in cyc_img], "max_defect": mp.nstr(max(cyc_def), 6)},
        "homoclinic": {"k": hom["k"], "points_hex": [[float(x).hex() for x in u] for u in full_d],
                       "images_mp": [i[0] for i in hom_img], "flight_times_mp": [i[1] for i in hom_img],
                       "max_defect": mp.nstr(max(hom_def), 6)},
        "y_star_mp": [mp.nstr(fwd[0][0], DIGITS), mp.nstr(fwd[0][3], DIGITS)],
    }
    if not args.no_write:
        write_result("orbit_refit_mp", payload, script=Path(__file__).resolve(),
                     params={"dps": reference.DPS, "order": reference.ORDER}, started=started, results_dir=DESIGN_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
