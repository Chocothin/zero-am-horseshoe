"""[수치] cycle·호모클리닉 점열 재고정과 4방향 불변 틀. 덮개 그래프 설계의 입력(orbit_refit.json)을 만든다.

호모클리닉은 착지 깊이 K_land 를 한 주기씩 늘리는 연속법으로 다중 사격한다. 뒤쪽 절반 u_{−k}=R u_k 의 단계 결손은
가역성을 따로 가정하지 않고 P̃ 를 직접 적분해 잰다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from t6hs import flow, frames, orbits  # noqa: E402
from t6hs.results import DESIGN_DIR, write_result  # noqa: E402

K_STEPS = (55, 71, 87, 103, 119, 135, 151, 167, 183)


def hexes(a) -> list:
    return [float(v).hex() for v in np.ravel(a)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    pk = orbits.load_package()
    cyc = orbits.refine_cycle(np.array(pk["cycle"]["points"]))
    fr = orbits.eigenframes(cyc)
    stored = np.array(pk["homoclinic"]["points"])[39:]
    continuation = []
    hom = None
    for k_land in K_STEPS:
        guess = orbits.tail_initial_guess(stored, k_land, cyc, fr)
        hom = orbits.solve_homoclinic(stored[0][[0, 3]], guess, cyc, fr)
        j = orbits.cycle_index_forward(k_land)
        landing = np.linalg.solve(fr.vectors[j], hom.forward[-1] - cyc.points[j])
        continuation.append({"k_land": k_land, "y_star_hex": [float(hom.forward[0][0]).hex(), float(hom.forward[0][3]).hex()],
                             "y_star": [hom.forward[0][0], hom.forward[0][3]], "max_defect": float(hom.defects.max()),
                             "landing_eig_coords": landing.tolist()})
        stored = hom.forward[:40]
    k_max = hom.k_land
    full = np.array([flow.REVERSOR @ hom.forward[k] for k in range(k_max, 0, -1)] + list(hom.forward))  # k=−K..K
    steps = [flow.next_syzygy(u, with_jacobian=True) for u in full]
    jac = np.array([s.jacobian for s in steps])
    defects = np.array([np.max(np.abs(steps[i].u - full[i + 1])) for i in range(len(full) - 1)])
    j_start = orbits.cycle_index_forward(k_max)
    j_start = (16 - j_start) % 16  # u_{−K}=R u_K 는 R p_j = p_{−j} 를 따라간다
    j_end = orbits.cycle_index_forward(k_max)
    of = frames.invariant_frames(jac[:-1], fr.vectors[j_start], fr.vectors[j_end])
    ks = np.arange(-k_max, k_max + 1)
    shadow = [((16 - orbits.cycle_index_forward(-k)) % 16) if k < 0 else orbits.cycle_index_forward(k) for k in ks]
    offsets = [np.linalg.solve(fr.vectors[j], u - cyc.points[j]).tolist() for j, u in zip(shadow, full)]
    payload = {
        "tag": "[수치] float64 다중 사격(DOP853 rtol 2.3e-14); 증명 근거 아님",
        "cycle": {"points_hex": [hexes(p) for p in cyc.points], "points": cyc.points, "residual": cyc.residual,
                  "jacobians": cyc.jacobians, "flight_times": cyc.flight_times, "min_distances": cyc.min_distances,
                  "eigenframes": fr.vectors, "step_multipliers": fr.step_multipliers,
                  "period_multipliers": fr.period_multipliers},
        "homoclinic": {"k_max": k_max, "k": ks, "points_hex": [hexes(u) for u in full], "points": full,
                       "jacobians": jac, "flight_times": [s.time for s in steps],
                       "min_distances": [s.min_distance for s in steps], "step_defects": defects,
                       "shadow_cycle_index": shadow, "offset_eig_coords": offsets,
                       "frames": of.vectors, "frame_step_matrices": of.step_matrices, "frame_offdiag": of.offdiag,
                       "angle_u_s_deg": of.angle_u_s_deg, "angle_in_u_deg": of.angle_in_u_deg,
                       "angle_in_s_deg": of.angle_in_s_deg},
        "continuation": continuation,
        "t2_y_star": pk["homoclinic"]["y_star"],
    }
    print("cycle residual", cyc.residual, "| homoclinic max step defect (both halves)", float(defects.max()))
    print("frame offdiag max", float(of.offdiag.max()), "| min angle u/s", float(of.angle_u_s_deg.min()),
          "| min angle in u", float(of.angle_in_u_deg.min()), "| min angle in s", float(of.angle_in_s_deg.min()))
    for c in continuation:
        print(c["k_land"], c["y_star"], c["max_defect"], c["landing_eig_coords"])
    if not args.no_write:
        write_result("orbit_refit", payload, script=Path(__file__).resolve(),
                     params={"k_steps": list(K_STEPS), "rtol": flow.RTOL, "atol": flow.ATOL}, started=started,
                     results_dir=DESIGN_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
