"""T3 c0 상 감싸기와 c1 도함수 감싸기의 폭을 단계마다 선형 예측과 비교한다 [CAP 감싸기 폭 / 선형 예측은 수치].

틀은 덮개 그래프와 같은 고유틀·불변 틀(단위 열)에 모든 성분 반경 r 을 곱한 것이다. 정규 좌표에서 선형 상의 반폭은
행마다 Σ_l |D_kl| 이므로, 감싸기 반폭/그 값이 1 에 가까울수록 꽉 조인다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from t6hs.results import DESIGN_DIR, T3_DIR, write_result  # noqa: E402

sys.path.insert(0, str(T3_DIR))
from t3cap import CoordinateSection, Failure, Frame, Interval, Settings, SystemParams, poincare_c0, poincare_c1  # noqa: E402

MAX_ENGINES = 3  # BRIEF rev3 CPU 예산
SP, SM = CoordinateSection.syzygy(+1), CoordinateSection.syzygy(-1)


def frame(center, vectors, r: Fraction) -> Frame:
    return Frame.make([Fraction(float(v)) for v in center],
                      [[Fraction(float(vectors[a][b])) * r for b in range(4)] for a in range(4)])


def measure(label, c_in, f_in, c_out, f_out, d_lin, r: Fraction) -> dict:
    fin, fout = frame(c_in, f_in, r), frame(c_out, f_out, r)
    box = tuple(Interval(Fraction(-1), Fraction(1)) for _ in range(4))
    point = tuple(Interval(Fraction(0), Fraction(0)) for _ in range(4))
    kw = dict(system=SystemParams.make(), section_in=SP, section_out=SM, frame_in=fin, frame_out=fout,
              settings=Settings())
    c0 = poincare_c0([box, point], **kw)
    c1 = poincare_c1([box], **kw)[0]
    lin_half = np.sum(np.abs(d_lin), axis=1)
    out = {"label": label, "r": str(r)}
    if isinstance(c0[0], Failure) or isinstance(c0[1], Failure) or isinstance(c1, Failure):
        out["failure"] = [getattr(x, "reason", None) for x in (c0[0], c0[1], c1)]
        return out
    out["c0_half_over_linear"] = [float(iv.width) / 2 / h for iv, h in zip(c0[0].image, lin_half)]
    out["c0_point_width_max"] = max(float(iv.width) for iv in c0[1].image)
    widths = np.array([[float(e.width) for e in row] for row in c1.derivative])
    out["c1_width_max"] = float(widths.max())
    out["c1_width_per_r"] = float(widths.max() / float(r))
    out["return_time_width_c0"] = float(c0[0].return_time.width)
    out["tube_min"] = float(c0[0].tube.min_distance_lower)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", default="1/10000000")
    parser.add_argument("--k-range", type=int, default=40, help="호모클리닉 단계 |k| 상한")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    r = Fraction(args.r)
    data = json.loads((DESIGN_DIR / "orbit_refit.json").read_text())
    cyc, hom = data["cycle"], data["homoclinic"]
    pts, fr = np.array(cyc["points"]), np.array(cyc["eigenframes"])
    tasks = []
    for i in range(16):
        j = (i + 1) % 16
        d = np.linalg.solve(fr[j], np.array(cyc["jacobians"][i]) @ fr[i])
        tasks.append((f"cycle {i}", pts[i], fr[i], pts[j], fr[j], d))
    ks = hom["k"]
    hp, hf, hj = np.array(hom["points"]), np.array(hom["frames"]), np.array(hom["jacobians"])
    for idx, k in enumerate(ks[:-1]):
        if abs(k) <= args.k_range and abs(k + 1) <= args.k_range:
            d = np.linalg.solve(hf[idx + 1], hj[idx] @ hf[idx])
            tasks.append((f"hom {k}", hp[idx], hf[idx], hp[idx + 1], hf[idx + 1], d))
    with ThreadPoolExecutor(MAX_ENGINES) as pool:
        rows = list(pool.map(lambda t: measure(*t, r), tasks))
    for row in rows:
        if "failure" in row:
            print(row["label"], "FAIL", row["failure"])
        else:
            print(f'{row["label"]:>9} c0/lin max {max(row["c0_half_over_linear"]):9.1f}  c1 width/r {row["c1_width_per_r"]:.3g}'
                  f'  point {row["c0_point_width_max"]:.2e}  tube {row["tube_min"]:.4f}')
    if not args.no_write:
        write_result("enclosure_diagnosis", {"tag": "[CAP 감싸기 폭] T3 c0·c1, 선형 예측은 [수치]", "r": str(r),
                                             "rows": rows}, script=Path(__file__).resolve(),
                     params={"r": str(r), "k_range": args.k_range}, started=started, results_dir=DESIGN_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
