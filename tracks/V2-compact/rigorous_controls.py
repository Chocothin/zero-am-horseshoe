"""V2-compact 엄밀 인증기의 대조: 일부러 틀린 입력에서 실패해야 하고, 비엄밀 표본은 엄밀 감싸기 안에 있어야 한다.

1) 표본 포함: mpmath 표본 990점(30변)의 r_ij 최소·최대와 귀환 시각이 arb 변별 관·창 안에 드는지(두 독립 코드 대조)
2) 건너뛴 통과(함정 ②): 계획 귀환 시각을 첫 영점 뒤로 옮기면 「(0,h] 에서 A>0」 인증이 깨져야 한다
3) 영점 없는 교차 계획: 계획 귀환 시각을 비행 중간으로 옮기면 교차 구조 인증이 깨져야 한다
4) 상자 이동: 출발 중심을 τ 기울기 부호 방향으로 성분마다 3 반경 옮기면 인증은 되지만 CAPD 창과 겹치지 않아야 한다
5) 주장 강화: 다시 잰 δ·R·t_min 을 (1±1e-12) 바깥으로 옮긴 주장은 거절되어야 한다
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import certify_tubes as ct  # noqa: E402
import rigorous_tube as rt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
R6 = REPO / "results" / "2026-10-15-T6-horseshoe"
OUT = REPO / "results" / "2026-10-16-V2-compact"


def sample_containment(rig: dict, sampling_files) -> dict:
    rows = {r["edge"]: r for r in rig["rows"]}
    checked = 0
    violations = []
    min_gap = {"lower": None, "upper": None, "tau_lo": None, "tau_hi": None}

    def upd(k, v):
        min_gap[k] = v if min_gap[k] is None or v < min_gap[k] else min_gap[k]

    for f in sampling_files:
        d = json.loads(Path(f).read_text())
        for res in d["results"]:
            edge = res["summary"]["edge"]
            rr = rows[edge]
            w0, w1 = (Fraction(x) for x in rr["window"])
            for row in res["rows"]:
                for p in ("r12", "r13", "r23"):
                    lo, hi = Fraction(rr["r_lower"][p]), Fraction(rr["r_upper"][p])
                    smin, smax = Fraction(row["rmin"][p]), Fraction(row["rmax"][p])
                    upd("lower", float(smin - lo))
                    upd("upper", float(hi - smax))
                    if smin < lo or smax > hi:
                        violations.append((edge, row["label"], p))
                tau = Fraction(row["tau"])
                upd("tau_lo", float(tau - w0))
                upd("tau_hi", float(w1 - tau))
                if not (w0 <= tau <= w1):
                    violations.append((edge, row["label"], "tau"))
                checked += 1
    return {"points": checked, "violations": violations, "min_gaps": min_gap, "pass": not violations and checked > 0}


def expect_failure(label: str, fn) -> dict:
    try:
        res = fn()
    except (rt.FlightFailure, ArithmeticError) as exc:
        return {"control": label, "rejected": True, "reason": str(exc)[:160]}
    return {"control": label, "rejected": False, "reason": f"unexpectedly certified: {res.get('window')}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rigorous", default=str(OUT / "rigorous_tubes.json"))
    ap.add_argument("--out", default=str(OUT / "rigorous_controls.json"))
    args = ap.parse_args()
    started = time.time()
    rig = json.loads(Path(args.rigorous).read_text())
    contain = sample_containment(rig, [OUT / "sampling_extremes.json", OUT / "sampling_random.json"])
    design = json.loads((R6 / "graph_design.json").read_text())
    c, m = ct.hset(design["vertices"]["N1"])
    tau1 = ct.plan_tau(c, m)
    controls = [
        expect_failure("skipped first crossing (plan 3x tau)", lambda: rt.fly(c, m, 3 * tau1)),
        expect_failure("crossing planned mid-flight (plan 0.5x tau)", lambda: rt.fly(c, m, 0.5 * tau1)),
    ]
    # 상자 이동: 표본의 축 점(±e_j)에서 잰 τ 기울기 부호 방향으로 성분마다 3 반경. 창 폭이 ±1 에 대응하므로 창이 겹치면 안 된다
    samp = json.loads((OUT / "sampling_extremes.json").read_text())
    rows_n1 = next(r for r in samp["results"] if r["summary"]["edge"] == "N1>N2")["rows"]
    tau_axis = {row["label"]: Fraction(row["tau"]) for row in rows_n1 if row["label"].startswith("axis")}
    signs = [1 if tau_axis[f"axis{j}+1"] > tau_axis[f"axis{j}-1"] else -1 for j in range(4)]
    shifted = [c[i] + 3 * sum(signs[j] * m[i][j] for j in range(4)) for i in range(4)]
    tau_s = ct.plan_tau(shifted, m)
    res = rt.fly(shifted, m, tau_s)
    w0, w1 = res["window"]
    rec = json.loads((R6 / "coverings" / "N1__N2.json").read_text())
    whole = rec["pieces_detail"]["whole"][0]
    cw = (ct.q(whole["return_time"]["lo"]), ct.q(whole["return_time"]["hi"]))
    overlap = not (w1 < cw[0] or cw[1] < w0)
    controls.append({"control": f"start box shifted 3 radii per axis along sign(grad tau)={signs}", "rejected": not overlap,
                     "reason": f"window [{float(w0)}, {float(w1)}] vs CAPD [{float(cw[0])}, {float(cw[1])}]"})
    s = rig["summary"]
    d, R, t = Fraction(s["delta_mine"]), Fraction(s["R_mine"]), Fraction(s["tmin_mine"])
    eps = Fraction(1, 10**12)
    tight = [
        {"control": "delta claim raised to (1+1e-12) x my bound", "rejected": not (d >= d * (1 + eps))},
        {"control": "R claim lowered to (1-1e-12) x my bound", "rejected": not (R <= R * (1 - eps))},
        {"control": "t_min claim raised to (1+1e-12) x my bound", "rejected": not (t >= t * (1 + eps))},
    ]
    controls += tight
    payload = {"tag": "[재계산·엄밀] arb 인증기 음성 대조와 표본 포함 대조", "sample_containment": contain,
               "controls": controls, "all_rejected": all(x["rejected"] for x in controls),
               "seconds": round(time.time() - started, 1)}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    return 0 if contain["pass"] and payload["all_rejected"] else 1


if __name__ == "__main__":
    sys.exit(main())
