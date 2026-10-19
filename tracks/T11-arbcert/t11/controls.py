"""G6 음성 대조: 정리 수준 주장마다 전용 대조 ≥2(강화 + 심은 입력)와 날카로움, 변 수준 덮개·원뿔 대조(변마다 3+3).

심은 입력은 저장 기록(흐름 관 행·상자 기록 JSON·설계)을 바꿔 load_records → assemble(exact_judge.judge_v5 포함) →
judge_claims 경로를 그대로 지난다. 상자 기록을 바꿀 때는 레포 밖 임시 폴더에 쓴다.
사용: t11cli.py controls [--out PATH]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import time
from fractions import Fraction
from math import isqrt
from pathlib import Path

import exact_judge as EJ
import v7exact as X

from . import assemble as A
from . import claims as C
from . import design as dz
from .paths import BOXES, RESULTS, TUBES

OUT = RESULTS / "controls.json"
EPS_STRONG = Fraction(1, 10**12)
EPS_PLANT = Fraction(1, 10**6)
GUARD_EXCEPTIONS = (ValueError,)    # 판정 경로가 일부러 내는 가드 예외(v7exact.iv lo>hi, design_signs 0 대각)만 거절로 센다


def guarded(action):
    """심은 입력을 실행해 (결과, 가드 예외)를 돌려준다. 그 밖의 예외는 심기 코드의 버그라 그대로 올린다."""
    try:
        return action(), None
    except GUARD_EXCEPTIONS as exc:
        return None, exc


class Planter:
    """기록 사본을 바꾸고 바뀐 상자 기록을 임시 폴더에 쓴다."""

    def __init__(self, records: dict, tmp: Path):
        self.records = copy.deepcopy(records)
        self.tmp = tmp
        self.n = 0

    def box(self, edge: str) -> dict:
        return json.loads(Path(self.records[edge]["box_path"]).read_text())

    def put_box(self, edge: str, rec: dict) -> None:
        self.n += 1
        p = self.tmp / f"plant{self.n}_{edge.replace('>', '__')}.json"
        p.write_text(json.dumps(rec))
        self.records[edge]["box_path"] = p


def widen_center(rec: dict, delta: Fraction) -> dict:
    """중심 상 성분마다 양쪽으로 delta 넓힌다. 평균값 상의 모든 성분이 delta 만큼 넓어져 덮개 여유가 정확히 delta 줄어든다."""
    out = copy.deepcopy(rec)
    out["center_image_exact"] = [[str(Fraction(lo) - delta), str(Fraction(hi) + delta)]
                                 for lo, hi in rec["center_image_exact"]]
    return out


def scale_entry(rec: dict, i: int, j: int, factor: Fraction) -> dict:
    out = copy.deepcopy(rec)
    lo, hi = (Fraction(x) * factor for x in rec["deriv_mine_exact"][i][j])
    out["deriv_mine_exact"][i][j] = [str(min(lo, hi)), str(max(lo, hi))]
    return out


def binding_rows(rec: dict) -> tuple[int, int]:
    """a* 를 정하는 안정 행 i∈{2,3} 과 b* 를 정하는 불안정 행 i∈{0,1}(Gershgorin 여유가 가장 작은 행)."""
    D = [[X.iv(rec["deriv_mine_exact"][i][j]) for j in range(4)] for i in range(4)]
    J = (1, 1, -1, -1)
    S = [[None] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(i, 4):
            acc = (Fraction(0), Fraction(0))
            for k in range(4):
                t = X.sq(D[k][i]) if i == j else X.mul(D[k][i], D[k][j])
                acc = X.add(acc, t if J[k] > 0 else X.neg(t))
            S[i][j] = S[j][i] = acc
    R = [sum((X.mag(S[i][j]) for j in range(4) if j != i), Fraction(0)) for i in range(4)]
    stable = max((2, 3), key=lambda i: R[i] - S[i][i][0])
    unstable = min((0, 1), key=lambda i: S[i][i][0] - R[i])
    return stable, unstable


def sqrt_digits(x: Fraction, digits: int, up: bool) -> Fraction:
    """√x 를 소수 digits 자리 격자에서 아래(up=False)·위로 자른 값(정수 제곱근, 정확)."""
    scale = 10 ** digits
    n = x * scale * scale
    k = isqrt(n.numerator // n.denominator)
    if up and Fraction(k * k) != n:
        k += 1
    return Fraction(k, scale)


def rate_thresholds(theorem: dict) -> dict:
    """쌍곡 상수 주장의 강화 문턱(바로 너머)과 날카로움 문턱(바로 안쪽): (너머, 안쪽)."""
    eps = C.epsilon(theorem)
    mu_s, mu_u = theorem["T7.mu_s"], theorem["T7.mu_u"]
    lower = {"T7.lambda_s": mu_s, "T7.C_s": 1 / eps, "T7.C_u": mu_u / eps}
    out = {k: (sqrt_digits(x, 30, False) * (1 - EPS_STRONG), sqrt_digits(x, 30, True) * (1 + EPS_STRONG))
           for k, x in lower.items()}
    out["T7.lambda_u"] = (sqrt_digits(mu_u, 30, True) * (1 + EPS_STRONG), sqrt_digits(mu_u, 30, False) * (1 - EPS_STRONG))
    return out


def theorem_controls(records: dict, design: dict, base: dict, tmp: Path) -> tuple[list, list]:
    m = base["measured"]
    checks = {c.name: c for c in base["checks"]}
    rows, sharp = [], []

    def verdict(result, claim, claimed=None, theorem=None):
        return next(c for c in C.judge_claims(result["measured"], claimed or C.CLAIMED, theorem or C.THEOREM)
                    if c.name == claim)

    def record(claim, kind, planted, result=None, claimed=None, theorem=None, exc=None):
        if exc is not None:
            rows.append({"claim": claim, "kind": kind, "planted": planted, "computed": None, "holds": False,
                         "rejected": True, "reason": f"가드 예외: {type(exc).__name__}: {exc}"})
            return
        chk = verdict(result, claim, claimed, theorem)
        rows.append({"claim": claim, "kind": kind, "planted": planted, "computed": str(chk.margin),
                     "computed_float": float(chk.margin), "holds": chk.holds, "rejected": not chk.holds,
                     "reason": "판정 거짓" if not chk.holds else "판정 참(누출)"})

    def run(planter=None, design_override=None):
        recs = planter.records if planter is not None else records
        return A.assemble(recs, design_override or design)

    # (가) 강화 대조와 날카로움: 상수를 다시 잰 T11 값 바로 너머·안쪽으로 옮긴다(rev12, T6·T7 상수를 받치는 주장도 같다)
    for name, chk in checks.items():
        if chk.relation in ("≥", "≤", ">", "<"):
            lower_side = chk.relation in ("≥", ">")
            beyond = chk.measured * (1 + EPS_STRONG) if lower_side else chk.measured * (1 - EPS_STRONG)
            inside = chk.measured * (1 - EPS_STRONG) if lower_side else chk.measured * (1 + EPS_STRONG)
            if name in C.REAL_CLAIMS:
                b_over, i_over = ({**C.CLAIMED, name: beyond}, None), ({**C.CLAIMED, name: inside}, None)
            else:
                b_over, i_over = (None, {**C.THEOREM, name: beyond}), (None, {**C.THEOREM, name: inside})
            record(name, "강화", f"상수 = 다시 잰 T11 값 ×(1{'+' if lower_side else '−'}1e-12)", base, *b_over)
            ok = verdict(base, name, *i_over).holds
            sharp.append({"claim": name, "inside_constant": str(inside), "holds": ok})
    for name, (beyond, inside) in rate_thresholds(C.THEOREM).items():
        record(name, "강화", f"상수 = 경계값(30자리) 바로 너머 {float(beyond):.15g}", base, None, {**C.THEOREM, name: beyond})
        sharp.append({"claim": name, "inside_constant": str(inside),
                      "holds": verdict(base, name, None, {**C.THEOREM, name: inside}).holds})

    def plant_box(claims, edge, label, fn):
        p = Planter(records, tmp)

        def act():
            p.put_box(edge, fn(p.box(edge)))
            return run(p)
        res, exc = guarded(act)
        for claim in claims:
            record(claim, "심은 입력", f"{edge}: {label}", res, exc=exc)

    def plant_tube(claims, edge, label, fn):
        p = Planter(records, tmp)

        def act():
            fn(p.records[edge]["tube"])
            return run(p)
        res, exc = guarded(act)
        for claim in claims:
            record(claim, "심은 입력", f"{edge}: {label}", res, exc=exc)

    def plant_design(claims, label, fn):
        d = copy.deepcopy(design)

        def act():
            fn(d)
            return run(design_override=d)
        res, exc = guarded(act)
        for claim in claims:
            record(claim, "심은 입력", label, res, exc=exc)

    def stable_row(rec):
        return binding_rows(rec)[0]

    def unstable_row(rec):
        return binding_rows(rec)[1]

    # (나) 심은 입력: 그 주장을 정한 변의 기록을 바꿔 조립 경로를 그대로 지난다
    e_cov = m.at["cover_margin"]
    slack = m.values["cover_margin"] - C.CLAIMED["cover_margin"] + EPS_PLANT
    plant_box(["cover_margin"], e_cov, f"중심 상 성분마다 ±{float(slack):.3e} 넓힘(여유를 상수 아래로)",
              lambda r: widen_center(r, slack))
    e_a, e_b = m.at["a_max"], m.at["b_min"]
    plant_box(["mu_s_margin"], e_a, "a* 를 정하는 안정 행의 대각 D(B)[i][i] ×(1+1e-3)",
              lambda r: scale_entry(r, stable_row(r), stable_row(r), Fraction(1001, 1000)))
    plant_box(["mu_u_margin"], e_b, "b* 를 정하는 불안정 행의 대각 D(B)[i][i] ×(1−1e-3)",
              lambda r: scale_entry(r, unstable_row(r), unstable_row(r), Fraction(999, 1000)))
    plant_box(["T7.mu_s", "T7.lambda_s", "T7.C_s"], e_a, "a* 를 정하는 안정 행의 대각 D(B)[i][i] ×1.05(a* 가 μ_s 를 넘게)",
              lambda r: scale_entry(r, stable_row(r), stable_row(r), Fraction(105, 100)))
    plant_box(["T7.mu_u", "T7.lambda_u", "T7.C_u"], e_b, "b* 를 정하는 불안정 행의 대각 D(B)[i][i] ×0.95(b* 가 μ_u 아래로)",
              lambda r: scale_entry(r, unstable_row(r), unstable_row(r), Fraction(95, 100)))
    e_g = m.at["g1"]
    a_g, b_g = Fraction(base["judged"][e_g]["a_star_exact"]), Fraction(base["judged"][e_g]["b_star_exact"])
    if 1 - a_g <= b_g - 1:
        plant_box(["g1"], e_g, "1−a* 쪽: 안정 행 대각 ×(1+1e-3)",
                  lambda r: scale_entry(r, stable_row(r), stable_row(r), Fraction(1001, 1000)))
    else:
        plant_box(["g1"], e_g, "b*−1 쪽: 불안정 행 대각 ×(1−1e-3)",
                  lambda r: scale_entry(r, unstable_row(r), unstable_row(r), Fraction(999, 1000)))
    e_d, p_d = m.at["delta"].split(":")
    plant_tube(["delta"], e_d, f"관 {p_d} 하한 = T11 상수 ×(1−1e-6)",
               lambda t: t["r_lower"].__setitem__(p_d, str(C.CLAIMED["delta"] * (1 - EPS_PLANT))))
    plant_tube(["T6.delta"], e_d, f"관 {p_d} 하한 = T6 상수 ×(1−1e-6)",
               lambda t: t["r_lower"].__setitem__(p_d, str(C.THEOREM["T6.delta"] * (1 - EPS_PLANT))))
    e_R, p_R = m.at["R"].split(":")
    plant_tube(["R"], e_R, f"관 {p_R} 상한 = T11 상수 ×(1+1e-6)",
               lambda t: t["r_upper"].__setitem__(p_R, str(C.CLAIMED["R"] * (1 + EPS_PLANT))))
    plant_tube(["T6.R"], e_R, f"관 {p_R} 상한 = T6 상수 ×(1+1e-6)",
               lambda t: t["r_upper"].__setitem__(p_R, str(C.THEOREM["T6.R"] * (1 + EPS_PLANT))))
    plant_tube(["t_min"], m.at["t_min"], "귀환 창 하한 = T11 상수 ×(1−1e-6)",
               lambda t: t["window"].__setitem__(0, str(C.CLAIMED["t_min"] * (1 - EPS_PLANT))))
    plant_tube(["T6.t_min"], m.at["t_min"], "귀환 창 하한 = T6 상수 ×(1−1e-6)",
               lambda t: t["window"].__setitem__(0, str(C.THEOREM["T6.t_min"] * (1 - EPS_PLANT))))
    plant_tube(["tau_max"], m.at["tau_max"], "귀환 창 상한 = 상수 ×(1+1e-6)",
               lambda t: t["window"].__setitem__(1, str(C.CLAIMED["tau_max"] * (1 + EPS_PLANT))))
    near = m.at["A3_gap"]
    plant_design(["A3_gap"], f"가장 가까운 정점 {near} 의 h-set 을 M+0 의 것으로 바꿈",
                 lambda d: d["vertices"].__setitem__(near, copy.deepcopy(d["vertices"]["M+0"])))
    p6, q6 = m.at["A6_gap"].split("·")
    plant_design(["A6_gap"], f"{q6} 의 h-set 을 {p6} 의 것으로 바꿈",
                 lambda d: d["vertices"].__setitem__(q6, copy.deepcopy(d["vertices"][p6])))
    # 정수 주장: 심은 입력 2개씩
    p = Planter(records, tmp)
    p.records["M+0>M+1"]["box_path"] = None
    res = run(p)
    for claim in ("A1", "T7.A5"):
        record(claim, "심은 입력", "M+0>M+1 상자 기록을 입력에서 뺌", res)
    n01 = json.loads(Path(records["N0>N1"]["box_path"]).read_text())
    m01 = Fraction(base["judged"]["N0>N1"]["cover_margin_mv_exact"])
    plant_box(["A1"], "N0>N1", "중심 상을 그 변의 덮개 여유 +1e-6 만큼 넓힘(그 변 덮개 거짓)",
              lambda r: widen_center(n01, m01 + EPS_PLANT))
    plant_box(["T7.A5"], "N0>N1", "D(B) 행 0↔3 교환(그 변 g(1) ≤ 0)", lambda r: swap_rows(r, 0, 3))
    plant_tube(["A2"], "N1>N2", "흐름 관 인증 실패로 바꿈",
               lambda t: t.update({"certified": False, "reason": "planted"}))
    plant_tube(["A2"], "N0>N1", "V2 귀환 창을 V5 외피 창 밖(창 폭 3배 뒤)으로 옮김",
               lambda t: t.__setitem__("window", [str(Fraction(t["window"][1]) + 3 * (Fraction(t["window"][1])
                                                  - Fraction(t["window"][0])) + EPS_PLANT),
                                                  str(Fraction(t["window"][1]) + 4 * (Fraction(t["window"][1])
                                                  - Fraction(t["window"][0])) + 2 * EPS_PLANT)]))
    plant_box(["A2"], "M+0>M+1", "V5 도착 근 하한을 음수로 바꿈",
              lambda r: {**r, "branch_out_hull_exact": ["-1/1000", r["branch_out_hull_exact"][1]]})
    plant_design(["A4"], "설계 변 N15>N0 제거",
                 lambda d: d.__setitem__("edges", [e for e in d["edges"] if (e["source"], e["target"]) != ("N15", "N0")]))
    plant_design(["A4"], "출발 접합 N8>M-148 을 N9>M-148 로 바꿈",
                 lambda d: [e.__setitem__("source", "N9") for e in d["edges"]
                            if (e["source"], e["target"]) == ("N8", "M-148")])
    return rows, sharp


def swap_rows(rec: dict, a: int, b: int) -> dict:
    out = copy.deepcopy(rec)
    for key in ("center_image_exact", "deriv_mine_exact", "image_whole_exact"):
        out[key][a], out[key][b] = out[key][b], out[key][a]
    return out


def edge_controls(records: dict, base: dict, signs: dict, tmp: Path) -> list:
    """변마다 덮개 대조 3개((가) 문턱을 여유×(1+1e-12) 로, (나-1) 상 좌표 1↔2, (나-2) 0↔3)와 원뿔 대조 3개
    ((가) 원뿔 여유 문턱, (나-1) D(B) 행 1↔2, (나-2) 행 0↔3). 교환은 저장 기록에 심어 judge_v5 로 다시 판정한다."""
    out = []
    for e, rec in records.items():
        path = rec["box_path"]
        if path is None:
            continue
        j = base["judged"][e]
        cov = Fraction(j["cover_margin_mv_exact"])
        cone = min(C.THEOREM["T7.mu_s"] - Fraction(j["a_star_exact"]), Fraction(j["b_star_exact"]) - C.THEOREM["T7.mu_u"])
        box = json.loads(Path(path).read_text())
        row = {"edge": e, "cover_margin": str(cov), "cone_margin": str(cone),
               "cover_threshold_rejected": not (cov > cov * (1 + EPS_STRONG)),
               "cone_threshold_rejected": not (cone > cone * (1 + EPS_STRONG)),
               "sharp_cover": cov > cov * (1 - EPS_STRONG), "sharp_cone": cone > cone * (1 - EPS_STRONG)}
        for tag, (a, b) in (("12", (1, 2)), ("03", (0, 3))):
            sw = swap_rows(box, a, b)
            p = tmp / f"swap{tag}_{e.replace('>', '__')}.json"
            p.write_text(json.dumps(sw))
            js = EJ.judge_v5(p, signs)
            row[f"cover_swap{tag}_rejected"] = not js["cover_proven_mv"]
            row[f"cone_swap{tag}_rejected"] = not js["cone_proven"]
            row[f"cover_swap{tag}_computed"] = js["cover_margin_mv"]
        out.append(row)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tubes", default=str(TUBES))
    ap.add_argument("--boxes", default=str(BOXES))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    started = time.time()
    design = dz.design()
    records = A.load_records(Path(args.tubes), Path(args.boxes), design)
    base = A.assemble(records, design)
    if not all(c.holds for c in base["checks"]):
        print("claims do not hold on the real data; controls are meaningless", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        rows, sharp = theorem_controls(records, design, base, Path(tmp))
        erows = edge_controls(records, base, EJ.design_signs(design), Path(tmp))
    claims = [c.name for c in base["checks"]]
    cover = {c: sum(1 for r in rows if r["claim"] == c and r["rejected"]) for c in claims}
    leaked = [r for r in rows if not r["rejected"]]
    edge_keys = ("cover_threshold_rejected", "cover_swap12_rejected", "cover_swap03_rejected",
                 "cone_threshold_rejected", "cone_swap12_rejected", "cone_swap03_rejected")
    edge_leaked = [(r["edge"], k) for r in erows for k in edge_keys if not r[k]]
    out = {
        "tag": "G6 음성 대조(정리 수준 주장마다 ≥2, 변 수준 덮개·원뿔 각 3)",
        "theorem_controls": rows, "sharpness": sharp,
        "controls_cover": cover, "controls_cover_ok": all(v >= 2 for v in cover.values()),
        "leaked": len(leaked), "sharp_all": all(s["holds"] for s in sharp),
        "edge_controls": {"edges": len(erows), "controls": len(erows) * len(edge_keys), "leaked": edge_leaked,
                          "sharp_cover_all": all(r["sharp_cover"] for r in erows),
                          "sharp_cone_all": all(r["sharp_cone"] for r in erows)},
        "edge_rows": erows,
        "seconds": round(time.time() - started, 1),
    }
    out["pass"] = (out["controls_cover_ok"] and not leaked and out["sharp_all"] and not edge_leaked
                   and out["edge_controls"]["sharp_cover_all"] and out["edge_controls"]["sharp_cone_all"])
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("controls_cover", "controls_cover_ok", "leaked", "sharp_all", "pass")},
                     ensure_ascii=False))
    print(json.dumps({k: v for k, v in out["edge_controls"].items() if k != "leaked"}, ensure_ascii=False),
          "edge leaked:", len(edge_leaked))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
