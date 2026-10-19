"""G8 독립 재계산: 이정표의 판정 부등식을 python-flint(Arb 공 산술)로 다시 판정한다.

입력은 milestone_figure8.json 에 정확 끝점으로 남긴 CAPD 감싸기(F(c), DF(X), 반환별 판별식·도착 근·반환 시간·흐름 관)다.
전처리 C 계산, Krawczyk 연산자 K, K ⊂ int X 판정, 사용 자격 부등식은 t3cap(정확 유리수)을 쓰지 않고 Arb 로만 한다.
CAPD 의 흐름·Poincaré·차트 감싸기 자체는 다시 만들지 않는다(그 범위는 CAPD 를 믿는다).
"""
from __future__ import annotations

import json
import sys
import time
from fractions import Fraction
from pathlib import Path

from flint import arb, arb_mat, ctx, fmpq

from common import RESULTS_DIR, TRACK_DIR, sha256_file, write_result
from reference import arb_flow

PREC_BITS = 256
SEGMENTS = 6
# 사실은 참인데 CAPD double F(c) 폭(약 2e-12)이 상자보다 커서만 실패하는 해상도 대조 — 폭 1e-36 인 독립 F 로는 증명되어야 한다
RESOLUTION_CONTROLS = {"neg: rho too small 1e-14"}


def exact(text: str) -> Fraction:
    """결과 파일의 정확 끝점 표기(16진 double 또는 p/q)를 유리수로. t3cap.export 와 따로 구현한다."""
    if text.lower().lstrip("+-").startswith("0x"):
        return Fraction(float.fromhex(text))
    return Fraction(text)


def ball(entry: dict) -> arb:
    """[lo, hi] 를 담는 Arb 공. 두 끝점의 합집합이라 바깥쪽이다."""
    lo, hi = exact(entry["lo"]), exact(entry["hi"])
    return arb(fmpq(lo.numerator, lo.denominator)).union(arb(fmpq(hi.numerator, hi.denominator)))


def certify(entries: list[dict], min_distance: Fraction) -> tuple[bool, list[str]]:
    """사용 자격 부등식을 Arb 비교(확실할 때만 참)로 다시 본다."""
    delta = arb(fmpq(min_distance.numerator, min_distance.denominator))
    failed = []
    for k, e in enumerate(entries):
        rt = ball(e["return_time"])
        checks = {
            "discriminant": ball(e["discriminant"]) > 0,
            "branch_out": ball(e["branch_out"]) > 0,
            "return_time": rt > 0,
            "tube_covers": ball(e["tube_t_end"]).lower() >= rt.upper(),
            "r12": ball(e["tube_r12"]) > delta,
            "r13": ball(e["tube_r13"]) > delta,
            "r23": ball(e["tube_r23"]) > delta,
        }
        failed += [f"{k}:{name}" for name, ok in checks.items() if not ok]
    return not failed, failed


def independent_f(inputs: dict) -> tuple[list[arb], int]:
    """F_k = P_k(c_k) - c_{k+1} 를 CAPD 없이 Arb 적분기(reference/arb_flow.py)로 감싼다. syzygy 방향은 +,-,+,-,+,-."""
    centers = [[Fraction(v) for v in c] for c in inputs["centers"]]
    md = arb_flow.Masses.of([Fraction(m) for m in inputs["masses"]])
    h, J = Fraction(inputs["energy"]), Fraction(inputs["angular_momentum"])
    out, steps = [], 0
    for k in range(SEGMENTS):
        start = 1 if k % 2 == 0 else -1
        x0 = arb_flow.chart_syzygy(centers[k], start, md, h, J)
        crossing = arb_flow.flow_to_next_syzygy(x0, md, direction=-start)
        u = arb_flow.reduced_chart_coords(crossing.state)
        out += [ui - arb_flow.q(ci) for ui, ci in zip(u, centers[(k + 1) % SEGMENTS])]
        steps += crossing.steps
    return out, steps


def krawczyk(inputs: dict, f_override: list[arb] | None = None) -> dict:
    rho = Fraction(inputs["rho"])
    n = 4 * SEGMENTS
    f = arb_mat([[v] for v in f_override]) if f_override is not None else arb_mat([[ball(e)] for e in inputs["f_center"]])
    df = arb_mat(n, n)
    for k, block in enumerate(inputs["derivative_blocks"]):
        nxt = (k + 1) % SEGMENTS
        for i in range(4):
            for j in range(4):
                df[4 * k + i, 4 * k + j] = ball(block[i][j])
            df[4 * k + i, 4 * nxt + i] = df[4 * k + i, 4 * nxt + i] - 1
    preconditioner = df.mid().inv().mid()   # 아무 정확 행렬이어도 결론은 엄밀하다
    radius = arb(fmpq(rho.numerator, rho.denominator))
    # Arb 공은 반경을 올림 저장하므로 식 안의 X-c 는 X 를 담는 공으로 쓴다(포함 단조성으로 K 를 바깥에서 감쌈).
    # 판정은 공 포함이 아니라 정확한 경계 +-rho 와의 확정 비교로 한다 — DF(X) 는 원래 상자 X 위에서만 쓰인다.
    x = arb_mat([[arb(0, radius)] for _ in range(n)])
    eye = arb_mat(n, n)
    for i in range(n):
        eye[i, i] = 1
    image = -(preconditioner * f) + (eye - preconditioner * df) * x

    def strictly_inside(bound: arb) -> bool:
        return all(image[i, 0] < bound and image[i, 0] > -bound for i in range(n))

    scale = Fraction(inputs["target_scale"])
    k_radius = max(max(abs(image[i, 0].lower()), abs(image[i, 0].upper())) for i in range(n))
    return {"inside": strictly_inside(radius),
            "inside_target": strictly_inside(radius * arb(fmpq(scale.numerator, scale.denominator))),
            "k_radius_over_rho": float((k_radius / radius).mid())}


def main() -> int:
    no_write = "--no-write" in sys.argv[1:]
    started = time.time()
    ctx.prec = PREC_BITS
    milestone = json.loads((RESULTS_DIR / "milestone_figure8.json").read_text())
    rows = []
    for case in milestone["cases"]:
        inputs = case.get("krawczyk_inputs")
        if inputs is None:
            rows.append({"case": case["case"], "status": "no inputs (engine failure)"})
            continue
        kv = krawczyk(inputs)
        cert_ok, failed = certify(inputs["certify"], Fraction(inputs["min_distance"]))
        proven_arb = kv["inside"] and kv["inside_target"] and cert_ok
        f_ind, steps = independent_f(inputs)
        kv_ind = krawczyk(inputs, f_ind)
        proven_ind = kv_ind["inside"] and kv_ind["inside_target"] and cert_ok
        # 두 감싸기는 같은 참값을 담아야 하므로 서로 겹치지 않으면 어느 한쪽이 틀린 것이다
        overlaps = all(a.overlaps(ball(e)) for a, e in zip(f_ind, inputs["f_center"]))
        expected_ind = case["case"] in RESOLUTION_CONTROLS or case["proven"]
        rows.append({
            "case": case["case"], "proven_arb": proven_arb, "krawczyk_inside_arb": kv["inside"],
            "certify_arb": cert_ok, "certify_failed_arb": failed[:6], "k_radius_over_rho_arb": kv["k_radius_over_rho"],
            "proven_independent_f": proven_ind, "k_radius_over_rho_independent_f": kv_ind["k_radius_over_rho"],
            "independent_f_overlaps_capd": overlaps,
            "independent_f_max_radius": float(max(v.rad() for v in f_ind)),
            "independent_f_steps": steps,
            "proven_exact_rational": case["proven"],
            "expected_independent_f": expected_ind,
            "agree": proven_arb == case["proven"] and proven_ind == expected_ind and overlaps,
        })
        print(json.dumps(rows[-1]), flush=True)
    disagreements = [r["case"] for r in rows if r.get("agree") is False]
    if no_write:
        return 1 if disagreements else 0
    write_result("arb_krawczyk", {"tag": "G8 독립 재계산 [CAP: Arb 공 산술, 입력은 CAPD 감싸기]", "prec_bits": PREC_BITS,
                                  "scope": "A: CAPD F(c)·DF(X) 를 받아 K·전처리·K⊂int X·사용 자격 부등식을 Arb 로 재판정. "
                                           "B: F(c) 를 CAPD 없이 Arb 적분기로 다시 감싸고 DF(X)·상자 흐름 관은 CAPD 것을 씀",
                                  "cases": rows, "disagreements": disagreements},
                 script=Path(__file__).resolve(), started=started,
                 params={"prec_bits": PREC_BITS,
                         "extra_code_sha256": {"reference/arb_flow.py": sha256_file(TRACK_DIR / "reference" / "arb_flow.py")}})
    return 1 if disagreements else 0


if __name__ == "__main__":
    sys.exit(main())
