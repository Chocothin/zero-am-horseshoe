"""정리 수준 주장: 보고서에 적는 T11 상수(CLAIMED)와 T6·T7 정리 상수(THEOREM), 다시 잰 값이 그 상수를 받치는지 정하는
judge_claims(한 곳에서 선언).

CLAIMED 는 데이터와 독립으로 여기에만 적는다. 실수 상수는 본 실행 측정값을 12자리에서 안전한 쪽으로 자른 값이다.
THEOREM 은 T6 assemble_theorem.py·T7 assemble_aprime.py 의 CLAIMED 가운데 CAPD 감싸기에 기대는 상수다(정확 계산만 쓰는
(A3)·(A4)·(A6)·ρ·ln ρ 는 T11 상수 A3_gap·A4·A6_gap 으로 따로 다시 판정한다).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

CLAIMED = {
    "cover_margin": Fraction("0.061029134539"),     # 314변 (W)(F) 여유의 최솟값 ≥
    "mu_s_margin": Fraction("0.012742629471"),      # μ_s − max a* ≥
    "mu_u_margin": Fraction("0.009357165084"),      # min b* − μ_u ≥
    "g1": Fraction("0.118342629471"),               # min_e min(b*−1, 1−a*) ≥ (μ=1 Gershgorin 여유)
    "delta": Fraction("0.217939990832"),            # 흐름 관 r_ij ≥
    "R": Fraction("7.274514881157"),                # 흐름 관 r_ij ≤
    "t_min": Fraction("0.276707990268"),            # 귀환 시각 ≥
    "tau_max": Fraction("14.795684705067"),         # 귀환 시각 ≤
    "A3_gap": Fraction("0.678638874080"),           # N_{M+0} 외피와 다른 312개 외피의 분리 간격 ≥
    "A6_gap": Fraction("0.565159173410"),           # cycle h-set 16개 외피의 쌍별 분리 간격 ≥
    "edges": 314,
    "degrees": {1: 312, -1: 2},
}

THEOREM = {
    "T6.delta": Fraction("0.217857228763"), "T6.R": Fraction("7.419986043806"), "T6.t_min": Fraction("0.276707990268"),
    "T7.mu_s": Fraction("0.8944"), "T7.mu_u": Fraction("1.1167"), "T7.lambda_s": Fraction("0.9458"),
    "T7.lambda_u": Fraction("1.0567"), "T7.C_s": Fraction("3.078"), "T7.C_u": Fraction("3.252"),
}

REAL_CLAIMS = ("cover_margin", "mu_s_margin", "mu_u_margin", "g1", "delta", "R", "t_min", "tau_max", "A3_gap", "A6_gap")
UPPER_CLAIMS = ("R", "tau_max")          # 다시 잰 값 ≤ 상수 꼴, 나머지 실수 주장은 ≥ 꼴
INTEGER_CLAIMS = ("A1", "A2", "A4", "T7.A5")
SUPPORT_CLAIMS = ("T6.delta", "T6.R", "T6.t_min", "T7.mu_s", "T7.mu_u")
RATE_CLAIMS = ("T7.lambda_s", "T7.lambda_u", "T7.C_s", "T7.C_u")
MEASURED = REAL_CLAIMS[:1] + ("a_max", "b_min") + REAL_CLAIMS[3:]


@dataclass(frozen=True)
class Measured:
    """314변 기록에서 다시 잰 값(정확 유리수)과 그 값을 정한 변·정점."""
    edges: int
    design_edges: int
    missing: tuple
    cover_unproven: tuple
    degrees: dict
    structure_failed: tuple
    g1_nonpositive: tuple
    graph_matches_expected: bool
    values: dict                      # MEASURED 이름 → Fraction
    at: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ClaimCheck:
    name: str
    holds: bool
    margin: Fraction                  # 부호 있는 여유(음수면 거짓). 정수 주장은 결손 개수의 음수
    measured: Fraction | None         # 상수와 비교한 다시 잰 값(정수·상수식 주장은 None)
    constant: Fraction | None
    relation: str                     # "≥" "≤" ">" "<" "정수" "상수식"
    detail: str


def epsilon(theorem: dict) -> Fraction:
    """T7 §2.3 의 ε = min(μ_u − 1, 1 − μ_s)."""
    return min(theorem["T7.mu_u"] - 1, 1 - theorem["T7.mu_s"])


def judge_claims(m: Measured, claimed: dict = CLAIMED, theorem: dict = THEOREM) -> list[ClaimCheck]:
    out = []
    v = m.values
    complete = m.edges == m.design_edges == claimed["edges"] and not m.missing

    def integer(name, bad, detail):
        out.append(ClaimCheck(name, complete and bad == 0 if name != "A4" else bad == 0, Fraction(-bad), None, None,
                              "정수", detail))

    degrees_ok = m.degrees == claimed["degrees"]
    integer("A1", len(m.cover_unproven) + len(m.missing) + abs(m.edges - claimed["edges"]) + (0 if degrees_ok else 1),
            f"edges {m.edges}/{claimed['edges']}, missing {len(m.missing)}, unproven {len(m.cover_unproven)}, "
            f"degrees {m.degrees}")
    integer("A2", len(m.structure_failed) + len(m.missing), f"structure failed {len(m.structure_failed)}")
    integer("A4", 0 if m.graph_matches_expected else 1, "edge set equals expected_edges(149, 8, 8)")
    integer("T7.A5", len(m.g1_nonpositive) + len(m.missing), f"edges with g_e(1) ≤ 0: {len(m.g1_nonpositive)}")
    derived = {"cover_margin": v["cover_margin"], "mu_s_margin": theorem["T7.mu_s"] - v["a_max"],
               "mu_u_margin": v["b_min"] - theorem["T7.mu_u"], **{n: v[n] for n in REAL_CLAIMS[3:]}}
    for name in REAL_CLAIMS:
        x, c = derived[name], claimed[name]
        upper = name in UPPER_CLAIMS
        mg = c - x if upper else x - c
        # 여유·이격·시각 하한은 상수가 양수여야 의미가 있다(상수 0 이하로는 주장을 받칠 수 없다)
        where = {"mu_s_margin": m.at.get("a_max"), "mu_u_margin": m.at.get("b_min")}.get(name, m.at.get(name))
        out.append(ClaimCheck(name, complete and c > 0 and mg >= 0, mg, x, c, "≤" if upper else "≥",
                              f"measured {float(x):.15g} at {where}"))
    mu_s, mu_u = theorem["T7.mu_s"], theorem["T7.mu_u"]
    support = {
        "T6.delta": (v["delta"], theorem["T6.delta"], "≥", v["delta"] - theorem["T6.delta"], theorem["T6.delta"] > 0),
        "T6.R": (v["R"], theorem["T6.R"], "≤", theorem["T6.R"] - v["R"], True),
        "T6.t_min": (v["t_min"], theorem["T6.t_min"], "≥", v["t_min"] - theorem["T6.t_min"], theorem["T6.t_min"] > 0),
        "T7.mu_s": (v["a_max"], mu_s, "<", mu_s - v["a_max"], 0 < mu_s < 1),
        "T7.mu_u": (v["b_min"], mu_u, ">", v["b_min"] - mu_u, mu_u > 1),
    }
    ok = {}
    for name, (x, c, rel, mg, sane) in support.items():
        holds = complete and sane and (mg > 0 if rel in ("<", ">") else mg >= 0)
        ok[name] = holds
        out.append(ClaimCheck(name, holds, mg, x, c, rel, f"T11 {float(x):.15g} vs theorem constant {c}"))
    eps = epsilon(theorem)
    ls, lu, cs, cu = (theorem[k] for k in RATE_CLAIMS)
    rate = {
        "T7.lambda_s": (ok["T7.mu_s"] and 0 < ls < 1, ls * ls - mu_s, "λ_s² ≥ μ_s"),
        "T7.lambda_u": (ok["T7.mu_u"] and lu > 1, mu_u - lu * lu, "λ_u² ≤ μ_u"),
        "T7.C_s": (ok["T7.mu_s"] and ok["T7.mu_u"], cs * cs * eps - 1, "c_s²ε ≥ 1"),
        "T7.C_u": (ok["T7.mu_s"] and ok["T7.mu_u"], cu * cu * eps - mu_u, "c_u²ε ≥ μ_u"),
    }
    for name, (pre, mg, text) in rate.items():
        out.append(ClaimCheck(name, pre and mg >= 0, mg, None, theorem[name], "상수식",
                              f"{text} (ε={eps}), μ 주장 {'성립' if pre else '불성립'}"))
    return out
