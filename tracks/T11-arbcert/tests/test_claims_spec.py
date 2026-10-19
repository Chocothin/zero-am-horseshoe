"""정리 수준 주장 판정(judge_claims) 명세 시험(G9). 기대값은 주장의 논리 형태(≥·≤ 는 등호 허용, T7 의 a*<μ_s·b*>μ_u 는
엄격 부등식, 314변 전부, 차수 개수)와 T6·T7 정리 상수의 정의(T7 §2.3 의 ε, §5.5 의 쌍곡 상수 부등식)에서 정했다."""
from __future__ import annotations

import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from t11 import claims as C  # noqa: E402

TH = C.THEOREM


def at_claims(**over):
    """측정값이 T11 상수와 정확히 같은 경우(덮개 여유·하한은 상수, a*·b* 는 μ 여유에서 역산)."""
    v = {"cover_margin": C.CLAIMED["cover_margin"], "a_max": TH["T7.mu_s"] - C.CLAIMED["mu_s_margin"],
         "b_min": TH["T7.mu_u"] + C.CLAIMED["mu_u_margin"],
         **{n: C.CLAIMED[n] for n in ("g1", "delta", "R", "t_min", "tau_max", "A3_gap", "A6_gap")}}
    v.update(over.pop("values", {}))
    base = dict(edges=314, design_edges=314, missing=(), cover_unproven=(), degrees={1: 312, -1: 2},
                structure_failed=(), g1_nonpositive=(), graph_matches_expected=True, values=v, at={})
    base.update(over)
    return C.Measured(**base)


def by_name(m, claimed=C.CLAIMED, theorem=TH):
    return {c.name: c for c in C.judge_claims(m, claimed, theorem)}


def test_all_claims_hold_when_measured_equals_constants():
    """측정값이 T11 상수와 같음 - 모든 주장 성립(≥·≤ 등호 허용, T11 상수는 T6·T7 상수보다 강하다)."""
    checks = by_name(at_claims())
    assert all(c.holds for c in checks.values()), [n for n, c in checks.items() if not c.holds]
    assert all(checks[n].margin == 0 for n in C.REAL_CLAIMS)


def test_each_T11_claim_fails_just_beyond_its_constant():
    """T11 실수 주장 - 하한은 값이 상수보다 작으면, 상한(R·τ 상한)은 크면 거짓, 다른 T11 주장은 그대로 성립."""
    tiny = F(1, 10 ** 30)
    moves = {"mu_s_margin": {"a_max": TH["T7.mu_s"] - C.CLAIMED["mu_s_margin"] + tiny},
             "mu_u_margin": {"b_min": TH["T7.mu_u"] + C.CLAIMED["mu_u_margin"] - tiny}}
    for name in C.REAL_CLAIMS:
        if name in moves:
            values = moves[name]
        else:
            values = {name: C.CLAIMED[name] + tiny if name in C.UPPER_CLAIMS else C.CLAIMED[name] - tiny}
        checks = by_name(at_claims(values=values))
        assert not checks[name].holds and checks[name].margin == -tiny
        assert all(checks[n].holds for n in C.REAL_CLAIMS if n != name)


def test_missing_edge_breaks_every_data_claim():
    """변 기록 하나가 빠짐 - 314변 전체에 대한 주장이라 그래프 주장(A4)과 상수식만으로 정해지지 않는 주장이 모두 거짓."""
    checks = by_name(at_claims(edges=313, missing=("M+0>M+1",)))
    assert checks["A4"].holds
    assert all(not c.holds for n, c in checks.items() if n != "A4")


def test_unproven_covering_or_wrong_degrees_breaks_A1():
    """덮개 하나 미인증 또는 차수 개수 다름 - A1 거짓(정리 A (A1): 모든 변 덮개, 차수 ±1)."""
    assert not by_name(at_claims(cover_unproven=("N0>N1",)))["A1"].holds
    assert not by_name(at_claims(degrees={1: 313, -1: 1}))["A1"].holds


def test_structure_graph_and_A5_failures_break_their_claims():
    """첫 통과 구조 실패 변 → A2, 설계 변 집합 불일치 → A4, g_e(1) ≤ 0 인 변 → T7.A5 가 거짓."""
    assert not by_name(at_claims(structure_failed=("N1>N2",)))["A2"].holds
    assert not by_name(at_claims(graph_matches_expected=False))["A4"].holds
    assert not by_name(at_claims(g1_nonpositive=("N0>N1",)))["T7.A5"].holds


def test_non_positive_constant_cannot_support_a_lower_claim():
    """상수 0 이하 - 여유·이격·시각 하한 주장으로 쓸 수 없어 거짓(δ=0 은 충돌 배제가 아니다)."""
    assert not by_name(at_claims(values={"delta": F(0)}), {**C.CLAIMED, "delta": F(0)})["delta"].holds
    assert not by_name(at_claims(values={"delta": F(0)}), C.CLAIMED, {**TH, "T6.delta": F(0)})["T6.delta"].holds


def test_T6_support_allows_equality_and_rejects_weaker_values():
    """T6 δ·t_min 은 하한(≥, 등호 허용), R 은 상한(≤) - T11 값이 그 상수와 같으면 받치고 넘으면 받치지 못한다."""
    eq = by_name(at_claims(values={"delta": TH["T6.delta"], "R": TH["T6.R"], "t_min": TH["T6.t_min"]}))
    assert eq["T6.delta"].holds and eq["T6.R"].holds and eq["T6.t_min"].holds
    tiny = F(1, 10 ** 30)
    weak = by_name(at_claims(values={"delta": TH["T6.delta"] - tiny, "R": TH["T6.R"] + tiny,
                                     "t_min": TH["T6.t_min"] - tiny}))
    assert not weak["T6.delta"].holds and not weak["T6.R"].holds and not weak["T6.t_min"].holds


def test_T7_mu_claims_are_strict():
    """T7 (A5) 의 a*_e<μ_s, b*_e>μ_u 는 엄격 부등식 - 같으면 받치지 못한다(T7 §4)."""
    eq = by_name(at_claims(values={"a_max": TH["T7.mu_s"], "b_min": TH["T7.mu_u"]}))
    assert not eq["T7.mu_s"].holds and not eq["T7.mu_u"].holds
    assert not by_name(at_claims(), C.CLAIMED, {**TH, "T7.mu_s": F(1)})["T7.mu_s"].holds
    assert not by_name(at_claims(), C.CLAIMED, {**TH, "T7.mu_u": F(1)})["T7.mu_u"].holds


def test_rate_constants_follow_T7_inequalities_and_need_mu_claims():
    """쌍곡 상수 - λ_s²≥μ_s, λ_u²≤μ_u, c_s²ε≥1, c_u²ε≥μ_u(ε=min(μ_u−1,1−μ_s)=0.1056)이고 기대는 μ 주장이 성립해야 한다."""
    assert C.epsilon(TH) == F("0.1056")
    ok = by_name(at_claims())
    assert all(ok[n].holds for n in C.RATE_CLAIMS)
    assert ok["T7.C_s"].margin == F("3.078") ** 2 * F("0.1056") - 1
    bad = by_name(at_claims(), C.CLAIMED, {**TH, "T7.lambda_s": F("0.9457"), "T7.lambda_u": F("1.0568"),
                                           "T7.C_s": F("3.077"), "T7.C_u": F("3.25")})
    assert not any(bad[n].holds for n in C.RATE_CLAIMS)
    no_mu = by_name(at_claims(values={"a_max": TH["T7.mu_s"]}))
    assert not no_mu["T7.lambda_s"].holds and not no_mu["T7.C_s"].holds and not no_mu["T7.C_u"].holds
    assert no_mu["T7.lambda_u"].holds


def test_integer_claim_margins_count_deficiencies():
    """정수 주장의 여유 = −(결손 개수). A1 은 미인증 + 누락 + |변 수−314| + (차수 불일치 1), A2 는 구조 실패 + 누락,
    A4 는 그래프 불일치 1, T7.A5 는 g_e(1)≤0 변 + 누락이다."""
    c = by_name(at_claims(cover_unproven=("a", "b"), degrees={1: 313, -1: 1}))
    assert c["A1"].margin == -3
    c = by_name(at_claims(edges=312, missing=("x", "y"), structure_failed=("s",), g1_nonpositive=("g",)))
    assert c["A1"].margin == -(2 + 2) and c["A2"].margin == -(1 + 2) and c["T7.A5"].margin == -(1 + 2)
    assert by_name(at_claims(graph_matches_expected=False))["A4"].margin == -1


def test_design_with_wrong_edge_count_supports_no_data_claim():
    """설계 변 수가 314 가 아님(빠진 기록은 없음) - 314변 그래프에 대한 주장이라 데이터 주장이 모두 거짓."""
    checks = by_name(at_claims(edges=313, design_edges=313))
    assert all(not c.holds for n, c in checks.items() if n != "A4")


def test_epsilon_is_the_smaller_gap():
    """ε = min(μ_u−1, 1−μ_s) - μ_u 쪽이 작은 경우에도 작은 쪽을 고른다(T7 §2.3)."""
    assert C.epsilon({**TH, "T7.mu_s": F("0.8"), "T7.mu_u": F("1.1")}) == F("0.1")
    assert C.epsilon(TH) == F("0.1056")


def test_theorem_constants_need_sane_signs():
    """T6 t_min 0, T7 μ_s 0 은 주장의 상수가 될 수 없다 - 측정값이 상수를 넘어도 거짓."""
    assert not by_name(at_claims(), C.CLAIMED, {**TH, "T6.t_min": F(0)})["T6.t_min"].holds
    neg_a = at_claims(values={"a_max": F(-1, 10)})
    assert not by_name(neg_a, C.CLAIMED, {**TH, "T7.mu_s": F(0)})["T7.mu_s"].holds


def test_rate_constants_must_lie_strictly_on_the_contracting_side():
    """λ_s<1·λ_u>1 이어야 한다(경계 1 은 수축·팽창이 아니다) - λ_s=1, λ_u=1 이면 부등식이 맞아도 거짓."""
    assert not by_name(at_claims(), C.CLAIMED, {**TH, "T7.lambda_s": F(1)})["T7.lambda_s"].holds
    assert not by_name(at_claims(), C.CLAIMED, {**TH, "T7.lambda_s": F(3, 2)})["T7.lambda_s"].holds
    assert not by_name(at_claims(), C.CLAIMED, {**TH, "T7.lambda_u": F(1)})["T7.lambda_u"].holds


def test_rate_constant_inequalities_allow_equality():
    """쌍곡 상수 부등식은 등호를 허용한다 - μ_s=3/4, μ_u=5/4(ε=1/4)에서 c_s=2 는 c_s²ε=1 로 성립한다."""
    th = {**TH, "T7.mu_s": F(3, 4), "T7.mu_u": F(5, 4), "T7.C_s": F(2), "T7.lambda_s": F(9, 10), "T7.lambda_u": F(11, 10)}
    m = at_claims(values={"a_max": F(1, 2), "b_min": F(3, 2)})
    c = by_name(m, C.CLAIMED, th)
    assert c["T7.C_s"].margin == 0 and c["T7.C_s"].holds
