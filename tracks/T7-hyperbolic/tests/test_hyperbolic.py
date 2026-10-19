"""정리 A′ 조립 명세 시험. 기대값은 주장의 논리 형태(전 변 인증, min b* > μ_u > 1, max a* < μ_s < 1, cycle h-set 쌍마다
분리 간격 > 0, 쌍곡 상수 λ_s² ≥ μ_s·λ_u² ≤ μ_u·c_s²ε ≥ 1·c_u²ε ≥ μ_u)와 ε=min(μ_u−1, 1−μ_s) 의 유도, 방향 반올림의 정의
(격자 위 가장 가까운 안전한 쪽 값), 그리고 다른 스크립트(certify_cones)가 낸 기록 값에서 정했다."""
import json
import random
from dataclasses import replace
from fractions import Fraction as F

import pytest

from t3cap.exact import Interval
from t7hb import cone, hyperbolic, stored
from t7hb.results import RESULTS_DIR, T6_DESIGN, T6_RESULTS, sha256_file



def claim(mu_s, mu_u):
    """μ 주장만 볼 때 쓰는 넉넉한 쌍곡 상수(λ_s=0.99, λ_u=1.01, c=10 은 ε ≥ 0.02 에서 모두 성립)."""
    return hyperbolic.Claimed(F(mu_s), F(mu_u), F("0.99"), F("1.01"), F(10), F(10))


CLAIM = claim("0.9", "1.1")


def point_diag(*d):
    return tuple(tuple(Interval(F(d[i]), F(d[i])) if i == j else Interval(F(0), F(0)) for j in range(4))
                 for i in range(4))


def cycle_hulls(spacing=F(10)):
    return {f"N{i}": [(spacing * i, spacing * i + 1)] + [(F(0), F(1))] * 3 for i in range(16)}


def inputs():
    return [stored.ConeInput("a>b", point_diag("1.2", "1.15", "0.8", "0.9"), ()),
            stored.ConeInput("b>c", point_diag("1.3", "1.12", "0.7", "0.85"), ())]


def test_extremes_and_edges_follow_definitions():
    """a* 최댓값·b* 최솟값과 그 변 - 정의(max_e a*_e, min_e b*_e)."""
    a = hyperbolic.assemble(inputs(), ["a>b", "b>c"], cycle_hulls())
    assert (a.lower_max, a.lower_edge) == (F("0.81"), "a>b")
    assert (a.upper_min, a.upper_edge) == (F("1.12") ** 2, "b>c")
    assert a.certified and not a.missing and hyperbolic.judge_claims(a, CLAIM)[0].holds


def test_missing_and_uncertified_edges_block_a5():
    """설계 변이 빠지거나 사용 불가·여유 음수인 변이 있으면 A5 는 거짓이다."""
    a = hyperbolic.assemble(inputs(), ["a>b", "b>c", "c>d"], cycle_hulls())
    assert a.missing == ("c>d",) and not hyperbolic.judge_claims(a, CLAIM)[0].holds
    bad = inputs() + [stored.ConeInput("c>d", point_diag("1.2", "0.95", "0.8", "0.9"), ())]
    a = hyperbolic.assemble(bad, ["a>b", "b>c", "c>d"], cycle_hulls())
    assert a.uncertified == ("c>d",) and not hyperbolic.judge_claims(a, CLAIM)[0].holds
    unusable = inputs() + [stored.ConeInput("c>d", point_diag("1.2", "1.2", "0.8", "0.8"), ("stale",))]
    a = hyperbolic.assemble(unusable, ["a>b", "b>c", "c>d", "d>e"], cycle_hulls())
    assert a.uncertified == ("c>d",) and a.missing == ("d>e",)
    assert hyperbolic.judge_claims(a, CLAIM)[0].margin == F(-2)


def test_zero_margin_edge_is_not_certified():
    """μ=1 의 여유가 정확히 0 인 변(d_1=1) - 엄격 양정치가 아니므로 인증하지 않는다."""
    edge = stored.ConeInput("a>b", point_diag("1.2", "1", "0.8", "0.9"), ())
    a = hyperbolic.assemble([edge], ["a>b"], cycle_hulls())
    assert a.edges[0].bounds.margin(F(1)) == 0 and a.uncertified == ("a>b",)


def test_no_edges_is_not_a_proof():
    """판정 입력이 하나도 없으면 A5 는 거짓이다(빈 전칭 금지)."""
    a = hyperbolic.assemble([], [], cycle_hulls())
    assert not hyperbolic.judge_claims(a, CLAIM)[0].holds


def test_mu_claims_are_strict_and_bracket_one():
    """μ_u 는 1 보다 크고 min b* 보다 엄격히 작아야, μ_s 는 0 과 1 사이이고 max a* 보다 엄격히 커야 성립한다."""
    a = hyperbolic.assemble(inputs(), ["a>b", "b>c"], cycle_hulls())
    names = lambda c: {x.name: x.holds for x in hyperbolic.judge_claims(a, c)}
    assert names(CLAIM)["mu_u"] and names(CLAIM)["mu_s"]
    assert not names(replace(CLAIM, mu_u=a.upper_min))["mu_u"]
    assert not names(replace(CLAIM, mu_u=F(1)))["mu_u"]
    assert not names(replace(CLAIM, mu_s=a.lower_max))["mu_s"]
    assert not names(replace(CLAIM, mu_s=F(1)))["mu_s"]
    negative = hyperbolic.Assembly(a.edges, (), F(-1), "a>b", a.upper_min, a.upper_edge, F(9), ("N0", "N1"))
    assert not {x.name: x.holds for x in hyperbolic.judge_claims(negative, replace(CLAIM, mu_s=F(0)))}["mu_s"]


def test_cycle_disjointness_uses_cycle_vertices_only_and_is_strict():
    """A6 는 N0..N15 쌍만 본다 - 사슬 외피가 겹쳐도 무관하고, cycle 외피가 닿으면(간격 0) 거짓이다."""
    hulls = {**cycle_hulls(), "M+1": [(F(0), F(200))] * 4}
    a = hyperbolic.assemble(inputs(), ["a>b", "b>c"], hulls)
    assert a.cycle_gap == F(9) and a.cycle_pair == ("N0", "N1")
    touching = {**cycle_hulls(), "N5": [(F(41), F(50))] + [(F(0), F(1))] * 3}
    a = hyperbolic.assemble(inputs(), ["a>b", "b>c"], touching)
    assert a.cycle_gap == 0 and not hyperbolic.judge_claims(a, CLAIM)[3].holds and not a.certified
    close = hyperbolic.assemble(inputs(), ["a>b", "b>c"], cycle_hulls(spacing=F(3, 2)))
    assert close.cycle_gap == F(1, 2) and close.certified and hyperbolic.judge_claims(close, CLAIM)[3].holds


def test_derived_eps_bounds_margin_at_one():
    """a*<μ_s, b*>μ_u 인 변은 μ=1 여유가 ε=min(μ_u−1, 1−μ_s) 보다 크다(대각 행렬 표본)."""
    rng = random.Random(4)
    c = claim("0.85", "1.1")
    eps = hyperbolic.derived_eps(c)
    assert eps == F("0.1")
    for _ in range(200):
        d = [F(rng.randint(1050, 1300), 1000), F(rng.randint(1050, 1300), 1000),
             F(rng.randint(500, 948), 1000), F(rng.randint(500, 948), 1000)]
        b = cone.cone_bounds(point_diag(*d))
        if b.lower < c.mu_s and b.upper > c.mu_u:
            assert b.margin(F(1)) > eps


def test_sqrt_bound_is_the_nearest_safe_grid_point():
    """√q 의 격자 반올림 - 위쪽은 제곱이 q 이상인 가장 작은 격자점, 아래쪽은 제곱이 q 이하인 가장 큰 격자점."""
    assert hyperbolic.sqrt_bound(F(4), 3, True) == hyperbolic.sqrt_bound(F(4), 3, False) == 2
    assert hyperbolic.sqrt_bound(F(0), 3, True) == 0
    assert hyperbolic.sqrt_bound(F(2), 2, False) == F("1.41") and hyperbolic.sqrt_bound(F(2), 2, True) == F("1.42")
    rng = random.Random(8)
    for _ in range(200):
        q, k = F(rng.randint(1, 10**6), rng.randint(1, 10**4)), rng.randint(0, 6)
        up, down, step = hyperbolic.sqrt_bound(q, k, True), hyperbolic.sqrt_bound(q, k, False), F(1, 10**k)
        assert up ** 2 >= q > (up - step) ** 2 and down ** 2 <= q < (down + step) ** 2
    with pytest.raises(ValueError):
        hyperbolic.sqrt_bound(F(-1), 3, True)


def test_reported_rate_constants_are_directed_roundings_of_the_mu_claims():
    """μ_s=0.8944, μ_u=1.1167, ε=0.1056 - 0.9458²≥μ_s>0.9457², 1.0567²≤μ_u<1.0568², 3.078²ε≥1>3.077²ε, 3.252²ε≥μ_u>3.251²ε."""
    mu_s, mu_u, eps = F("0.8944"), F("1.1167"), F("0.1056")
    assert F("0.9458") ** 2 >= mu_s > F("0.9457") ** 2 and F("1.0567") ** 2 <= mu_u < F("1.0568") ** 2
    assert F("3.078") ** 2 * eps >= 1 > F("3.077") ** 2 * eps and F("3.252") ** 2 * eps >= mu_u > F("3.251") ** 2 * eps
    assert hyperbolic.rate_constants(mu_s, mu_u) == {"lambda_s": F("0.9458"), "lambda_u": F("1.0567"),
                                                     "c_s": F("3.078"), "c_u": F("3.252")}


def test_rate_claims_accept_exact_equality():
    """한 변·μ_s=0.64·μ_u=1.44 이면 ε=0.36 이고 λ_s=0.8, λ_u=1.2, c_s=5/3, c_u=2 가 추정식의 등호다 - 등호는 성립(≥·≤)이다."""
    a = hyperbolic.assemble([stored.ConeInput("a>b", point_diag("1.3", "1.3", "0.7", "0.7"), ())], ["a>b"], cycle_hulls())
    exact = hyperbolic.Claimed(F("0.64"), F("1.44"), F("0.8"), F("1.2"), F(5, 3), F(2))
    checks = hyperbolic.judge_claims(a, exact)
    assert all(c.holds for c in checks)
    assert {c.name: c.margin for c in checks if c.name in ("lambda_s", "lambda_u", "C_s", "C_u")} == {
        "lambda_s": 0, "lambda_u": 0, "C_s": 0, "C_u": 0}


def test_rate_claims_follow_the_estimates_and_lean_on_mu_claims():
    """λ_s²≥μ_s·λ_s<1, λ_u²≤μ_u·λ_u>1, c_s²ε≥1, c_u²ε≥μ_u 이고 기대는 μ 주장이 성립할 때만 쌍곡 상수 주장이 성립한다."""
    a = hyperbolic.assemble(inputs(), ["a>b", "b>c"], cycle_hulls())
    holds = lambda c: {x.name: x.holds for x in hyperbolic.judge_claims(a, c)}
    exact = replace(CLAIM, lambda_s=hyperbolic.sqrt_bound(F("0.9"), 6, True), lambda_u=hyperbolic.sqrt_bound(F("1.1"), 6, False),
                    c_s=hyperbolic.sqrt_bound(F(10), 6, True), c_u=hyperbolic.sqrt_bound(F(11), 6, True))
    assert all(holds(exact)[n] for n in ("lambda_s", "lambda_u", "C_s", "C_u"))
    assert not holds(replace(exact, lambda_s=hyperbolic.sqrt_bound(F("0.9"), 6, False)))["lambda_s"]
    assert not holds(replace(exact, lambda_s=F(1)))["lambda_s"]
    assert not holds(replace(exact, lambda_u=hyperbolic.sqrt_bound(F("1.1"), 6, True)))["lambda_u"]
    assert not holds(replace(exact, lambda_u=F(1)))["lambda_u"]
    assert not holds(replace(exact, c_s=hyperbolic.sqrt_bound(F(10), 6, False)))["C_s"]
    assert not holds(replace(exact, c_u=hyperbolic.sqrt_bound(F(11), 6, False)))["C_u"]
    broken = replace(exact, mu_s=a.lower_max)
    assert not holds(broken)["lambda_s"] and not holds(broken)["C_s"] and not holds(broken)["C_u"]
    broken = replace(exact, mu_u=a.upper_min)
    assert not holds(broken)["lambda_u"] and not holds(broken)["C_u"] and not holds(broken)["C_s"]
    margins = {x.name: x.margin for x in hyperbolic.judge_claims(a, exact)}
    assert margins["lambda_s"] == exact.lambda_s ** 2 - F("0.9") and margins["C_u"] == exact.c_u ** 2 * F("0.1") - F("1.1")


def test_real_assembly_matches_independent_per_edge_record():
    """T6 기록 314개 조립 - 극값 변과 값이 certify_cones 가 변마다 낸 기록(cones.json)의 극값과 같다."""
    design = json.loads((T6_DESIGN / "graph_design.json").read_text())
    dh = sha256_file(T6_DESIGN / "graph_design.json")
    ins = [stored.cone_input(stored.load(T6_RESULTS / "coverings" / f"{e['source']}__{e['target']}.json"), dh)
           for e in design["edges"]]
    hulls = {f"N{i}": [(F(10 * i), F(10 * i + 1))] * 4 for i in range(16)}
    a = hyperbolic.assemble(ins, [f"{e['source']}>{e['target']}" for e in design["edges"]], hulls)
    rows = json.loads((RESULTS_DIR / "cones.json").read_text())["rows"]
    lo = max(rows, key=lambda r: F(r["lower_a"]))
    up = min(rows, key=lambda r: F(r["upper_b"]))
    assert (a.lower_edge, a.lower_max) == (lo["edge"], F(lo["lower_a"])) == ("M-147>M-146", F(lo["lower_a"]))
    assert (a.upper_edge, a.upper_min) == (up["edge"], F(up["upper_b"])) == ("M+145>M+146", F(up["upper_b"]))
    assert len(a.edges) == 314 and not a.uncertified
