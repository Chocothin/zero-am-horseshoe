"""정리 A 조립층 명세 시험. 기대값은 정리의 가정 목록(덮개 증명·그래프·M₀ 분리·δ>0·귀환 시각>0)과
독립 출처 값(알려진 근, mpmath findroot, 설계 훑기 결과)에서 정했다."""
from fractions import Fraction as Q

import dataclasses

import mpmath as mp
from t3cap.exact import Interval

from t6hs import assembly
from t6hs.covering import PieceImage, face_box, unit_box
from t6hs.records import CoveringInput

GAP = Q(1, 2**79)
EYE = [[Q(int(i == j)) for j in range(4)] for i in range(4)]


def iv(lo, hi):
    return Interval(Q(lo), Q(hi))


def mpq(x, dps):
    with mp.workdps(dps):
        return mp.mpf(x.numerator) / x.denominator


def test_rho_lower_bound_pins_known_roots():
    """x⁻¹+x⁻¹=1 의 근 2, x⁻¹+x⁻²=1 의 근 황금비 - 이분법 하한이 근 아래 2⁻⁷⁹ 안(정확 근과 대조)"""
    lo = assembly.rho_lower_bound(1, 1)
    assert lo <= 2 and 2 - lo < GAP
    with mp.workdps(50):
        phi = (1 + mp.sqrt(5)) / 2
        assert 0 <= phi - mpq(assembly.rho_lower_bound(1, 2), 50) < mpq(GAP, 50)


def test_rho_lower_bound_for_the_graph_matches_an_independent_root():
    """순환 16·298 - mpmath findroot(60자리)로 구한 근과 2⁻⁷⁹ 안에서 같고 근을 넘지 않음(값 고정 1.0073969…)"""
    with mp.workdps(60):
        root = mp.findroot(lambda x: x ** -16 + x ** -298 - 1, mp.mpf("1.0074"))
        assert 0 <= root - mpq(assembly.rho_lower_bound(16, 298), 60) < mpq(GAP, 60)
        assert abs(root - mp.mpf("1.0073969")) < mp.mpf("1e-7")


def test_ln_lower_bound_stays_below_ln():
    """ln 2·ln(3/2)·ln ρ(16,298) 의 정확 하한 - mpmath 200자리 로그(독립 출처)보다 엄격히 작고 1e-25 안"""
    rho = assembly.rho_lower_bound(16, 298)
    with mp.workdps(200):
        for r in (Q(2), Q(3, 2), rho):
            assert 0 < mp.log(mpq(r, 200)) - mpq(assembly.ln_lower_bound(r), 200) < mp.mpf("1e-25")
        assert abs(mpq(assembly.ln_lower_bound(rho), 200) - mp.mpf("0.0073696936384425")) < mp.mpf("1e-16")


def test_expected_edges_and_loops_of_a_small_graph():
    """K=2, 출발·도착 p₈ - 변 집합과 순환 길이(16, 4)가 손으로 쓴 그래프와 같음"""
    hand = {(f"N{i}", f"N{(i + 1) % 16}") for i in range(16)} | {
        ("N8", "M-1"), ("M-1", "M+0"), ("M+0", "M+1"), ("M+1", "N8")}
    assert assembly.expected_edges(2, 8, 8) == hand
    assert assembly.loop_lengths(2, 8, 8) == (16, 4)


def test_loop_lengths_match_the_design_scan():
    """K=149(p₈→p₈)·K=150(p₇→p₉)·K=140(p₁→p₁₅) - 설계 훑기(graph_design_scan.json, design.py)가 낸 순환과 같음"""
    assert assembly.loop_lengths(149, 8, 8) == (16, 298)
    assert assembly.loop_lengths(150, 7, 9) == (16, 314)
    assert assembly.loop_lengths(140, 1, 15) == (16, 282)


def test_hull_and_disjointness():
    """단위 상자 둘 - 3 떨어지면 서로소, 2(닿음)·1.5(겹침)이면 서로소 아님, 한 축만 떨어져도 서로소(인자 순서 양쪽)"""
    eye = [[Q(int(i == j)) for j in range(4)] for i in range(4)]
    h0 = assembly.hull([Q(0)] * 4, eye)
    assert h0 == [(Q(-1), Q(1))] * 4
    shifted = lambda d: assembly.hull([Q(d), Q(0), Q(0), Q(0)], eye)
    assert assembly.disjoint(h0, shifted(3))
    assert not assembly.disjoint(h0, shifted(2)) and not assembly.disjoint(h0, shifted(Q(3, 2)))
    assert assembly.disjoint(shifted(0), assembly.hull([Q(0), Q(0), Q(0), Q(5)], eye))
    assert assembly.disjoint(shifted(3), h0)
    assert not assembly.disjoint(shifted(2), h0) and not assembly.disjoint(shifted(Q(3, 2)), h0)


def good_input(edge, tube_lower=Q(1, 4), tube_upper=Q(7), rt=Q(1, 4), problems=()):
    img = (iv(-2, 2), iv(-2, 2), iv(Q(-1, 2), Q(1, 2)), iv(Q(-1, 2), Q(1, 2)))
    far = {(0, -1): (iv(-3, -2), iv(-1, 1), iv(0, 0), iv(0, 0)), (0, 1): (iv(2, 3), iv(-1, 1), iv(0, 0), iv(0, 0)),
           (1, -1): (iv(-1, 1), iv(-3, -2), iv(0, 0), iv(0, 0)), (1, 1): (iv(-1, 1), iv(2, 3), iv(0, 0), iv(0, 0))}
    faces = {k: [PieceImage(face_box(*k), v, True)] for k, v in far.items()}
    return CoveringInput(edge, [PieceImage(unit_box(), img, True)], faces, (1, 1), tube_lower, tube_upper, rt, problems)


def setup(k=2):
    edges = assembly.expected_edges(k, 8, 8)
    eye = [[Q(int(i == j)) for j in range(4)] for i in range(4)]
    names = {n for e in edges for n in e}
    hulls = {n: assembly.hull([Q(10 * idx), Q(0), Q(0), Q(0)], eye) for idx, n in enumerate(sorted(names))}
    return edges, hulls, [good_input(f"{s}>{t}") for s, t in sorted(edges)]


def test_assembly_certifies_only_when_every_hypothesis_holds():
    """정상 입력 - 증명. 가정 하나씩 깨면(변 실패·저장값 위반·누락·그래프 불일치·M₀ 겹침·닿음·δ=0·귀환 시각=0) - 모두 거짓"""
    edges, hulls, inputs = setup()
    assert assembly.assemble(inputs, edges, 2, 8, 8, hulls).certified
    bad_verdict = list(inputs)
    bad_verdict[0] = CoveringInput(inputs[0].edge, inputs[0].whole, inputs[0].faces, (1, -1), Q(1, 4), Q(7), Q(1, 4), ())
    variants = [
        (bad_verdict, edges, hulls, 8),
        (inputs[:-1] + [good_input(inputs[-1].edge, problems=("stored usage",))], edges, hulls, 8),
        (inputs[1:], edges, hulls, 8),
        (inputs, edges - {("N0", "N1")}, hulls, 8),
        (inputs, edges, {**hulls, "N0": hulls["M+0"]}, 8),
        (inputs, edges, {**hulls, "N0": assembly.hull([Q(2), Q(0), Q(0), Q(0)], EYE)}, 8),
        (inputs[:-1] + [good_input(inputs[-1].edge, tube_lower=Q(0))], edges, hulls, 8),
        (inputs[:-1] + [good_input(inputs[-1].edge, rt=Q(0))], edges, hulls, 8),
        (inputs, edges, hulls, 7),
    ]
    for ins, eds, hls, dep in variants:
        assert not assembly.assemble(ins, eds, 2, dep, 8, hls).certified


def test_assembly_constants_are_extrema_over_coverings():
    """δ=흐름 관 하한의 최솟값, R=상한의 최댓값, 귀환 시각 하한=최솟값 - 합성 입력에서 손으로 셈한 값과 같음"""
    edges, hulls, inputs = setup()
    inputs[3] = good_input(inputs[3].edge, tube_lower=Q(1, 5) + Q(1, 100), tube_upper=Q(9), rt=Q(3, 10))
    inputs[5] = good_input(inputs[5].edge, tube_lower=Q(1, 3), tube_upper=Q(8), rt=Q(1, 5))
    a = assembly.assemble(inputs, edges, 2, 8, 8, hulls)
    assert (a.delta, a.rmax, a.tmin) == (Q(21, 100), Q(9), Q(1, 5))
    # 외피 중심이 x 축으로 10 씩 떨어진 단위 상자라 가장 가까운 이웃과의 분리 간격은 10−2
    assert (a.m0_gap, a.m0_overlaps, a.graph_diff) == (Q(8), (), 0)


def test_separation_is_the_largest_axis_gap():
    """단위 상자와 x 축으로 3·2·1.5 옮긴 상자 - 분리 간격 1·0(닿음)·−1/2(겹침), 인자 순서와 무관"""
    h0 = assembly.hull([Q(0)] * 4, EYE)
    for d, gap in ((Q(3), Q(1)), (Q(2), Q(0)), (Q(3, 2), Q(-1, 2))):
        h = assembly.hull([d, Q(0), Q(0), Q(0)], EYE)
        assert assembly.separation(h0, h) == gap == assembly.separation(h, h0)


def test_floor_decimal_rounds_down_and_pads():
    """2/3·1/3·5/2·10⁻⁶·123.456 - 반올림이 아니라 버림(0.66666), 앞자리 0 채움(0.000), 정수부 유지(123.45)"""
    cases = ((Q(2, 3), 5, "0.66666"), (Q(1, 3), 5, "0.33333"), (Q(5, 2), 3, "2.500"), (Q(1, 10**6), 3, "0.000"),
             (Q(123456, 1000), 2, "123.45"))
    for x, digits, text in cases:
        assert assembly.floor_decimal(x, digits) == text


def test_bisection_count_and_theorem_strings_are_pinned():
    """근 2 를 [1,2] 에서 80번 이분하면 하한은 정확히 2−2⁻⁸⁰(F(mid)=2/mid>1 이라 매번 오름). 정리 A 의 ρ·ln ρ 하한
    문자열은 theorem_A.json 에 적은 값과 정확히 같음(재현 고정)"""
    assert assembly.rho_lower_bound(1, 1) == 2 - Q(1, 2**80)
    rho = assembly.rho_lower_bound(16, 298)
    assert rho == Q(1217868143156043955884963, 2**80)
    assert assembly.floor_decimal(assembly.ln_lower_bound(rho)) == "0.0073696936384425343859428"


def claimed_for(a, **over):
    base = assembly.Claimed(delta=a.delta, rmax=a.rmax, tmin=a.tmin, rho=a.rho_lower,
                            ln_rho=assembly.ln_lower_bound(a.rho_lower))
    return dataclasses.replace(base, **over)


def verdicts(a, claimed):
    return {c.name: c for c in assembly.judge_claims(a, claimed)}


def test_claims_hold_at_their_certified_values_and_fail_just_beyond():
    """주장 상수가 다시 잰 값과 같으면 성립(경계 포함, 여유 0), 그 너머로 옮기면 그 주장만 거짓이고 여유는 옮긴 만큼 음수"""
    edges, hulls, inputs = setup()
    a = assembly.assemble(inputs, edges, 2, 8, 8, hulls)
    base = verdicts(a, claimed_for(a))
    assert len(base) == 7 and all(c.holds for c in base.values())
    assert (base["delta"].margin, base["R"].margin, base["t_min"].margin) == (0, 0, 0)
    tiny = Q(1, 10**9)
    beyond = {"delta": dict(delta=a.delta + tiny), "R": dict(rmax=a.rmax - tiny), "t_min": dict(tmin=a.tmin + tiny),
              "ln_rho": dict(ln_rho=assembly.ln_lower_bound(a.rho_lower) + tiny)}
    for name, over in beyond.items():
        res = verdicts(a, claimed_for(a, **over))
        assert not res[name].holds and res[name].margin == -tiny
        assert all(c.holds for n, c in res.items() if n != name)
    res = verdicts(a, claimed_for(a, rho=a.rho_lower + Q(1, 10**6)))
    assert not res["rho"].holds and res["rho"].margin < 0 and not res["ln_rho"].holds


def test_claims_need_positive_constants_and_supported_rho():
    """δ·t_min 주장 0, ρ 주장 1, ρ 가 받쳐지지 않으면 ln 주장도 - 거짓(이격·귀환·엔트로피가 양수여야 정리의 뜻이 있음).
    근이 유리수 2 인 순환 (1,1) 에서는 ρ ≥ 2 가 정확히 성립(F(2)=1)하고 2 를 넘으면 거짓"""
    edges, hulls, inputs = setup()
    a = assembly.assemble(inputs, edges, 2, 8, 8, hulls)
    for name, over in (("delta", dict(delta=Q(0))), ("t_min", dict(tmin=Q(0))), ("rho", dict(rho=Q(1)))):
        assert not verdicts(a, claimed_for(a, **over))[name].holds
    assert not verdicts(a, claimed_for(a, rho=Q(2), ln_rho=Q(0)))["ln_rho"].holds
    two = dataclasses.replace(a, loops=(1, 1))
    assert verdicts(two, claimed_for(two, rho=Q(2), ln_rho=Q(0)))["rho"].holds
    assert not verdicts(two, claimed_for(two, rho=Q(2) + Q(1, 10**9), ln_rho=Q(0)))["rho"].holds


def test_structure_and_missing_data_claims():
    """M₀ 외피가 닿음·설계 변 하나 빠짐·덮개 기록 없음 - (A3)·(A4)·상수 주장 거짓, 여유는 간격 0·대칭차 −1·없음(None)"""
    edges, hulls, inputs = setup()
    touching = assembly.assemble(inputs, edges, 2, 8, 8, {**hulls, "N0": assembly.hull([Q(2), Q(0), Q(0), Q(0)], EYE)})
    res = verdicts(touching, claimed_for(touching))
    assert not res["A3_m0_disjoint"].holds and res["A3_m0_disjoint"].margin == 0 and touching.m0_overlaps == ("N0",)
    dropped = assembly.assemble(inputs, edges - {("N0", "N1")}, 2, 8, 8, hulls)
    res = verdicts(dropped, claimed_for(dropped))
    assert not res["A4_graph"].holds and res["A4_graph"].margin == -1
    good = assembly.assemble(inputs, edges, 2, 8, 8, hulls)
    empty = assembly.assemble([], edges, 2, 8, 8, hulls)
    res = verdicts(empty, claimed_for(good))
    assert all(not res[n].holds and res[n].margin is None for n in ("delta", "R", "t_min"))


def test_small_positive_separation_still_counts_as_disjoint():
    """M₀ 외피와 이웃의 간격 1/2(0 보다 크고 1 보다 작음) - 서로소: 겹침 목록이 비고 정리 가정·(A3) 주장 성립, 여유 1/2"""
    edges, hulls, inputs = setup()
    near = {**hulls, "N0": assembly.hull([Q(5, 2), Q(0), Q(0), Q(0)], EYE)}
    a = assembly.assemble(inputs, edges, 2, 8, 8, near)
    assert a.certified and a.m0_overlaps == () and a.m0_gap == Q(1, 2)
    check = verdicts(a, claimed_for(a))["A3_m0_disjoint"]
    assert check.holds and check.margin == Q(1, 2)
