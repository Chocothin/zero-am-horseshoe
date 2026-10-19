"""정확 판정층 명세 시험(G9). 기대값은 ZG2004 Theorem 15 의 덮개 조건(T6 REPORT §4.1 대응), Gershgorin 원판 정리,
구간 산술의 포함 성질, 방향 반올림의 정의, T6 설계의 부호 규칙에서 정했다. 구현을 보고 기대값을 정하지 않았다."""
from __future__ import annotations

import json
import random
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import exact_judge as EJ  # noqa: E402
import v7exact as X  # noqa: E402

UNIT = (F(-1), F(1))


def test_parse_hex_is_exact_beyond_53_bits():
    """16진 끝점 - 53비트를 넘는 가수도 반올림 없이 정확한 유리수(함정 ①: 끝점을 float 로 바꾸지 않는다)."""
    assert X.parse("0x1.8p+0") == F(3, 2)
    assert X.parse("-0x1.0000000000000001p+0") == -(1 + F(1, 2 ** 64))
    assert X.parse("0x1p-1074") == F(1, 2 ** 1074)
    assert X.parse("123/456") == F(123, 456)
    assert X.parse("-0.25") == F(-1, 4)


def test_interval_rejects_reversed_endpoints():
    """lo > hi 인 구간 - 거부(잘못된 기록이 빈 구간으로 판정을 통과하지 않게)."""
    with pytest.raises(ValueError):
        X.iv({"lo": "1", "hi": "0"})


def test_interval_operations_contain_all_point_results():
    """구간 곱·제곱·합·부호 반전 - 끝점 사이 모든 점의 결과를 담는다(포함 단조성)."""
    rnd = random.Random(11)
    for _ in range(300):
        a = sorted(F(rnd.randint(-50, 50), rnd.randint(1, 9)) for _ in range(2))
        b = sorted(F(rnd.randint(-50, 50), rnd.randint(1, 9)) for _ in range(2))
        x = a[0] + (a[1] - a[0]) * F(rnd.randint(0, 100), 100)
        y = b[0] + (b[1] - b[0]) * F(rnd.randint(0, 100), 100)
        assert X.contains(X.mul(tuple(a), tuple(b)), (x * y, x * y))
        assert X.contains(X.sq(tuple(a)), (x * x, x * x))
        assert X.contains(X.add(tuple(a), tuple(b)), (x + y, x + y))
        assert X.contains(X.neg(tuple(a)), (-x, -x))
        assert X.mag(tuple(a)) == max(abs(a[0]), abs(a[1]))


def test_square_of_interval_straddling_zero_starts_at_zero():
    """0 을 품은 구간의 제곱 - 하한 0(제곱 감싸기 {a²: a∈[a]}, 곱 감싸기보다 좁다)."""
    assert X.sq((F(-2), F(3))) == (F(0), F(9))
    assert X.mul((F(-2), F(3)), (F(-2), F(3))) == (F(-6), F(9))


def test_affine_image_contains_images_of_points():
    """center + D·piece - 중심 구간·행렬 구간·조각의 모든 점 c+A·w 를 담는다(평균값 형식의 상)."""
    rnd = random.Random(5)
    c = [(F(rnd.randint(-9, 0), 10), F(rnd.randint(1, 9), 10)) for _ in range(4)]
    D = [[tuple(sorted((F(rnd.randint(-30, 30), 10), F(rnd.randint(-30, 30), 10)))) for _ in range(4)] for _ in range(4)]
    img = X.affine_image(c, D, [UNIT] * 4)
    for _ in range(200):
        cp = [lo + (hi - lo) * F(rnd.randint(0, 10), 10) for lo, hi in c]
        A = [[lo + (hi - lo) * F(rnd.randint(0, 10), 10) for lo, hi in row] for row in D]
        w = [F(rnd.randint(-10, 10), 10) for _ in range(4)]
        for i in range(4):
            v = cp[i] + sum(A[i][j] * w[j] for j in range(4))
            assert img[i][0] <= v <= img[i][1]


def box(w0, w1, w2, w3):
    return [(F(a), F(b)) for a, b in (w0, w1, w2, w3)]


def test_cover_W_holds_by_unstable_component_outside():
    """(W) 불안정 성분 하나가 통째로 |w′|>1 - 여유 = 1 까지 거리(T6 §4.1 (W) 첫 갈래)."""
    w, w_in, _ = X.cover_margins(box(("3/2", "2"), ("-5", "5"), ("-5", "5"), ("-5", "5")), {}, (1, 1))
    assert w == F(1, 2) and w_in < 0


def test_cover_W_holds_by_both_stable_components_inside():
    """(W) 안정 성분 둘이 통째로 (−1,1) 안 - 여유 = 경계까지의 거리(T6 §4.1 (W) 둘째 갈래)."""
    w, w_in, _ = X.cover_margins(box(("-5", "5"), ("-5", "5"), ("-1/2", "1/4"), ("-3/4", "0")), {}, (1, 1))
    assert w == F(1, 4) and w_in == F(1, 4)


def test_cover_W_touching_boundary_is_not_proven():
    """(W) 안정 성분이 경계 1 에 닿음 - 여유 0, 엄격 부등식이라 성립하지 않는다(판정은 > 0)."""
    w, _, _ = X.cover_margins(box(("-5", "5"), ("-5", "5"), ("-1/2", "1"), ("0", "0")), {}, (1, 1))
    assert w == 0


def test_cover_W_fails_when_one_stable_component_leaves():
    """(W) 안정 성분 하나가 (−1,1) 밖으로 나가고 불안정 성분도 1 을 품음 - 여유 음수(덮개 판정 거짓)."""
    w, _, _ = X.cover_margins(box(("1/2", "3/2"), ("-2", "2"), ("-1/2", "1/2"), ("9/10", "11/10")), {}, (1, 1))
    assert w < 0


def test_cover_faces_follow_sign_times_side():
    """(F) 출구 면 {w_j=σ} 에서 s_j·σ·w′_j > 1 - s_jσ>0 이면 하한−1, <0 이면 −1−상한(T6 §4.1 (F))."""
    img_pos = box(("6/5", "13/10"), ("0", "0"), ("0", "0"), ("0", "0"))
    img_neg = box(("0", "0"), ("-7/5", "-11/10"), ("0", "0"), ("0", "0"))
    _, _, faces = X.cover_margins(box(*[("0", "0")] * 4), {(0, 1): img_pos, (1, 1): img_neg}, (1, -1))
    assert faces[(0, 1)] == F(1, 5)
    assert faces[(1, 1)] == F(1, 10)
    _, _, wrong = X.cover_margins(box(*[("0", "0")] * 4), {(0, 1): img_pos}, (-1, -1))
    assert wrong[(0, 1)] < 0


def test_face_piece_is_the_whole_face():
    """출구 면 조각 - 꼭짓점이 아니라 면 전체 {w_j=σ}×[−1,1]³(함정 ⑥)."""
    assert X.face_piece(1, -1) == [UNIT, (F(-1), F(-1)), UNIT, UNIT]


def test_gershgorin_diagonal_matrix_by_hand():
    """대각 D=diag(d) - S=DᵀJD=diag(d0²,d1²,−d2²,−d3²), R=0 이라 a*=max(d2²,d3²), b*=min(d0²,d1²)."""
    d = [F(3, 2), F(6, 5), F(4, 5), F(9, 10)]
    D = [[(d[i], d[i]) if i == j else (F(0), F(0)) for j in range(4)] for i in range(4)]
    a, b = X.gershgorin(D)
    assert a == max(d[2] ** 2, d[3] ** 2) and b == min(d[0] ** 2, d[1] ** 2)


def test_gershgorin_off_diagonal_by_hand():
    """비대각 성분 하나 - S 의 (0,2) 성분 = d00·d02(J₀=+1) 가 R_0·R_2 에 더해진다(손 계산)."""
    D = [[(F(2), F(2)), (F(0), F(0)), (F(1, 10), F(1, 10)), (F(0), F(0))],
         [(F(0), F(0)), (F(3, 2), F(3, 2)), (F(0), F(0)), (F(0), F(0))],
         [(F(0), F(0)), (F(0), F(0)), (F(1, 2), F(1, 2)), (F(0), F(0))],
         [(F(0), F(0)), (F(0), F(0)), (F(0), F(0)), (F(1, 2), F(1, 2))]]
    # S00=4, S11=9/4, S22=1/100−1/4, S33=−1/4, S02=S20=2·(1/10)=1/5
    a, b = X.gershgorin(D)
    assert b == min(F(4) - F(1, 5), F(9, 4)) and a == max(F(1, 5) - (F(1, 100) - F(1, 4)), F(1, 4))


def test_gershgorin_bound_holds_for_point_matrices():
    """구간 행렬 안 임의의 A - λ_min(AᵀJA − μJ) ≥ min(b*−μ, μ−a*) (Gershgorin 원판 정리, T7 §2.1)."""
    rnd = random.Random(3)
    J = np.diag([1.0, 1.0, -1.0, -1.0])
    base = np.diag([1.3, 1.15, 0.85, 0.7])
    D = [[(F(base[i][j]) - F(1, 50), F(base[i][j]) + F(1, 50)) for j in range(4)] for i in range(4)]
    a, b = X.gershgorin(D)
    assert a < 1 < b
    for mu in (float(a) + 1e-3, 1.0, float(b) - 1e-3):
        bound = min(float(b) - mu, mu - float(a))
        for _ in range(100):
            A = np.array([[float(lo + (hi - lo) * F(rnd.randint(0, 20), 20)) for lo, hi in row] for row in D])
            assert np.linalg.eigvalsh(A.T @ J @ A - mu * J).min() >= bound - 1e-12


def test_directed_rounding_is_floor_and_ceiling():
    """방향 반올림 - down 은 −∞ 쪽, up 은 +∞ 쪽 12자리 격자(정의)."""
    assert X.fstr(F(1, 3), 3, "down") == "0.333" and X.fstr(F(1, 3), 3, "up") == "0.334"
    assert X.fstr(F(-1, 3), 3, "down") == "-0.334" and X.fstr(F(-1, 3), 3, "up") == "-0.333"
    assert X.fstr(F(1, 4), 2, "up") == "0.25"


def test_design_signs_follow_linear_diagonal_and_reject_zero():
    """덮개 부호 - 설계 linear 대각의 부호(T6 certify_graph 규칙), 0 대각은 거부."""
    d = {"edges": [{"source": "A", "target": "B", "linear": [[2.0, 0, 0, 0], [0, -3.0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}]}
    assert EJ.design_signs(d) == {"A>B": [1, -1]}
    d["edges"][0]["linear"][1][1] = 0.0
    with pytest.raises(ValueError):
        EJ.design_signs(d)


def synthetic_record(tmp_path, center, diag, off=F(0)):
    D = [[[str(diag[i]), str(diag[i])] if i == j else [str(off), str(off)] for j in range(4)] for i in range(4)]
    rec = {"edge": "A>B", "center_image_exact": [[str(c), str(c)] for c in center], "deriv_mine_exact": D,
           "image_whole_exact": [[str(center[i] - abs(diag[i]) - 3 * abs(off)), str(center[i] + abs(diag[i]) + 3 * abs(off))]
                                 for i in range(4)],
           "a_star_exact": "0", "b_star_exact": "0", "covering_margin": 0, "a_star": "0", "b_star": "0"}
    p = tmp_path / "A__B.json"
    p.write_text(json.dumps(rec))
    return p


def test_judge_v5_margin_for_diagonal_map_by_hand(tmp_path):
    """대각 사상 P̂(w)=diag(2, 3/2, 1/2, 1/4)w - 상자 상 불안정 [−2,2]·[−3/2,3/2] 은 1 을 품고 안정 둘이 안쪽(여유 1/2),
    출구 면 여유 min(2,3/2)−1=1/2, 그래서 덮개 여유 1/2 이고 a*=1/4, b*=9/4(손 계산)."""
    p = synthetic_record(tmp_path, [F(0)] * 4, [F(2), F(3, 2), F(1, 2), F(1, 4)])
    r = EJ.judge_v5(p, {"A>B": [1, 1]})
    assert F(r["cover_margin_mv_exact"]) == F(1, 2) and r["cover_proven_mv"]
    assert F(r["a_star_exact"]) == F(1, 4) and F(r["b_star_exact"]) == F(9, 4) and r["cone_proven"]


def test_judge_v5_wrong_sign_fails_faces(tmp_path):
    """부호를 뒤집은 덮개 - 출구 면 조건이 깨져 덮개 판정 거짓(부호는 판정 입력이다)."""
    p = synthetic_record(tmp_path, [F(0)] * 4, [F(2), F(3, 2), F(1, 2), F(1, 4)])
    assert not EJ.judge_v5(p, {"A>B": [-1, 1]})["cover_proven_mv"]


def test_judge_v5_offset_center_reduces_margin(tmp_path):
    """중심 상을 불안정 방향으로 1/10 옮김 - 한쪽 출구 면 여유가 정확히 1/10 줄어든다(평균값 상의 평행 이동)."""
    p = synthetic_record(tmp_path, [F(1, 10), F(0), F(0), F(0)], [F(2), F(3, 2), F(1, 2), F(1, 4)])
    r = EJ.judge_v5(p, {"A>B": [1, 1]})
    assert F(r["cover_margin_mv_exact"]) == F(1, 2)
    p2 = synthetic_record(tmp_path, [F(0), F(1, 10), F(0), F(0)], [F(2), F(3, 2), F(1, 2), F(1, 4)])
    assert F(EJ.judge_v5(p2, {"A>B": [1, 1]})["cover_margin_mv_exact"]) == F(1, 2) - F(1, 10)


def design_with_linear(diag):
    lin = [[float(diag[i]) if i == j else 0.0 for j in range(4)] for i in range(4)]
    return {"edges": [{"source": "A", "target": "B", "linear": lin}]}


def test_design_sign_rule_makes_exit_faces_hold_for_orientation_reversing_map(tmp_path):
    """T6 §4.1 (F) s_j·σ·w′_j>1 과 (69) deg=s₀s₁ - 첫 불안정 방향을 뒤집는 사상 diag(−2, 3/2, 1/2, 1/4) 에서
    설계 규칙(부호=linear 대각 부호)이 s=(−1,+1) 을 주고, 그 부호로 출구 면 조건이 성립하며 차수는 −1, 반대 부호는 실패한다."""
    diag = [F(-2), F(3, 2), F(1, 2), F(1, 4)]
    signs = EJ.design_signs(design_with_linear(diag))
    assert signs == {"A>B": [-1, 1]}
    p = synthetic_record(tmp_path, [F(0)] * 4, diag)
    r = EJ.judge_v5(p, signs)
    assert r["cover_proven_mv"] and all(F(v) > 0 for v in r["faces_mv"].values())
    assert signs["A>B"][0] * signs["A>B"][1] == -1
    assert not EJ.judge_v5(p, {"A>B": [1, 1]})["cover_proven_mv"]


def test_design_sign_rule_is_the_only_sign_choice_that_covers(tmp_path):
    """무작위 대각 사상(불안정 |λ|∈[6/5,3], 안정 |λ|≤4/5, 부호 무작위) - 설계 규칙의 부호에서만 (F) 가 성립한다(T6 §4.1)."""
    rnd = random.Random(17)
    for _ in range(40):
        diag = [F(rnd.choice((-1, 1)) * rnd.randint(12, 30), 10) for _ in range(2)]
        diag += [F(rnd.choice((-1, 1)) * rnd.randint(0, 8), 10) for _ in range(2)]
        p = synthetic_record(tmp_path, [F(0)] * 4, diag)
        signs = EJ.design_signs(design_with_linear(diag))
        assert EJ.judge_v5(p, signs)["cover_proven_mv"]
        for s0 in (-1, 1):
            for s1 in (-1, 1):
                if [s0, s1] != signs["A>B"]:
                    assert not EJ.judge_v5(p, {"A>B": [s0, s1]})["cover_proven_mv"]


def test_contains_requires_both_endpoints():
    """구간 포함 - 바깥 하한 ≤ 안 하한이고 안 상한 ≤ 바깥 상한일 때만 참(한쪽만 맞으면 거짓)."""
    assert X.contains((F(0), F(3)), (F(1), F(2)))
    assert not X.contains((F(0), F(2)), (F(1), F(3)))
    assert not X.contains((F(1), F(3)), (F(0), F(2)))
    assert not X.contains((F(0), F(1)), (F(2), F(3)))


def test_directed_rounding_default_digits_and_zero():
    """방향 반올림 - 기본 15자리, 0 은 부호 없이 0.000(정의)."""
    assert X.fstr(F(1, 3)) == "0.333333333333333"
    assert X.fstr(F(0), 3, "down") == "0.000" and X.fstr(F(0), 3, "up") == "0.000"
    assert X.fstr(F(-1, 10 ** 5), 3, "up") == "0.000" and X.fstr(F(-1, 10 ** 5), 3, "down") == "-0.001"


def test_gershgorin_uses_lower_end_of_binding_unstable_row():
    """b* 는 불안정 행 i∈{0,1} 의 S_ii 하한 − R_i 의 최솟값 - 0행이 정하는 경우와 구간 대각의 하한(손 계산)."""
    D = [[(F(1), F(11, 10)), (F(0), F(0)), (F(0), F(0)), (F(0), F(0))],
         [(F(0), F(0)), (F(2), F(2)), (F(0), F(0)), (F(0), F(0))],
         [(F(0), F(0)), (F(0), F(0)), (F(1, 2), F(1, 2)), (F(0), F(0))],
         [(F(0), F(0)), (F(0), F(0)), (F(0), F(0)), (F(1, 4), F(1, 4))]]
    a, b = X.gershgorin(D)
    assert b == 1 and a == F(1, 4)


def test_cover_W_by_unstable_component_below_minus_one_and_second_unstable_component():
    """(W) 불안정 성분이 통째로 −1 아래(여유 −1−상한), 둘째 불안정 성분 w′₁ 만 밖인 경우도 (W) 가 성립한다."""
    w, _, _ = X.cover_margins(box(("-2", "-3/2"), ("-5", "5"), ("-5", "5"), ("-5", "5")), {}, (1, 1))
    assert w == F(1, 2)
    w1, _, _ = X.cover_margins(box(("-5", "5"), ("3/2", "2"), ("-5", "5"), ("-5", "5")), {}, (1, 1))
    assert w1 == F(1, 2)


def diag_record(tmp_path, diag, name="A__B.json"):
    D = [[[str(diag[i][0]), str(diag[i][1])] if i == j else ["0", "0"] for j in range(4)] for i in range(4)]
    rec = {"edge": "A>B", "center_image_exact": [["0", "0"]] * 4, "deriv_mine_exact": D,
           "image_whole_exact": [[str(-max(abs(x) for x in diag[i])), str(max(abs(x) for x in diag[i]))] for i in range(4)],
           "a_star_exact": "1/4", "b_star_exact": "9/4", "covering_margin": 0, "a_star": "0", "b_star": "0"}
    p = tmp_path / name
    p.write_text(json.dumps(rec))
    return p


def test_judge_v5_reported_fields_follow_their_definitions(tmp_path):
    """judge_v5 의 보고 칸 - 12자리 방향 반올림 문자열, μ 여유 = μ_s−a*·b*−μ_u, 덮개 여유 = (W)·면 여유의 최솟값,
    V5 대각 부호와 일치 여부, 저장값과의 같음 여부(diag(2, 3/2, 1/2, 1/4) 손 계산)."""
    p = diag_record(tmp_path, [(F(2), F(2)), (F(3, 2), F(3, 2)), (F(1, 2), F(1, 2)), (F(1, 4), F(1, 4))])
    r = EJ.judge_v5(p, {"A>B": [1, 1]})
    assert r["cover_margin_mv"] == "0.500000000000" and r["W_mv"] == "0.500000000000"
    assert r["faces_mv"] == {"w0-1": "1.000000000000", "w0+1": "1.000000000000", "w1-1": "0.500000000000",
                             "w1+1": "0.500000000000"}
    assert r["a_star"] == "0.250000000000" and r["b_star"] == "2.250000000000"
    assert r["mu_s_margin"] == "0.644400000000" and r["mu_u_margin"] == "1.133300000000"
    assert r["signs_from_D5_diag"] == [1, 1] and r["signs_agree"]
    assert r["a_star_equals_v5"] and r["b_star_equals_v5"]
    assert r["W_stored_whole"] == X.fstr(F(1, 2), 12, "down")


def test_judge_v5_diag_signs_need_strict_sign_of_interval(tmp_path):
    """V5 대각 부호 - 대각 구간 하한 > 0 이면 +1, 상한 < 0 이면 −1, 0 에 닿으면 0(엄격)."""
    p = diag_record(tmp_path, [(F(0), F(1)), (F(-1), F(0)), (F(1, 2), F(1, 2)), (F(1, 4), F(1, 4))])
    r = EJ.judge_v5(p, {"A>B": [1, -1]})
    assert r["signs_from_D5_diag"] == [0, 0] and not r["signs_agree"]
    p2 = diag_record(tmp_path, [(F(2), F(3)), (F(-3), F(-2)), (F(1, 2), F(1, 2)), (F(1, 4), F(1, 4))], "C__D.json")
    r2 = EJ.judge_v5(p2, {"A>B": [1, -1]})
    assert r2["signs_from_D5_diag"] == [1, -1] and r2["signs_agree"]


def test_judge_v5_cover_proven_is_strict(tmp_path):
    """덮개 판정은 여유 > 0 - 안정 성분 상이 경계 ±1 에 닿아 여유가 정확히 0 이면 증명이 아니다."""
    p = diag_record(tmp_path, [(F(2), F(2)), (F(3, 2), F(3, 2)), (F(1), F(1)), (F(1, 4), F(1, 4))])
    r = EJ.judge_v5(p, {"A>B": [1, 1]})
    assert F(r["cover_margin_mv_exact"]) == 0 and not r["cover_proven_mv"]


def test_judge_v5_cone_needs_both_strict_inequalities(tmp_path, monkeypatch):
    """원뿔 판정은 a*<μ_s 그리고 b*>μ_u(엄격) - 같으면 거짓, 한쪽만 맞아도 거짓."""
    p = diag_record(tmp_path, [(F(2), F(2)), (F(3, 2), F(3, 2)), (F(1, 2), F(1, 2)), (F(1, 4), F(1, 4))])
    monkeypatch.setattr(EJ, "MU_S", F(1, 4))
    monkeypatch.setattr(EJ, "MU_U", F(2))
    assert not EJ.judge_v5(p, {"A>B": [1, 1]})["cone_proven"]
    monkeypatch.setattr(EJ, "MU_S", F(1, 2))
    monkeypatch.setattr(EJ, "MU_U", F(9, 4))
    assert not EJ.judge_v5(p, {"A>B": [1, 1]})["cone_proven"]
    monkeypatch.setattr(EJ, "MU_U", F(2))
    assert EJ.judge_v5(p, {"A>B": [1, 1]})["cone_proven"]


def test_exact_judge_summary_reports_minimum_cover_margin(tmp_path):
    """exact_judge 요약 - 변 기록 여러 개의 덮개 여유 최솟값과 그 변을 12자리 내림으로 낸다."""
    d = tmp_path / "boxes"
    d.mkdir()
    diag_record(d, [(F(2), F(2)), (F(3, 2), F(3, 2)), (F(1, 2), F(1, 2)), (F(1, 4), F(1, 4))])
    rec = json.loads((d / "A__B.json").read_text())
    rec["edge"] = "N0>N1"
    (d / "N0__N1.json").write_text(json.dumps(rec))
    (d / "A__B.json").unlink()
    out = tmp_path / "judge.json"
    EJ.main(["--v5dir", str(d), "--out", str(out)])
    s = json.loads(out.read_text())["v5_stage2"]
    assert s["min_cover_mv_value_down"] == "0.500000000000" and s["min_cover_mv"] == "N0>N1"
