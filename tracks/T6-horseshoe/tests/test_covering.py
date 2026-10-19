"""덮개 판정층 명세 시험. 기대값은 ZG2004 Theorem 15 의 조건 (64)~(69) 와 「조각 합집합 = 상자」 전제에서 정했다."""
from fractions import Fraction as Q

import pytest
from t3cap.exact import Interval

from t6hs import covering
from t6hs.covering import PieceImage, covering_verdict, exactly_covers, face_box, unit_box


def iv(lo, hi):
    return Interval(Q(lo), Q(hi))


def linear_image(piece, diag):
    return tuple(Interval(min(a * p.lo, a * p.hi), max(a * p.lo, a * p.hi)) for a, p in zip(map(Q, diag), piece))


def linear_case(diag, signs=None, certified=True):
    signs = signs or (1 if diag[0] > 0 else -1, 1 if diag[1] > 0 else -1)
    whole = [PieceImage(unit_box(), linear_image(unit_box(), diag), certified)]
    faces = {(j, s): [PieceImage(face_box(j, s), linear_image(face_box(j, s), diag), True)]
             for j in covering.UNSTABLE for s in (-1, 1)}
    return covering_verdict(whole, faces, signs)


def test_hyperbolic_diagonal_map_is_a_covering():
    """불안정 |a|>1·안정 |a|<1 인 대각 선형 사상 - 증명, 차수=부호 곱(Theorem 15 의 모든 조건이 엄격 부등식으로 성립)"""
    v = linear_case((Q(13, 10), Q(-12, 10), Q(1, 2), Q(-7, 10)))
    assert v.proven and v.degree == -1


def test_exit_face_on_the_boundary_is_rejected():
    """불안정 배율이 정확히 1 - 실패((65)(68) 은 출구 면 상이 M 밖, 즉 |w'|>1 엄격 부등식을 요구)"""
    v = linear_case((Q(1), Q(2), Q(1, 2), Q(1, 2)))
    assert not v.proven and any(c.name.startswith("face_w0") for c in v.failed)


def test_stable_image_reaching_the_entry_set_is_rejected():
    """안정 배율이 정확히 1 이고 그 조각이 불안정 방향 밖도 아님 - 실패((66) f(N)∩M⁺=∅ 를 보일 수 없음)"""
    v = linear_case((Q(2), Q(2), Q(1), Q(1, 2)))
    assert not v.proven and "whole_box_images_ok" in [c.name for c in v.failed]


def test_wrong_orientation_sign_is_rejected():
    """음의 배율에 부호 +1 을 주장 - 실패(차수 계산의 직선 호모토피가 출구 면 부호와 맞아야 함)"""
    v = linear_case((Q(2), Q(-2), Q(1, 2), Q(1, 2)), signs=(1, 1))
    assert not v.proven


def test_uncertified_engine_result_is_rejected():
    """사용 자격 검사를 못 넘은 조각 - 실패(감싸기가 엄밀하지 않으면 부등식 판정이 무의미)"""
    assert not linear_case((Q(2), Q(2), Q(1, 2), Q(1, 2)), certified=False).proven


def nonlinear_pieces(split):
    """w'=(3w0, 1.5w1, 0.5w2+2w0², 0.5w3): |w0|>split 조각은 불안정 방향으로 밖이고 안정 성분은 커도 된다."""
    edges = [Q(-1), -split, split, Q(1)]
    pieces = []
    for a, b in zip(edges, edges[1:]):
        p = (iv(a, b), iv(-1, 1), iv(-1, 1), iv(-1, 1))
        sq = Interval(Q(0) if a <= 0 <= b else min(a * a, b * b), max(a * a, b * b))
        img = (iv(3 * a, 3 * b), iv(Q(-3, 2), Q(3, 2)), Interval(Q(-1, 2) + 2 * sq.lo, Q(1, 2) + 2 * sq.hi),
               iv(Q(-1, 2), Q(1, 2)))
        pieces.append(PieceImage(p, img, True))
    return pieces


def nonlinear_faces():
    out = {}
    for s in (-1, 1):
        out[(0, s)] = [PieceImage(face_box(0, s), (iv(3 * s, 3 * s), iv(Q(-3, 2), Q(3, 2)), iv(Q(3, 2), Q(5, 2)),
                                                    iv(Q(-1, 2), Q(1, 2))), True)]
        out[(1, s)] = [PieceImage(face_box(1, s), (iv(-3, 3), iv(Q(3, 2) * s, Q(3, 2) * s), iv(Q(-1, 2), Q(5, 2)),
                                                    iv(Q(-1, 2), Q(1, 2))), True)]
    return out


def test_pieces_outside_in_unstable_direction_may_leave_stable_bounds():
    """불안정 방향으로 통째로 밖인 조각은 안정 성분이 ±1 을 넘어도 됨 - 증명(S(M)⁻={‖p‖>1} 이 열린집합이라 (64)(66) 이 성립)"""
    v = covering_verdict(nonlinear_pieces(Q(2, 5)), nonlinear_faces(), (1, 1))
    assert v.proven


def test_piece_straddling_the_unstable_boundary_with_large_stable_image_is_rejected():
    """|w'0| 가 1 을 걸치는 조각의 안정 성분이 ±1 을 넘음 - 실패(그 조각 상이 M⁺ 와 만날 수 있음)"""
    v = covering_verdict(nonlinear_pieces(Q(7, 10)), nonlinear_faces(), (1, 1))
    assert not v.proven


def test_missing_piece_breaks_the_cover():
    """조각 하나를 뺌 - 실패(빠진 부분의 상은 아무것도 판정되지 않음)"""
    pieces = nonlinear_pieces(Q(2, 5))[:2]
    assert not covering_verdict(pieces, nonlinear_faces(), (1, 1)).proven


def test_overlapping_pieces_with_the_right_total_volume_do_not_cover():
    """부피 합은 맞지만 내부가 겹치고 일부가 빔 - 덮지 않음(부피 조건만으로는 합집합=상자가 아님)"""
    a = (iv(-1, Q(1, 2)), iv(-1, 1), iv(-1, 1), iv(-1, 1))
    b = (iv(0, Q(1, 2)), iv(-1, 1), iv(-1, 1), iv(-1, 1))
    assert not exactly_covers([a, b], unit_box())


def test_piece_sticking_out_of_the_box_is_rejected():
    """상자 밖으로 나간 조각 - 덮개로 인정하지 않음(그 조각의 판정은 h-set 밖의 점을 포함)"""
    a = (iv(-1, Q(3, 2)), iv(-1, 1), iv(-1, 1), iv(-1, 1))
    assert not exactly_covers([a], unit_box())


def test_dyadic_grid_covers_box_and_faces():
    """2^4 격자 조각과 면의 2^3 격자 조각 - 정확히 덮음"""
    halves = [iv(-1, 0), iv(0, 1)]
    grid = [(a, b, c, d) for a in halves for b in halves for c in halves for d in halves]
    assert exactly_covers(grid, unit_box())
    assert not exactly_covers(grid[1:], unit_box())
    face = [(iv(1, 1), b, c, d) for b in halves for c in halves for d in halves]
    assert exactly_covers(face, face_box(0, 1))


def test_missing_exit_face_is_a_usage_error():
    """출구 면 4개 중 하나가 없음 - ValueError(조건 (65) 는 N⁻ 전체를 요구)"""
    faces = nonlinear_faces()
    del faces[(1, -1)]
    with pytest.raises(ValueError):
        covering_verdict(nonlinear_pieces(Q(2, 5)), faces, (1, 1))


def test_empty_piece_list_is_not_a_cover():
    """조각이 하나도 없음 - 덮지 않고 판정도 실패(빈 all() 이 참이 되어 거짓 증명이 되는 경로를 막음)"""
    assert not exactly_covers([], unit_box())
    assert not covering_verdict([], nonlinear_faces(), (1, 1)).proven


def test_piece_of_wrong_dimension_is_rejected():
    """3차원 조각을 4차원 상자에 줌 - 덮개로 인정하지 않음"""
    assert not exactly_covers([(iv(-1, 1), iv(-1, 1), iv(-1, 1))], unit_box())


def faces_from(images):
    """출구 면마다 조각 하나와 주어진 상(없으면 충분히 밖)."""
    far = {(0, -1): (iv(-3, -2), iv(-1, 1), iv(0, 0), iv(0, 0)), (0, 1): (iv(2, 3), iv(-1, 1), iv(0, 0), iv(0, 0)),
           (1, -1): (iv(-1, 1), iv(-3, -2), iv(0, 0), iv(0, 0)), (1, 1): (iv(-1, 1), iv(2, 3), iv(0, 0), iv(0, 0))}
    far.update(images)
    return {k: [PieceImage(face_box(*k), img, True)] for k, img in far.items()}


INSIDE = PieceImage(unit_box(), (iv(-2, 2), iv(-2, 2), iv(Q(-1, 2), Q(1, 2)), iv(Q(-1, 2), Q(1, 2))), True)


def test_one_sided_exit_face_boundary_is_rejected_on_each_side():
    """면 하나만 상 끝점이 정확히 ±1 - 실패(나머지 면이 모두 밖이어도 (65) 의 엄격 부등식이 깨짐)"""
    for key, img in (((0, 1), (iv(1, 2), iv(-1, 1), iv(0, 0), iv(0, 0))),
                     ((0, -1), (iv(-2, -1), iv(-1, 1), iv(0, 0), iv(0, 0))),
                     ((1, 1), (iv(-1, 1), iv(1, 2), iv(0, 0), iv(0, 0))),
                     ((1, -1), (iv(-1, 1), iv(-2, -1), iv(0, 0), iv(0, 0)))):
        assert not covering_verdict([INSIDE], faces_from({key: img}), (1, 1)).proven
    assert covering_verdict([INSIDE], faces_from({}), (1, 1)).proven


def test_one_sided_stable_boundary_is_rejected():
    """안정 성분 상의 한쪽 끝만 정확히 ±1 이고 불안정 방향으로 밖이 아님 - 실패((66) 은 ‖q′‖<1 엄격)"""
    for lo, hi in ((-1, Q(1, 2)), (Q(-1, 2), 1)):
        for k in (2, 3):
            img = [iv(-2, 2), iv(-2, 2), iv(Q(-1, 2), Q(1, 2)), iv(Q(-1, 2), Q(1, 2))]
            img[k] = iv(lo, hi)
            assert not covering_verdict([PieceImage(unit_box(), tuple(img), True)], faces_from({}), (1, 1)).proven


def test_outside_test_needs_the_whole_piece_beyond_one():
    """불안정 성분 상이 [1,3]·[−3,−1]·[0.2,0.9] 인 조각의 안정 성분이 큼 - 실패(통째로 |w′|>1 일 때만 S(M)⁻ 안)"""
    for u in ((1, 3), (-3, -1), (Q(1, 5), Q(9, 10)), (Q(-9, 10), Q(-1, 5))):
        img = (iv(*u), iv(Q(-1, 2), Q(1, 2)), iv(-5, 5), iv(-5, 5))
        assert not covering_verdict([PieceImage(unit_box(), img, True)], faces_from({}), (1, 1)).proven


def test_every_piece_must_pass_not_just_one():
    """조각 여럿 중 하나만 사용 자격·면 조건을 못 넘음 - 실패(all 이 any 로 약해지면 거짓 증명)"""
    halves = [(iv(-1, 0), iv(-1, 1), iv(-1, 1), iv(-1, 1)), (iv(0, 1), iv(-1, 1), iv(-1, 1), iv(-1, 1))]
    good = lambda p, ok=True: PieceImage(p, INSIDE.image, ok)
    assert covering_verdict([good(halves[0]), good(halves[1])], faces_from({}), (1, 1)).proven
    assert not covering_verdict([good(halves[0]), good(halves[1], False)], faces_from({}), (1, 1)).proven
    face_halves = [(iv(1, 1), iv(-1, 0), iv(-1, 1), iv(-1, 1)), (iv(1, 1), iv(0, 1), iv(-1, 1), iv(-1, 1))]
    out_img = (iv(2, 3), iv(-1, 1), iv(0, 0), iv(0, 0))
    in_img = (iv(Q(1, 2), 3), iv(-1, 1), iv(0, 0), iv(0, 0))
    faces = faces_from({})
    faces[(0, 1)] = [PieceImage(face_halves[0], out_img, True), PieceImage(face_halves[1], in_img, True)]
    assert not covering_verdict([INSIDE], faces, (1, 1)).proven
    faces[(0, 1)] = [PieceImage(face_halves[0], out_img, True), PieceImage(face_halves[1], out_img, False)]
    assert not covering_verdict([INSIDE], faces, (1, 1)).proven


def test_signs_must_be_plus_or_minus_one():
    """부호 0·2 - ValueError(차수 계산의 선형 사상 diag(2s₀,2s₁) 이 정의되지 않음)"""
    for signs in ((1, 0), (2, 1), (0, 0)):
        with pytest.raises(ValueError):
            covering_verdict([INSIDE], faces_from({}), signs)


def test_cover_checks_on_small_boxes_and_face_axes():
    """폭 1 이하 상자, 면의 퇴화 축에서 두께를 가진 조각, 면 위에서 겹치는 조각, 한 조각만 차원이 다름 - 덮개 판단이 명세대로"""
    small = (iv(0, 1), iv(0, 1), iv(0, 1), iv(0, 1))
    assert exactly_covers(covering.bisect_box(small), small) and len(covering.bisect_box(small)) == 16
    assert not exactly_covers(covering.bisect_box(small)[1:], small)
    thick = (iv(Q(9, 10), 1), iv(-1, 1), iv(-1, 1), iv(-1, 1))
    assert not exactly_covers([thick], face_box(0, 1))
    a = (iv(1, 1), iv(-1, Q(1, 2)), iv(-1, 1), iv(-1, 1))
    b = (iv(1, 1), iv(0, Q(1, 2)), iv(-1, 1), iv(-1, 1))
    assert not exactly_covers([a, b], face_box(0, 1))
    shifted = (iv(Q(-3, 2), Q(1, 2)), iv(-1, 1), iv(-1, 1), iv(-1, 1))
    assert not exactly_covers([shifted], unit_box())
    assert not exactly_covers([unit_box(), (iv(-1, 1), iv(-1, 1), iv(-1, 1))], unit_box())


HALVES = [(iv(-1, 0), iv(-1, 1), iv(-1, 1), iv(-1, 1)), (iv(0, 1), iv(-1, 1), iv(-1, 1), iv(-1, 1))]


def test_certified_margin_is_the_smallest_slack():
    """여유 = 조각별 여유의 최솟값 - 손으로 셈한 값(안쪽 1/2, 비대칭 안정 상 1/4, 양·음 쪽 바깥 조각 1/4, 부호 −1 면 1/5)과 같음"""
    assert covering.certified_margin([INSIDE], faces_from({}), (1, 1)) == Q(1, 2)
    lopsided = PieceImage(unit_box(), (iv(-2, 2), iv(-2, 2), iv(Q(-1, 2), Q(3, 4)), iv(Q(-1, 2), Q(1, 2))), True)
    assert covering.certified_margin([lopsided], faces_from({}), (1, 1)) == Q(1, 4)
    for u0 in (iv(Q(5, 4), 3), iv(-3, Q(-5, 4))):
        out = PieceImage(HALVES[1], (u0, iv(-1, 1), iv(-5, 5), iv(-5, 5)), True)
        half = PieceImage(HALVES[0], INSIDE.image, True)
        assert covering.certified_margin([half, out], faces_from({}), (1, 1)) == Q(1, 4)
    flipped = faces_from({(0, 1): (iv(-3, Q(-6, 5)), iv(-1, 1), iv(0, 0), iv(0, 0)),
                          (0, -1): (iv(Q(7, 5), 3), iv(-1, 1), iv(0, 0), iv(0, 0))})
    assert covering_verdict([INSIDE], flipped, (-1, 1)).proven
    assert covering.certified_margin([INSIDE], flipped, (-1, 1)) == Q(1, 5)


def test_thresholds_move_the_decision_exactly_at_the_margin():
    """문턱을 여유만큼 조이면 실패, 조금 덜 조이면 증명 - 출구 면·바깥 조각(불안정 문턱)과 안쪽 조각(안정 문턱)에서 따로"""
    faces = faces_from({(0, 1): (iv(Q(13, 10), 3), iv(-1, 1), iv(0, 0), iv(0, 0))})
    assert covering.certified_margin([INSIDE], faces, (1, 1)) == Q(3, 10)
    assert not covering_verdict([INSIDE], faces, (1, 1), unstable_threshold=Q(13, 10)).proven
    assert covering_verdict([INSIDE], faces, (1, 1), unstable_threshold=Q(129, 100)).proven
    assert not covering_verdict([INSIDE], faces, (1, 1), stable_threshold=Q(1, 2)).proven
    assert covering_verdict([INSIDE], faces, (1, 1), stable_threshold=Q(51, 100)).proven
    out = [PieceImage(h, (iv(Q(5, 4), 3), iv(-1, 1), iv(-5, 5), iv(-5, 5)), True) for h in HALVES]
    assert not covering_verdict(out, faces_from({}), (1, 1), unstable_threshold=Q(5, 4)).proven
    assert covering_verdict(out, faces_from({}), (1, 1), unstable_threshold=Q(124, 100)).proven
    neg = [PieceImage(h, (iv(-3, Q(-5, 4)), iv(-1, 1), iv(-5, 5), iv(-5, 5)), True) for h in HALVES]
    assert not covering_verdict(neg, faces_from({}), (1, 1), unstable_threshold=Q(5, 4)).proven
    assert covering_verdict(neg, faces_from({}), (1, 1), unstable_threshold=Q(124, 100)).proven
