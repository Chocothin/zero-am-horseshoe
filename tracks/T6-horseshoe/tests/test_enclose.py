"""평균값 형식 조립·조각 이분 명세 시험. 기대값은 평균값 정리 f(w)−f(c)=∫Df·(w−c) 와 「Df(조각)⊂D」 가정에서 정했다."""
from fractions import Fraction as Q

from t3cap.exact import Interval

from t6hs.covering import bisect_box, exactly_covers, face_box, unit_box
from t6hs.enclose import hset_from_design, mean_value_image


def iv(lo, hi):
    return Interval(Q(lo), Q(hi))


def pt(x):
    return Interval(Q(x), Q(x))


ZERO = pt(0)


def test_linear_map_image_is_exact():
    """선형 사상(점 행렬 D) - 평균값 형식 상이 정확한 선형 상과 같음(구간 산술 낭비 없음)"""
    d = [[pt(2), ZERO, ZERO, ZERO], [ZERO, pt(-3), ZERO, ZERO], [ZERO, ZERO, pt(Q(1, 2)), ZERO], [ZERO, ZERO, ZERO, pt(0)]]
    img = mean_value_image((pt(1), pt(0), pt(0), pt(5)), d, unit_box(), (Q(0),) * 4)
    assert img == (iv(-1, 3), iv(-3, 3), iv(Q(-1, 2), Q(1, 2)), iv(5, 5))


def test_quadratic_component_is_enclosed():
    """f0=w0² 를 도함수 범위 [−2,2] 로 감쌈 - 표본 점의 참값이 모두 상 안(평균값 정리)"""
    d = [[iv(-2, 2), ZERO, ZERO, ZERO]] + [[ZERO] * 4 for _ in range(3)]
    img = mean_value_image((pt(0), pt(0), pt(0), pt(0)), d, unit_box(), (Q(0),) * 4)
    for k in range(-10, 11):
        assert img[0].contains(Q(k, 10) ** 2)


def test_off_centre_piece_uses_its_own_centre():
    """중심이 0 이 아닌 조각 [1/2,1] - 편차 (w−c) 를 그 조각 중심 3/4 에서 잰다"""
    piece = (iv(Q(1, 2), 1), iv(-1, 1), iv(-1, 1), iv(-1, 1))
    d = [[pt(4), ZERO, ZERO, ZERO]] + [[ZERO] * 4 for _ in range(3)]
    img = mean_value_image((pt(3), pt(0), pt(0), pt(0)), d, piece, (Q(3, 4), Q(0), Q(0), Q(0)))
    assert img[0] == iv(2, 4)


def test_degenerate_face_axis_does_not_contribute():
    """출구 면 조각(w1=1 고정) - 그 축의 도함수 열은 상 폭에 기여하지 않음"""
    d = [[ZERO, iv(-100, 100), ZERO, ZERO]] + [[ZERO] * 4 for _ in range(3)]
    face = face_box(1, 1)
    img = mean_value_image((pt(7), pt(0), pt(0), pt(0)), d, face, (Q(0), Q(1), Q(0), Q(0)))
    assert img[0] == pt(7)


def test_bisection_keeps_the_union_and_respects_face_axes():
    """상자 이분 16조각과 면 이분 8조각 - 합집합이 원래 상자·면과 정확히 같음"""
    assert len(bisect_box(unit_box())) == 16 and exactly_covers(bisect_box(unit_box()), unit_box())
    face = face_box(0, -1)
    halves = bisect_box(face)
    assert len(halves) == 8 and exactly_covers(halves, face)


def test_face_piece_enclosed_from_the_box_centre():
    """단일 중심 방식: 면 w0=+1 을 상자 중심 0 에서 감쌈 - 편차가 (1,·,·,·) 라 상이 P(0)+D_00 쪽으로 감(부호가 뒤집히면 틀림)"""
    d = [[pt(2), ZERO, ZERO, ZERO]] + [[ZERO] * 4 for _ in range(3)]
    img = mean_value_image((pt(0), pt(0), pt(0), pt(0)), d, face_box(0, 1), (Q(0),) * 4)
    assert img[0] == iv(2, 2)


def test_design_vertex_becomes_frame_times_radii():
    """설계 정점(중심·단위 방향 16진, 반경) - 행렬 열 j 가 방향 열 j 에 반경 r_j 를 곱한 정확 유리수(행 배율이 아님)"""
    frame = [[1.0, 0.5, 0.0, 0.0], [0.0, 1.0, 0.0, 0.25], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    vertex = {"center_hex": [x.hex() for x in (2.5, -0.75, 0.0, 1.0)],
              "frame_hex": [[x.hex() for x in row] for row in frame], "radii": [0.5, 0.25, 2.0, 4.0]}
    h = hset_from_design("N0", vertex)
    assert h.center == (Q(5, 2), Q(-3, 4), Q(0), Q(1))
    assert h.matrix[0] == (Q(1, 2), Q(1, 8), Q(0), Q(0)) and h.matrix[1] == (Q(0), Q(1, 4), Q(0), Q(1))
