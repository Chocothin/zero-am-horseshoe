"""원뿔 판정층 명세 시험. 기대값은 정의(V_μ=AᵀJA−μJ 의 양정치, J=diag(1,1,−1,−1))와 Gershgorin 원판 정리,
구간 산술의 포함 성질에서 정했다. 구현을 보고 기대값을 정하지 않았다."""
import random
from fractions import Fraction as F

import pytest

from t3cap.exact import Interval
from t7hb import cone

J = (1, 1, -1, -1)


def point(m):
    return tuple(tuple(Interval(F(x), F(x)) for x in row) for row in m)


def diag(*d):
    return [[d[i] if i == j else 0 for j in range(4)] for i in range(4)]


def exact_v(a, mu=F(1)):
    """실행렬 A 의 AᵀJA − μJ(정확 유리수)."""
    return [[sum(J[k] * F(a[k][i]) * F(a[k][j]) for k in range(4)) - (mu * J[i] if i == j else 0) for j in range(4)]
            for i in range(4)]


def positive_definite(m):
    """선행 주소행렬식이 모두 양수 ⟺ 양정치(Sylvester), 정확 유리수 소거."""
    a = [list(map(F, row)) for row in m]
    n = len(a)
    for c in range(n):
        if a[c][c] <= 0:
            return False
        for r in range(c + 1, n):
            f = a[r][c] / a[c][c]
            a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return True


def test_square_encloses_all_squares_and_is_tight():
    """0 을 품는 구간·양·음 구간 - 끝점 제곱으로 정확히 감싼다(구간 함수 {a²} 의 상)."""
    assert cone.square(Interval(F(-2), F(3))) == Interval(F(0), F(9))
    assert cone.square(Interval(F(2), F(3))) == Interval(F(4), F(9))
    assert cone.square(Interval(F(-3), F(-2))) == Interval(F(4), F(9))
    rng = random.Random(1)
    for _ in range(200):
        lo, hi = sorted(F(rng.randint(-50, 50), rng.randint(1, 9)) for _ in range(2))
        iv = cone.square(Interval(lo, hi))
        t = lo + (hi - lo) * F(rng.randint(0, 100), 100)
        assert iv.lo <= t * t <= iv.hi


def test_gram_of_point_matrix_equals_exact_product():
    """점 구간 행렬 - S 는 AᵀJA 그 자체(정의)."""
    rng = random.Random(2)
    a = [[F(rng.randint(-9, 9), rng.randint(1, 5)) for _ in range(4)] for _ in range(4)]
    s = cone.gram(point(a))
    v = exact_v(a, F(0))
    assert len(s) == 4 and all(len(row) == 4 for row in s)
    assert all(s[i][j].lo == s[i][j].hi == v[i][j] for i in range(4) for j in range(4))


def test_diagonal_matrix_range_is_exact_squares():
    """대각 A - V_μ 는 대각(d_u²−μ, μ−d_s²)이라 양정치 μ 구간은 정확히 (max d_s², min d_u²) 다."""
    b = cone.cone_bounds(point(diag(F("1.2"), F("1.1"), F("0.8"), F("0.9"))))
    assert b.lower == F("0.81") and b.upper == F("1.21")
    assert b.margin(F(1)) == F("0.19")
    v = cone.cone_verdict(point(diag(F("1.2"), F("1.1"), F("0.8"), F("0.9"))))
    assert v.proven and v.margin == F("0.19")


def test_off_diagonal_intervals_match_hand_computed_bounds():
    """D=diag(2,2,1/2,1/2) 에 D_01∈[−1/10,1/10], D_23∈[−1/10,1/10] 을 넣으면 정의대로 손으로 푼 S=DᵀJD 는
    S_00=4, S_01=2·D_01=[−1/5,1/5], S_11=D_01²+4=[4,401/100], S_22=−1/4, S_23=−D_23/2=[−1/20,1/20], S_33=−D_23²−1/4=[−26/100,−1/4],
    나머지 0 이다. 그래서 R=(1/5,1/5,1/20,1/20), b*=min(4−1/5, 4−1/5)=19/5, a*=max(1/20+1/4, 1/20+26/100)=31/100,
    g(1)=min(b*−1, 1−a*)=69/100 이다(Gershgorin 행 여유의 정의)."""
    tenth = Interval(F(-1, 10), F(1, 10))
    d = [[Interval(F(x), F(x)) for x in row] for row in diag(2, 2, F(1, 2), F(1, 2))]
    d[0][1], d[2][3] = tenth, tenth
    b = cone.cone_bounds(tuple(tuple(row) for row in d))
    assert b.diag_lower == (F(4), F(4), F(-1, 4), F(-26, 100))
    assert b.radii == (F(1, 5), F(1, 5), F(1, 20), F(1, 20))
    assert b.upper == F(19, 5) and b.lower == F(31, 100) and b.margin(F(1)) == F(69, 100)


def test_weak_unstable_contraction_fails_at_mu_one():
    """불안정 성분이 줄면(|d_1|<1) V_1 대각 하나가 음수 - 양정치가 아니다(Rayleigh 몫 e_1)."""
    v = cone.cone_verdict(point(diag(F("1.2"), F("0.9"), F("0.8"), F("0.9"))))
    assert not v.proven and v.margin < 0


def test_unstable_to_stable_coupling_breaks_positivity():
    """불안정 열에 큰 안정 성분이 섞이면 (V_1)_00 = 1.44 − 1 − 1 < 0 - 양정치가 아니다."""
    a = diag(F("1.2"), F("1.2"), F("0.5"), F("0.5"))
    a[2][0] = F(1)
    assert not positive_definite(exact_v(a))
    assert not cone.cone_verdict(point(a)).proven


def test_gershgorin_margin_lower_bounds_every_member():
    """구간 행렬이 인증되면 그 안의 모든 실행렬에서 λ_min(AᵀJA−J) ≥ 여유(Gershgorin 원판 정리)."""
    rng = random.Random(3)
    centre = diag(F("1.5"), F("1.3"), F("0.7"), F("0.6"))
    width = F(1, 50)
    d = tuple(tuple(Interval(F(centre[i][j]) - width, F(centre[i][j]) + width) for j in range(4)) for i in range(4))
    v = cone.cone_verdict(d)
    assert v.proven
    for _ in range(100):
        a = [[d[i][j].lo + (d[i][j].hi - d[i][j].lo) * F(rng.randint(0, 64), 64) for j in range(4)] for i in range(4)]
        m = exact_v(a)
        shifted = [[m[i][j] - (v.margin * F(999, 1000) if i == j else 0) for j in range(4)] for i in range(4)]
        assert positive_definite(shifted)


def test_threshold_is_strict():
    """문턱과 여유가 같으면 인증하지 않는다(정리는 엄격 양정치를 요구)."""
    d = point(diag(F("1.2"), F("1.1"), F("0.8"), F("0.9")))
    assert not cone.cone_verdict(d, threshold=F("0.19")).proven
    assert cone.cone_verdict(d, threshold=F("0.189")).proven


def test_mu_enters_rows_with_signature_sign():
    """대각 A - 여유는 min(min_u d_u² − μ, μ − max_s d_s²) 라서 μ 가 (0.81, 1.21) 밖이면 음수다(V_μ 의 대각)."""
    b = cone.cone_bounds(point(diag(F("1.2"), F("1.1"), F("0.8"), F("0.9"))))
    assert b.margin(F("1.2")) == F("0.01") and b.margin(F("1.22")) == F("-0.01")
    assert b.margin(F("0.82")) == F("0.01") and b.margin(F("0.80")) == F("-0.01")


def test_shape_guard():
    """4×4 가 아닌 감싸기는 판정 전에 거부한다(치명 가드: 실패를 삼키지 않음)."""
    with pytest.raises(ValueError):
        cone.cone_verdict(point([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))


def test_ragged_rows_are_rejected():
    """행이 4개라도 한 행의 길이가 다르면 거부한다(색인 오류로 새지 않게)."""
    with pytest.raises(ValueError):
        cone.cone_verdict(point([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1]]))
