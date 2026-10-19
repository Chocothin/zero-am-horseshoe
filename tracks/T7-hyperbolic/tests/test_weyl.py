"""G8 독립 판정 명세 시험. 기대값은 Sylvester 판정법(피벗이 모두 양수 ⟺ 양정치), Weyl 부등식, 중점·반경 감싸기의
포함 성질에서 정했다."""
import random
from fractions import Fraction as F

import pytest

from t3cap.exact import Interval
from t7hb import weyl

J = (1, 1, -1, -1)


def point(m):
    return tuple(tuple(Interval(F(x), F(x)) for x in row) for row in m)


def diag(*d):
    return [[F(d[i]) if i == j else F(0) for j in range(4)] for i in range(4)]


def test_cholesky_matches_known_definiteness():
    """대각 양수·2×2 고유값 (3,1) 은 양정치, 영 피벗·고유값 (3,−1) 은 아니다."""
    assert weyl.cholesky_positive(diag(1, 2, 3, 4))
    assert not weyl.cholesky_positive(diag(1, 2, 0, 4))
    assert weyl.cholesky_positive([[F(2), F(1)], [F(1), F(2)]])
    assert not weyl.cholesky_positive([[F(1), F(2)], [F(2), F(1)]])


def test_cholesky_on_gram_matrices():
    """BᵀB + I/10 은 양정치, −BᵀB 는 양정치가 아니다(정의)."""
    rng = random.Random(5)
    for _ in range(50):
        b = [[F(rng.randint(-9, 9), rng.randint(1, 4)) for _ in range(4)] for _ in range(4)]
        g = [[sum(b[k][i] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
        assert weyl.cholesky_positive([[g[i][j] + (F(1, 10) if i == j else 0) for j in range(4)] for i in range(4)])
        assert not weyl.cholesky_positive([[-g[i][j] for j in range(4)] for i in range(4)])


def test_point_diagonal_range():
    """대각 점 행렬 diag(1.2,1.1,0.8,0.9) - V_μ 양정치 ⟺ 0.81 < μ < 1.21(경계에서는 피벗 0)."""
    d = point(diag(F("1.2"), F("1.1"), F("0.8"), F("0.9")))
    assert weyl.certifies(d, F(1)) and weyl.certifies(d, F("1.2")) and weyl.certifies(d, F("0.82"))
    assert not weyl.certifies(d, F("1.21")) and not weyl.certifies(d, F("0.81"))


def test_certified_interval_matrix_has_only_positive_definite_members():
    """판정이 참이면 감싸기 안 모든 실행렬 A 에서 AᵀJA − μJ 가 양정치다(Weyl 부등식)."""
    rng = random.Random(6)
    centre = diag(F("1.4"), F("1.25"), F("0.7"), F("0.75"))
    r = F(1, 40)
    d = tuple(tuple(Interval(centre[i][j] - r, centre[i][j] + r) for j in range(4)) for i in range(4))
    assert weyl.certifies(d, F(1))
    for _ in range(100):
        a = [[d[i][j].lo + (d[i][j].hi - d[i][j].lo) * F(rng.randint(0, 32), 32) for j in range(4)] for i in range(4)]
        v = [[sum(J[k] * a[k][i] * a[k][j] for k in range(4)) - (J[i] if i == j else 0) for j in range(4)]
             for i in range(4)]
        assert weyl.cholesky_positive(v)


def test_wide_enclosure_is_not_certified():
    """대각 반경 0.3 이면 감싸기 안에 |d_1|<1 인 행렬이 있어 양정치가 아니므로 판정은 거짓이어야 한다."""
    centre = diag(F("1.2"), F("1.2"), F("0.8"), F("0.8"))
    d = tuple(tuple(Interval(centre[i][j] - (F("0.3") if i == j else 0), centre[i][j] + (F("0.3") if i == j else 0))
                    for j in range(4)) for i in range(4))
    assert not weyl.certifies(d, F(1))


def test_one_wide_entry_blocks_certification():
    """D_00∈[1.0,1.4] 이면 D_00=1 인 원소에서 (V_1)_00 = 0 이라 엄격 양정치가 아니므로 판정은 거짓이어야 한다."""
    centre = diag(F("1.2"), F("1.2"), F("0.8"), F("0.8"))
    d = tuple(tuple(Interval(centre[i][j] - (F("0.2") if (i, j) == (0, 0) else 0),
                             centre[i][j] + (F("0.2") if (i, j) == (0, 0) else 0)) for j in range(4)) for i in range(4))
    assert not weyl.certifies(d, F(1))


def test_quadratic_radius_term_is_kept():
    """D_c=diag(1.5,1.5,0.5,0.5), D_20∈[−1.2,1.2] - 원소 D_20=1.2 에서 V_1 이 양정치가 아니므로(2×2 소행렬식 < 0) 거짓이어야 한다.
    중점이 0 인 성분에서는 반경 제곱항 D_rᵀD_r 만이 이 위험을 담는다."""
    centre = diag(F("1.5"), F("1.5"), F("0.5"), F("0.5"))
    member = [row[:] for row in centre]
    member[2][0] = F("1.2")
    v = [[sum(J[k] * member[k][i] * member[k][j] for k in range(4)) - (J[i] if i == j else 0) for j in range(4)]
         for i in range(4)]
    assert not weyl.cholesky_positive(v)
    d = tuple(tuple(Interval(centre[i][j] - (F("1.2") if (i, j) == (2, 0) else 0),
                             centre[i][j] + (F("1.2") if (i, j) == (2, 0) else 0)) for j in range(4)) for i in range(4))
    assert not weyl.certifies(d, F(1))


def test_shape_guard():
    """4×4 가 아닌 감싸기는 거부한다(치명 가드)."""
    with pytest.raises(ValueError):
        weyl.certifies(point([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), F(1))
    with pytest.raises(ValueError):
        weyl.certifies(point([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1]]), F(1))
