"""엔진의 축소 벡터장·에너지·거리·차트 감싸기를 명세(표준 8차원 Jacobi 계와 축소 좌표의 정의)에서 대조한다.

축소 좌표 y_k(X) 는 8차원 상태 X 의 함수이므로, 표준 흐름을 따라 잰 시간 미분 d/dt y_k(X(t)) 가
엔진 벡터장의 감싸기 안에 있어야 한다. 기준 값은 엔진과 독립인 mpmath 구현으로 만든다.
"""
from __future__ import annotations

import random
from fractions import Fraction

import mpmath as mp
import pytest

from reference import mp_reference as ref
from t3cap import CoordinateSection, Failure, Interval, Settings, SystemParams, chart, vector_field
from t3cap.exact import point_box


def mp_value(q: Fraction) -> mp.mpf:
    return mp.mpf(q.numerator) / q.denominator


def frac(x) -> Fraction:
    sign, man, exp, _ = mp.mpf(x)._mpf_
    v = Fraction(int(man)) * Fraction(2) ** int(exp)
    return -v if sign else v


def random_reduced_state(rng: random.Random) -> list[Fraction]:
    return [Fraction(rng.randint(30, 60), 10), Fraction(rng.randint(-20, 20), 10), Fraction(rng.randint(-10, 10), 10),
            Fraction(rng.randint(-8, 8), 10), Fraction(rng.randint(-8, 8), 10), Fraction(rng.randint(-8, 8), 10)]


@pytest.mark.parametrize("arithmetic", ["double", "mp:128"])
def test_reduced_vector_field_is_the_pushforward_of_the_standard_jacobi_flow(arithmetic):
    """MP 경로는 끝점을 방향 반올림한 10진으로 내보내므로 그 입출력까지 함께 대조된다."""
    mp.mp.dps = 50
    rng = random.Random(11)
    eps = mp.mpf("1e-20")
    for masses, J in (((1, 1, 1), Fraction(0)), ((1, 2, Fraction(3, 2)), Fraction(1, 7)),
                      ((Fraction(9, 10), Fraction(11, 10), 1), Fraction(-2, 5))):
        md = ref.MassData.of(masses)
        system = SystemParams.make(masses=masses, energy=Fraction(-1, 2), angular_momentum=J)
        states = [random_reduced_state(rng) for _ in range(5)]
        enclosures = vector_field([point_box(s) for s in states], system=system,
                                  settings=Settings(arithmetic=arithmetic))
        for y, enc in zip(states, enclosures):
            assert not isinstance(enc, Failure), enc
            X = ref.reduced_to_state([mp_value(v) for v in y], mp_value(J))
            F = [c[1] for c in ref.taylor_coefficients(X, md, 2)]
            ahead = [X[i] + eps * F[i] for i in range(8)]
            behind = [X[i] - eps * F[i] for i in range(8)]
            for k in range(6):
                rate = (ref.reduced_coordinate(ahead, k) - ref.reduced_coordinate(behind, k)) / (2 * eps)
                iv = enc.value[k]
                # 중심차분 오차 ~eps^2 는 감싸기 폭(~1e-15)보다 훨씬 작다
                assert iv.lo - Fraction(1, 10**30) <= frac(rate) <= iv.hi + Fraction(1, 10**30), (masses, y, k)
            assert enc.energy.contains(frac(ref.jacobi_energy(X, md)))
            r12, r13, r23 = ref.distances(X, md)
            assert enc.r12.contains(frac(r12)) and enc.r13.contains(frac(r13)) and enc.r23.contains(frac(r23))


def test_chart_hull_contains_independent_chart_and_energy_level():
    mp.mp.dps = 40
    md = ref.MassData.of((1, 1, 1))
    system = SystemParams.make()
    for section in (CoordinateSection.syzygy(+1), CoordinateSection.syzygy(-1),
                    CoordinateSection(3, Fraction(0), 1, 5, 1)):
        u = [Fraction(5), Fraction(3, 10), Fraction(1, 10), Fraction(2, 5)]
        res = chart([point_box(u)], system=system, section=section)[0]
        assert not isinstance(res, Failure), res
        st = ref.chart_state(u, coord=section.coord, value=section.value, solved=section.solved_momentum,
                             branch=section.branch, free=section.free_coords, md=md, h=Fraction(-1, 2), J=0)
        for k in range(6):
            assert res.hull[k].contains(frac(ref.reduced_coordinate(st, k))), (section, k)
        assert res.energy.contains(Fraction(-1, 2))
        assert res.discriminant.lo > 0


def test_chart_refuses_points_off_the_energy_surface():
    """함정 ⑤: 에너지가 모자라 근이 없는 점(판별식 <= 0)은 감싸기 대신 실패로 끝나야 한다."""
    system = SystemParams.make()
    u = [Fraction(5), Fraction(0), Fraction(3), Fraction(3)]  # 운동에너지만으로 h=-1/2 를 넘는다
    res = chart([point_box(u)], system=system, section=CoordinateSection.syzygy(+1))[0]
    assert isinstance(res, Failure) and "discriminant" in res.reason


def test_chart_derivative_contains_finite_difference_of_independent_chart():
    mp.mp.dps = 40
    md = ref.MassData.of((1, 1, 1))
    section = CoordinateSection.syzygy(+1)
    u = [Fraction(5), Fraction(3, 10), Fraction(1, 10), Fraction(2, 5)]
    box = tuple(Interval(q - Fraction(1, 10**6), q + Fraction(1, 10**6)) for q in u)
    res = chart([box], system=SystemParams.make(), section=section)[0]
    assert not isinstance(res, Failure)
    eps = Fraction(1, 10**12)
    for j in range(4):
        up = [u[i] + (eps if i == j else 0) for i in range(4)]
        dn = [u[i] - (eps if i == j else 0) for i in range(4)]
        kw = dict(coord=2, value=Fraction(0), solved=5, branch=1, free=(0, 1, 3, 4), md=md, h=Fraction(-1, 2), J=0)
        py_up = ref.reduced_coordinate(ref.chart_state(up, **kw), 5)
        py_dn = ref.reduced_coordinate(ref.chart_state(dn, **kw), 5)
        assert res.dpsi[5][j].contains(frac((py_up - py_dn) / (2 * mp_value(eps))))


def test_rotation_angle_enclosure_contains_reference_direction_of_rho1():
    """함정 ⑦: rotation=True 의 theta 는 반환 시점 rho1 방향각(출발 게이지 theta=0)을 감싸야 한다."""
    from t3cap import Frame, poincare_c0

    mp.mp.dps = 32
    md = ref.MassData.of((1, 1, 1))
    u = (Fraction(5148567982527431, 10**15), Fraction(0), Fraction(0), Fraction(432690650760826, 10**15))
    res = poincare_c0([point_box((0, 0, 0, 0))], system=SystemParams.make(), section_in=CoordinateSection.syzygy(+1),
                      section_out=CoordinateSection.syzygy(-1), frame_in=Frame.make(u, ((1, 0, 0, 0), (0, 1, 0, 0),
                                                                                       (0, 0, 1, 0), (0, 0, 0, 1))),
                      rotation=True)[0]
    assert not isinstance(res, Failure), res
    st = ref.chart_state(list(u), coord=2, value=Fraction(0), solved=5, branch=1, free=(0, 1, 3, 4), md=md,
                         h=Fraction(-1, 2), J=0)
    crossing = ref.flow_to_section(st, md, coord=2, value=Fraction(0), direction=-1, returns=1)
    assert res.theta is not None and res.theta.contains(frac(mp.atan2(crossing.state[1], crossing.state[0])))
