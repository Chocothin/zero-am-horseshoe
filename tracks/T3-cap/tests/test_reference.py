"""기준 해(mpmath 표준 Jacobi 적분기)의 명세 시험: 정확해 재현, 관성 뉴턴 힘과의 일치, 정밀도 독립성."""
from __future__ import annotations

import random
from fractions import Fraction

import mpmath as mp

from reference import mp_reference as ref


def inertial_forces(X, masses):
    m = [ref.to_mpf(Fraction(v)) for v in masses]
    md = ref.MassData.of(masses)
    rho1, rho2 = (X[0], X[1]), (X[2], X[3])
    # rho1 = q2-q1, rho2 = q3-(m1 q1+m2 q2)/(m1+m2), 질량중심 0
    M = sum(m)
    q3 = [rho2[k] * (m[0] + m[1]) / M for k in range(2)]
    c12 = [-rho2[k] * m[2] / M for k in range(2)]
    q1 = [c12[k] - md.a * rho1[k] for k in range(2)]
    q2 = [c12[k] + md.b * rho1[k] for k in range(2)]
    qs = [q1, q2, q3]
    acc = []
    for i in range(3):
        a = [mp.mpf(0), mp.mpf(0)]
        for j in range(3):
            if i != j:
                d = [qs[j][k] - qs[i][k] for k in range(2)]
                r3 = mp.sqrt(d[0] ** 2 + d[1] ** 2) ** 3
                a = [a[k] + m[j] * d[k] / r3 for k in range(2)]
        acc.append(a)
    pi1 = [md.mu1 * (acc[1][k] - acc[0][k]) for k in range(2)]
    pi2 = [md.mu2 * (acc[2][k] - (m[0] * acc[0][k] + m[1] * acc[1][k]) / (m[0] + m[1])) for k in range(2)]
    return pi1 + pi2


def test_jacobi_forces_equal_inertial_newton_forces_for_unequal_masses():
    mp.mp.dps = 40
    rng = random.Random(3)
    for masses in ((1, 1, 1), (1, 2, Fraction(3, 2)), (Fraction(7, 10), 1, Fraction(13, 10))):
        md = ref.MassData.of(masses)
        for _ in range(4):
            X = [mp.mpf(rng.uniform(-2, 2)) for _ in range(8)]
            rate = [c[1] for c in ref.taylor_coefficients(X, md, 2)]
            expected = inertial_forces(X, masses)
            assert max(abs(rate[4 + k] - expected[k]) for k in range(4)) < mp.mpf("1e-35")


def test_lagrange_rotating_equilateral_solution_returns_after_one_revolution():
    """정삼각형 상대평형(각속도^2 = 3/s^3, 단위 질량)은 뉴턴 방정식의 정확해다."""
    mp.mp.dps = 40
    md = ref.MassData.of((1, 1, 1))
    s = mp.mpf(2)
    w = mp.sqrt(3 / s ** 3)
    R = s / mp.sqrt(3)
    q = [(R * mp.cos(2 * mp.pi * k / 3), R * mp.sin(2 * mp.pi * k / 3)) for k in range(3)]
    v = [(-w * p[1], w * p[0]) for p in q]
    rho1 = [q[1][k] - q[0][k] for k in range(2)]
    rho2 = [q[2][k] - (q[0][k] + q[1][k]) / 2 for k in range(2)]
    pi1 = [(v[1][k] - v[0][k]) / 2 for k in range(2)]
    pi2 = [mp.mpf(2) / 3 * (v[2][k] - (v[0][k] + v[1][k]) / 2) for k in range(2)]
    X0 = rho1 + rho2 + pi1 + pi2
    T = 2 * mp.pi / w
    x, t, tol = list(X0), mp.mpf(0), mp.mpf(10) ** -38
    while True:
        c = ref.taylor_coefficients(x, md, 30)
        h = ref.step_size(c, tol)
        if t + h >= T:
            x = ref.evaluate(c, T - t)
            break
        x, t = ref.evaluate(c, h), t + h
    assert max(abs(a - b) for a, b in zip(x, X0)) < mp.mpf("1e-30")


def test_mass_constants_follow_the_precision_in_use_not_construction_time():
    """회귀: 15자리에서 만든 MassData 를 40자리에서 써도 mu2=2/3 이 40자리로 정확해야 한다(REPORT 정정 기록 1)."""
    mp.mp.dps = 15
    md = ref.MassData.of((1, 1, 1))
    mp.mp.dps = 40
    assert abs(md.mu2 * 3 - 2) < mp.mpf("1e-38")


def test_chart_state_lies_on_energy_level_and_section():
    mp.mp.dps = 40
    md = ref.MassData.of((1, 1, 1))
    u = [Fraction(5), Fraction(1, 10), Fraction(-1, 20), Fraction(2, 5)]
    st = ref.chart_state(u, coord=2, value=Fraction(0), solved=5, branch=1, free=(0, 1, 3, 4), md=md,
                         h=Fraction(-1, 2), J=Fraction(1, 100))
    assert abs(ref.jacobi_energy(st, md) + mp.mpf(1) / 2) < mp.mpf("1e-35")
    assert abs(ref.angular_momentum(st) - mp.mpf(1) / 100) < mp.mpf("1e-35")
    assert ref.reduced_coordinate(st, 2) == 0
