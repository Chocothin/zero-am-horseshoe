"""G5 기준 적분기(t11/g5ref.py) 시험. 기대값은 Lagrange 정삼각 회전 정확해, 에너지 보존, 단면·차트의 정의, 유한 차분에서 정했다.
기준 적분기는 감사 사본과 독립이어야 하므로 이 시험도 사본을 import 하지 않는다."""
from __future__ import annotations

import sys
from fractions import Fraction as F
from pathlib import Path

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from t11 import design as dz  # noqa: E402
from t11 import g5ref as G  # noqa: E402


def jacobi_from_cartesian(q, v):
    r1 = (q[2] - q[0], q[3] - q[1])
    r2 = (q[4] - (q[0] + q[2]) / 2, q[5] - (q[1] + q[3]) / 2)
    v1 = (v[2] - v[0], v[3] - v[1])
    v2 = (v[4] - (v[0] + v[2]) / 2, v[5] - (v[1] + v[3]) / 2)
    return [*r1, *r2, *v1, *v2]


def lagrange(t):
    a = mp.mpf(3) / 2
    R = a / mp.sqrt(3)
    w = mp.sqrt(3 / a ** 3)
    th = [mp.mpf(1) / 10 + 2 * mp.pi * k / 3 + w * t for k in range(3)]
    q = [c for k in range(3) for c in (R * mp.cos(th[k]), R * mp.sin(th[k]))]
    v = [c for k in range(3) for c in (-w * R * mp.sin(th[k]), w * R * mp.cos(th[k]))]
    return jacobi_from_cartesian(q, v)


def energy(y):
    U = sum(1 / mp.sqrt(dx * dx + dy * dy) for dx, dy in (
        (y[0], y[1]), (y[2] + y[0] / 2, y[3] + y[1] / 2), (y[2] - y[0] / 2, y[3] - y[1] / 2)))
    return (y[4] ** 2 + y[5] ** 2) / 4 + (y[6] ** 2 + y[7] ** 2) / 3 - U


def test_taylor_matches_lagrange_rotation_in_jacobi_coordinates():
    """정삼각 회전해 - Jacobi 좌표 Taylor 32차 다항식(s=1/10)이 정확해와 1e-35 안에서 같다(운동 방정식 유도 확인)."""
    mp.mp.dps = 45
    Y, _, _, _ = G.taylor(lagrange(0), [], 32)
    s = mp.mpf(1) / 10
    exact = lagrange(s)
    assert max(abs(G.horner(Y[c], s) - exact[c]) for c in range(8)) < mp.mpf(10) ** -35


def test_lift_is_on_section_at_energy_and_chart_roundtrips():
    """들어올림 - A=ρ₁×ρ₂=0, Ȧ>0(Σ⁺), 에너지 −1/2, J=0 이고 차트로 되돌리면 u 다(단면·차트 정의)."""
    mp.mp.dps = 45
    c, _ = dz.hset("M+0")
    u = [G.frac_to_mpf(x) for x in c]
    y = [s.v for s in G.lift([G.Dual(x, []) for x in u])]
    assert y[1] == 0 and y[3] == 0
    adot = y[4] * y[3] - y[5] * y[2] + y[0] * y[7] - y[1] * y[6]
    assert adot > 0
    assert abs(energy(y) + mp.mpf(1) / 2) < mp.mpf(10) ** -40
    J = y[0] * y[5] / 2 - y[1] * y[4] / 2 + (y[2] * y[7] - y[3] * y[6]) * 2 / 3
    assert abs(J) < mp.mpf(10) ** -40
    assert max(abs(a - b) for a, b in zip(G.chart(y), u)) < mp.mpf(10) ** -40


def test_flight_conserves_energy_and_tangent_matches_finite_difference():
    """N1⟹N2 중심 비행 - 도착 에너지 −1/2(1e-30 안), 접벡터로 낸 dP̃/dw₀ 가 중심차분(보폭 1e-6)과 1e-25 안에서 같다."""
    mp.mp.dps = 45
    c, M = dz.hset("N1")
    u0 = [G.frac_to_mpf(x) for x in c]
    cols = [[G.frac_to_mpf(M[i][j]) for j in range(4)] for i in range(4)]
    fl = G.fly(u0, cols)
    assert fl.adot_tau < 0 and fl.min_A_before > 0
    assert abs(energy(fl.y_out) + mp.mpf(1) / 2) < mp.mpf(10) ** -30
    eps = mp.mpf(10) ** -6
    plus = G.fly([u0[i] + eps * cols[i][0] for i in range(4)], cols, tangents=False)
    minus = G.fly([u0[i] - eps * cols[i][0] for i in range(4)], cols, tangents=False)
    for i in range(4):
        fd = (plus.u_out[i] - minus.u_out[i]) / (2 * eps)
        assert abs(fd - fl.D_chart[i][0]) <= mp.mpf(10) ** -25 * (1 + abs(fl.D_chart[i][0]))


def test_exact_conversion_keeps_sign():
    """mpf → 유리수 변환 - 부호와 값이 정확하다(가수 부호를 버리는 man_exp 함정)."""
    mp.mp.dps = 30
    assert G.mpf_to_frac(mp.mpf(-3) / 8) == F(-3, 8)
    assert G.mpf_to_frac(mp.mpf(0)) == 0
