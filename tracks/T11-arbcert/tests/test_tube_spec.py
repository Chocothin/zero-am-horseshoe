"""흐름 관 인증기(감사 V2 사본 rigorous_tube.py) 명세 시험(G9). 기대값은 Lagrange 정삼각 회전 정확해, 에너지·각운동량·
총운동량 보존과 단면 정의(y=0, J=0), 다항식 부호의 정의, Picard–Lindelöf 사전 감싸기 조건에서 정했다."""
from __future__ import annotations

import sys
from fractions import Fraction as F
from pathlib import Path

import mpmath
from flint import arb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rigorous_tube as rt  # noqa: E402
from t11 import design as dz  # noqa: E402


def lagrange_state():
    a = arb(3) / 2
    R = a / arb(3).sqrt()
    w = (3 / a ** 3).sqrt()
    th = [arb(1) / 10 + 2 * arb.pi() * k / 3 for k in range(3)]
    q = [c for k in range(3) for c in (R * th[k].cos(), R * th[k].sin())]
    v = [c for k in range(3) for c in (-w * R * th[k].sin(), w * R * th[k].cos())]
    return q, v, R, w, th


def test_taylor_series_matches_lagrange_rotation():
    """Lagrange 정삼각 회전해(한 변 3/2, ω²=3/a³) - 30차 Taylor 다항식(s=1/10)이 정확해와 겹치고, 1e-20 옮긴 값과는 서로소."""
    q, v, R, w, th = lagrange_state()
    Q, V, S, A = rt.taylor(q, v, 30, rt.ArbOps)
    s = arb(1) / 10
    tiny = arb(0, 1e-30)
    for k in range(3):
        ph = w * s + th[k]
        exact = (R * ph.cos(), R * ph.sin())
        for c in range(2):
            got = rt.horner(Q[2 * k + c], s) + tiny
            assert got.overlaps(exact[c])
            assert not got.overlaps(exact[c] + arb("1e-20"))


def test_lagrange_pair_distance_series_is_constant():
    """정삼각 회전해 - 쌍 거리 제곱 S 의 1차 이상 계수가 모두 0 을 담는다(거리 불변)."""
    q, v, *_ = lagrange_state()
    _, _, S, _ = rt.taylor(q, v, 12, rt.ArbOps)
    for p in range(3):
        assert all(abs(S[p][k]) < arb("1e-30") for k in range(1, 13))


def test_initial_state_lies_on_section_with_energy_and_zero_momenta():
    """h-set N1 상자의 들어올림 - y=0 이라 A₀ 가 정확히 0, 에너지 −1/2·J=0·총운동량 0 을 감싸고 판별식이 양수(함정 ⑤⑦)."""
    c, M = dz.hset("N1")
    q, v, disc = rt.initial_state(c, M)
    assert disc > 0
    qb, vb = [x.ball() for x in q], [x.ball() for x in v]
    for i in (1, 3, 5):
        assert qb[i].is_zero()
    kin = sum((x * x for x in vb), arb(0)) / 2
    pot = sum((1 / ((qb[2 * j] - qb[2 * i]) ** 2 + (qb[2 * j + 1] - qb[2 * i + 1]) ** 2).sqrt()
               for i, j in rt.PAIRS), arb(0))
    assert (kin - pot).contains(arb(-1) / 2)
    J = sum((qb[2 * i] * vb[2 * i + 1] - qb[2 * i + 1] * vb[2 * i] for i in range(3)), arb(0))
    assert J.contains(0) and (vb[0] + vb[2] + vb[4]).contains(0) and (vb[1] + vb[3] + vb[5]).contains(0)


def test_classify_reports_sign_only_when_certain():
    """A(s)=s−1/2 - [0,1/4] 에서 −1, [3/4,1] 에서 +1, 영점을 품은 [1/4,3/4] 에서 0(미정)."""
    A = [rt.Aff.const(arb(-1) / 2), rt.Aff.const(arb(1))]
    assert rt.classify(A, rt.ZERO, F(0), F(1, 4))[0] == -1
    assert rt.classify(A, rt.ZERO, F(3, 4), F(1))[0] == 1
    assert rt.classify(A, rt.ZERO, F(1, 4), F(3, 4))[0] == 0


def test_poly_range_contains_sampled_values():
    """다항식 범위의 평균값 감싸기 - 구간 안 표본 값을 모두 담는다."""
    coeffs = [arb(1), arb(-3), arb(2), arb(5) / 7, arb(-1) / 3]
    a, b = arb(1) / 5, arb(3) / 5
    rng = rt.poly_range(coeffs, a, b, rt.ZERO, 5)
    for k in range(21):
        s = a + (b - a) * k / 20
        assert rng.contains(rt.horner(coeffs, s))


def test_apriori_enclosure_satisfies_picard_condition_and_contains_flight():
    """사전 감싸기 E - X0+[0,h]·f(E) ⊆ E 를 다시 만족하고, mpmath 고정밀 해를 [0,h] 동안 담는다."""
    q, v, *_ = lagrange_state()
    X0 = q + v
    h = arb(1) / 50
    E = rt.apriori(X0, h)
    assert E is not None
    I = arb(0).union(h)
    fE = rt.vector_field_box(E)
    assert all(e.contains(x + I * f) for e, x, f in zip(E, X0, fE))
    mpmath.mp.dps = 40

    def rhs(t, y):
        qq, vv = y[:6], y[6:]
        acc = [mpmath.mpf(0)] * 6
        for i, j in rt.PAIRS:
            dx, dy = qq[2 * j] - qq[2 * i], qq[2 * j + 1] - qq[2 * i + 1]
            r3 = (dx * dx + dy * dy) ** mpmath.mpf(1.5)
            acc[2 * i] += dx / r3
            acc[2 * i + 1] += dy / r3
            acc[2 * j] -= dx / r3
            acc[2 * j + 1] -= dy / r3
        return list(vv) + acc

    y0 = [mpmath.mpf(x.mid().str(45, radius=False)) for x in X0]
    sol = mpmath.odefun(rhs, 0, y0, tol=mpmath.mpf(10) ** -30, degree=30)
    for t in ("0", "0.01", "0.02"):
        y = sol(mpmath.mpf(t))
        assert all(E[c].contains(arb(mpmath.nstr(y[c], 35))) for c in range(12))
