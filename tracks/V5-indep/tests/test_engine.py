"""엔진 부품 시험. 기대값은 독립 출처(mpmath 고정밀 ODE 해, 유한 차분, 정의)에서 정했다."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import mpmath
import pytest
from flint import arb, ctx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v5.charts import D, chart_out, lift  # noqa: E402
from v5.flow import FirstOrder, apriori, syz, syz_grad, syz_hess  # noqa: E402
from v5.jets import J2  # noqa: E402
from v5.taylor import PAIRS2, coefficients, coefficients2, evaluate  # noqa: E402

X0 = ["-1", "0.1", "1", "-0.05", "0.2", "0.3", "0.1", "0.4", "-0.3", "0.2", "0.2", "-0.6"]


@pytest.fixture(autouse=True)
def precision():
    old = ctx.prec
    ctx.prec = 256
    yield
    ctx.prec = old


def newton_rhs(t, y):
    q, v = y[:6], y[6:]
    acc = [mpmath.mpf(0)] * 6
    for i, j in ((0, 1), (0, 2), (1, 2)):
        dx, dy = q[2 * j] - q[2 * i], q[2 * j + 1] - q[2 * i + 1]
        r3 = (dx * dx + dy * dy) ** mpmath.mpf(1.5)
        acc[2 * i] += dx / r3
        acc[2 * i + 1] += dy / r3
        acc[2 * j] -= dx / r3
        acc[2 * j + 1] -= dy / r3
    return list(v) + acc


def test_taylor_polynomial_matches_mpmath_ode():
    """Taylor 30차 다항식(t=0.05)이 mpmath odefun(허용 1e-60) 해와 1e-30 안에서 같다 - 계수 점화식의 정확성."""
    mpmath.mp.prec = 256
    X, _ = coefficients([arb(v) for v in X0], [], 31)
    sol = mpmath.odefun(newton_rhs, 0, [mpmath.mpf(v) for v in X0], tol=mpmath.mpf(10) ** -60, degree=60)
    yt = sol(mpmath.mpf("0.05"))
    xe = evaluate(X, [arb(0)] * 12, 30, arb("0.05"))
    assert max(abs(mpmath.mpf(xe[c].mid().str(60, radius=False)) - yt[c]) for c in range(12)) < 1e-30


def test_tangent_and_second_order_match_finite_differences():
    """1계 접벡터와 2계 변분이 유한 차분(보폭 1e-40)과 1e-33 안에서 같다 - 이원수·2계 점화식의 정확성."""
    rnd = random.Random(1)
    x0 = [arb(v) for v in X0]
    dirs = [[arb(rnd.uniform(-1, 1)) for _ in range(12)] for _ in range(4)]
    _, TT, WW = coefficients2(x0, dirs, [[arb(0)] * 12 for _ in PAIRS2], 30)
    _, T1 = coefficients(x0, dirs, 30)
    tt, eps = arb("0.05"), arb(10) ** -40
    for q, (a, b) in enumerate(PAIRS2):
        xp = [x0[c] + eps * dirs[b][c] for c in range(12)]
        _, Tp = coefficients(xp, dirs, 30)
        Ta = evaluate(T1[a], [arb(0)] * 12, 30, tt)
        Tap = evaluate(Tp[a], [arb(0)] * 12, 30, tt)
        Wq = evaluate(WW[q], [arb(0)] * 12, 30, tt)
        assert max(float(abs(Wq[c] - (Tap[c] - Ta[c]) / eps).upper()) for c in range(12)) < 1e-33
    for a in range(4):
        assert max(float(abs(TT[a][c][k] - T1[a][c][k]).upper()) for c in range(12) for k in range(31)) < 1e-50


def test_ball_coefficients_contain_point_coefficients():
    """공 초기값의 계수가 그 공 안 임의 점의 계수를 담는다 - 포함 단조성(감싸기의 기본 가정)."""
    box = [arb(v) + arb(0, 1e-6) for v in X0]
    Xb, _ = coefficients(box, [], 12)
    rnd = random.Random(7)
    for _ in range(5):
        pt = [arb(v) + arb(rnd.uniform(-1e-6, 1e-6)) for v in X0]
        Xp, _ = coefficients(pt, [], 12)
        assert all(Xb[c][k].contains(Xp[c][k]) for c in range(12) for k in range(13))


def test_apriori_box_contains_true_flight():
    """사전 감싸기 Y 가 [0,h] 동안의 참 궤적(mpmath)을 담는다 - Picard 판정의 결론."""
    mpmath.mp.prec = 200
    h = arb("0.02")
    Y = apriori([arb(v) for v in X0], h, FirstOrder(0).field)
    assert Y is not None
    sol = mpmath.odefun(newton_rhs, 0, [mpmath.mpf(v) for v in X0], tol=mpmath.mpf(10) ** -40, degree=40)
    for s in ("0", "0.005", "0.013", "0.02"):
        y = sol(mpmath.mpf(s))
        assert all(Y[c].contains(arb(mpmath.nstr(y[c], 50))) for c in range(12))


def test_syzygy_function_derivatives():
    """∇s·u 와 D²s[u,u] 가 s 의 방향 미분(정의)과 같다 - s 는 2차식이라 정확히 일치."""
    rnd = random.Random(3)
    x = [arb(rnd.uniform(-1, 1)) for _ in range(12)]
    u = [arb(rnd.uniform(-1, 1)) for _ in range(12)]
    t = arb("0.001")
    s0, s1, s2 = syz(x), syz([a + t * b for a, b in zip(x, u)]), syz([a + 2 * t * b for a, b in zip(x, u)])
    g = sum((gi * ui for gi, ui in zip(syz_grad(x), u)), arb(0))
    hss = syz_hess(u, u)
    assert abs(((s1 - s0) - t * g) - t * t * hss / 2) < arb(10) ** -60
    assert abs((s2 - 2 * s1 + s0) - t * t * hss) < arb(10) ** -60


def test_lift_energy_momentum_and_chart_roundtrip():
    """들어올린 상태가 에너지 −1/2, 총운동량 0, J=0, y=0 을 만족하고 차트로 되돌리면 u 다 - 들어올림의 정의."""
    u = [arb("2.69110835645908928884876"), arb("0.01"), arb("0.003"), arb("-0.76932156313958619620")]
    state, disc = lift(u, arb(-1) / 2)
    from v5.flow import energy
    assert energy(state).contains(arb(-1) / 2) and energy(state).rad() < 1e-60
    assert all(abs(sum((state[6 + 2 * i + c] for i in range(3)), arb(0))) < 1e-70 for c in (0, 1))
    J = sum((state[2 * i] * state[7 + 2 * i] - state[2 * i + 1] * state[6 + 2 * i] for i in range(3)), arb(0))
    assert abs(J) < 1e-70
    assert syz(state).is_zero()
    back = chart_out(state)
    assert all(abs(b - a) < 1e-70 for a, b in zip(u, back))


def test_jet_arithmetic_matches_finite_differences():
    """J2 의 기울기·헤세가 유한 차분과 같다(차트 사상에 적용) - 2계 연쇄 법칙."""
    rnd = random.Random(5)
    x = [arb(rnd.uniform(0.5, 1.5)) for _ in range(12)]
    x[2] = x[0] + 1
    G = [[arb(rnd.uniform(-1, 1)) for _ in range(12)] for _ in range(4)]
    jets = [J2.var(x[c], [G[a][c] for a in range(4)]) for c in range(12)]
    out = chart_out(jets)
    eps1, eps2 = arb(10) ** -30, arb(10) ** -20      # 2계는 반올림 오차/eps² 때문에 보폭을 키운다

    def f(w):
        return chart_out([x[c] + sum((G[a][c] * w[a] for a in range(4)), arb(0)) for c in range(12)])

    base = f([arb(0)] * 4)
    for a in range(4):
        fa = f([eps1 if k == a else arb(0) for k in range(4)])
        for i in range(4):
            assert abs((fa[i] - base[i]) / eps1 - out[i].g[a]) < 1e-25
        wa = [eps2 if k == a else arb(0) for k in range(4)]
        fa2 = f(wa)
        for b in range(4):
            wb = [eps2 if k == b else arb(0) for k in range(4)]
            fb, fab = f(wb), f([wa[k] + wb[k] for k in range(4)])
            for i in range(4):
                fd = (fab[i] - fa2[i] - fb[i] + base[i]) / (eps2 * eps2)
                assert abs(fd - out[i].H[a][b]) < 1e-15


def test_dual_numbers_on_lift_match_jets():
    """이원수(D)와 제트(J2)가 들어올림에서 같은 1계 미분을 낸다 - 두 미분 구현의 일치."""
    base = [arb("2.7"), arb("0.02"), arb("0.001"), arb("-0.77")]
    M = [[arb(1) if i == j else arb(0) for j in range(4)] for i in range(4)]
    sd, _ = lift([D(base[i], M[i]) for i in range(4)], arb(-1) / 2)
    sj, _ = lift([J2.var(base[i], M[i]) for i in range(4)], arb(-1) / 2)
    for a, b in zip(sd, sj):
        assert all(abs(p - q) < 1e-70 for p, q in zip(a.g, b.g))
