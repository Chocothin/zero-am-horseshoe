"""독립 엄밀 적분기(python-flint Arb): 표준 8차원 Jacobi 뉴턴 방정식의 점 초기값 흐름과 syzygy 단면 통과를 감싼다.

CAPD·t3cap 과 코드를 공유하지 않는다. 한 단계는 Picard 사전 감싸기 W(x+[0,h]f(W) ⊂ int W), Taylor 다항식,
Lagrange 나머지 F_N(W)h^N 로 감싼다. 단면 통과는 W 위 횡단성(ds/dt 가 0 을 포함하지 않음)과 시간 구간 Newton 으로 증명한다.
상자(C1) 감싸기는 하지 않는다 — 점 초기값 전용이다.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from flint import arb, fmpq

ALPHA = arb(-3) / 2


def sq(v: arb) -> arb:
    """제곱. python-flint 의 v**2 는 0 을 담은 공에서 nan 을 내므로 곱셈으로 쓴다."""
    return v * v


def q(v: Fraction | int) -> arb:
    v = Fraction(v)
    return arb(fmpq(v.numerator, v.denominator))


@dataclass(frozen=True)
class Masses:
    m1: arb
    m2: arb
    m3: arb
    a: arb
    b: arb
    mu1: arb
    mu2: arb

    @staticmethod
    def of(masses: Sequence[Fraction | int]) -> "Masses":
        m1, m2, m3 = (Fraction(m) for m in masses)
        return Masses(q(m1), q(m2), q(m3), q(m2 / (m1 + m2)), q(m1 / (m1 + m2)), q(m1 * m2 / (m1 + m2)),
                      q((m1 + m2) * m3 / (m1 + m2 + m3)))


def taylor(x: Sequence[arb], order: int, md: Masses) -> list[list[arb]]:
    """X=(rho1x,rho1y,rho2x,rho2y,pi1x,pi1y,pi2x,pi2y) 의 Taylor 계수 [성분][0..order]. 공 산술이라 x 가 공이면 계수도 감싸기다."""
    c = [[v] for v in x]
    pairs = (("12", arb(1), arb(0)), ("13", md.a, arb(1)), ("23", -md.b, arb(1)))
    g = {"12": md.m1 * md.m2, "13": md.m1 * md.m3, "23": md.m2 * md.m3}
    d = {k: ([], []) for k, *_ in pairs}
    s = {k: [] for k, *_ in pairs}
    w = {k: [] for k, *_ in pairs}
    wd = {k: ([], []) for k, *_ in pairs}
    for k in range(order):
        for key, c1, c2 in pairs:
            dx, dy = d[key]
            dx.append(c1 * c[0][k] + c2 * c[2][k])
            dy.append(c1 * c[1][k] + c2 * c[3][k])
            sk = s[key]
            sk.append(sum((dx[l] * dx[k - l] + dy[l] * dy[k - l] for l in range(k + 1)), arb(0)))
            wk = w[key]
            if k == 0:
                if not sk[0] > 0:
                    raise ArithmeticError("pair distance not proven positive")
                wk.append(1 / (sk[0] * sk[0].sqrt()))
            else:
                acc = sum(((ALPHA * j - (k - j)) * sk[j] * wk[k - j] for j in range(1, k + 1)), arb(0))
                wk.append(acc / (k * sk[0]))
            wdx, wdy = wd[key]
            wdx.append(sum((wk[l] * dx[k - l] for l in range(k + 1)), arb(0)))
            wdy.append(sum((wk[l] * dy[k - l] for l in range(k + 1)), arb(0)))
        f12 = [g["12"] * v[k] for v in wd["12"]]
        f13 = [g["13"] * v[k] for v in wd["13"]]
        f23 = [g["23"] * v[k] for v in wd["23"]]
        inv = arb(1) / (k + 1)
        c[0].append(c[4][k] / md.mu1 * inv)
        c[1].append(c[5][k] / md.mu1 * inv)
        c[2].append(c[6][k] / md.mu2 * inv)
        c[3].append(c[7][k] / md.mu2 * inv)
        c[4].append((-f12[0] - md.a * f13[0] + md.b * f23[0]) * inv)
        c[5].append((-f12[1] - md.a * f13[1] + md.b * f23[1]) * inv)
        c[6].append((-f13[0] - f23[0]) * inv)
        c[7].append((-f13[1] - f23[1]) * inv)
    return c


def field(x: Sequence[arb], md: Masses) -> list[arb]:
    return [col[1] for col in taylor(x, 1, md)]


def energy(x: Sequence[arb], md: Masses) -> arb:
    kin = (sq(x[4]) + sq(x[5])) / (2 * md.mu1) + (sq(x[6]) + sq(x[7])) / (2 * md.mu2)
    r12 = (sq(x[0]) + sq(x[1])).sqrt()
    r13 = (sq(x[2] + md.a * x[0]) + sq(x[3] + md.a * x[1])).sqrt()
    r23 = (sq(x[2] - md.b * x[0]) + sq(x[3] - md.b * x[1])).sqrt()
    return kin - md.m1 * md.m2 / r12 - md.m1 * md.m3 / r13 - md.m2 * md.m3 / r23


def distances(x: Sequence[arb], md: Masses) -> tuple[arb, arb, arb]:
    r12 = (sq(x[0]) + sq(x[1])).sqrt()
    r13 = (sq(x[2] + md.a * x[0]) + sq(x[3] + md.a * x[1])).sqrt()
    r23 = (sq(x[2] - md.b * x[0]) + sq(x[3] - md.b * x[1])).sqrt()
    return r12, r13, r23


def syzygy(x: Sequence[arb]) -> arb:
    """rho1 x rho2 — 축소 좌표 y=(rho1 x rho2)/|rho1| 와 부호·영점이 같다."""
    return x[0] * x[3] - x[1] * x[2]


def syzygy_rate(x: Sequence[arb], md: Masses) -> arb:
    """d/dt (rho1 x rho2) = (pi1/mu1) x rho2 + rho1 x (pi2/mu2)."""
    return (x[4] * x[3] - x[5] * x[2]) / md.mu1 + (x[0] * x[7] - x[1] * x[6]) / md.mu2


def chart_syzygy(u: Sequence[Fraction], direction: int, md: Masses, h: Fraction, J: Fraction) -> list[arb]:
    """syzygy 차트 u=(r,x,pr,px) -> 8차원 상태(theta=0). py 는 H 가 py 의 2차식이라는 사실로 세 점에서 맞춰 푼다."""
    r, x, pr, px = (q(v) for v in u)
    Jq = q(J)

    def state(py: arb) -> list[arb]:
        return [r, arb(0), x, arb(0), pr, (Jq - x * py) / r, px, py]

    e_m, e_0, e_p = (energy(state(arb(v)), md) for v in (-1, 0, 1))
    qa = (e_p + e_m) / 2 - e_0
    qb = (e_p - e_m) / 2
    disc = sq(qb) - 4 * qa * (e_0 - q(h))
    if not disc > 0:
        raise ArithmeticError("chart discriminant not proven positive")
    return state((-qb + direction * disc.sqrt()) / (2 * qa))


def reduced_chart_coords(x: Sequence[arb]) -> list[arb]:
    """8차원 상태의 syzygy 차트 좌표 (r, x, pr, px) = (|rho1|, e.rho2, e.pi1, e.pi2)."""
    r = (sq(x[0]) + sq(x[1])).sqrt()
    ex, ey = x[0] / r, x[1] / r
    return [r, ex * x[2] + ey * x[3], ex * x[4] + ey * x[5], ex * x[6] + ey * x[7]]


def apriori(x: Sequence[arb], h: arb, md: Masses) -> list[arb] | None:
    """x + [0,h] f(W) ⊂ int W 인 W 를 찾는다. 성립하면 [0,h] 동안 해가 Y=x+[0,h]f(W) 안에 있다(Picard–Lindelöf)."""
    tau = h * (arb(0, 1) + 1) / 2   # [0, h]
    fx = field(x, md)
    # 출발점에서 도함수가 0 인 성분도 W 위에서는 움직이므로 보폭에 비례한 절대 여유를 모든 성분에 준다
    slack = h * (max(abs(v).upper() for v in fx) / 5 + arb(10) ** -30)
    Y = [xi + tau * fi for xi, fi in zip(x, fx)]
    for _ in range(8):
        W = [yi + arb(0, 1) * (abs(yi - xi).upper() / 5 + slack) for yi, xi in zip(Y, x)]
        Y_next = [xi + tau * fi for xi, fi in zip(x, field(W, md))]
        if all(wi.contains_interior(yi) for wi, yi in zip(W, Y_next)):
            return Y_next
        Y = [yi.union(yn) for yi, yn in zip(Y, Y_next)]
    return None


def evaluate(coeffs: Sequence[Sequence[arb]], remainder: Sequence[arb], order: int, t: arb) -> list[arb]:
    """sum_{k<order} c_k t^k + R t^order (t 는 [0,h] 안의 공)."""
    out = []
    for comp, rem in zip(coeffs, remainder):
        acc = rem
        for k in range(order - 1, -1, -1):
            acc = acc * t + comp[k]
        out.append(acc)
    return out


@dataclass(frozen=True)
class ArbCrossing:
    state: list[arb]
    time: arb
    min_distance: arb
    steps: int


def flow_to_next_syzygy(x0: Sequence[arb], md: Masses, *, direction: int, order: int = 30,
                        tolerance_exp: int = -40, max_steps: int = 20000) -> ArbCrossing:
    """단면 위 x0 에서 출발해 다음 syzygy 통과(방향 direction) 상태를 감싼다. 첫 단계의 영점은 출발점(횡단성으로 유일)이다."""
    x = list(x0)
    t = arb(0)
    dmin = min(distances(x, md), key=lambda v: v.mid())
    tol = arb(10) ** tolerance_exp
    for step in range(max_steps):
        c = taylor(x, order, md)
        mags = [max(abs(comp[k]).upper() for comp in c) for k in (order - 1, order)]
        # 마지막 두 계수로 어림한 수렴 반경의 절반 — 사전 감싸기와 나머지를 여유 있게 잡는다
        h = (min((tol / mg) ** (arb(1) / k) for mg, k in zip(mags, (order - 1, order)) if mg > 0) / 2).mid()
        while True:
            W = apriori(x, h, md)
            if W is not None:
                break
            h = h / 2
            if h < arb(10) ** -12:
                raise ArithmeticError("a priori enclosure failed")
        cW = taylor(W, order, md)
        rem = [comp[order] for comp in cW]
        dmin = min([dmin] + list(distances(W, md)), key=lambda v: v.lower())
        s_W = syzygy(W)
        x_next = evaluate(c, rem, order, h)
        s_start = syzygy(x)
        if step > 0 and s_start.contains(0):
            raise ArithmeticError("section sign at step start undecided")
        if s_W.contains(0):
            rate = syzygy_rate(W, md)
            if rate.contains(0):
                raise ArithmeticError("transversality not proven on step enclosure")
            s_end = syzygy(x_next)
            crossed = step > 0 and not s_end.contains(0) and ((s_end > 0) != (s_start > 0))
            if step > 0 and s_end.contains(0):
                raise ArithmeticError("section sign at step end undecided")
            if crossed:
                if (rate > 0) != (direction > 0):
                    raise ArithmeticError("crossing in the wrong direction")
                T = h * (arb(0, 1) + 1) / 2
                for _ in range(40):
                    tm = T.mid()
                    gm = syzygy(evaluate(c, rem, order, tm))
                    newton = tm - gm / rate
                    T = T.intersection(newton)
                state = evaluate(c, rem, order, T)
                return ArbCrossing(state, t + T, dmin, step + 1)
        x = x_next
        t = t + h
    raise ArithmeticError("next syzygy not reached")
