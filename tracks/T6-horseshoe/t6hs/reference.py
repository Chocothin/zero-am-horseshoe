"""G5 기준 해 [수치, 비엄밀 고정밀]: 표준 Jacobi 8차원 뉴턴 방정식을 mpmath Taylor 급수로 적분한다.

CAPD 엔진의 축소 좌표 벡터장·차트와 독립이다. 차트 u→상태는 J=0·H=h 식(d=−b f/a, f=+√(N/D))을 직접 쓴다.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import mpmath as mp

DPS = 40
ORDER = 30
# d_p = w1 ρ1 + w2 ρ2 (d12=ρ1, d13=ρ2+ρ1/2, d23=ρ2−ρ1/2), 힘은 −w_a d_p |d_p|^{-3}
PAIRS = ((Fraction(1), Fraction(0)), (Fraction(1, 2), Fraction(1)), (Fraction(-1, 2), Fraction(1)))


def _q(x: Fraction) -> mp.mpf:
    return mp.mpf(x.numerator) / x.denominator


def to_mpf(x) -> mp.mpf:
    return _q(x) if isinstance(x, Fraction) else mp.mpf(x)


def taylor(state: Sequence[mp.mpf], order: int) -> list[list[mp.mpf]]:
    """해의 Taylor 계수 c[k][성분], k=0..order. s_p=|d_p|², v_p=s_p^{−3/2} 는 s v' = (−3/2) s' v 재귀로 구한다."""
    alpha = mp.mpf(-3) / 2
    inv_mu = (mp.mpf(2), mp.mpf(2), mp.mpf(3) / 2, mp.mpf(3) / 2)
    w = [(_q(a), _q(b)) for a, b in PAIRS]
    c = [list(state)]
    d = [[] for _ in PAIRS]
    s = [[] for _ in PAIRS]
    v = [[] for _ in PAIRS]
    for k in range(order):
        ck = c[k]
        force = [mp.mpf(0)] * 4
        for p, (w1, w2) in enumerate(w):
            d[p].append((w1 * ck[0] + w2 * ck[2], w1 * ck[1] + w2 * ck[3]))
            s[p].append(mp.fsum(d[p][j][0] * d[p][k - j][0] + d[p][j][1] * d[p][k - j][1] for j in range(k + 1)))
            if k == 0:
                v[p].append(s[p][0] ** alpha)
            else:
                v[p].append(mp.fsum((alpha * j - (k - j)) * s[p][j] * v[p][k - j] for j in range(1, k + 1))
                            / (k * s[p][0]))
            fx = mp.fsum(d[p][j][0] * v[p][k - j] for j in range(k + 1))
            fy = mp.fsum(d[p][j][1] * v[p][k - j] for j in range(k + 1))
            force[0] -= w1 * fx
            force[1] -= w1 * fy
            force[2] -= w2 * fx
            force[3] -= w2 * fy
        c.append([ck[4 + i] * inv_mu[i] / (k + 1) for i in range(4)] + [force[i] / (k + 1) for i in range(4)])
    return c


def evaluate(c, h):
    n = len(c) - 1
    val = list(c[n])
    der = [mp.mpf(0)] * 8
    for k in range(n - 1, -1, -1):
        der = [der[i] * h + val[i] for i in range(8)]
        val = [val[i] * h + c[k][i] for i in range(8)]
    return val, der


def step_size(c, tol):
    n = len(c) - 1
    r = []
    for k in (n - 1, n):
        mag = max(abs(x) for x in c[k])
        if mag > 0:
            r.append((tol / mag) ** (mp.mpf(1) / k))
    return min(r) * mp.mpf("0.7") if r else mp.mpf("0.1")


def collinear(x):
    return x[0] * x[3] - x[1] * x[2]


def distances(x):
    return [mp.sqrt((_q(a) * x[0] + _q(b) * x[2]) ** 2 + (_q(a) * x[1] + _q(b) * x[3]) ** 2) for a, b in PAIRS]


def lift(u, energy=Fraction(-1, 2)):
    a, b, cc, e = (to_mpf(x) for x in u)
    pot = 1 / a + 1 / abs(b + a / 2) + 1 / abs(b - a / 2)
    numer = 2 * (to_mpf(energy) + pot) - 2 * cc ** 2 - mp.mpf(3) / 2 * e ** 2
    denom = 2 * b ** 2 / a ** 2 + mp.mpf(3) / 2
    if numer <= 0:
        raise ValueError("chart point off the transversal domain (N<=0)")
    f = mp.sqrt(numer / denom)
    return [a, mp.mpf(0), b, mp.mpf(0), cc, -b * f / a, e, f]


def project(x):
    a = mp.sqrt(x[0] ** 2 + x[1] ** 2)
    nx, ny = x[0] / a, x[1] / a
    return [a, x[2] * nx + x[3] * ny, x[4] * nx + x[5] * ny, x[6] * nx + x[7] * ny]


@dataclass(frozen=True)
class Flight:
    u_out: list
    time: mp.mpf
    min_distance: list            # r12, r13, r23 표본 최솟값(출발~반환)
    max_distance: list
    steps: int
    g_min_before: mp.mpf          # 첫 표본부터 반환 직전 표본까지 g=ρ1×ρ2 의 최솟값(>0 이면 그 사이 영점 표본 없음)
    samples_before: int


def next_syzygy(u, samples: int = 8, t_max: float = 200.0) -> Flight:
    """lift(u)(g 증가 통과)에서 t>0 의 첫 g 하강 영점까지. 단계 다항식 위 Newton 으로 근을 찾는다."""
    with mp.workdps(DPS):
        tol = mp.mpf(10) ** (-(DPS - 3))
        x = lift(u)
        t = mp.mpf(0)
        lo = distances(x)
        hi = list(lo)
        g_min, n_samples = None, 0
        for steps in range(1, 200000):
            c = taylor(x, ORDER)
            h = step_size(c, tol)
            prev_tau, prev_g = mp.mpf(0), collinear(x)
            for j in range(1, samples + 1):
                tau = h * j / samples
                y, _ = evaluate(c, tau)
                g = collinear(y)
                # 출발점은 g=0 이고 바로 g>0 이 되므로, 하강 영점은 g 가 양에서 음으로 바뀌는 곳이다
                if prev_g > 0 and g <= 0 and t + tau > 0:
                    root = _root(c, prev_tau, tau)
                    end, _ = evaluate(c, root)
                    for i, r in enumerate(distances(end)):
                        lo[i], hi[i] = min(lo[i], r), max(hi[i], r)
                    return Flight(project(end), t + root, lo, hi, steps, g_min, n_samples)
                g_min = g if g_min is None else min(g_min, g)
                n_samples += 1
                for i, r in enumerate(distances(y)):
                    lo[i], hi[i] = min(lo[i], r), max(hi[i], r)
                prev_tau, prev_g = tau, g
            x, _ = evaluate(c, h)
            t += h
            if t > t_max:
                break
        raise RuntimeError("no syzygy before t_max")


def _root(c, lo, hi):
    tol = mp.mpf(10) ** (-(DPS - 4))
    tau = (lo + hi) / 2
    for _ in range(200):
        y, dy = evaluate(c, tau)
        g = collinear(y)
        if abs(hi - lo) < tol or abs(g) < tol:
            return tau
        dg = dy[0] * y[3] + y[0] * dy[3] - dy[1] * y[2] - y[1] * dy[2]
        if g > 0:
            lo = tau
        else:
            hi = tau
        cand = tau - g / dg
        tau = cand if lo < cand < hi else (lo + hi) / 2
    raise RuntimeError("syzygy root did not converge")


def jacobian(u, step=Fraction(1, 10**15)) -> tuple[Flight, list[list[mp.mpf]], list[mp.mpf]]:
    """중심 비행, u 좌표 중심차분 야코비안과 귀환 시각 기울기(40자리에서 절단·반올림 오차 모두 1e-24 이하)."""
    with mp.workdps(DPS):
        base = next_syzygy(u)
        cols, grad = [], []
        h = to_mpf(step)
        for j in range(4):
            up = [to_mpf(v) + (h if i == j else 0) for i, v in enumerate(u)]
            um = [to_mpf(v) - (h if i == j else 0) for i, v in enumerate(u)]
            fp, fm = next_syzygy(up), next_syzygy(um)
            cols.append([(a - b) / (2 * h) for a, b in zip(fp.u_out, fm.u_out)])
            grad.append((fp.time - fm.time) / (2 * h))
        return base, [[cols[j][i] for j in range(4)] for i in range(4)], grad
