"""V2-compact 비엄밀 표본용 고정밀 적분기: 데카르트 뉴턴 방정식(평면 3체, 질량 1, G=1)의 Taylor 급수 방법(mpmath).

원래 트랙 코드와 공유하는 것이 없다. 좌표 변환은 CLAUDE.md 의 Jacobi 정의(ρ₁=q₂−q₁, ρ₂=q₃−(q₁+q₂)/2)를 따른다.
"""
from __future__ import annotations

from fractions import Fraction

import mpmath as mp

PAIRS = ((0, 1), (0, 2), (1, 2))


def mu1():
    return mp.mpf(1) / 2


def mu2():
    return mp.mpf(2) / 3


def to_mpf(x) -> mp.mpf:
    if isinstance(x, Fraction):
        return mp.mpf(x.numerator) / x.denominator
    return mp.mpf(x)


# ---------------------------------------------------------------- 차트 ↔ 관성 좌표

def potential_reduced(r, x, y=0):
    return 1 / r + 1 / mp.sqrt((x + r / 2) ** 2 + y**2) + 1 / mp.sqrt((x - r / 2) ** 2 + y**2)


def lift_sigma_plus(u, h=None):
    """차트 u=(r,x,p_r,p_x) → Σ⁺(y=0, ẏ>0) 위 축소 상태. J=0, y=0 에서 qa·p_y²+qc=h 를 p_y>0 으로 푼다."""
    h = mp.mpf(-1) / 2 if h is None else h
    r, x, pr, px = u
    qa = 1 / (2 * mu2()) + x**2 / (2 * mu1() * r**2)
    qc = pr**2 / (2 * mu1()) + px**2 / (2 * mu2()) - potential_reduced(r, x)
    disc = -4 * qa * (qc - h)
    if disc <= 0:
        raise ValueError("discriminant not positive")
    py = mp.sqrt(disc) / (2 * qa)
    return (r, x, mp.mpf(0), pr, px, py), disc


def reduced_to_inertial(state, J=None):
    """축소 상태(θ=0 틀) → 관성 q_i, v_i. ρ₁=(r,0), ρ₂=(x,y), π₁=(p_r, L/r), π₂=(p_x,p_y), L=J−x p_y+y p_x."""
    J = mp.mpf(0) if J is None else J
    r, x, y, pr, px, py = state
    L = J - x * py + y * px
    rho1 = (r, mp.mpf(0))
    rho2 = (x, y)
    rd1 = (pr / mu1(), (L / r) / mu1())
    rd2 = (px / mu2(), py / mu2())
    q = [None] * 6
    v = [None] * 6
    for c in range(2):
        q[c] = -rho1[c] / 2 - rho2[c] / 3
        q[2 + c] = rho1[c] / 2 - rho2[c] / 3
        q[4 + c] = 2 * rho2[c] / 3
        v[c] = -rd1[c] / 2 - rd2[c] / 3
        v[2 + c] = rd1[c] / 2 - rd2[c] / 3
        v[4 + c] = 2 * rd2[c] / 3
    return q, v


def inertial_to_reduced(q, v):
    """관성 → 축소 (r,x,y,p_r,p_x,p_y). e=ρ₁/|ρ₁| 틀에서 성분을 잰다."""
    rho1 = (q[2] - q[0], q[3] - q[1])
    rho2 = (q[4] - (q[0] + q[2]) / 2, q[5] - (q[1] + q[3]) / 2)
    rd1 = (v[2] - v[0], v[3] - v[1])
    rd2 = (v[4] - (v[0] + v[2]) / 2, v[5] - (v[1] + v[3]) / 2)
    r = mp.sqrt(rho1[0] ** 2 + rho1[1] ** 2)
    e = (rho1[0] / r, rho1[1] / r)
    ep = (-e[1], e[0])
    pi1 = (mu1() * rd1[0], mu1() * rd1[1])
    pi2 = (mu2() * rd2[0], mu2() * rd2[1])

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1]

    return (r, dot(e, rho2), dot(ep, rho2), dot(e, pi1), dot(e, pi2), dot(ep, pi2))


def invariants(q, v):
    kin = sum(v[k] ** 2 for k in range(6)) / 2
    pot = 0
    for i, j in PAIRS:
        pot -= 1 / mp.sqrt((q[2 * j] - q[2 * i]) ** 2 + (q[2 * j + 1] - q[2 * i + 1]) ** 2)
    ang = sum(q[2 * i] * v[2 * i + 1] - q[2 * i + 1] * v[2 * i] for i in range(3))
    mom = (v[0] + v[2] + v[4], v[1] + v[3] + v[5])
    return kin + pot, ang, mom


# ---------------------------------------------------------------- Taylor 계수(자동 미분 점화식)

def taylor(q, v, order):
    """q(t0+s)=Σ Q[c][k] s^k 의 계수, 쌍 거리 제곱 계수 S[p][k], 넓이 지시자 A=(q₂−q₁)×(q₃−q₁) 계수."""
    zero = mp.mpf(0)
    Q = [[q[c]] + [zero] * order for c in range(6)]
    V = [[v[c]] + [zero] * order for c in range(6)]
    dx = [[zero] * (order + 1) for _ in PAIRS]
    dy = [[zero] * (order + 1) for _ in PAIRS]
    S = [[zero] * (order + 1) for _ in PAIRS]
    W = [[zero] * (order + 1) for _ in PAIRS]
    alpha = mp.mpf(-3) / 2
    for k in range(order + 1):
        for p, (i, j) in enumerate(PAIRS):
            dx[p][k] = Q[2 * j][k] - Q[2 * i][k]
            dy[p][k] = Q[2 * j + 1][k] - Q[2 * i + 1][k]
            S[p][k] = mp.fsum(dx[p][l] * dx[p][k - l] + dy[p][l] * dy[p][k - l] for l in range(k + 1))
        if k == order:
            break
        acc = [zero] * 6
        for p, (i, j) in enumerate(PAIRS):
            if k == 0:
                W[p][0] = S[p][0] ** alpha
            else:
                W[p][k] = mp.fsum((alpha * l - (k - l)) * S[p][l] * W[p][k - l]
                                  for l in range(1, k + 1)) / (k * S[p][0])
            fx = mp.fsum(dx[p][l] * W[p][k - l] for l in range(k + 1))
            fy = mp.fsum(dy[p][l] * W[p][k - l] for l in range(k + 1))
            acc[2 * i] += fx
            acc[2 * i + 1] += fy
            acc[2 * j] -= fx
            acc[2 * j + 1] -= fy
        for c in range(6):
            Q[c][k + 1] = V[c][k] / (k + 1)
            V[c][k + 1] = acc[c] / (k + 1)
    # PAIRS[0]=(1,2) 의 d=q₂−q₁, PAIRS[1]=(1,3) 의 d=q₃−q₁
    A = [mp.fsum(dx[0][l] * dy[1][k - l] - dy[0][l] * dx[1][k - l] for l in range(k + 1)) for k in range(order + 1)]
    return Q, V, S, A


def horner(coeffs, s):
    acc = mp.mpf(0)
    for c in reversed(coeffs):
        acc = acc * s + c
    return acc


def dpoly(coeffs):
    return [k * coeffs[k] for k in range(1, len(coeffs))]


def step_size(Q, V, order):
    """Jorba–Zou 식 보폭: ρ_m=(1/‖x^{[m]}‖)^{1/m} 의 작은 쪽에 e^{-2} 를 곱한다."""
    def norm(m):
        return max(abs(Q[c][m]) for c in range(6)) + max(abs(V[c][m]) for c in range(6))

    rhos = [(1 / norm(m)) ** (mp.mpf(1) / m) for m in (order - 1, order) if norm(m) > 0]
    return min(rhos) * mp.e ** -2


def extremes_on(coeffs, a, b, grid=48):
    """다항식(거리 제곱)의 [a,b] 위 최소·최대(격자 + 도함수 Newton 다듬기)와 그 위치."""
    der = dpoly(coeffs)
    der2 = dpoly(der)
    pts = [a + (b - a) * mp.mpf(m) / grid for m in range(grid + 1)]
    vals = [horner(coeffs, s) for s in pts]
    cands = [(vals[m], pts[m]) for m in range(grid + 1)]
    tol = mp.mpf(10) ** (-mp.mp.dps + 4)
    for m in range(1, grid):
        if (vals[m] - vals[m - 1]) * (vals[m + 1] - vals[m]) <= 0:
            s = pts[m]
            for _ in range(60):
                d2 = horner(der2, s)
                if d2 == 0:
                    break
                ns = s - horner(der, s) / d2
                if ns < pts[m - 1] or ns > pts[m + 1]:
                    break
                done = abs(ns - s) < tol
                s = ns
                if done:
                    break
            cands.append((horner(coeffs, s), s))
    lo = min(cands, key=lambda c: c[0])
    hi = max(cands, key=lambda c: c[0])
    return lo, hi


def refine_zero(A, lo, hi):
    """A(lo)>0≥A(hi) 인 구간에서 이분법+Newton."""
    dA = dpoly(A)
    tol = mp.mpf(10) ** (-mp.mp.dps + 4)
    s = (lo + hi) / 2
    for _ in range(200):
        val = horner(A, s)
        if val > 0:
            lo = s
        else:
            hi = s
        der = horner(dA, s)
        ns = s - val / der if der != 0 else (lo + hi) / 2
        if not (lo < ns < hi):
            ns = (lo + hi) / 2
        if abs(ns - s) < tol or hi - lo < tol:
            return ns
        s = ns
    return s


def fly_to_syzygy(q, v, *, order=36, tmax=60, grid=64):
    """t=0 의 Σ⁺ 상태(A=0, Ȧ>0)에서 첫 syzygy(A 의 +→− 영점)까지 적분한다.

    [0,τ] 위 r_ij 최소·최대와 그 시각, τ 의 상태, (0,τ) 격자점의 A 표본(다른 영점이 없는지 보는 용도)을 돌려준다.
    """
    t = mp.mpf(0)
    rmin2 = [mp.inf] * 3
    rmax2 = [mp.mpf(0)] * 3
    tmin_at = [None] * 3
    tmax_at = [None] * 3
    steps = 0
    samples = []  # (t, A) 격자 표본
    first = True
    while t < tmax:
        Q, V, S, A = taylor(q, v, order)
        h = step_size(Q, V, order)
        pts = [h * mp.mpf(m) / grid for m in range(grid + 1)]
        vals = [horner(A, s) for s in pts]
        end = h
        zero_found = False
        for m in range(1, grid + 1):
            if vals[m] <= 0:
                if first and m == 1:
                    raise RuntimeError("syzygy inside the first grid cell")
                end = refine_zero(A, pts[m - 1], pts[m])
                zero_found = True
                break
            samples.append((t + pts[m], vals[m]))
        for p in range(3):
            (lo, slo), (hi, shi) = extremes_on(S[p], mp.mpf(0), end)
            if lo < rmin2[p]:
                rmin2[p], tmin_at[p] = lo, t + slo
            if hi > rmax2[p]:
                rmax2[p], tmax_at[p] = hi, t + shi
        newq = [horner(Q[c], end) for c in range(6)]
        newv = [horner(V[c], end) for c in range(6)]
        steps += 1
        if zero_found:
            return {"tau": t + end, "rmin": [mp.sqrt(x) for x in rmin2], "rmax": [mp.sqrt(x) for x in rmax2],
                    "tmin_at": tmin_at, "tmax_at": tmax_at, "q": newq, "v": newv, "steps": steps,
                    "Adot_at_tau": horner(dpoly(A), end), "samples": samples}
        q, v = newq, newv
        t += end
        first = False
    raise RuntimeError("no syzygy before tmax")
