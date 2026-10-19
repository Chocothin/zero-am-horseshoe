"""G5 기준 적분기(비엄밀, mpmath): Jacobi 좌표 8차원 뉴턴 방정식의 Taylor 급수 적분과 1계 변분(접벡터 4개).

감사 사본(v5·rigorous_tube·nbody_mp)과 코드를 공유하지 않고 좌표도 다르다(사본은 관성 12차원).
표준 Jacobi 좌표(CLAUDE.md): ρ₁=q₂−q₁, ρ₂=q₃−(q₁+q₂)/2, μ₁=1/2, μ₂=2/3. 쌍 벡터 d₁₂=ρ₁, d₁₃=ρ₂+ρ₁/2, d₂₃=ρ₂−ρ₁/2.
운동 방정식 ρ̈₁=−2d₁₂/r₁₂³−d₁₃/r₁₃³+d₂₃/r₂₃³, ρ̈₂=−(3/2)(d₁₃/r₁₃³+d₂₃/r₂₃³) 은 q̈_i=Σ_j(q_j−q_i)/r_ij³ 에서 직접 유도했다.
syzygy 지시자 A=ρ₁×ρ₂ 이고, 차트는 (r, x, p_r, p_x)=(|ρ₁|, e·ρ₂, e·π₁, e·π₂), e=ρ₁/|ρ₁| 다.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import mpmath as mp

# 쌍 p 의 d_p = C1[p]·ρ₁ + C2[p]·ρ₂, 가속도 기여 ρ̈₁ += A1[p]·d_p/|d_p|³, ρ̈₂ += A2[p]·d_p/|d_p|³
PAIR_NAMES = ("r12", "r13", "r23")


def _consts():
    half = mp.mpf(1) / 2
    return ((mp.mpf(1), half, -half), (mp.mpf(0), mp.mpf(1), mp.mpf(1)),
            (mp.mpf(-2), mp.mpf(-1), mp.mpf(1)), (mp.mpf(0), -mp.mpf(3) / 2, -mp.mpf(3) / 2))


def frac_to_mpf(q: Fraction) -> mp.mpf:
    return mp.mpf(q.numerator) / q.denominator


def mpf_to_frac(x: mp.mpf) -> Fraction:
    """mpf 의 정확한 이진 값. mpf.man_exp 는 가수 부호를 버리므로 _mpf_ (부호, 가수, 지수, 비트 수)를 쓴다."""
    x = mp.mpf(x)
    if not mp.isfinite(x):
        raise ValueError(f"not a finite number: {x}")
    sign, man, exp, _ = x._mpf_
    val = Fraction(int(man)) * (Fraction(2) ** int(exp))
    return -val if sign else val


def _conv(a, b, k):
    return mp.fdot(a[: k + 1], b[k::-1])


def _conv1(a, b, k):
    return mp.fdot(a[1: k + 1], b[k - 1:: -1])


def _power_next(s, w, k, alpha):
    """w=s^α 의 k 차 계수: k s₀ w_k = Σ_{l=1..k} (α l − (k−l)) s_l w_{k−l}."""
    coeffs = [alpha * l - (k - l) for l in range(1, k + 1)]
    return mp.fdot(coeffs, [s[l] * w[k - l] for l in range(1, k + 1)]) / (k * s[0])


def taylor(y, xis, order):
    """y(8), xis(m×8) 의 Taylor 계수 0..order. 반환 (Y[c][k], XI[a][c][k], A[k], S[p][k])."""
    C1, C2, A1, A2 = _consts()
    m = len(xis)
    Y = [[v] for v in y]
    XI = [[[v] for v in xi] for xi in xis]
    dx = [[] for _ in range(3)]
    dy = [[] for _ in range(3)]
    S = [[] for _ in range(3)]
    G = [[] for _ in range(3)]
    Z = [[] for _ in range(3)]
    tdx = [[[] for _ in range(3)] for _ in range(m)]
    tdy = [[[] for _ in range(3)] for _ in range(m)]
    sig = [[[] for _ in range(3)] for _ in range(m)]
    U = [[[] for _ in range(3)] for _ in range(m)]
    m32, m52 = -mp.mpf(3) / 2, -mp.mpf(5) / 2
    for k in range(order + 1):
        for p in range(3):
            dx[p].append(C1[p] * Y[0][k] + C2[p] * Y[2][k])
            dy[p].append(C1[p] * Y[1][k] + C2[p] * Y[3][k])
            S[p].append(_conv(dx[p], dx[p], k) + _conv(dy[p], dy[p], k))
        if k == order:
            break
        acc = [mp.mpf(0)] * 4
        tacc = [[mp.mpf(0)] * 4 for _ in range(m)]
        for p in range(3):
            if k == 0:
                G[p].append(S[p][0] ** m32)
                Z[p].append(S[p][0] ** m52)
            else:
                G[p].append(_power_next(S[p], G[p], k, m32))
                Z[p].append(_power_next(S[p], Z[p], k, m52))
            fx = _conv(dx[p], G[p], k)
            fy = _conv(dy[p], G[p], k)
            acc[0] += A1[p] * fx
            acc[1] += A1[p] * fy
            acc[2] += A2[p] * fx
            acc[3] += A2[p] * fy
            for a in range(m):
                T = XI[a]
                tdx[a][p].append(C1[p] * T[0][k] + C2[p] * T[2][k])
                tdy[a][p].append(C1[p] * T[1][k] + C2[p] * T[3][k])
                sig[a][p].append(_conv(dx[p], tdx[a][p], k) + _conv(dy[p], tdy[a][p], k))
                U[a][p].append(_conv(sig[a][p], Z[p], k))
                # δ(d s^{-3/2}) = δd·s^{-3/2} − 3 d (d·δd) s^{-5/2}
                gx = _conv(tdx[a][p], G[p], k) - 3 * _conv(dx[p], U[a][p], k)
                gy = _conv(tdy[a][p], G[p], k) - 3 * _conv(dy[p], U[a][p], k)
                ta = tacc[a]
                ta[0] += A1[p] * gx
                ta[1] += A1[p] * gy
                ta[2] += A2[p] * gx
                ta[3] += A2[p] * gy
        inv = mp.mpf(1) / (k + 1)
        for c in range(4):
            Y[c].append(Y[4 + c][k] * inv)
        for c in range(4):
            Y[4 + c].append(acc[c] * inv)
        for a in range(m):
            T = XI[a]
            for c in range(4):
                T[c].append(T[4 + c][k] * inv)
            for c in range(4):
                T[4 + c].append(tacc[a][c] * inv)
    A = [_conv(Y[0], Y[3], k) - _conv(Y[1], Y[2], k) for k in range(order + 1)]
    return Y, XI, A, S


def horner(coeffs, s):
    acc = mp.mpf(0)
    for c in reversed(coeffs):
        acc = acc * s + c
    return acc


def dpoly(coeffs):
    return [k * coeffs[k] for k in range(1, len(coeffs))]


def field(y):
    """벡터장 f(y) (8성분)."""
    Y, _, _, _ = taylor(y, [], 1)
    return [Y[c][1] for c in range(8)]


# ---------------------------------------------------------------- 이원수(값 + 방향 미분 n 개)

class Dual:
    __slots__ = ("v", "g")

    def __init__(self, v, g):
        self.v, self.g = v, g

    def _o(self, o):
        return o if isinstance(o, Dual) else Dual(o, [mp.mpf(0)] * len(self.g))

    def __add__(self, o):
        o = self._o(o)
        return Dual(self.v + o.v, [a + b for a, b in zip(self.g, o.g)])

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, [-a for a in self.g])

    def __sub__(self, o):
        return self + (-self._o(o))

    def __rsub__(self, o):
        return self._o(o) + (-self)

    def __mul__(self, o):
        o = self._o(o)
        return Dual(self.v * o.v, [a * o.v + self.v * b for a, b in zip(self.g, o.g)])

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = self._o(o)
        q = self.v / o.v
        return Dual(q, [(a - q * b) / o.v for a, b in zip(self.g, o.g)])

    def __rtruediv__(self, o):
        return self._o(o) / self

    def sqrt(self):
        r = mp.sqrt(self.v)
        return Dual(r, [a / (2 * r) for a in self.g])


def lift(u, h=Fraction(-1, 2)):
    """차트 u=(r,x,p_r,p_x)(Dual) → Σ⁺(A=0, Ȧ>0) 위 Jacobi 상태 8성분(Dual). J=0, 에너지 h."""
    r, x, pr, px = u
    U = 1 / r + 1 / (x + r / 2) * sgn(x + r / 2) + 1 / (x - r / 2) * sgn(x - r / 2)
    Aq = x * x / (r * r) + mp.mpf(3) / 4
    C = pr * pr + mp.mpf(3) / 4 * px * px - U
    num = frac_to_mpf(h) - C
    if not num.v > 0:
        raise ValueError("energy level not reachable with p_y real")
    py = (num / Aq).sqrt()
    zero = r * 0
    # v = ρ̇ = π/μ: v₁ = 2π₁ = (2p_r, −2x p_y/r), v₂ = (3/2)π₂
    return [r, zero, x, zero, 2 * pr, -2 * x * py / r, mp.mpf(3) / 2 * px, mp.mpf(3) / 2 * py]


def sgn(d: Dual) -> int:
    if d.v == 0:
        raise ValueError("collision in the chart (|x ± r/2| = 0)")
    return 1 if d.v > 0 else -1


def chart(y):
    """Jacobi 상태(Dual 또는 mpf) → (r, x, p_r, p_x)."""
    r1x, r1y, r2x, r2y, v1x, v1y, v2x, v2y = y
    r = sqrt_any(r1x * r1x + r1y * r1y)
    ex, ey = r1x / r, r1y / r
    return [r, ex * r2x + ey * r2y, (ex * v1x + ey * v1y) / 2, (ex * v2x + ey * v2y) * 2 / 3]


def sqrt_any(v):
    return v.sqrt() if isinstance(v, Dual) else mp.sqrt(v)


# ---------------------------------------------------------------- 한 점의 첫 syzygy 비행

@dataclass
class Flight:
    tau: mp.mpf
    u_out: list            # 도착 차트(Σ⁻ 쪽 첫 syzygy)
    D_chart: list          # dP̃/dw (4×4), 열 j 는 출발 정규 좌표 w_j 방향
    adot_tau: mp.mpf       # 도착 시각의 Ȧ (음수여야 한다)
    rmin: dict
    rmax: dict
    min_A_before: mp.mpf   # (0, τ) 격자점에서 A 의 최솟값(첫 단계 첫 격자점 제외)
    steps: int
    disc: mp.mpf
    y_out: list            # 도착 Jacobi 상태(보존량 시험용)


def fly(u0, Mcols, *, order=32, tol=None, grid=24, tmax=80, tangents=True):
    """u0(4, mpf) 에서 출발해 첫 syzygy(A 의 +→− 영점)까지. Mcols 는 출발 정규 좌표 방향 du/dw_j(4×4, 열 j)."""
    tol = tol if tol is not None else mp.mpf(10) ** (-(mp.mp.dps - 5))
    m = 4 if tangents else 0
    ud = [Dual(u0[i], [Mcols[i][j] for j in range(4)] if tangents else []) for i in range(4)]
    st = lift(ud)
    y = [s.v for s in st]
    xis = [[s.g[a] for s in st] for a in range(m)]
    # 판별식 D=4A(h−C) 의 값(사본의 disc 와 같은 양)
    r, x, pr, px = u0
    U = 1 / r + 1 / abs(x + r / 2) + 1 / abs(x - r / 2)
    disc = 4 * (x * x / (r * r) + mp.mpf(3) / 4) * (mp.mpf(-1) / 2 - (pr * pr + mp.mpf(3) / 4 * px * px - U))
    t = mp.mpf(0)
    rmin = {n: mp.inf for n in PAIR_NAMES}
    rmax = {n: mp.mpf(0) for n in PAIR_NAMES}
    min_A = mp.inf
    first = True
    steps = 0
    while t < tmax:
        Y, XI, A, S = taylor(y, xis, order)
        norms = []
        for kk in (order - 1, order):
            v = max(abs(Y[c][kk]) / (1 + abs(Y[c][0])) for c in range(8))
            if v > 0:
                norms.append((tol / v) ** (mp.mpf(1) / kk))
        h = min(norms) / 2
        pts = [h * j / grid for j in range(grid + 1)]
        vals = [horner(A, s) for s in pts]
        end = None
        for j in range(1, grid + 1):
            if vals[j] <= 0:
                if first and j == 1:
                    raise RuntimeError("syzygy inside the first grid cell of the flight")
                end = _root(A, pts[j - 1], pts[j])
                break
            if not (first and j == 1):
                min_A = min(min_A, vals[j])
        last = end if end is not None else h
        _track(S, last, grid, rmin, rmax)
        steps += 1
        if end is not None:
            yt = [horner(Y[c], end) for c in range(8)]
            xt = [[horner(XI[a][c], end) for c in range(8)] for a in range(m)]
            return _arrive(t + end, yt, xt, A, end, rmin, rmax, min_A, steps, disc)
        y = [horner(Y[c], h) for c in range(8)]
        xis = [[horner(XI[a][c], h) for c in range(8)] for a in range(m)]
        t += h
        first = False
    raise RuntimeError("no syzygy before tmax")


def _root(A, lo, hi):
    """A(lo)>0≥A(hi) 인 구간에서 이분법과 Newton."""
    dA = dpoly(A)
    eps = mp.mpf(10) ** (-(mp.mp.dps - 3))
    s = (lo + hi) / 2
    for _ in range(400):
        val = horner(A, s)
        if val > 0:
            lo = s
        else:
            hi = s
        der = horner(dA, s)
        ns = s - val / der if der != 0 else (lo + hi) / 2
        if not (lo < ns < hi):
            ns = (lo + hi) / 2
        if abs(ns - s) <= eps * (1 + abs(s)) or hi - lo <= eps * (1 + abs(s)):
            return ns
        s = ns
    raise RuntimeError("root refinement did not converge")


def _track(S, last, grid, rmin, rmax):
    """[0, last] 에서 r_ij²(다항식)의 격자 값과 격자 극값 근처 Newton 극점으로 최소·최대를 갱신한다."""
    for p, name in enumerate(PAIR_NAMES):
        pts = [last * j / grid for j in range(grid + 1)]
        vals = [horner(S[p], s) for s in pts]
        cands = list(vals)
        d1, d2 = dpoly(S[p]), dpoly(dpoly(S[p]))
        for j in range(1, grid):
            if (vals[j] - vals[j - 1]) * (vals[j + 1] - vals[j]) <= 0:
                s = pts[j]
                for _ in range(40):
                    dd = horner(d2, s)
                    if dd == 0:
                        break
                    ns = s - horner(d1, s) / dd
                    if not (pts[j - 1] <= ns <= pts[j + 1]):
                        break
                    s = ns
                cands.append(horner(S[p], s))
        lo, hi = mp.sqrt(min(cands)), mp.sqrt(max(cands))
        rmin[name] = min(rmin[name], lo)
        rmax[name] = max(rmax[name], hi)


def _arrive(tau, yt, xt, A, end, rmin, rmax, min_A, steps, disc):
    f = field(yt)
    grad = [yt[3], -yt[2], -yt[1], yt[0], 0, 0, 0, 0]       # ∇A, A=ρ₁ₓρ₂ᵧ−ρ₁ᵧρ₂ₓ
    adot = mp.fdot(grad, f)
    m = len(xt)
    cols = []
    for a in range(m):
        dtau = -mp.fdot(grad, xt[a]) / adot
        cols.append([xt[a][c] + f[c] * dtau for c in range(8)])
    yd = [Dual(yt[c], [cols[a][c] for a in range(m)]) for c in range(8)]
    out = chart(yd)
    D = [[out[i].g[a] for a in range(m)] for i in range(4)]
    return Flight(tau, [o.v for o in out], D, adot, rmin, rmax, min_A, steps, disc, yt)
