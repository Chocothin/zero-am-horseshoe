"""엄밀 Taylor 적분기: Picard 사전 감싸기, Lagrange 나머지, syzygy 단면 통과(횡단성 + 시간 구간 Newton).

상태는 평탄한 성분 목록 z 이고 앞 12성분이 관성 상태 x 다(뒤는 1계·2계 변분). 여러 궤적을 같은 보폭으로 함께
적분하고, 사전 감싸기와 나머지 계수는 궤적들의 외피 하나로 한 번만 계산한다(외피가 모든 궤적의 초기값을 담는다).
단계 [t_k, t_k+h] 에서 z(t_k+τ) ∈ Σ_{n≤N} c_n(z_k) τ^n + c_{N+1}(Y) τ^{N+1} 이다. Y 는 사전 감싸기이고, 자율계라
z^{(N+1)}(ξ)/(N+1)! = c_{N+1}(z(ξ)) ∈ c_{N+1}(Y) 이므로 Lagrange 나머지를 담는다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field as dfield

from flint import arb

from .taylor import PAIRS2, coefficients, coefficients2, evaluate

ZERO = arb(0)


def unit_interval(h: arb) -> arb:
    """[0, h] 를 담는 공(반경 올림 때문에 조금 넓을 수 있으나 감싸기로는 안전하다)."""
    return arb(h / 2, h / 2)


def syz(x):
    """s = (q2 − q1) × (q3 − q1) = ρ₁ × ρ₂ = |ρ₁|·y. 세 물체가 일직선이면 0."""
    return (x[2] - x[0]) * (x[5] - x[1]) - (x[3] - x[1]) * (x[4] - x[0])


def syz_rate(x):
    """ds/dt = (v2 − v1) × (q3 − q1) + (q2 − q1) × (v3 − v1)."""
    return ((x[8] - x[6]) * (x[5] - x[1]) - (x[9] - x[7]) * (x[4] - x[0])
            + (x[2] - x[0]) * (x[11] - x[7]) - (x[3] - x[1]) * (x[10] - x[6]))


def syz_grad(x):
    """∇s (12성분)."""
    g = [ZERO] * 12
    ax, ay = x[2] - x[0], x[3] - x[1]
    bx, by = x[4] - x[0], x[5] - x[1]
    g[2], g[3], g[4], g[5] = by, -bx, -ay, ax
    g[0] = -by + ay
    g[1] = bx - ax
    return g


def syz_hess(u, v):
    """D²s[u, v] (s 는 위치의 2차식)."""
    au, av = (u[2] - u[0], u[3] - u[1]), (v[2] - v[0], v[3] - v[1])
    bu, bv = (u[4] - u[0], u[5] - u[1]), (v[4] - v[0], v[5] - v[1])
    return (au[0] * bv[1] - au[1] * bv[0]) + (av[0] * bu[1] - av[1] * bu[0])


def sq(v):
    return v * v


def safe_distances(x):
    """0 을 담은 공에서도 nan 을 내지 않도록 곱셈으로 제곱한다."""
    r12 = (sq(x[2] - x[0]) + sq(x[3] - x[1])).sqrt()
    r13 = (sq(x[4] - x[0]) + sq(x[5] - x[1])).sqrt()
    r23 = (sq(x[4] - x[2]) + sq(x[5] - x[3])).sqrt()
    return r12, r13, r23


def energy(x, masses=(1, 1, 1)):
    m = [arb(v) for v in masses]
    kin = sum((m[i] * (sq(x[6 + 2 * i]) + sq(x[7 + 2 * i])) for i in range(3)), ZERO) / 2
    r12, r13, r23 = safe_distances(x)
    return kin - m[0] * m[1] / r12 - m[0] * m[2] / r13 - m[1] * m[2] / r23


def hull(values):
    out = values[0]
    for v in values[1:]:
        out = out.union(v)
    return out


class FirstOrder:
    """z = x(12) + 접벡터 m 개(각 12)."""

    def __init__(self, m: int, masses=(1, 1, 1)):
        self.m, self.masses = m, masses

    def coeffs(self, z, order):
        x, T = z[:12], [z[12 + 12 * a: 24 + 12 * a] for a in range(self.m)]
        X, TT = coefficients(x, T, order, self.masses)
        return X + [c for t in TT for c in t]

    def field(self, z):
        return [c[1] for c in self.coeffs(z, 1)]


class SecondOrder:
    """z = x(12) + 접벡터 4개 + 2계 변분 10개(PAIRS2 순서)."""

    def __init__(self, masses=(1, 1, 1)):
        self.masses = masses

    def coeffs(self, z, order):
        x = z[:12]
        T = [z[12 + 12 * a: 24 + 12 * a] for a in range(4)]
        W = [z[60 + 12 * q: 72 + 12 * q] for q in range(len(PAIRS2))]
        X, TT, WW = coefficients2(x, T, W, order, self.masses)
        return X + [c for t in TT for c in t] + [c for t in WW for c in t]

    def field(self, z):
        return [c[1] for c in self.coeffs(z, 1)]


@dataclass
class Crossing:
    index: int            # 출발점 다음 몇 번째 영점인가(1부터)
    time: arb
    state: list           # 평탄 상태 z(통과 시각)
    rate: arb             # 그 단계 사전 감싸기 위 ds/dt
    step: int
    tau: arb              # 단계 안 통과 시각
    step_start: list      # 그 단계 시작 상태(외피 비행에서 상 외피를 다시 감쌀 때 씀)


@dataclass
class Traj:
    z: list
    sign: int = 0
    crossings: list = dfield(default_factory=list)
    sample_min: list = dfield(default_factory=lambda: [None, None, None])
    sample_max: list = dfield(default_factory=lambda: [None, None, None])
    active: bool = True


@dataclass
class FlightLog:
    steps: int = 0
    halvings: int = 0
    t_end: arb = ZERO
    tube_lo: list = dfield(default_factory=lambda: [None, None, None])     # 사전 감싸기 위 r_ij(엄밀 바깥 경계)
    tube_hi: list = dfield(default_factory=lambda: [None, None, None])
    max_rem: float = 0.0
    seconds: float = 0.0
    min_rate_abs: float = float("inf")


def _abs_up(v: arb) -> float:
    return float(abs(v).upper())


def apriori(z, h, F):
    """z + [0,h]·F(W) ⊂ int W 인 W 를 찾고 Y = z + [0,h]·F(W) 를 돌려준다(Picard–Lindelöf). 못 찾으면 None."""
    tau = unit_interval(h)
    Fz = F(z)
    Y = [zi + tau * fi for zi, fi in zip(z, Fz)]
    hf = float(h)
    for _ in range(12):
        W = []
        for yi, zi, fi in zip(Y, z, Fz):
            rad = 0.25 * _abs_up(yi - zi) + hf * (0.05 * _abs_up(fi) + 1e-25 * _abs_up(zi)) + 1e-300
            W.append(yi + arb(0, rad))
        try:
            FW = F(W)
        except (ArithmeticError, ValueError, ZeroDivisionError):
            return None                   # 후보 상자가 충돌 등 정의역 밖을 품으면 보폭을 줄인다
        Yn = [zi + tau * fi for zi, fi in zip(z, FW)]
        if all(wi.contains_interior(yi) for wi, yi in zip(W, Yn)):
            return Yn
        Y = [a.union(b) for a, b in zip(Y, Yn)]
    return None


def _newton(C, rem, order, h, rate):
    tau = unit_interval(h)
    for it in range(80):
        mid = arb(tau.mid())
        g = syz(evaluate(C[:12], rem[:12], order, mid))
        new = tau.intersection(mid - g / rate)
        done = new.rad() >= tau.rad() * 0.999 and it > 3
        tau = new
        if done:
            break
    return tau


def rem_ok(CY, z, h, order, tol):
    """사전 감싸기 위 나머지가 허용치 안인가. 점 계수로 고른 보폭에서도 넓은 Y 위 (N+1)차 계수가 크게 과대평가되면
    감싸기는 엄밀하되 넓어지므로 보폭을 줄인다(위치·속도 성분 기준)."""
    remx = max(_abs_up(c[order + 1]) for c in CY[:12]) * float(h) ** (order + 1)
    return remx <= 1e3 * tol * (1 + max(_abs_up(v) for v in z[:12]))


def flow_to_time(z0, system, t_end, *, order, tol, max_steps=200000):
    """고정 시각 t_end(정확 이진수) 까지 적분한다(검증용). 마지막 단계는 t_end 에 정확히 닿게 줄인다."""
    z = list(z0)
    t = ZERO
    steps = 0
    while t < t_end:
        C = system.coeffs(z, order + 1)
        mk = {k: max(_abs_up(c[k]) / (1 + _abs_up(c[0])) for c in C[:12]) for k in (order, order + 1)}
        h = arb(float(min((tol / v) ** (1.0 / k) for k, v in mk.items() if v > 0)) * 0.5)
        rest = t_end - t
        if h > rest:
            h = rest
        while True:
            Y = apriori(z, h, system.field)
            if Y is not None:
                CY = system.coeffs(Y, order + 1)
                if rem_ok(CY, z, h, order, tol):
                    break
            h = arb(float(h) * 0.5)
        z = evaluate(C, [c[order + 1] for c in CY], order, h)
        t = t + h
        steps += 1
        if steps > max_steps:
            raise ArithmeticError("max_steps reached")
    if not (t - t_end).is_zero():
        raise ArithmeticError("time accumulation is not exact")
    return z, steps


def integrate(trajs, system, *, order, tol, crossings_wanted, directions, max_steps=200000,
              samples_per_step=6, track_until=1, h_max=None, log=None):
    """모든 궤적이 crossings_wanted 번째 영점을 지날 때까지 적분한다. directions[i] 는 i+1 번째 영점의 ds/dt 부호.

    출발 단계에서는 s(x0)=0 이고 사전 감싸기 위 ds/dt 부호가 −directions[0] 이어야 한다(출발 단면, (0,h] 에 다른
    영점 없음). track_until 번째 영점까지 r_ij 표본(통과 시각 이전만)과 사전 감싸기 범위를 모은다.
    """
    started = time.time()
    log = log or FlightLog()
    t = ZERO
    first = True
    for step in range(max_steps):
        live = [tr for tr in trajs if tr.active]
        if not live:
            break
        coeffs = [system.coeffs(tr.z, order + 1) for tr in live]
        mk = {order: 0.0, order + 1: 0.0}
        for C in coeffs:
            for k in mk:
                mk[k] = max(mk[k], max(_abs_up(c[k]) / (1 + _abs_up(c[0])) for c in C[:12]))
        h_est = min((tol / v) ** (1.0 / k) for k, v in mk.items() if v > 0)
        if h_max is not None:
            h_est = min(h_est, h_max)
        h = arb(float(h_est) * 0.5)
        zh = [hull([tr.z[c] for tr in live]) for c in range(len(live[0].z))]
        while True:
            Y = apriori(zh, h, system.field)
            ok = Y is not None
            rate = None
            if ok:
                sY = syz(Y[:12])
                if sY.contains(0):
                    rate = syz_rate(Y[:12])
                    if rate.contains(0):
                        ok = False
            if ok:
                CY = system.coeffs(Y, order + 1)
                ok = rem_ok(CY, zh, h, order, tol)
            if ok:
                rem = [c[order + 1] for c in CY]
                nxt = [evaluate(C, rem, order, h) for C in coeffs]
                if rate is not None and any(syz(zn[:12]).contains(0) for zn in nxt) and not first:
                    ok = False
                if first and rate is not None and any(syz(zn[:12]).contains(0) for zn in nxt):
                    ok = False
            if ok:
                break
            h = arb(float(h) * 0.5)
            log.halvings += 1
            if float(h) < 1e-14:
                raise ArithmeticError(f"step control failed at t={t.str(10)}")
        log.max_rem = max(log.max_rem, max(_abs_up(r) for r in rem[:12]) * float(h) ** (order + 1))
        if rate is not None:
            log.min_rate_abs = min(log.min_rate_abs, float(abs(rate).lower()))
        if any(len(tr.crossings) < track_until for tr in live):
            d = safe_distances(Y[:12])
            for i in range(3):
                lo, hi = d[i].lower(), d[i].upper()
                log.tube_lo[i] = lo if log.tube_lo[i] is None else min(log.tube_lo[i], lo)
                log.tube_hi[i] = hi if log.tube_hi[i] is None else max(log.tube_hi[i], hi)
        for C, tr, zn in zip(coeffs, live, nxt):
            tracking = len(tr.crossings) < track_until
            limit = h
            if first:
                if rate is None or not syz(tr.z[:12]).contains(0):
                    raise ArithmeticError("departure: start is not on the section")
                dep = 1 if rate > 0 else -1
                if dep != -directions[0]:
                    raise ArithmeticError("departure in the wrong direction")
                tr.sign = dep
            elif rate is not None:
                end_sign = 1 if syz(zn[:12]) > 0 else -1
                if end_sign != tr.sign:
                    want = directions[len(tr.crossings)]
                    got = 1 if rate > 0 else -1
                    if got != want:
                        raise ArithmeticError(f"crossing {len(tr.crossings) + 1} in the wrong direction")
                    tau = _newton(C, rem, order, h, rate)
                    zc = evaluate(C, rem, order, tau)
                    tr.crossings.append(Crossing(len(tr.crossings) + 1, t + tau, zc, rate, step, tau, list(tr.z)))
                    tr.sign = end_sign
                    if tracking and len(tr.crossings) == track_until:
                        limit = arb(tau.lower())
                    if len(tr.crossings) >= crossings_wanted:
                        tr.active = False
            if tracking and samples_per_step > 0:
                for q in range(samples_per_step + 1):
                    tq = limit * q / samples_per_step
                    xs = evaluate(C[:12], rem[:12], order, tq)
                    for i, di in enumerate(safe_distances(xs)):
                        lo, hi = tr.sample_min[i], tr.sample_max[i]
                        tr.sample_min[i] = di if lo is None or di.upper() < lo.upper() else lo
                        tr.sample_max[i] = di if hi is None or di.lower() > hi.lower() else hi
            tr.z = zn
        first = False
        t = t + h
        log.steps += 1
        log.t_end = t
    else:
        raise ArithmeticError("max_steps reached")
    log.seconds += time.time() - started
    return log
