"""[수치] float64 syzygy 사상 P̃ 와 그 미분. 설계·초기값 전용이며 증명 근거로 쓰지 않는다.

좌표: 표준 Jacobi x=(ρ1,ρ2,π1,π2), μ1=1/2, μ2=2/3, h=−1/2, J=0. 단면 차트 u=(a,b,c,e)=(|ρ1|, ρ2·n, π1·n, π2·n),
n=ρ1/|ρ1| 이고 T3 API 의 (r,x,pr,px) 와 같다. 출발점은 g=ρ1×ρ2 가 증가하며 지나는 쪽(T3 syzygy(+1)),
도착은 g 가 감소하며 지나는 다음 영점(T3 syzygy(−1))이다. 두 방향은 반사 y→−y 로 u 가 같아 P̃ 는 u 위의 사상이다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

MU1, MU2 = 0.5, 2.0 / 3.0
INV_MASS = np.array([1 / MU1, 1 / MU1, 1 / MU2, 1 / MU2])
ENERGY = -0.5
# scipy 는 100·eps 미만 rtol 을 올려 쓰므로 실제로 지켜지는 가장 작은 값
RTOL, ATOL = 2.3e-14, 1e-14
REVERSOR = np.diag([1.0, -1.0, -1.0, 1.0])  # R(a,b,c,e)=(a,−b,−c,e)
# (ρ1 계수, ρ2 계수): d12=ρ1, d13=ρ2+ρ1/2, d23=ρ2−ρ1/2
PAIRS = ((1.0, 0.0), (0.5, 1.0), (-0.5, 1.0))


class SectionError(RuntimeError):
    """단면 차트 밖(N≤0)이거나 다음 syzygy 를 찾지 못함."""


def _pair_vectors(x):
    return [(w1 * x[0] + w2 * x[2], w1 * x[1] + w2 * x[3]) for w1, w2 in PAIRS]


def rhs(_t, x):
    force = np.zeros(4)
    for (w1, w2), (dx, dy) in zip(PAIRS, _pair_vectors(x)):
        s = dx * dx + dy * dy
        k = s ** -1.5
        force[0] -= w1 * k * dx
        force[1] -= w1 * k * dy
        force[2] -= w2 * k * dx
        force[3] -= w2 * k * dy
    return np.concatenate([x[4:8] * INV_MASS, force])


def rhs_jacobian(x):
    jac = np.zeros((8, 8))
    jac[0:4, 4:8] = np.diag(INV_MASS)
    for (w1, w2), (dx, dy) in zip(PAIRS, _pair_vectors(x)):
        d = np.array([dx, dy])
        s = d @ d
        # ∂(−d s^{-3/2})/∂d
        block = -(np.eye(2) * s ** -1.5 - 3.0 * np.outer(d, d) * s ** -2.5)
        coef = np.array([w1, w2])
        for a in range(2):
            for b in range(2):
                jac[4 + 2 * a:6 + 2 * a, 2 * b:2 * b + 2] += coef[a] * coef[b] * block
    return jac


def _rhs_with_stm(t, z):
    x = z[:8]
    phi = z[8:].reshape(8, 8)
    return np.concatenate([rhs(t, x), (rhs_jacobian(x) @ phi).ravel()])


def transverse_terms(u, energy=ENERGY):
    a, b, c, e = u
    potential = 1.0 / a + 1.0 / abs(b + a / 2) + 1.0 / abs(b - a / 2)
    numer = 2.0 * (energy + potential) - c * c / MU1 - e * e / MU2
    denom = b * b / (MU1 * a * a) + 1.0 / MU2
    return numer, denom


def lift(u, energy=ENERGY):
    a, b, c, e = u
    numer, denom = transverse_terms(u, energy)
    if numer <= 0 or a <= 0:
        raise SectionError(f"no transversal crossing at u={u}: N={numer}")
    f = np.sqrt(numer / denom)
    return np.array([a, 0.0, b, 0.0, c, -b * f / a, e, f])


def lift_jacobian(u, energy=ENERGY, step=1e-7):
    u = np.asarray(u, dtype=float)
    cols = [(lift(u + step * e) - lift(u - step * e)) / (2 * step) for e in np.eye(4)]
    return np.column_stack(cols)


def project(x):
    a = np.hypot(x[0], x[1])
    n = x[0:2] / a
    return np.array([a, x[2:4] @ n, x[4:6] @ n, x[6:8] @ n])


def project_jacobian(x):
    a = np.hypot(x[0], x[1])
    n = x[0:2] / a
    jac = np.zeros((4, 8))
    jac[0, 0:2] = n
    for row, sl in ((1, slice(2, 4)), (2, slice(4, 6)), (3, slice(6, 8))):
        w = x[sl]
        jac[row, 0:2] = (w - (w @ n) * n) / a
        jac[row, sl] = n
    return jac


def collinearity(x):
    return x[0] * x[3] - x[1] * x[2]


def collinearity_gradient(x):
    return np.array([x[3], -x[2], -x[1], x[0], 0.0, 0.0, 0.0, 0.0])


def _event(_t, z):
    return z[0] * z[3] - z[1] * z[2]


_event.terminal = True
_event.direction = -1


@dataclass(frozen=True)
class Step:
    u: np.ndarray
    time: float
    jacobian: np.ndarray | None
    min_distance: float


def _min_distance(ys):
    r = [np.hypot(w1 * ys[0] + w2 * ys[2], w1 * ys[1] + w2 * ys[3]) for w1, w2 in PAIRS]
    return float(min(v.min() for v in r))


def next_syzygy(u, with_jacobian=False, t_max=200.0, energy=ENERGY) -> Step:
    """P̃(u). 끝점은 사건 시각에서 한 번의 시간 Newton 으로 g=0 에 맞춘다."""
    x0 = lift(np.asarray(u, dtype=float), energy)
    z0 = np.concatenate([x0, np.eye(8).ravel()]) if with_jacobian else x0
    fun = _rhs_with_stm if with_jacobian else rhs
    sol = solve_ivp(fun, (0.0, t_max), z0, method="DOP853", rtol=RTOL, atol=ATOL, events=_event, dense_output=False)
    if sol.status != 1 or sol.t_events[0].size == 0:
        raise SectionError(f"no syzygy within t_max from u={u}")
    t_hit = float(sol.t_events[0][0])
    z_end = sol.y_events[0][0]
    x_end = z_end[:8]
    f_end = rhs(0.0, x_end)
    grad = collinearity_gradient(x_end)
    dt = -collinearity(x_end) / (grad @ f_end)
    x_end = x_end + f_end * dt
    jac = None
    if with_jacobian:
        stm = z_end[8:].reshape(8, 8)
        f_end = rhs(0.0, x_end)
        grad = collinearity_gradient(x_end)
        along = np.eye(8) - np.outer(f_end, grad) / (grad @ f_end)
        jac = project_jacobian(x_end) @ along @ stm @ lift_jacobian(u, energy)
    return Step(project(x_end), t_hit + dt, jac, _min_distance(sol.y))


def orbit(u, n, with_jacobian=False):
    out, cur = [], np.asarray(u, dtype=float)
    for _ in range(n):
        st = next_syzygy(cur, with_jacobian)
        out.append(st)
        cur = st.u
    return out
