"""변 v⟹w 의 정규 좌표 사상 P̂(w) = M_w⁻¹(P̃(c_v + M_v w) − c_w) 와 그 미분을 점마다 엄밀하게 감싼다."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from flint import arb, arb_mat

from . import data
from .charts import D, lift, poincare_derivative, reduced_py_rate
from .flow import FirstOrder, FlightLog, Traj, energy, integrate

HALF = Fraction(-1, 2)


@dataclass
class PointResult:
    w: tuple
    image: list            # P̂(w) 4성분(공)
    deriv: list            # DP̂(w) 4×4(공)
    t1: arb                # 첫 영점(반환) 시각
    t2: arb | None         # 둘째 영점 시각
    disc: arb              # 출발 차트 판별식(T3 정의 D = 4A(h − C))
    branch_out: arb        # −∂H/∂p_y (도착 Σ⁻, branch −1)
    energy_out: arb
    sample_min: list       # r_ij 표본 최솟값 감싸기(참 최솟값의 위 경계)
    sample_max: list


def setup(edge: str):
    d = data.design()
    src, dst = edge.split(">")
    return data.hset(d["vertices"][src]), data.hset(d["vertices"][dst])


def initial(cv, Mv, w):
    u = [cv[i] + sum(Mv[i][j] * w[j] for j in range(4)) for i in range(4)]
    ud = [D(data.q2arb(u[i]), [data.q2arb(Mv[i][a]) for a in range(4)]) for i in range(4)]
    state, disc = lift(ud, data.q2arb(HALF))
    return [s.v for s in state], [[s.g[a] for s in state] for a in range(4)], disc.v


def normalize(Mw, cw, u, Dm):
    """w' = M_w⁻¹(u − c_w), DP̂ = M_w⁻¹ DP̃ (Arb 선형 해)."""
    A = arb_mat([[data.q2arb(v) for v in row] for row in Mw])
    rhs = arb_mat([[u[i] - data.q2arb(cw[i])] + [Dm[i][a] for a in range(len(Dm[i]))] for i in range(4)])
    sol = A.solve(rhs)
    return [sol[i, 0] for i in range(4)], [[sol[i, 1 + a] for a in range(len(Dm[i]))] for i in range(4)]


def run_points(edge: str, ws, *, order=30, tol=1e-40, second_zero=True, samples_per_step=6):
    (cv, Mv), (cw, Mw) = setup(edge)
    trajs, discs = [], []
    for w in ws:
        x0, T0, disc = initial(cv, Mv, w)
        trajs.append(Traj(x0 + [v for t in T0 for v in t]))
        discs.append(disc)
    log = FlightLog()
    integrate(trajs, FirstOrder(4), order=order, tol=tol, crossings_wanted=2 if second_zero else 1,
              directions=[-1, 1], samples_per_step=samples_per_step, log=log)
    out = []
    for w, tr, disc in zip(ws, trajs, discs):
        c1 = tr.crossings[0]
        x, T = c1.state[:12], [c1.state[12 + 12 * a: 24 + 12 * a] for a in range(4)]
        u, Dm = poincare_derivative(x, T)
        img, der = normalize(Mw, cw, u, Dm)
        t2 = tr.crossings[1].time if second_zero else None
        out.append(PointResult(tuple(w), img, der, c1.time, t2, disc, -reduced_py_rate(x), energy(x),
                               tr.sample_min, tr.sample_max))
    return out, log


def vertices():
    out = []
    for a in (-1, 1):
        for b in (-1, 1):
            for c in (-1, 1):
                for e in (-1, 1):
                    out.append((Fraction(a), Fraction(b), Fraction(c), Fraction(e)))
    return out


CENTER = (Fraction(0),) * 4
