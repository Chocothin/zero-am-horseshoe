"""2단계: 출발 h-set 전체(정규 좌표 B=[−1,1]⁴)에서 P̂ 의 C¹ 감싸기를 CAPD 없이 만든다.

1. 중심 점 비행(좁음): P̂(0), DP̂(0).
2. 외피 비행: 초기값 (x, ∂x/∂w, ∂²x/∂w²) 를 B 전체의 구간으로 두고 (x, V, W) 계를 적분한다. 첫 영점 구조(출발 후
   다른 영점 없음, 하강 통과 하나)를 B 전체에서 인증하고, 통과 외피에서 첫 통과 사상의 2계 미분 [D²P̂(B)] 를 감싼다.
3. 평균값 형식: DP̂(w) = DP̂(0) + ∫₀¹ D²P̂(sw)[w,·]ds ∈ DP̂(0) + [D²P̂(B)]·B (B 볼록, 0∈B).
   P̂(Q) ⊂ P̂(0) + [DP̂(B)]·Q (Q ⊂ B: 상자 자체와 출구 면). 외피 비행의 직접 감싸기와 교집합을 취한다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from flint import arb, arb_mat, ctx

from . import data
from .charts import chart_out, lift
from .edge import HALF, normalize, run_points, setup, CENTER
from .flow import FlightLog, SecondOrder, Traj, integrate, syz_grad, syz_hess, syz_rate
from .jets import J2
from .taylor import PAIRS2, field

UNIT = arb(0, 1)   # [−1, 1]


def box_initial(cv, Mv):
    """u(B) = c + M·B 를 값 구간·기울기 M·헤세 0 의 제트로 두고 들어올린다."""
    ud = []
    for i in range(4):
        val = data.q2arb(cv[i]) + sum((data.q2arb(Mv[i][j]) * UNIT for j in range(4)), arb(0))
        ud.append(J2.var(val, [data.q2arb(Mv[i][a]) for a in range(4)]))
    state, disc = lift(ud, data.q2arb(HALF))
    x0 = [s.v for s in state]
    T0 = [[s.g[a] for s in state] for a in range(4)]
    W0 = [[s.H[a][b] for s in state] for (a, b) in PAIRS2]
    return x0, T0, W0, disc.v


def crossing_jets(z):
    """통과 외피 z=(x, T, W) 에서 y(w) = x(τ(w); w) 의 제트와 τ 미분. s(y(w)) ≡ 0 을 두 번 미분해 τ_a, τ_ab 를 푼다."""
    x = z[:12]
    T = [z[12 + 12 * a: 24 + 12 * a] for a in range(4)]
    W = {ab: z[60 + 12 * q: 72 + 12 * q] for q, ab in enumerate(PAIRS2)}
    f, DfT = field(x, T)
    _, Dff = field(x, [f])
    Dff = Dff[0]
    g = syz_grad(x)
    sf = sum((gi * fi for gi, fi in zip(g, f)), arb(0))
    tau = [-sum((gi * ti for gi, ti in zip(g, T[a])), arb(0)) / sf for a in range(4)]
    Dy = [[T[a][c] + f[c] * tau[a] for c in range(12)] for a in range(4)]
    D2y = {}
    tau2 = {}
    for (a, b), Wab in W.items():
        rest = [Wab[c] + DfT[b][c] * tau[a] + DfT[a][c] * tau[b] + Dff[c] * tau[a] * tau[b] for c in range(12)]
        tab = -(sum((gi * ri for gi, ri in zip(g, rest)), arb(0)) + syz_hess(Dy[a], Dy[b])) / sf
        tau2[(a, b)] = tab
        D2y[(a, b)] = [rest[c] + f[c] * tab for c in range(12)]
    jets = []
    for c in range(12):
        H = [[D2y[(min(a, b), max(a, b))][c] for b in range(4)] for a in range(4)]
        jets.append(J2(x[c], [Dy[a][c] for a in range(4)], H))
    return jets, tau, tau2, sf


@dataclass
class BoxResult:
    center_image: list
    center_deriv: list
    hull_image: list          # 외피 비행 직접 감싸기(정규 좌표)
    hull_deriv: list          # 외피 비행 직접 DP̂ 감싸기
    d2: list                  # [D²P̂(B)] 4×4×4 (i, j, b)
    deriv_mv: list            # DP̂(0) + [D²P̂(B)]·B
    deriv: list               # 최종(교집합)
    tau_hull: arb
    tube_lo: list
    tube_hi: list
    branch_out_hull: arb
    disc_hull: arb
    seconds_center: float
    seconds_hull: float
    steps_hull: int
    halvings_hull: int
    max_rem_hull: float


def enclose_box(edge: str, *, prec_center=256, order_center=30, tol_center=1e-40, prec_hull=128,
                order_hull=24, tol_hull=1e-28) -> BoxResult:
    (cv, Mv), (cw, Mw) = setup(edge)
    ctx.prec = prec_center
    t0 = time.time()
    (c,), _ = run_points(edge, [CENTER], order=order_center, tol=tol_center, second_zero=False, samples_per_step=0)
    sec_c = time.time() - t0
    ctx.prec = prec_hull
    t0 = time.time()
    x0, T0, W0, disc = box_initial(cv, Mv)
    z0 = x0 + [v for t in T0 for v in t] + [v for t in W0 for v in t]
    tr = Traj(z0)
    log = FlightLog()
    integrate([tr], SecondOrder(), order=order_hull, tol=tol_hull, crossings_wanted=1, directions=[-1],
              samples_per_step=0, log=log)
    cr = tr.crossings[0]
    jets, tau, tau2, sf = crossing_jets(cr.state)
    u = chart_out(jets)
    Du = [[u[i].g[a] for a in range(4)] for i in range(4)]
    img_h, der_h = normalize(Mw, cw, [ui.v for ui in u], Du)
    A = arb_mat([[data.q2arb(v) for v in row] for row in Mw])
    d2 = [[[None] * 4 for _ in range(4)] for _ in range(4)]
    for b in range(4):
        sol = A.solve(arb_mat([[u[i].H[a][b] for a in range(4)] for i in range(4)]))
        for i in range(4):
            for j in range(4):
                d2[i][j][b] = sol[i, j]
    deriv_mv = [[c.deriv[i][j] + sum((d2[i][j][b] * UNIT for b in range(4)), arb(0)) for j in range(4)]
                for i in range(4)]
    deriv = [[deriv_mv[i][j].intersection(der_h[i][j]) for j in range(4)] for i in range(4)]
    x = cr.state[:12]
    r1 = ((x[2] - x[0]) * (x[2] - x[0]) + (x[3] - x[1]) * (x[3] - x[1])).sqrt()
    sec_h = time.time() - t0
    return BoxResult(c.image, c.deriv, img_h, der_h, d2, deriv_mv, deriv, cr.time, log.tube_lo, log.tube_hi,
                     -syz_rate(x) / r1, disc, sec_c, sec_h, log.steps, log.halvings, log.max_rem)


def image_of(res: BoxResult, piece):
    """P̂(0) + [DP̂(B)]·piece (piece ⊂ B, 성분 공) 와 외피 직접 감싸기의 교집합."""
    out = []
    for i in range(4):
        v = res.center_image[i] + sum((res.deriv[i][j] * piece[j] for j in range(4)), arb(0))
        out.append(v.intersection(res.hull_image[i]))
    return out
