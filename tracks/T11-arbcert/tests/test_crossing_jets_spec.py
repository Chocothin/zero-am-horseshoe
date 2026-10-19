"""2계 통과 사상 시험(G9, R1 발견에 따른 보강). 감사 V5 사본의 boxmap.crossing_jets 로 점 하나의 D²P̃(w) 를 내고, G5 기준
적분기(t11/g5ref.py: Jacobi 좌표, 사본과 코드 공유 없음)로 잰 1계 도함수의 중심차분과 대조한다.

기대값은 2계 연쇄 법칙의 정의 D²P̃ = ∂(DP̃)/∂w 에서 정했다. 귀환 시각의 2계 도함수 항(f·τ_ab)이나 다른 2계 항이 틀리면
차이가 D²P̃ 자체와 같은 크기가 되어 실패한다(변이 사본으로 확인한 기록은 results/.../crossing_jets_mutant.json)."""
from __future__ import annotations

import sys
from fractions import Fraction as F
from pathlib import Path

import mpmath as mp
from flint import ctx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from t11 import design as dz  # noqa: E402
from t11 import g5ref as G  # noqa: E402
from v5 import data  # noqa: E402
from v5.boxmap import crossing_jets  # noqa: E402
from v5.charts import chart_out, lift  # noqa: E402
from v5.flow import FlightLog, SecondOrder, Traj, integrate  # noqa: E402
from v5.jets import J2  # noqa: E402
from v5.taylor import PAIRS2  # noqa: E402

DIRECTIONS = (0, 3)        # 강불안정·강안정 방향의 중심차분(헤세는 대칭이라 열 둘이 16성분씩을 준다)


def point_hessian(edge: str, w):
    """출발 정규 좌표 점 w 에서 2계 변분 비행 → crossing_jets → chart_out. 반환 u_i 의 값·기울기·헤세(공)."""
    old = ctx.prec
    ctx.prec = 128
    try:
        c, M = dz.hset(edge.split(">")[0])
        ud = [J2.var(data.q2arb(c[i] + sum(M[i][j] * w[j] for j in range(4))), [data.q2arb(M[i][a]) for a in range(4)])
              for i in range(4)]
        state, _ = lift(ud, data.q2arb(F(-1, 2)))
        z0 = [s.v for s in state] + [s.g[a] for a in range(4) for s in state] + [s.H[a][b] for (a, b) in PAIRS2
                                                                                 for s in state]
        tr = Traj(z0)
        integrate([tr], SecondOrder(), order=24, tol=1e-28, crossings_wanted=1, directions=[-1], samples_per_step=0,
                  log=FlightLog())
        jets, _, _, _ = crossing_jets(tr.crossings[0].state)
        return chart_out(jets)
    finally:
        ctx.prec = old


def reference_hessian_column(edge: str, w, b: int, eps: F):
    """기준 적분기의 dP̃/dw 를 w ± eps·e_b 에서 내어 중심차분한 ∂²P̃/∂w_a∂w_b (a=0..3, 성분 i)."""
    mp.mp.dps = 45
    c, M = dz.hset(edge.split(">")[0])
    cols = [[G.frac_to_mpf(M[i][j]) for j in range(4)] for i in range(4)]
    out = []
    for sgn in (1, -1):
        wp = [w[j] + (sgn * eps if j == b else 0) for j in range(4)]
        u0 = [G.frac_to_mpf(c[i] + sum(M[i][j] * wp[j] for j in range(4))) for i in range(4)]
        out.append(G.fly(u0, cols).D_chart)
    e = G.frac_to_mpf(eps)
    return [[(out[0][i][a] - out[1][i][a]) / (2 * e) for a in range(4)] for i in range(4)]


def check_edge(edge: str, w):
    u = point_hessian(edge, w)
    scale = max(float(abs(u[i].H[a][b]).upper()) for i in range(4) for a in range(4) for b in range(4))
    worst = 0.0
    for b in DIRECTIONS:
        fd = reference_hessian_column(edge, w, b, F(1, 10 ** 4))
        for i in range(4):
            for a in range(4):
                h = u[i].H[a][b]
                gap = abs(mp.mpf(h.mid().str(40, radius=False)) - fd[i][a])
                worst = max(worst, float(gap) - float(h.rad()))
    return worst, scale


def test_second_order_crossing_map_matches_finite_difference_cycle_edge():
    """cycle 변 N1⟹N2 의 중심 근처 점 - D²P̃ 감싸기와 기준 적분기 중심차분의 차가 헤세 크기의 1e-9 이하."""
    worst, scale = check_edge("N1>N2", [F(1, 3), F(-1, 5), F(1, 7), F(-1, 2)])
    assert worst <= 1e-9 * scale, (worst, scale)


def test_second_order_crossing_map_matches_finite_difference_chain_edge():
    """사슬 변 M+0⟹M+1(소풍 구간)의 점 - 같은 대조가 헤세 크기의 1e-9 안에서 맞는다."""
    worst, scale = check_edge("M+0>M+1", [F(-1, 4), F(2, 3), F(-3, 5), F(1, 9)])
    assert worst <= 1e-9 * scale, (worst, scale)
