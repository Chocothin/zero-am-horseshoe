"""관성 좌표 평면 3체 계의 Taylor 계수(공 산술). CAPD·T3·T6·T7 코드와 공유하는 것이 없다.

상태 x = (X1,Y1,X2,Y2,X3,Y3, U1,V1,U2,V2,U3,V3): 위치 q_i 와 속도 v_i.
운동 방정식: q_i' = v_i, v_i' = Σ_{j≠i} m_j (q_j − q_i)/|q_j − q_i|³ (G=1).
접벡터 ξ 는 변분 방정식 ξ' = Df(x)ξ 를 따른다. 계수는 이원수(값 + 방향 미분) 점화식으로 같이 구한다.
공 산술의 포함 단조성 때문에 x·ξ 가 공이면 계수도 그 공 위 모든 초기값의 계수를 감싼다.
"""
from __future__ import annotations

from operator import mul

from flint import arb

PAIRS = ((0, 1), (0, 2), (1, 2))
ZERO = arb(0)


def _conv(a, b, k):
    """Σ_{l=0..k} a_l b_{k−l}."""
    return sum(map(mul, a[: k + 1], b[k::-1]), ZERO)


def _conv1(a, b, k):
    """Σ_{l=1..k} a_l b_{k−l}."""
    return sum(map(mul, a[1 : k + 1], b[k - 1 :: -1]), ZERO)


def coefficients(x, tangents, order, masses=(1, 1, 1)):
    """x(12)·tangents(m×12) 의 Taylor 계수 0..order. 반환 (X, TT): X[c][k], TT[a][c][k]."""
    m = [arb(v) for v in masses]
    alpha1 = arb(-1) / 2          # α+1, α = −3/2 (w = s^α)
    X = [[v] for v in x]
    TT = [[[v] for v in t] for t in tangents]
    nt = len(tangents)
    npair = len(PAIRS)
    dx = [[] for _ in range(npair)]
    dy = [[] for _ in range(npair)]
    s = [[] for _ in range(npair)]
    js = [[ZERO] for _ in range(npair)]    # j·s_j (j≥1), 거듭제곱 점화식용
    w = [[] for _ in range(npair)]
    z = [[] for _ in range(npair)]         # z = w/s = s^(−5/2)
    tdx = [[[] for _ in range(npair)] for _ in range(nt)]
    tdy = [[[] for _ in range(npair)] for _ in range(nt)]
    ts = [[[] for _ in range(npair)] for _ in range(nt)]
    tw = [[[] for _ in range(npair)] for _ in range(nt)]
    for k in range(order):
        acc = [ZERO] * 6
        tacc = [[ZERO] * 6 for _ in range(nt)]
        for p, (i, j) in enumerate(PAIRS):
            dxp, dyp, sp, wp, zp = dx[p], dy[p], s[p], w[p], z[p]
            dxp.append(X[2 * j][k] - X[2 * i][k])
            dyp.append(X[2 * j + 1][k] - X[2 * i + 1][k])
            sk = _conv(dxp, dxp, k) + _conv(dyp, dyp, k)
            sp.append(sk)
            if k == 0:
                if not sk > 0:
                    raise ArithmeticError("pair distance not proven positive")
                wk = 1 / (sk * sk.sqrt())
                zk = wk / sk
            else:
                js[p].append(k * sk)
                # k s0 w_k = Σ_{j=1..k} ((α+1) j − k) s_j w_{k−j}
                wk = (alpha1 * _conv1(js[p], wp, k) - k * _conv1(sp, wp, k)) / (k * sp[0])
                zk = (wk - _conv1(sp, zp, k)) / sp[0]
            wp.append(wk)
            zp.append(zk)
            fx = _conv(wp, dxp, k)
            fy = _conv(wp, dyp, k)
            acc[2 * i] += m[j] * fx
            acc[2 * i + 1] += m[j] * fy
            acc[2 * j] -= m[i] * fx
            acc[2 * j + 1] -= m[i] * fy
            for a in range(nt):
                T = TT[a]
                tdxp, tdyp, tsp, twp = tdx[a][p], tdy[a][p], ts[a][p], tw[a][p]
                tdxp.append(T[2 * j][k] - T[2 * i][k])
                tdyp.append(T[2 * j + 1][k] - T[2 * i + 1][k])
                tsp.append(2 * (_conv(dxp, tdxp, k) + _conv(dyp, tdyp, k)))
                twp.append(-3 * _conv(zp, tsp, k) / 2)
                gx = _conv(twp, dxp, k) + _conv(wp, tdxp, k)
                gy = _conv(twp, dyp, k) + _conv(wp, tdyp, k)
                ta = tacc[a]
                ta[2 * i] += m[j] * gx
                ta[2 * i + 1] += m[j] * gy
                ta[2 * j] -= m[i] * gx
                ta[2 * j + 1] -= m[i] * gy
        inv = arb(1) / (k + 1)
        for c in range(6):
            X[c].append(X[6 + c][k] * inv)
        for c in range(6):
            X[6 + c].append(acc[c] * inv)
        for a in range(nt):
            T = TT[a]
            for c in range(6):
                T[c].append(T[6 + c][k] * inv)
            for c in range(6):
                T[6 + c].append(tacc[a][c] * inv)
    return X, TT


PAIRS2 = tuple((a, b) for a in range(4) for b in range(a, 4))


def coefficients2(x, tangents, seconds, order, masses=(1, 1, 1)):
    """2계 변분까지: ξ_a' = Df ξ_a, W_ab' = Df W_ab + D²f[ξ_a, ξ_b] (ab ∈ PAIRS2). 반환 (X, TT, WW).

    짝 p 의 d = q_j − q_i, s = |d|², w = s^α(α=−3/2), F = w d 에 대해
    s_a = 2 d·d_a, w_a = α s^(α−1) s_a, F_a = w_a d + w d_a,
    s_ab = 2(d_a·d_b + d·d_ab), w_ab = α s^(α−1) s_ab + α(α−1) s^(α−2) s_a s_b,
    F_ab = w_ab d + w_a d_b + w_b d_a + w d_ab.
    """
    m = [arb(v) for v in masses]
    alpha = arb(-3) / 2
    alpha1 = arb(-1) / 2
    aa1 = arb(15) / 4                 # α(α−1)
    nt = len(tangents)
    npw = len(seconds)
    pairs2 = PAIRS2[:npw]
    X = [[v] for v in x]
    TT = [[[v] for v in t] for t in tangents]
    WW = [[[v] for v in t] for t in seconds]
    npair = len(PAIRS)
    dx = [[] for _ in range(npair)]
    dy = [[] for _ in range(npair)]
    s = [[] for _ in range(npair)]
    js = [[ZERO] for _ in range(npair)]
    w = [[] for _ in range(npair)]
    z = [[] for _ in range(npair)]       # s^(α−1)
    y = [[] for _ in range(npair)]       # s^(α−2)
    tdx = [[[] for _ in range(npair)] for _ in range(nt)]
    tdy = [[[] for _ in range(npair)] for _ in range(nt)]
    ts = [[[] for _ in range(npair)] for _ in range(nt)]
    tw = [[[] for _ in range(npair)] for _ in range(nt)]
    wdx = [[[] for _ in range(npair)] for _ in range(npw)]
    wdy = [[[] for _ in range(npair)] for _ in range(npw)]
    wsab = [[[] for _ in range(npair)] for _ in range(npw)]
    pab = [[[] for _ in range(npair)] for _ in range(npw)]
    wwab = [[[] for _ in range(npair)] for _ in range(npw)]
    for k in range(order):
        acc = [ZERO] * 6
        tacc = [[ZERO] * 6 for _ in range(nt)]
        wacc = [[ZERO] * 6 for _ in range(npw)]
        for p, (i, j) in enumerate(PAIRS):
            dxp, dyp, sp, wp, zp, yp = dx[p], dy[p], s[p], w[p], z[p], y[p]
            dxp.append(X[2 * j][k] - X[2 * i][k])
            dyp.append(X[2 * j + 1][k] - X[2 * i + 1][k])
            sk = _conv(dxp, dxp, k) + _conv(dyp, dyp, k)
            sp.append(sk)
            if k == 0:
                if not sk > 0:
                    raise ArithmeticError("pair distance not proven positive")
                wk = 1 / (sk * sk.sqrt())
                zk = wk / sk
                yk = zk / sk
            else:
                js[p].append(k * sk)
                wk = (alpha1 * _conv1(js[p], wp, k) - k * _conv1(sp, wp, k)) / (k * sp[0])
                zk = (wk - _conv1(sp, zp, k)) / sp[0]
                yk = (zk - _conv1(sp, yp, k)) / sp[0]
            wp.append(wk)
            zp.append(zk)
            yp.append(yk)
            fx = _conv(wp, dxp, k)
            fy = _conv(wp, dyp, k)
            acc[2 * i] += m[j] * fx
            acc[2 * i + 1] += m[j] * fy
            acc[2 * j] -= m[i] * fx
            acc[2 * j + 1] -= m[i] * fy
            for a in range(nt):
                T = TT[a]
                tdxp, tdyp, tsp, twp = tdx[a][p], tdy[a][p], ts[a][p], tw[a][p]
                tdxp.append(T[2 * j][k] - T[2 * i][k])
                tdyp.append(T[2 * j + 1][k] - T[2 * i + 1][k])
                tsp.append(2 * (_conv(dxp, tdxp, k) + _conv(dyp, tdyp, k)))
                twp.append(alpha * _conv(zp, tsp, k))
                gx = _conv(twp, dxp, k) + _conv(wp, tdxp, k)
                gy = _conv(twp, dyp, k) + _conv(wp, tdyp, k)
                ta = tacc[a]
                ta[2 * i] += m[j] * gx
                ta[2 * i + 1] += m[j] * gy
                ta[2 * j] -= m[i] * gx
                ta[2 * j + 1] -= m[i] * gy
            for q, (a, b) in enumerate(pairs2):
                Wq = WW[q]
                wdxp, wdyp = wdx[q][p], wdy[q][p]
                wdxp.append(Wq[2 * j][k] - Wq[2 * i][k])
                wdyp.append(Wq[2 * j + 1][k] - Wq[2 * i + 1][k])
                sab = 2 * (_conv(tdx[a][p], tdx[b][p], k) + _conv(tdy[a][p], tdy[b][p], k)
                           + _conv(dxp, wdxp, k) + _conv(dyp, wdyp, k))
                wsab[q][p].append(sab)
                pab[q][p].append(_conv(ts[a][p], ts[b][p], k))
                wab = alpha * _conv(zp, wsab[q][p], k) + aa1 * _conv(yp, pab[q][p], k)
                wwab[q][p].append(wab)
                twa, twb = tw[a][p], tw[b][p]
                gx = (_conv(wwab[q][p], dxp, k) + _conv(twa, tdx[b][p], k) + _conv(twb, tdx[a][p], k)
                      + _conv(wp, wdxp, k))
                gy = (_conv(wwab[q][p], dyp, k) + _conv(twa, tdy[b][p], k) + _conv(twb, tdy[a][p], k)
                      + _conv(wp, wdyp, k))
                wa = wacc[q]
                wa[2 * i] += m[j] * gx
                wa[2 * i + 1] += m[j] * gy
                wa[2 * j] -= m[i] * gx
                wa[2 * j + 1] -= m[i] * gy
        inv = arb(1) / (k + 1)
        for c in range(6):
            X[c].append(X[6 + c][k] * inv)
        for c in range(6):
            X[6 + c].append(acc[c] * inv)
        for blocks, accs in ((TT, tacc), (WW, wacc)):
            for a, T in enumerate(blocks):
                for c in range(6):
                    T[c].append(T[6 + c][k] * inv)
                for c in range(6):
                    T[6 + c].append(accs[a][c] * inv)
    return X, TT, WW


def field(x, tangents=(), masses=(1, 1, 1)):
    """벡터장 f(x) 와 접벡터장 Df(x)ξ (1차 계수)."""
    X, TT = coefficients(x, tangents, 1, masses)
    return [c[1] for c in X], [[c[1] for c in T] for T in TT]


def evaluate(coeffs, remainder, order, t):
    """Σ_{k≤order} c_k t^k + R t^(order+1) (Horner, t 는 공)."""
    out = []
    for comp, rem in zip(coeffs, remainder):
        acc = rem
        for k in range(order, -1, -1):
            acc = acc * t + comp[k]
        out.append(acc)
    return out
