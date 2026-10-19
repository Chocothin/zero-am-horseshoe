"""[재계산·엄밀] V2-compact 독립 흐름 관 인증: 엔진·CAPD 없이 python-flint arb(공 산술)로 직접 쓴 Taylor 적분기.

방법
- 상자 B=[−1,1]⁴ 의 점 w 를 u=c+M·w 로 차트에 놓고 Σ⁺ 로 들어올린 뒤 관성 좌표(질량 1, θ=0 틀)로 옮긴다.
- 상태를 w 에 대한 1차 꼴 x(w) ∈ c + g·w + r(아핀 꼴, g 는 4성분, r 은 나머지 공)로 들고 다닌다.
  곱·함수는 2차 항을 r 에 바깥쪽으로 넣으므로 모든 w∈B 에서 포함이 유지된다(래핑은 2차 항에만 생긴다).
- 단계마다 Taylor 계수(자동 미분 점화식, 차수 n)를 아핀 꼴로 계산하고, Lagrange 나머지는
  1차 Picard 사전 감싸기 E(X₀+[0,h]·f(E)⊆E)에서 잰 (n+1)차 계수로 넣는다.
- r_ij² 과 syzygy 지시자 A=(q₂−q₁)×(q₃−q₁)=ρ₁×ρ₂ 의 단계 다항식을 부분 구간마다 평균값 형식으로 감싸
  흐름 관 r_ij 상·하한과 (0,τ) 의 A>0, 교차 창의 Ȧ<0 을 인증한다.
원래 트랙 코드(t3cap·t6hs)와 CAPD 를 쓰지 않는다. 입력은 graph_design.json 의 h-set 원자료뿐이다.
"""
from __future__ import annotations

import math
from fractions import Fraction

from flint import arb, ctx

ctx.prec = 128
UNIT = arb(0, 1)          # [−1,1] 을 담는 공
PAIRS = ((0, 1), (0, 2), (1, 2))
ZERO = arb(0)


def ball_hull(lo: arb, hi: arb) -> arb:
    return lo.union(hi)


def interval(a: Fraction, b: Fraction) -> arb:
    """[a,b] 를 담는 공(끝점 정확 유리수)."""
    return arb(a.numerator) / a.denominator if a == b else (
        (arb(a.numerator) / a.denominator).union(arb(b.numerator) / b.denominator))


def qarb(x: Fraction) -> arb:
    return arb(x.numerator) / x.denominator


# ---------------------------------------------------------------- w 에 대한 아핀 꼴

class Aff:
    __slots__ = ("c", "g", "r", "_s")

    def __init__(self, c: arb, g, r: arb):
        self.c = c
        self.g = g
        self.r = r
        self._s = None

    @staticmethod
    def const(x) -> "Aff":
        return Aff(x if isinstance(x, arb) else arb(x), (ZERO, ZERO, ZERO, ZERO), ZERO)

    def sig(self) -> arb:
        """g·w 의 범위 [−σ,σ] (w∈[−1,1]⁴)."""
        if self._s is None:
            g = self.g
            self._s = UNIT * (abs(g[0]) + abs(g[1]) + abs(g[2]) + abs(g[3]))
        return self._s

    def ball(self) -> arb:
        return self.c + self.sig() + self.r

    def __add__(self, o):
        if isinstance(o, Aff):
            a, b = self.g, o.g
            return Aff(self.c + o.c, (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]), self.r + o.r)
        return Aff(self.c + o, self.g, self.r)

    __radd__ = __add__

    def __sub__(self, o):
        if isinstance(o, Aff):
            a, b = self.g, o.g
            return Aff(self.c - o.c, (a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]), self.r - o.r)
        return Aff(self.c - o, self.g, self.r)

    def __rsub__(self, o):
        return Aff(o - self.c, tuple(-x for x in self.g), -self.r)

    def __neg__(self):
        return Aff(-self.c, tuple(-x for x in self.g), -self.r)

    def scale(self, k) -> "Aff":
        g = self.g
        return Aff(self.c * k, (g[0] * k, g[1] * k, g[2] * k, g[3] * k), self.r * k)

    def __mul__(self, o):
        if not isinstance(o, Aff):
            return self.scale(o)
        return aff_dot(((self, o),))

    __rmul__ = __mul__


def aff_dot(pairs, coeffs=None) -> Aff:
    """Σ k_l·a_l·b_l 를 한 번에(중간 객체 없이) 아핀 꼴로. 2차 교차항은 나머지 공에 넣는다."""
    c = ZERO
    g0 = g1 = g2 = g3 = ZERO
    r = ZERO
    for idx, (a, b) in enumerate(pairs):
        k = None if coeffs is None else coeffs[idx]
        ac, bc, ag, bg = a.c, b.c, a.g, b.g
        sa, sb = a.sig(), b.sig()
        cc = ac * bc
        t0 = ac * bg[0] + bc * ag[0]
        t1 = ac * bg[1] + bc * ag[1]
        t2 = ac * bg[2] + bc * ag[2]
        t3 = ac * bg[3] + bc * ag[3]
        rr = ac * b.r + bc * a.r + sa * sb + sa * b.r + sb * a.r + a.r * b.r
        if k is not None:
            cc, t0, t1, t2, t3, rr = cc * k, t0 * k, t1 * k, t2 * k, t3 * k, rr * k
        c += cc
        g0 += t0
        g1 += t1
        g2 += t2
        g3 += t3
        r += rr
    return Aff(c, (g0, g1, g2, g3), r)


def aff_func(a: Aff, f, f1, f2) -> Aff:
    """f(c+δ) = f(c) + f'(c)δ + f''(ξ)δ²/2, ξ∈c+D, δ∈D. f·f1·f2 는 arb 공에서의 값·1계·2계 도함수."""
    D = a.sig() + a.r
    d1 = f1(a.c)
    rng = a.c + D
    r = d1 * a.r + f2(rng) * (D * D) / 2
    g = a.g
    return Aff(f(a.c), (d1 * g[0], d1 * g[1], d1 * g[2], d1 * g[3]), r)


def aff_inv(a: Aff) -> Aff:
    return aff_func(a, lambda x: 1 / x, lambda x: -1 / (x * x), lambda x: 2 / (x * x * x))


def aff_sqrt(a: Aff) -> Aff:
    return aff_func(a, lambda x: x.sqrt(), lambda x: 1 / (2 * x.sqrt()), lambda x: -1 / (4 * x * x.sqrt()))


def aff_pow_m32(a: Aff) -> Aff:
    return aff_func(a, lambda x: x.rsqrt() ** 3, lambda x: arb(-3) / 2 * x.rsqrt() ** 5,
                    lambda x: arb(15) / 4 * x.rsqrt() ** 7)


def aff_pow_mhalf(a: Aff) -> Aff:
    return aff_func(a, lambda x: x.rsqrt(), lambda x: arb(-1) / 2 * x.rsqrt() ** 3,
                    lambda x: arb(3) / 4 * x.rsqrt() ** 5)


# ---------------------------------------------------------------- Taylor 계수 점화식(아핀 꼴·공 공용)

class AffOps:
    zero = Aff.const(0)

    @staticmethod
    def dot(pairs, coeffs=None):
        return aff_dot(pairs, coeffs)

    @staticmethod
    def pow_m32(x):
        return aff_pow_m32(x)

    @staticmethod
    def inv(x):
        return aff_inv(x)

    @staticmethod
    def scale(x, k):
        return x.scale(k)


class ArbOps:
    zero = ZERO

    @staticmethod
    def dot(pairs, coeffs=None):
        s = ZERO
        for idx, (a, b) in enumerate(pairs):
            s += a * b if coeffs is None else coeffs[idx] * a * b
        return s

    @staticmethod
    def pow_m32(x):
        if not x > 0:
            raise ArithmeticError("distance square not positive on the enclosure")
        return x.rsqrt() ** 3

    @staticmethod
    def inv(x):
        return 1 / x

    @staticmethod
    def scale(x, k):
        return x * k


INV_INT = [None] + [arb(1) / k for k in range(1, 200)]


def taylor(q, v, order, ops):
    """q(t+s)=Σ Q[c][k] s^k. 쌍 거리 제곱 S[p], 지시자 A=(q₂−q₁)×(q₃−q₁) 의 계수도 돌려준다(차수 order 까지)."""
    Q = [[q[c]] for c in range(6)]
    V = [[v[c]] for c in range(6)]
    dx = [[], [], []]
    dy = [[], [], []]
    S = [[], [], []]
    W = [[], [], []]
    invS0 = [None, None, None]
    for k in range(order + 1):
        for p, (i, j) in enumerate(PAIRS):
            dx[p].append(Q[2 * j][k] - Q[2 * i][k])
            dy[p].append(Q[2 * j + 1][k] - Q[2 * i + 1][k])
            X, Y = dx[p], dy[p]
            S[p].append(ops.dot([(X[l], X[k - l]) for l in range(k + 1)] + [(Y[l], Y[k - l]) for l in range(k + 1)]))
        if k == order:
            break
        acc = [ops.zero] * 6
        for p, (i, j) in enumerate(PAIRS):
            if k == 0:
                W[p].append(ops.pow_m32(S[p][0]))
                invS0[p] = ops.inv(S[p][0])
            else:
                # w=s^{-3/2}: k s₀ w_k = Σ_{l=1}^k (αl − (k−l)) s_l w_{k−l}, α=−3/2 → 계수 −(l+2k)/2
                coeffs = [arb(-(l + 2 * k)) / 2 for l in range(1, k + 1)]
                tot = ops.dot([(S[p][l], W[p][k - l]) for l in range(1, k + 1)], coeffs)
                W[p].append(ops.scale(ops.dot([(tot, invS0[p])]), INV_INT[k]))
            fx = ops.dot([(dx[p][l], W[p][k - l]) for l in range(k + 1)])
            fy = ops.dot([(dy[p][l], W[p][k - l]) for l in range(k + 1)])
            acc[2 * i] = acc[2 * i] + fx
            acc[2 * i + 1] = acc[2 * i + 1] + fy
            acc[2 * j] = acc[2 * j] - fx
            acc[2 * j + 1] = acc[2 * j + 1] - fy
        for c in range(6):
            Q[c].append(ops.scale(V[c][k], INV_INT[k + 1]))
            V[c].append(ops.scale(acc[c], INV_INT[k + 1]))
    A = [ops.dot([(dx[0][l], dy[1][k - l]) for l in range(k + 1)])
         - ops.dot([(dy[0][l], dx[1][k - l]) for l in range(k + 1)]) for k in range(order + 1)]
    return Q, V, S, A


# ---------------------------------------------------------------- 사전 감싸기(1차 Picard)

def vector_field_box(x):
    """x=(q₁x,q₁y,q₂x,q₂y,q₃x,q₃y, v…) 공 → (v, a)."""
    q, v = x[:6], x[6:]
    acc = [ZERO] * 6
    for i, j in PAIRS:
        dxx = q[2 * j] - q[2 * i]
        dyy = q[2 * j + 1] - q[2 * i + 1]
        s = dxx * dxx + dyy * dyy
        if not s > 0:
            raise ArithmeticError("collision not excluded on the enclosure")
        w = s.rsqrt() ** 3
        acc[2 * i] += dxx * w
        acc[2 * i + 1] += dyy * w
        acc[2 * j] -= dxx * w
        acc[2 * j + 1] -= dyy * w
    return list(v) + acc


def inflate(b: arb, factor: float = 1.5, eps: float = 1e-30) -> arb:
    m = b.mid()
    rad = b.rad()
    return m + UNIT * (arb(rad) * arb(factor) + arb(eps))


def apriori(X0, h: arb):
    """X0+[0,h]·f(E) ⊆ E 인 E 를 찾는다(Picard–Lindelöf: 해가 [0,h] 동안 E 안에 머문다)."""
    I = arb(0).union(h)
    f0 = vector_field_box(X0)
    E = [inflate(x + I * fx) for x, fx in zip(X0, f0)]
    for _ in range(12):
        fE = vector_field_box(E)
        new = [x + I * fx for x, fx in zip(X0, fE)]
        if all(e.contains(n) for e, n in zip(E, new)):
            # E 는 검증됐다. 더 좁은 new 도 직접 검증되면 그것을 쓴다(공 산술은 포함 단조를 글자 그대로 보장하지 않는다)
            for _ in range(2):
                f_new = vector_field_box(new)
                newer = [x + I * fx for x, fx in zip(X0, f_new)]
                if not all(nw.contains(nr) for nw, nr in zip(new, newer)):
                    break
                E, new = new, newer
            return E
        # 포함이 깨진 성분만 넓힌다(모든 성분을 넓히면 위치 상자가 커져 가속도 상자가 따라 커지며 발산한다)
        E = [e if e.contains(n) else inflate(n.union(e)) for n, e in zip(new, E)]
    return None


# ---------------------------------------------------------------- 다항식 범위

def horner(coeffs, s):
    acc = ZERO
    for c in reversed(coeffs):
        acc = acc * s + c
    return acc


def poly_range(coeffs, a: arb, b: arb, rem: arb, rem_power: int):
    """Σ coeffs_j s^j + rem·s^{rem_power} 의 s∈[a,b] 위 범위(평균값 형식 + 나머지)."""
    m = (a + b) / 2
    S = a.union(b)
    der = [j * coeffs[j] for j in range(1, len(coeffs))]
    val = horner(coeffs, m) + horner(der, S) * (S - m)
    return val + rem * S ** rem_power


# ---------------------------------------------------------------- 들어올림

MU1 = arb(1) / 2
MU2 = arb(2) / 3


def initial_state(center, matrix, h_energy=Fraction(-1, 2)):
    """상자 B 전체를 아핀 꼴 관성 상태로: u=c+M w → Σ⁺ 들어올림(J=0) → q, v. 판별식 D 의 공도 돌려준다."""
    u = [Aff(qarb(center[i]), tuple(qarb(matrix[i][j]) for j in range(4)), ZERO) for i in range(4)]
    r, x, pr, px = u
    inv_r = aff_inv(r)
    qa = Aff.const(arb(3) / 4) + x * x * inv_r * inv_r            # 1/(2μ₂) + x²/(2μ₁ r²)
    U = inv_r + aff_pow_mhalf((x + r.scale(arb(1) / 2)) * (x + r.scale(arb(1) / 2))) \
        + aff_pow_mhalf((x - r.scale(arb(1) / 2)) * (x - r.scale(arb(1) / 2)))
    qc = pr * pr + (px * px).scale(arb(3) / 4) - U                # p_r²/(2μ₁) + p_x²/(2μ₂) − U
    disc = (qa * (qc - qarb(h_energy))).scale(arb(-4))
    if not disc.ball() > 0:
        raise ArithmeticError("chart discriminant not certified positive")
    py = aff_sqrt(disc) * aff_inv(qa).scale(arb(1) / 2)
    L = -(x * py)                                                  # J − x p_y + y p_x, y=0, J=0
    rd1 = (pr.scale(2), (L * inv_r).scale(2))                      # ρ̇₁ = π₁/μ₁
    rd2 = (px.scale(arb(3) / 2), py.scale(arb(3) / 2))             # ρ̇₂ = π₂/μ₂
    zero = Aff.const(0)
    rho1 = (r, zero)
    rho2 = (x, zero)
    third = arb(1) / 3
    q = [(-rho1[0]).scale(arb(1) / 2) - rho2[0].scale(third), (-rho1[1]).scale(arb(1) / 2) - rho2[1].scale(third),
         rho1[0].scale(arb(1) / 2) - rho2[0].scale(third), rho1[1].scale(arb(1) / 2) - rho2[1].scale(third),
         rho2[0].scale(2 * third), rho2[1].scale(2 * third)]
    v = [(-rd1[0]).scale(arb(1) / 2) - rd2[0].scale(third), (-rd1[1]).scale(arb(1) / 2) - rd2[1].scale(third),
         rd1[0].scale(arb(1) / 2) - rd2[0].scale(third), rd1[1].scale(arb(1) / 2) - rd2[1].scale(third),
         rd2[0].scale(2 * third), rd2[1].scale(2 * third)]
    return q, v, disc.ball()


# ---------------------------------------------------------------- 한 변의 비행

def float_center_plan(q, v):
    """계획용(비엄밀) 중심 상태 float."""
    return [float(a.c.mid()) for a in q], [float(a.c.mid()) for a in v]


def step_estimate(Q, V, order):
    def norm(m):
        return max(abs(float(Q[c][m].c.mid())) for c in range(6)) + max(abs(float(V[c][m].c.mid())) for c in range(6))
    rhos = [(1 / norm(m)) ** (1 / m) for m in (order - 1, order) if norm(m) > 0]
    return min(rhos)


def dyadic_down(x: float, bits: int = 30) -> Fraction:
    e = math.floor(math.log2(x))
    scale = 2 ** (bits - e)
    return Fraction(math.floor(x * scale), scale)


class FlightFailure(RuntimeError):
    pass


def classify(A_aff, Arem, a: Fraction, b: Fraction, divided: bool = False):
    """[a,b] 위 A(또는 A/s) 의 부호: +1, −1, 0(미정). A 의 w 의존은 1차 꼴로 유지한다."""
    sa, sb = qarb(a), qarb(b)
    if divided:
        cs = [x.c for x in A_aff[1:]]
        gs = [[x.g[i] for x in A_aff[1:]] for i in range(4)]
        rs = [x.r for x in A_aff[1:]]
        power = len(A_aff) - 1
    else:
        cs = [x.c for x in A_aff]
        gs = [[x.g[i] for x in A_aff] for i in range(4)]
        rs = [x.r for x in A_aff]
        power = len(A_aff)
    S = sa.union(sb)
    val = poly_range(cs, sa, sb, Arem, power)
    for i in range(4):
        val += UNIT * abs(horner(gs[i], S))
    val += horner(rs, S)
    if val > 0:
        return 1, val
    if val < 0:
        return -1, val
    return 0, val


def adot_range(A_aff, Arem, a: Fraction, b: Fraction):
    """Ȧ(t+s) = Σ_{j<n}(j+1)A_{j+1}s^j + (n+1)A^{[n+1]}(E)·[0,s]^n 의 s∈[a,b] 범위."""
    n = len(A_aff) - 1
    S = qarb(a).union(qarb(b))
    der = [(j + 1) * A_aff[j + 1].ball() for j in range(n)]
    return horner(der, S) + (n + 1) * Arem * S ** n


def fly(center, matrix, tau_plan: float, order: int = 18, kappa: float = 0.08, sub: int = 32, max_steps: int = 20000):
    """상자 전체의 첫 syzygy 까지 흐름 관과 귀환 창을 인증한다. 실패하면 FlightFailure."""
    q, v, disc = initial_state(center, matrix)
    t = Fraction(0)
    rlo = [None, None, None]
    rhi = [None, None, None]
    first = True
    steps = 0
    planned_cross = False
    while steps < max_steps:
        Q, V, S, A = taylor(q, v, order, AffOps)
        if first and not (A[0].c.is_zero() and A[0].r.is_zero() and all(x.is_zero() for x in A[0].g)):
            raise FlightFailure("initial set is not exactly on the section (A_0 != 0)")
        rho = step_estimate(Q, V, order)
        hn = rho * kappa
        d = tau_plan - float(t)
        if d > 1.5 * hn:
            h = dyadic_down(hn)
            crossing = False
        elif d > 0.6 * hn:
            h = dyadic_down(d - 0.5 * hn)
            crossing = False
        else:
            h = dyadic_down(2 * d)
            crossing = True
        harb = qarb(h)
        X0 = [x.ball() for x in q] + [x.ball() for x in v]
        E = apriori(X0, harb)
        tries = 0
        while E is None:
            if crossing or tries > 6:
                raise FlightFailure(f"a priori enclosure failed at t={float(t)} h={float(h)}")
            h = h / 2
            harb = qarb(h)
            E = apriori(X0, harb)
            tries += 1
        QE, VE, SE, AE = taylor(E[:6], E[6:], order + 1, ArbOps)
        remQ = [QE[c][order + 1] for c in range(6)]
        remV = [VE[c][order + 1] for c in range(6)]
        remS = [SE[p][order + 1] for p in range(3)]
        remA = AE[order + 1]
        Iord = arb(0).union(harb) ** (order + 1)
        # 지시자: 교차 단계가 아니면 (0,h] 전체에서 A>0(첫 단계는 A/s>0)
        s_end = h
        window = None
        if not crossing:
            for m in range(sub):
                a, b = h * m / sub, h * (m + 1) / sub
                sign, val = classify(A, remA, a, b, divided=first)
                if sign != 1:
                    raise FlightFailure(f"syzygy indicator not certified positive at t={float(t)}+[{float(a)},{float(b)}]")
        else:
            if first:
                raise FlightFailure("crossing planned in the first step")
            final = []
            for m in range(sub):
                a, b = h * m / sub, h * (m + 1) / sub
                final.append((a, b, classify(A, remA, a, b)[0]))
            undecided = [k for k, (_, _, sg) in enumerate(final) if sg == 0]
            if not undecided:
                raise FlightFailure(f"no undecided piece in crossing step at t={float(t)}")
            i0, i1 = undecided[0], undecided[-1]
            if not (all(sg == 1 for _, _, sg in final[:i0]) and all(sg == -1 for _, _, sg in final[i1 + 1:])
                    and i0 > 0 and i1 < sub - 1):
                raise FlightFailure(f"crossing structure not certified at t={float(t)}")
            tol = h * Fraction(1, 2**48)
            # 창 왼끝: 가장 왼쪽 미정 조각을 이분한다(왼쪽 반이 양이면 오른쪽 반으로)
            a, b = final[i0][0], final[i0][1]
            while b - a > tol:
                mid = (a + b) / 2
                sg = classify(A, remA, a, mid)[0]
                if sg == 1:
                    a = mid
                elif sg == 0:
                    b = mid
                else:
                    raise FlightFailure("negative piece left of the crossing window")
            s_lo = a
            # 창 오른끝: 가장 오른쪽 미정 조각을 이분한다(오른쪽 반이 음이면 왼쪽 반으로)
            a, b = final[i1][0], final[i1][1]
            while b - a > tol:
                mid = (a + b) / 2
                sg = classify(A, remA, mid, b)[0]
                if sg == -1:
                    b = mid
                elif sg == 0:
                    a = mid
                else:
                    raise FlightFailure("positive piece right of the crossing window")
            s_hi = b
            adot = adot_range(A, remA, s_lo, s_hi)
            if not adot < 0:
                raise FlightFailure("crossing not certified transversal (dA/dt may vanish)")
            window = (t + s_lo, t + s_hi)
            s_end = s_hi
        # 흐름 관: [0, s_end] 를 sub 조각으로
        for p in range(3):
            coeffs = [x.ball() for x in S[p]]
            for m in range(sub):
                a, b = s_end * m / sub, s_end * (m + 1) / sub
                rng = poly_range(coeffs, qarb(a), qarb(b), remS[p], order + 1)
                lo = rng.lower()
                hi = rng.upper()
                if not lo > 0:
                    raise FlightFailure("r_ij^2 lower bound not positive")
                rlo[p] = lo if rlo[p] is None or lo < rlo[p] else rlo[p]
                rhi[p] = hi if rhi[p] is None or hi > rhi[p] else rhi[p]
        steps += 1
        if window is not None:
            return {"window": window, "r2_lower": rlo, "r2_upper": rhi, "steps": steps, "disc": disc}
        # 상태 갱신: x(t+h) ∈ Σ Q_k h^k + rem·[0,h]^{n+1}
        hp = [arb(1)]
        for _ in range(order):
            hp.append(hp[-1] * harb)
        newq = []
        newv = []
        for c in range(6):
            accq = aff_dot([(Q[c][k], Aff.const(hp[k])) for k in range(order + 1)])
            accv = aff_dot([(V[c][k], Aff.const(hp[k])) for k in range(order + 1)])
            newq.append(Aff(accq.c, accq.g, accq.r + remQ[c] * Iord))
            newv.append(Aff(accv.c, accv.g, accv.r + remV[c] * Iord))
        q, v = newq, newv
        t += h
        first = False
    raise FlightFailure("too many steps")
