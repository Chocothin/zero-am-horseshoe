"""syzygy 차트 ↔ 관성 상태, 그리고 단면 사상의 미분. 이원수(값 + 4방향 미분)로 값과 미분을 함께 감싼다.

차트 u = (r, x, p_r, p_x) 는 Jacobi 좌표(ρ₁ = q₂ − q₁, ρ₂ = q₃ − (q₁+q₂)/2, π₁ = ρ₁'/2, π₂ = (2/3)ρ₂')의
회전 불변량 (|ρ₁|, e·ρ₂, e·π₁, e·π₂), e = ρ₁/|ρ₁| 이다(등질량). 출발(Σ⁺)에서는 ρ₁ = (r,0), ρ₂ = (x,0),
π₁ = (p_r, −x p_y/r)(J=0), π₂ = (p_x, p_y), p_y = +√((h − C)/A) 로 들어올린다. 에너지 식은 아래에서 직접 유도했다:
E = |π₁|² + (3/4)|π₂|² − U = A p_y² + C, A = x²/r² + 3/4, C = p_r² + (3/4)p_x² − U.
"""
from __future__ import annotations

from flint import arb, arb_mat

ZERO = arb(0)


class D:
    """이원수: 값 v 와 방향 미분 g(리스트)."""

    __slots__ = ("v", "g")

    def __init__(self, v, g):
        self.v = v
        self.g = g

    @staticmethod
    def const(v, n):
        return D(v, [ZERO] * n)

    def __add__(self, o):
        if isinstance(o, D):
            return D(self.v + o.v, [a + b for a, b in zip(self.g, o.g)])
        return D(self.v + o, list(self.g))

    __radd__ = __add__

    def __neg__(self):
        return D(-self.v, [-a for a in self.g])

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, D):
            return D(self.v * o.v, [a * o.v + self.v * b for a, b in zip(self.g, o.g)])
        return D(self.v * o, [a * o for a in self.g])

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, D):
            inv = 1 / o.v
            q = self.v * inv
            return D(q, [(a - q * b) * inv for a, b in zip(self.g, o.g)])
        return D(self.v / o, [a / o for a in self.g])

    def __rtruediv__(self, o):
        inv = 1 / self.v
        q = o * inv
        return D(q, [-q * a * inv for a in self.g])

    def sqrt(self):
        if not self.v > 0:
            raise ArithmeticError("sqrt argument not proven positive")
        r = self.v.sqrt()
        return D(r, [a / (2 * r) for a in self.g])


def sqr(v):
    return v * v


def lift(u, energy):
    """차트 u(4, D) → 관성 상태 12성분(D). 판별식 D_T3 = 4A(h − C) 도 돌려준다(T3 정의와 같은 양)."""
    r, x, pr, px = u
    rho13 = (sqr(x + r / 2)).sqrt()      # |x + r/2|
    rho23 = (sqr(x - r / 2)).sqrt()      # |x − r/2|
    U = 1 / r + 1 / rho13 + 1 / rho23
    A = sqr(x) / sqr(r) + arb(3) / 4
    C = sqr(pr) + arb(3) / 4 * sqr(px) - U
    num = energy - C
    disc = 4 * A * num
    py = (num / A).sqrt()
    p1x, p1y = pr, -(x * py) / r
    p2x, p2y = px, py
    # q1 = −ρ₁/2 − ρ₂/3, q2 = ρ₁/2 − ρ₂/3, q3 = 2ρ₂/3 ; v1 = −π₁ − π₂/2, v2 = π₁ − π₂/2, v3 = π₂
    zero = r * 0                          # 값·미분이 정확히 0(구간 r 에서도 r − r 와 달리 넓어지지 않음)
    third = arb(1) / 3
    q1x = -r / 2 - x * third
    q2x = r / 2 - x * third
    q3x = 2 * x * third
    v1x = -p1x - p2x / 2
    v1y = -p1y - p2y / 2
    v2x = p1x - p2x / 2
    v2y = p1y - p2y / 2
    state = [q1x, zero, q2x, zero, q3x, zero, v1x, v1y, v2x, v2y, p2x, p2y]
    return state, disc


def chart_out(x):
    """관성 상태(D 또는 arb) → (r, x, p_r, p_x) = (|ρ₁|, e·ρ₂, e·π₁, e·π₂)."""
    r1x, r1y = x[2] - x[0], x[3] - x[1]
    r2x, r2y = x[4] - (x[0] + x[2]) / 2, x[5] - (x[1] + x[3]) / 2
    p1x, p1y = (x[8] - x[6]) / 2, (x[9] - x[7]) / 2
    p2x = (x[10] - (x[6] + x[8]) / 2) * 2 / 3
    p2y = (x[11] - (x[7] + x[9]) / 2) * 2 / 3
    r = (sqr(r1x) + sqr(r1y)).sqrt()
    ex, ey = r1x / r, r1y / r
    return [r, ex * r2x + ey * r2y, ex * p1x + ey * p1y, ex * p2x + ey * p2y]


def reduced_py_rate(x):
    """도착점에서 ∂H/∂p_y = dy/dt = (ds/dt)/r (s = r·y, y=0). T3 branch_out = branch·∂H/∂p_y 와 비교한다."""
    from .flow import syz_rate
    r1x, r1y = x[2] - x[0], x[3] - x[1]
    r = (sqr(r1x) + sqr(r1y)).sqrt()
    return syz_rate(x) / r


def poincare_derivative(xc, Tc, masses=(1, 1, 1)):
    """첫 통과 사상의 미분: d/dw x(τ(w); w) = T + f·∂τ/∂w, ∂τ/∂w = −(∇s·T)/(∇s·f). 반환 (u', DP̃ 4×m)."""
    from .flow import syz_grad
    from .taylor import field
    fx, _ = field(xc, [], masses)
    g = syz_grad(xc)
    sf = sum((gi * fi for gi, fi in zip(g, fx)), ZERO)
    m = len(Tc)
    cols = []
    for a in range(m):
        sT = sum((gi * ti for gi, ti in zip(g, Tc[a])), ZERO)
        dtau = -sT / sf
        cols.append([ti + fi * dtau for ti, fi in zip(Tc[a], fx)])
    xd = [D(xc[c], [cols[a][c] for a in range(m)]) for c in range(12)]
    out = chart_out(xd)
    u = [o.v for o in out]
    Dm = [[o.g[a] for a in range(m)] for o in out]
    return u, Dm


def solve(M, rhs_cols):
    """M⁻¹ · rhs (공 행렬, 엄밀 선형 해)."""
    A = arb_mat(M)
    B = arb_mat(rhs_cols)
    return A.solve(B)
