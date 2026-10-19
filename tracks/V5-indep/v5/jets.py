"""2계 제트(값, 기울기, 헤세)의 공 산술. 값이 구간이면 그 구간의 모든 점에서 기울기·헤세를 감싼다(연쇄 법칙의 포함 단조성)."""
from __future__ import annotations

from flint import arb

ZERO = arb(0)


class J2:
    __slots__ = ("v", "g", "H")

    def __init__(self, v, g, H):
        self.v, self.g, self.H = v, g, H

    @property
    def n(self):
        return len(self.g)

    @staticmethod
    def var(v, g):
        n = len(g)
        return J2(v, list(g), [[ZERO] * n for _ in range(n)])

    def _lift(self, o):
        return o if isinstance(o, J2) else J2(o if isinstance(o, arb) else arb(o), [ZERO] * self.n,
                                              [[ZERO] * self.n for _ in range(self.n)])

    def __add__(self, o):
        o = self._lift(o)
        n = self.n
        return J2(self.v + o.v, [self.g[i] + o.g[i] for i in range(n)],
                  [[self.H[i][j] + o.H[i][j] for j in range(n)] for i in range(n)])

    __radd__ = __add__

    def __neg__(self):
        n = self.n
        return J2(-self.v, [-a for a in self.g], [[-self.H[i][j] for j in range(n)] for i in range(n)])

    def __sub__(self, o):
        return self + (-self._lift(o))

    def __rsub__(self, o):
        return self._lift(o) + (-self)

    def __mul__(self, o):
        n = self.n
        if not isinstance(o, J2):
            c = o if isinstance(o, arb) else arb(o)
            return J2(self.v * c, [a * c for a in self.g], [[self.H[i][j] * c for j in range(n)] for i in range(n)])
        a, b = self, o
        return J2(a.v * b.v, [a.g[i] * b.v + a.v * b.g[i] for i in range(n)],
                  [[a.H[i][j] * b.v + a.g[i] * b.g[j] + a.g[j] * b.g[i] + a.v * b.H[i][j] for j in range(n)]
                   for i in range(n)])

    __rmul__ = __mul__

    def recip(self):
        n = self.n
        r = 1 / self.v
        r2 = r * r
        r3 = r2 * r
        return J2(r, [-a * r2 for a in self.g],
                  [[2 * self.g[i] * self.g[j] * r3 - self.H[i][j] * r2 for j in range(n)] for i in range(n)])

    def __truediv__(self, o):
        if not isinstance(o, J2):
            c = o if isinstance(o, arb) else arb(o)
            return self * (1 / c)
        return self * o.recip()

    def __rtruediv__(self, o):
        return self.recip() * (o if isinstance(o, arb) else arb(o))

    def sqrt(self):
        if not self.v > 0:
            raise ArithmeticError("sqrt argument not proven positive")
        n = self.n
        r = self.v.sqrt()
        r3 = r * r * r
        return J2(r, [a / (2 * r) for a in self.g],
                  [[self.H[i][j] / (2 * r) - self.g[i] * self.g[j] / (4 * r3) for j in range(n)] for i in range(n)])
