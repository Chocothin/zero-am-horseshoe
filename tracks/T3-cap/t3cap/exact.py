"""정확 유리수 끝점 구간. 엔진 출력(16진 double·방향 반올림 10진)을 손실 없이 받아 판정층에서 반올림 없이 계산한다."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Iterable, Sequence, Union

RationalLike = Union[int, Fraction, str]
DOUBLE_EXACT_LIMIT = 1 << 53
_HEX_FLOAT = re.compile(r"([+-]?)0[xX]([0-9a-fA-F]*)(?:\.([0-9a-fA-F]*))?(?:[pP]([+-]?[0-9]{1,7}))?")


def is_hex_text(text: str) -> bool:
    return text.lower().lstrip("+-").startswith("0x")


def exact_hex_double(text: str) -> Fraction:
    """16진 double 문자열('0x1.8p+1')을 정확 유리수로. double 로 정확히 나타나지 않는 토큰(53비트 초과·범위 밖)은
    float.fromhex 처럼 최근접으로 반올림하지 않고 ValueError 로 거부한다."""
    m = _HEX_FLOAT.fullmatch(text)
    if m is None or len(text) > 64 or not (m.group(2) or m.group(3)):
        raise ValueError(f"not a hex float: {text!r}")
    sign, whole, frac, exponent = m.group(1), m.group(2), m.group(3) or "", int(m.group(4) or "0")
    mantissa = int(whole + frac, 16)
    try:
        nearest = Fraction(float.fromhex(text))
    except OverflowError as exc:
        raise ValueError(f"hex token {text!r} is outside the double range") from exc
    # 0 으로 내려앉는 토큰을 먼저 걸러 아래 시프트의 지수가 double 범위 근처로 묶이게 한다
    if nearest == 0 and mantissa != 0:
        raise ValueError(f"hex token {text!r} underflows to zero")
    scale = exponent - 4 * len(frac)
    value = Fraction(mantissa << scale) if scale >= 0 else Fraction(mantissa, 1 << -scale)
    if (-value if sign == "-" else value) != nearest:
        raise ValueError(f"hex token {text!r} is not exactly a double (it would be rounded)")
    return nearest


def rational(value: RationalLike) -> Fraction:
    """정수·Fraction·문자열('1/3', '-0.5', '0x1.8p+1')을 정확 유리수로. float 는 함정 ① 이라 받지 않는다."""
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError("float is not accepted: build constants from strings or Fractions (trap 1)")
    if isinstance(value, (int, Fraction)):
        return Fraction(value)
    if isinstance(value, str):
        text = value.strip()
        if is_hex_text(text):
            return exact_hex_double(text)
        return Fraction(Decimal(text)) if "/" not in text else Fraction(text)
    raise TypeError(f"unsupported rational input: {type(value).__name__}")


def round_down_double(q: Fraction) -> float:
    f = float(q)
    return f if Fraction(f) <= q else math.nextafter(f, -math.inf)


def round_up_double(q: Fraction) -> float:
    f = float(q)
    return f if Fraction(f) >= q else math.nextafter(f, math.inf)


def engine_token(q: Fraction, *, upper: bool) -> str:
    """엔진 끝점 토큰: 53비트 안의 유리수는 'p/q' 그대로(엔진이 자기 산술로 감싼다), 아니면 바깥쪽 16진 double."""
    if abs(q.numerator) <= DOUBLE_EXACT_LIMIT and q.denominator <= DOUBLE_EXACT_LIMIT:
        return f"{q.numerator}/{q.denominator}"
    return (round_up_double(q) if upper else round_down_double(q)).hex()


def parse_engine_endpoint(token: str) -> Fraction:
    if is_hex_text(token):
        return exact_hex_double(token)
    return Fraction(Decimal(token))


@dataclass(frozen=True)
class Interval:
    lo: Fraction
    hi: Fraction

    def __post_init__(self) -> None:
        if not isinstance(self.lo, Fraction) or not isinstance(self.hi, Fraction):
            raise TypeError("Interval endpoints must be Fraction")
        if self.lo > self.hi:
            raise ValueError(f"empty interval [{self.lo}, {self.hi}]")

    @staticmethod
    def point(value: RationalLike) -> "Interval":
        q = rational(value)
        return Interval(q, q)

    @staticmethod
    def of(lo: RationalLike, hi: RationalLike) -> "Interval":
        return Interval(rational(lo), rational(hi))

    @staticmethod
    def around(center: RationalLike, radius: RationalLike) -> "Interval":
        c, r = rational(center), rational(radius)
        return Interval(c - r, c + r)

    @staticmethod
    def from_engine(lo: str, hi: str) -> "Interval":
        return Interval(parse_engine_endpoint(lo), parse_engine_endpoint(hi))

    @property
    def width(self) -> Fraction:
        return self.hi - self.lo

    @property
    def mid(self) -> Fraction:
        return (self.lo + self.hi) / 2

    @property
    def radius(self) -> Fraction:
        return self.width / 2

    def contains(self, q: Fraction) -> bool:
        return self.lo <= q <= self.hi

    def contains_interval(self, other: "Interval") -> bool:
        return self.lo <= other.lo and other.hi <= self.hi

    def strictly_contains_interval(self, other: "Interval") -> bool:
        return self.lo < other.lo and other.hi < self.hi

    def hull(self, other: "Interval") -> "Interval":
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    def magnitude(self) -> Fraction:
        return max(abs(self.lo), abs(self.hi))

    def __add__(self, other: "Interval") -> "Interval":
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __sub__(self, other: "Interval") -> "Interval":
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def __neg__(self) -> "Interval":
        return Interval(-self.hi, -self.lo)

    def __mul__(self, other: "Interval") -> "Interval":
        products = (self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi)
        return Interval(min(products), max(products))

    def scale(self, q: Fraction) -> "Interval":
        return Interval(min(self.lo * q, self.hi * q), max(self.lo * q, self.hi * q))

    def engine_tokens(self) -> str:
        return f"{engine_token(self.lo, upper=False)} {engine_token(self.hi, upper=True)}"

    def __repr__(self) -> str:
        return f"[{float(self.lo):.17g}, {float(self.hi):.17g}]"


Box = tuple[Interval, ...]
IntervalMatrix = tuple[tuple[Interval, ...], ...]
RationalMatrix = tuple[tuple[Fraction, ...], ...]


def box(*intervals: Interval) -> Box:
    return tuple(intervals)


def box_around(center: Sequence[RationalLike], radius: Sequence[RationalLike] | RationalLike) -> Box:
    radii = list(radius) if isinstance(radius, (list, tuple)) else [radius] * len(center)
    return tuple(Interval.around(c, r) for c, r in zip(center, radii))


def point_box(center: Sequence[RationalLike]) -> Box:
    return tuple(Interval.point(c) for c in center)


def box_mid(b: Box) -> tuple[Fraction, ...]:
    return tuple(i.mid for i in b)


def box_contains_point(b: Box, p: Sequence[Fraction]) -> bool:
    return all(i.contains(q) for i, q in zip(b, p, strict=True))


def identity(n: int) -> RationalMatrix:
    return tuple(tuple(Fraction(int(i == j)) for j in range(n)) for i in range(n))


def matrix_inverse(m: RationalMatrix) -> RationalMatrix:
    """정확 유리수 Gauss-Jordan. 특이 행렬이면 ValueError."""
    n = len(m)
    a = [list(row) + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(m)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if a[r][col] != 0), None)
        if pivot is None:
            raise ValueError("singular matrix")
        a[col], a[pivot] = a[pivot], a[col]
        inv = 1 / a[col][col]
        a[col] = [v * inv for v in a[col]]
        for r in range(n):
            if r != col and a[r][col] != 0:
                factor = a[r][col]
                a[r] = [v - factor * w for v, w in zip(a[r], a[col])]
    return tuple(tuple(row[n:]) for row in a)


def rational_matrix_from_floats(rows: Iterable[Iterable[float]]) -> RationalMatrix:
    """부동소수 행렬(예: 근사 역행렬)을 그 double 값 그대로의 정확 유리수 행렬로. 판정 결과는 이 행렬 선택과 무관하게 엄밀하다."""
    return tuple(tuple(Fraction(float(v)) for v in row) for row in rows)


def imat_vec(m: IntervalMatrix, v: Box) -> Box:
    return tuple(_sum(m_ij * v_j for m_ij, v_j in zip(row, v, strict=True)) for row in m)


def rmat_ivec(m: RationalMatrix, v: Box) -> Box:
    return tuple(_sum(v_j.scale(m_ij) for m_ij, v_j in zip(row, v, strict=True)) for row in m)


def rmat_imat(a: RationalMatrix, b: IntervalMatrix) -> IntervalMatrix:
    if any(len(row) != len(b) for row in a):
        raise ValueError("inner dimension mismatch")
    cols = len(b[0])
    return tuple(
        tuple(_sum(b[k][j].scale(a[i][k]) for k in range(len(b))) for j in range(cols)) for i in range(len(a))
    )


def _sum(items: Iterable[Interval]) -> Interval:
    total = Interval(Fraction(0), Fraction(0))
    for item in items:
        total = total + item
    return total
