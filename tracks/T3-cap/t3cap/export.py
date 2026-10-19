"""결과를 구간 끝점 JSON 으로. 끝점마다 정확 표현(16진 double 또는 p/q)과 바깥쪽 반올림 10진을 함께 쓴다."""
from __future__ import annotations

import dataclasses
import math
from decimal import ROUND_CEILING, ROUND_FLOOR, Context, Decimal
from fractions import Fraction
from typing import Any

from .exact import Interval, exact_hex_double, is_hex_text

DECIMAL_DIGITS = 20


def exact_text(q: Fraction) -> str:
    if q.denominator & (q.denominator - 1) == 0:
        f = float(q)
        if math.isfinite(f) and Fraction(f) == q:
            return f.hex()
    return f"{q.numerator}/{q.denominator}"


def decimal_text(q: Fraction, *, upper: bool) -> str:
    ctx = Context(prec=DECIMAL_DIGITS, rounding=ROUND_CEILING if upper else ROUND_FLOOR)
    return str(ctx.divide(Decimal(q.numerator), Decimal(q.denominator)))


def interval_json(iv: Interval) -> dict[str, str]:
    return {
        "lo": exact_text(iv.lo),
        "hi": exact_text(iv.hi),
        "lo_dec": decimal_text(iv.lo, upper=False),
        "hi_dec": decimal_text(iv.hi, upper=True),
    }


def to_json(obj: Any) -> Any:
    if isinstance(obj, Interval):
        return interval_json(obj)
    if isinstance(obj, Fraction):
        return {"exact": exact_text(obj), "dec": decimal_text(obj, upper=False)}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {f.name: to_json(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        out["type"] = type(obj).__name__
        return out
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    return obj


def interval_from_json(d: dict[str, str]) -> Interval:
    """exact 필드만 신뢰한다(10진 필드는 사람이 읽는 용도)."""
    return Interval(_parse_exact(d["lo"]), _parse_exact(d["hi"]))


def _parse_exact(text: str) -> Fraction:
    if is_hex_text(text):
        return exact_hex_double(text)
    return Fraction(text)
