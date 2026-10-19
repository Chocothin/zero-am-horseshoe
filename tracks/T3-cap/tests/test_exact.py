"""정확 유리수 구간 연산의 명세 시험: 결과는 모든 원소 조합의 값을 담고, 반올림이 없으므로 끝점이 실제 극값과 같다."""
from __future__ import annotations

import itertools
import random
import sys
from fractions import Fraction

import pytest

from t3cap.exact import (
    Interval,
    exact_hex_double,
    identity,
    imat_vec,
    matrix_inverse,
    parse_engine_endpoint,
    rational,
    rmat_imat,
    rmat_ivec,
    round_down_double,
    round_up_double,
)
from t3cap.export import interval_from_json

F = Fraction


def rand_interval(rng: random.Random) -> Interval:
    a, b = F(rng.randint(-50, 50), rng.randint(1, 9)), F(rng.randint(-50, 50), rng.randint(1, 9))
    return Interval(min(a, b), max(a, b))


def grid(iv: Interval) -> list[Fraction]:
    return [iv.lo, iv.hi, (iv.lo + iv.hi) / 2, iv.lo + iv.width / 3]


@pytest.mark.parametrize("op", ["add", "sub", "mul"])
def test_binary_operations_are_exact_hulls_of_pointwise_results(op):
    rng = random.Random(5)
    f = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y}[op]
    for _ in range(200):
        a, b = rand_interval(rng), rand_interval(rng)
        res = {"add": a + b, "sub": a - b, "mul": a * b}[op]
        values = [f(x, y) for x, y in itertools.product(grid(a), grid(b))]
        assert all(res.contains(v) for v in values)
        corners = [f(x, y) for x, y in itertools.product((a.lo, a.hi), (b.lo, b.hi))]
        assert res.lo == min(corners) and res.hi == max(corners)


def test_negation_and_scaling_are_exact():
    rng = random.Random(6)
    for _ in range(100):
        a = rand_interval(rng)
        q = F(rng.randint(-9, 9), rng.randint(1, 5))
        assert -a == Interval(-a.hi, -a.lo)
        s = a.scale(q)
        assert s.lo == min(a.lo * q, a.hi * q) and s.hi == max(a.lo * q, a.hi * q)


def test_containment_predicates_follow_closed_and_open_semantics():
    a = Interval(F(0), F(1))
    assert a.contains(F(0)) and a.contains(F(1)) and not a.contains(F(-1, 10**30))
    assert a.contains_interval(Interval(F(0), F(1))) and not a.strictly_contains_interval(Interval(F(0), F(1)))
    assert a.strictly_contains_interval(Interval(F(1, 3), F(2, 3)))
    assert not a.strictly_contains_interval(Interval(F(1, 3), F(1)))
    assert a.hull(Interval(F(2), F(3))) == Interval(F(0), F(3))
    assert Interval(F(-3), F(2)).magnitude() == F(3)
    with pytest.raises(ValueError):
        Interval(F(1), F(0))


def test_exact_matrix_inverse_and_products():
    m = ((F(2), F(1), F(0)), (F(1), F(3), F(1)), (F(0), F(1), F(4)))
    inv = matrix_inverse(m)
    prod = tuple(tuple(sum(m[i][k] * inv[k][j] for k in range(3)) for j in range(3)) for i in range(3))
    assert prod == identity(3)
    with pytest.raises(ValueError):
        matrix_inverse(((F(1), F(2)), (F(2), F(4))))
    v = (Interval(F(-1), F(1)), Interval(F(2), F(3)), Interval(F(0), F(0)))
    out = rmat_ivec(m, v)
    assert out[0] == Interval(F(0), F(5))
    im = tuple(tuple(Interval(x, x) for x in row) for row in m)
    assert imat_vec(im, v) == out
    assert rmat_imat(identity(3), im) == im


def test_outward_double_rounding_brackets_rationals():
    for q in (F(1, 3), F(-2, 7), F(10**20 + 1, 3), F(5, 8)):
        lo, hi = round_down_double(q), round_up_double(q)
        assert F(lo) <= q <= F(hi)
        assert (lo == hi) == (F(lo) == q)


def test_directed_rounding_is_identity_on_representable_rationals():
    """RD(q) 는 q 이하 최대 double 이므로 q 가 double 이면 q 자신이다(RU 도 같다)."""
    for x in (0.625, -3.0, 1e-300, 2.0**60):
        q = F(x)
        assert round_down_double(q) == x and round_up_double(q) == x


def test_interval_requires_fraction_endpoints_on_both_sides():
    with pytest.raises(TypeError):
        Interval(F(1), 2)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Interval(1, F(2))  # type: ignore[arg-type]


def test_containment_rejects_partial_overlap_on_each_side():
    a = Interval(F(0), F(1))
    assert not a.contains_interval(Interval(F(-1, 10), F(1, 2)))
    assert not a.contains_interval(Interval(F(1, 2), F(11, 10)))
    assert not a.strictly_contains_interval(Interval(F(0), F(2, 3)))


def test_matrix_vector_products_reject_dimension_mismatch():
    m = ((F(1), F(2)), (F(3), F(4)))
    with pytest.raises(ValueError):
        rmat_ivec(m, (Interval(F(0), F(1)),))
    with pytest.raises(ValueError):
        imat_vec(tuple(tuple(Interval(x, x) for x in row) for row in m), (Interval(F(0), F(1)),))


def test_rational_interval_matrix_product_rejects_a_single_ragged_row():
    b = tuple(tuple(Interval(F(v), F(v)) for v in row) for row in ((1, 2), (3, 4)))
    with pytest.raises(ValueError):
        rmat_imat(((F(1), F(0)), (F(1),)), b)


def test_hex_tokens_exact_as_double_keep_their_value_on_every_path():
    """double 로 정확히 나타나는 16진 토큰 - 꼴이 달라도 그 값 그대로(비정규 수·최댓값·음의 0 포함), 세 해석 경로가 같음"""
    cases = {"0x1.8p+1": F(3), "0x1.8000p+1": F(3), "0x0001.8p+1": F(3), "0X1.8P+1": F(3), "+0x1.8p1": F(3),
             "0x10": F(16), "0x1p+0": F(1), "0x.8": F(1, 2), "0x.8p1": F(1),
             "-0x1.999999999999ap-4": -F(0.1), "0x1.0000000000001p+0": 1 + F(2) ** -52,
             "0x1p-1074": F(2) ** -1074, "0x0.0000000000001p-1022": F(2) ** -1074,
             "0x1.fffffffffffffp+1023": F(sys.float_info.max), "-0x0p+0": F(0)}
    for text, q in cases.items():
        assert exact_hex_double(text) == q and rational(text) == q and parse_engine_endpoint(text) == q
        assert interval_from_json({"lo": text, "hi": text}) == Interval(q, q)


def test_hex_tokens_that_need_rounding_are_refused_on_every_path():
    """가수 53비트 초과·범위 밖 토큰 - rational·엔진 끝점·JSON 끝점 모두 ValueError(최근접으로 반올림하면 점 구간이 참값을 놓친다)"""
    for text in ("0x1.00000000000001p+0", "0x1.00000000000008p+0", "0x1p+1024", "0x1p-1075", "0x1.8p-1074",
                 "0x20000000000001"):
        for parse in (exact_hex_double, rational, parse_engine_endpoint, lambda t: interval_from_json({"lo": t, "hi": t})):
            with pytest.raises(ValueError):
                parse(text)


def test_malformed_hex_tokens_are_refused():
    """16진 꼴이 아닌 토큰(자릿수 없음·잘못된 글자·지수 없음·소수점 둘·지수 8자리) - ValueError"""
    for text in ("0x", "0x.", "0xg1", "0x1p", "0x1p+", "0x1.2.3", "0x1p+12345678"):
        with pytest.raises(ValueError):
            exact_hex_double(text)


def test_hex_token_length_cap_is_64_characters():
    """길이 상한(엔진과 같은 64자) - 64자 토큰(값 1)은 받고 65자 토큰은 값이 double 이어도 거부"""
    longest = "0x1." + "0" * 57 + "p+0"
    assert len(longest) == 64 and exact_hex_double(longest) == 1
    with pytest.raises(ValueError):
        exact_hex_double("0x1." + "0" * 58 + "p+0")
