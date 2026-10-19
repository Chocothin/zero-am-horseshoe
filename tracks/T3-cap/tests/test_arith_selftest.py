"""구간 산술의 바깥쪽 반올림을 정확 유리수와 대조한다(arm64 NATIVE·x86_64 filib 둘 다).

명세: 결과 구간은 정확값(들)을 포함한다. 점 사칙연산은 끝점이 정확값의 아래·위 방향 반올림과 같아야 한다 —
반올림 모드 전환이 컴파일러 최적화로 사라지면 이 엄격성 검사가 먼저 깨진다. sqrt 는 NATIVE 에서 엄격,
filib 에서 1 ulp 이내 바깥(실측: filib sqrt 는 RU 보다 1 ulp 넓은 경우가 있다).
"""
from __future__ import annotations

import math
import subprocess
from fractions import Fraction
from math import isqrt

import pytest

N_CASES = "3000"


def rd(q: Fraction) -> float:
    f = float(q)
    return f if Fraction(f) <= q else math.nextafter(f, -math.inf)


def ru(q: Fraction) -> float:
    f = float(q)
    return f if Fraction(f) >= q else math.nextafter(f, math.inf)


def sqrt_rd(a: Fraction) -> float:
    """sqrt(a) 아래쪽 최근접 double: lo^2 <= a 인 최대 double."""
    guess = float(math.sqrt(float(a)))
    while Fraction(guess) ** 2 > a:
        guess = math.nextafter(guess, -math.inf)
    while Fraction(math.nextafter(guess, math.inf)) ** 2 <= a:
        guess = math.nextafter(guess, math.inf)
    return guess


def sqrt_ru(a: Fraction) -> float:
    lo = sqrt_rd(a)
    return lo if Fraction(lo) ** 2 == a else math.nextafter(lo, math.inf)


def parse(line: str) -> tuple[str, list[Fraction]]:
    op, *vals = line.split()
    return op, [Fraction(float.fromhex(v)) for v in vals]


def check_lines(lines: list[str], sqrt_slack_ulps: int = 0) -> dict[str, int]:
    """sqrt_slack_ulps: filib sqrt 는 최근접 결과를 1 ulp 넓히는 방식이라 엄격한 방향 반올림 대신 그만큼의 여유를 준다."""
    counts: dict[str, int] = {}
    for line in lines:
        op, v = parse(line)
        counts[op] = counts.get(op, 0) + 1
        if op in ("add", "sub", "mul", "div"):
            a, b, lo, hi = v
            exact = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[op]
            assert lo <= exact <= hi, f"enclosure broken: {line}"
            assert float(lo) == rd(exact) and float(hi) == ru(exact), f"not directed-rounded: {line}"
        elif op == "sqrt":
            a, _, lo, hi = v
            assert lo * lo <= a <= hi * hi, f"sqrt enclosure broken: {line}"
            tight_lo, tight_hi = sqrt_rd(a), sqrt_ru(a)
            for _ in range(sqrt_slack_ulps):
                tight_lo, tight_hi = math.nextafter(tight_lo, -math.inf), math.nextafter(tight_hi, math.inf)
            assert tight_lo <= float(lo) <= sqrt_rd(a) and sqrt_ru(a) <= float(hi) <= tight_hi, \
                f"sqrt not directed-rounded within {sqrt_slack_ulps} ulp: {line}"
        elif op in ("iadd", "isub", "imul", "idiv"):
            a1, a2, b1, b2, lo, hi = v
            f = {"iadd": lambda x, y: x + y, "isub": lambda x, y: x - y, "imul": lambda x, y: x * y,
                 "idiv": lambda x, y: x / y}[op]
            if op == "isub":
                corners = [a1 - b2, a2 - b1]
            else:
                corners = [f(x, y) for x in (a1, a2) for y in (b1, b2)]
            assert lo <= min(corners) and max(corners) <= hi, f"interval enclosure broken: {line}"
        elif op == "isqrt":
            a1, a2, _, _, lo, hi = v
            assert lo * lo <= a1 and a2 <= hi * hi, f"interval sqrt enclosure broken: {line}"
        elif op == "matvec":
            *pairs, lo, hi = v
            exact = sum(pairs[2 * j] * pairs[2 * j + 1] for j in range(3))
            assert lo <= exact <= hi, f"matvec enclosure broken: {line}"
        else:
            pytest.fail(f"unknown op {op}")
    return counts


@pytest.mark.parametrize("config", ["native-strict", "x86filib-strict"])
def test_outward_rounding_matches_exact_arithmetic(config, request):
    bindir = request.getfixturevalue("native_dir" if config == "native-strict" else "x86_dir")
    out = subprocess.run([str(bindir / "arith_selftest"), N_CASES], capture_output=True, text=True, check=True)
    counts = check_lines(out.stdout.splitlines(), sqrt_slack_ulps=0 if config == "native-strict" else 1)
    assert counts["add"] == int(N_CASES) and counts["matvec"] == 3 * (int(N_CASES) // 10)


def test_checker_rejects_round_to_nearest_output():
    """음성 대조: 반올림 전환이 없는(최근접) 결과를 주면 검사기가 실패해야 한다."""
    a, b = 0.1, 0.2
    exact = Fraction(a) + Fraction(b)
    nearest = float(a + b)
    line = f"add {a.hex()} {b.hex()} {nearest.hex()} {nearest.hex()}"
    assert Fraction(nearest) != exact
    with pytest.raises(AssertionError):
        check_lines([line])


def test_checker_rejects_one_ulp_too_wide():
    """음성 대조: 바깥쪽이지만 1 ulp 더 넓은(pred/succ) 결과도 방향 반올림이 아니므로 잡아낸다."""
    a, b = 0.1, 0.2
    exact = Fraction(a) + Fraction(b)
    lo, hi = math.nextafter(rd(exact), -math.inf), ru(exact)
    with pytest.raises(AssertionError):
        check_lines([f"add {a.hex()} {b.hex()} {lo.hex()} {hi.hex()}"])


def test_isqrt_helper_is_exact():
    for n in (2, 3, 10, 12345):
        a = Fraction(n)
        lo = sqrt_rd(a)
        assert Fraction(lo) ** 2 <= a < Fraction(math.nextafter(lo, math.inf)) ** 2
    assert isqrt(16) == 4


def enclosure_failures(lines: list[str]) -> int:
    """점 사칙연산에서 정확값이 [lo, hi] 밖에 놓인 줄 수(엄격성과 무관한 순수 포함 위반)."""
    bad = 0
    for line in lines:
        op, v = parse(line)
        if op in ("add", "sub", "mul", "div"):
            a, b, lo, hi = v
            exact = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[op]
            bad += not (lo <= exact <= hi)
    return bad


def test_disabling_the_rounding_switch_breaks_enclosure_and_is_detected(native_dir):
    """G6 음성 대조: 같은 CAPD 구간 템플릿에서 반올림 전환(msr fpcr)만 없애면 포함이 깨지고 검사기가 실패해야 한다."""
    out = subprocess.run([str(native_dir / "arith_selftest_noround"), "1000"], capture_output=True, text=True,
                         check=True)
    lines = out.stdout.splitlines()
    assert enclosure_failures(lines) > 100
    with pytest.raises(AssertionError):
        check_lines(lines)
    rounded = subprocess.run([str(native_dir / "arith_selftest"), "1000"], capture_output=True, text=True, check=True)
    assert enclosure_failures(rounded.stdout.splitlines()) == 0
