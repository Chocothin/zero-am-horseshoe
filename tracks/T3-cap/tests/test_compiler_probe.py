"""컴파일러 탐침 검사(engine/compiler_probe.cpp, 엔진과 같은 플래그로 빌드).

명세: 컴파일 시점에 값이 정해지는 피연산자로 계산한 구간도 정확값을 담아야 한다. 단일 사칙연산은 끝점이 정확값의
아래·위 방향 반올림과 같아야 한다. 컴파일러가 반올림 전환을 무시하고 상수를 최근접으로 접으면 [n, n] 이 나와 정확값을
놓친다. 피연산자는 C 의 double 리터럴이므로 정확값은 그 double 값(Fraction(0.1) 등)으로 계산한다.
"""
from __future__ import annotations

import math
import subprocess
from fractions import Fraction

import pytest

from test_arith_selftest import rd, ru, sqrt_rd, sqrt_ru

F = Fraction
EXACT_SINGLE = {
    "div_1_3": F(1, 3),
    "div_1_10": F(1, 10),
    "add_01_02": F(0.1) + F(0.2),
    "sub_1_1e-20": 1 - F(1e-20),
    "mul_01_3": F(0.1) * 3,
    "mul_01_07": F(0.1) * F(0.7),
    "sqr_01": F(0.1) ** 2,
}
EXACT_COMPOUND = {
    "muladd_01_07_03": F(0.1) * F(0.7) + F(0.3),
    "matvec_0": F(0.1) * 3 + F(0.2) * F(0.1),
    "matvec_1": F(0.3) * 3 + F(0.7) * F(0.1),
    "map_div_3": F(1, 3),
    "map_muladd_01_02": F(0.1) + F(0.2),
    "map_sub_mul_07": (1 - F(1e-20)) * F(0.7),
}
SQRT_TWO = ("sqrt_2", "map_sqrt_2")
ROUNDED_NAMES = set(EXACT_SINGLE) | set(EXACT_COMPOUND) | set(SQRT_TWO)


def probe_lines(binary) -> dict[str, tuple[Fraction, Fraction]]:
    out = subprocess.run([str(binary)], capture_output=True, text=True, check=True, timeout=120)
    rows = {}
    for line in out.stdout.splitlines():
        name, lo, hi = line.split()
        rows[name] = (F(float.fromhex(lo)), F(float.fromhex(hi)))
    return rows


def check_probes(rows: dict[str, tuple[Fraction, Fraction]], sqrt_slack_ulps: int = 0) -> None:
    """포함은 모든 탐침에서, 방향 반올림의 엄격성은 단일 사칙연산과 √2(filib 은 1 ulp 여유)에서 본다."""
    for name, (lo, hi) in rows.items():
        if name in EXACT_SINGLE:
            q = EXACT_SINGLE[name]
            assert lo <= q <= hi, f"{name}: [{lo}, {hi}] misses the exact value"
            assert float(lo) == rd(q) and float(hi) == ru(q), f"{name}: not directed-rounded"
        elif name in EXACT_COMPOUND:
            q = EXACT_COMPOUND[name]
            assert lo <= q <= hi, f"{name}: [{lo}, {hi}] misses the exact value"
        elif name in SQRT_TWO:
            assert lo * lo <= 2 <= hi * hi, f"{name}: [{lo}, {hi}] misses sqrt(2)"
            down, up = sqrt_rd(F(2)), sqrt_ru(F(2))
            for _ in range(sqrt_slack_ulps):
                down, up = math.nextafter(down, -math.inf), math.nextafter(up, math.inf)
            assert down <= float(lo) and float(hi) <= up, f"{name}: wider than {sqrt_slack_ulps} ulp beyond directed"
        else:
            pytest.fail(f"unknown probe {name}")


@pytest.mark.parametrize("config", ["native-strict", "x86filib-strict"])
def test_constant_operand_probes_enclose_exact_values(config, request):
    """엔진과 같은 플래그(-O2 -frounding-math -ffp-contract=off) 빌드 - 상수 피연산자 15종이 모두 정확값을 담음(접기 없음)"""
    bindir = request.getfixturevalue("native_dir" if config == "native-strict" else "x86_dir")
    rows = probe_lines(bindir / "compiler_probe")
    assert set(rows) == ROUNDED_NAMES
    check_probes(rows, sqrt_slack_ulps=0 if config == "native-strict" else 1)


def test_probe_build_without_rounding_switch_is_rejected(native_dir):
    """반올림 전환을 뺀 같은 탐침 - 정확값을 놓치는 탐침이 생기고 검사기가 실패함(검사기가 접기를 잡을 수 있다는 대조)"""
    rows = probe_lines(native_dir / "compiler_probe_noround")
    misses = [n for n, (lo, hi) in rows.items() if n in EXACT_SINGLE and not lo <= EXACT_SINGLE[n] <= hi]
    assert len(misses) >= 4
    with pytest.raises(AssertionError):
        check_probes(rows)


def test_checker_rejects_nearest_folded_constant():
    """1/3 을 최근접 double 하나로 접은 결과 [n, n] - 포함이 깨져 실패(컴파일러 상수 접기의 모습)"""
    n = F(1 / 3)
    assert n != F(1, 3)
    with pytest.raises(AssertionError):
        check_probes({"div_1_3": (n, n)})
