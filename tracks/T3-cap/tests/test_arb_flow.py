"""독립 Arb 적분기의 명세 시험: 에너지 보존(감싸기가 h 를 담음), 고정밀 기준 해의 통과 상태 포함, 출발점 이동 음성 대조."""
from __future__ import annotations

from fractions import Fraction

import mpmath as mp
from flint import arb, ctx

from reference import arb_flow as af
from reference import mp_reference as ref

U0 = (Fraction(5148567982527431, 10**15), Fraction(0), Fraction(0), Fraction(432690650760826, 10**15))
H = Fraction(-1, 2)


def arb_crossing(u):
    ctx.prec = 192
    md = af.Masses.of((1, 1, 1))
    x0 = af.chart_syzygy(u, +1, md, H, Fraction(0))
    return af.flow_to_next_syzygy(x0, md, direction=-1), md


def reference_coords(u) -> list[mp.mpf]:
    mp.mp.dps = 50
    md = ref.MassData.of((1, 1, 1))
    st = ref.chart_state(list(u), coord=2, value=Fraction(0), solved=5, branch=1, free=(0, 1, 3, 4), md=md, h=H, J=0)
    c = ref.flow_to_section(st, md, coord=2, value=Fraction(0), direction=-1, returns=1, order=40)
    return [ref.reduced_coordinate(c.state, k) for k in (0, 1, 3, 4)], c.time


def contains_mp(ball: arb, value: mp.mpf) -> bool:
    return ball.contains(arb(mp.nstr(value, 48)))


def test_energy_enclosure_at_crossing_contains_the_energy_level():
    crossing, md = arb_crossing(U0)
    assert af.energy(crossing.state, md).contains(af.q(H))


def test_crossing_enclosure_contains_independent_high_precision_reference():
    crossing, _ = arb_crossing(U0)
    coords = af.reduced_chart_coords(crossing.state)
    expected, t = reference_coords(U0)
    # 기준 해 오차(약 1e-45)가 Arb 반경(약 1e-36)보다 훨씬 작아 포함이 의미 있다
    assert all(contains_mp(b, v) for b, v in zip(coords, expected))
    assert contains_mp(crossing.time, t)


def test_shifted_start_excludes_the_original_reference():
    """음성 대조: 출발점 r 을 1e-20 옮기면 감싸기(반경 약 1e-36)는 원래 궤도의 통과 상태를 담지 않아야 한다."""
    moved = (U0[0] + Fraction(1, 10**20),) + U0[1:]
    crossing, _ = arb_crossing(moved)
    coords = af.reduced_chart_coords(crossing.state)
    expected, _ = reference_coords(U0)
    assert not all(contains_mp(b, v) for b, v in zip(coords, expected))
