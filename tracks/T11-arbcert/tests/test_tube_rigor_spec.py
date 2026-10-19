"""흐름 관 인증기(감사 V2 사본) 엄밀성 시험(G9, R1-07 보강). 기대값은 감싸기의 정의(차수와 보폭에 관계없이 참 해를 담는다)와
함정 ②(첫 syzygy 를 시간창 안에서 정확히 한 번 횡단)에서 정했다. 기준 해는 G5 기준 적분기(t11/g5ref.py, 사본과 코드 공유 없음)다.

- 낮은 차수(8)에서는 Taylor 절단 오차가 커서, 단계 갱신의 Lagrange 나머지를 빠뜨리면 귀환 창이 참 귀환 시각을 놓친다.
- 교차 창의 Ȧ 감싸기가 0 을 품으면(횡단성 미인증) 인증하지 않아야 한다. 이 조건을 실제 동역학으로 만들기 어려워서
  adot_range 의 결과에 0 보다 큰 값을 합친 고장 주입으로 시험한다.
"""
from __future__ import annotations

import sys
from fractions import Fraction as F
from pathlib import Path

import mpmath as mp
import pytest
from flint import arb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import certify_tubes as ct  # noqa: E402
import rigorous_tube as rt  # noqa: E402
from t11 import design as dz  # noqa: E402
from t11 import g5ref as G  # noqa: E402

ZERO_BOX = [[F(0)] * 4 for _ in range(4)]


def reference(vertex: str):
    mp.mp.dps = 45
    c, _ = dz.hset(vertex)
    fl = G.fly([G.frac_to_mpf(x) for x in c], [[mp.mpf(0)] * 4 for _ in range(4)], tangents=False)
    return G.mpf_to_frac(fl.tau), {k: G.mpf_to_frac(v) for k, v in fl.rmin.items()}, \
        {k: G.mpf_to_frac(v) for k, v in fl.rmax.items()}


@pytest.mark.parametrize("order", [18, 8])
def test_point_flight_encloses_reference_return_time_and_distances(order):
    """N1 중심 점의 첫 syzygy 비행(차수 18·8) - 귀환 창이 기준 해 τ 를, 관 r_ij² 범위가 기준 해의 r_ij 최소·최대를 담는다."""
    c, _ = dz.hset("N1")
    tau, rmin, rmax = reference("N1")
    res = rt.fly(c, ZERO_BOX, ct.plan_tau(c, ZERO_BOX), order=order)
    w0, w1 = res["window"]
    assert w0 <= tau <= w1
    for k, name in enumerate(("r12", "r13", "r23")):
        lo2 = ct.exact(res["r2_lower"][k])
        hi2 = ct.exact(res["r2_upper"][k])
        assert lo2 <= rmin[name] ** 2 and rmax[name] ** 2 <= hi2


def test_crossing_without_certified_transversality_is_rejected(monkeypatch):
    """교차 창의 Ȧ 감싸기가 0 을 품음(고장 주입) - fly 는 창을 내지 않고 FlightFailure 로 끝난다(정확히 한 번 횡단, 함정 ②)."""
    c, M = dz.hset("N1")
    original = rt.adot_range
    monkeypatch.setattr(rt, "adot_range", lambda *a, **k: original(*a, **k).union(arb(1)))
    with pytest.raises(rt.FlightFailure, match="transversal"):
        rt.fly(c, M, ct.plan_tau(c, M))


def test_certified_crossing_has_negative_adot_on_window(monkeypatch):
    """인증된 교차 창 - 그 창 위 Ȧ 감싸기가 음수(Ȧ<0 이 실제로 성립한 경우에만 창을 낸다)."""
    c, M = dz.hset("N1")
    calls = []
    original = rt.adot_range

    def spy(*a, **k):
        out = original(*a, **k)
        calls.append(out)
        return out

    monkeypatch.setattr(rt, "adot_range", spy)
    rt.fly(c, M, ct.plan_tau(c, M))
    assert calls and all(x < 0 for x in calls)
