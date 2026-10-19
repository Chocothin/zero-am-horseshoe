"""조각 사용 자격(usage_failures) 명세 시험. 기대값은 「c0 가 (0,T_lo) 에 영점이 없음을 보이고, c1 창이 둘째 영점 창보다
앞서면 c1 은 첫 통과를 미분했다」는 논법과 certify_return 의 가설(판별식>0, 도착 근>0, 귀환>0, 흐름 관이 반환을 덮음, r_ij>δ)에서 정했다."""
from fractions import Fraction as Q

from t3cap.api import C0Enclosure, C1Enclosure, Crossings, Failure, TubeBounds
from t3cap.exact import Interval

from t6hs.enclose import usage_failures

DELTA = Q(1, 5)


def iv(lo, hi):
    return Interval(Q(lo), Q(hi))


def tube(r12=iv("0.3", "2"), t_end=iv("1.01", "1.02")):
    return TubeBounds(r12, iv("0.4", "3"), iv("0.5", "3"), 10, t_end, Crossings(1, 0, 1, iv(1, "1.001")))


def c0(rt=iv(1, "1.001"), disc=iv("0.5", "0.6"), branch=iv(1, 2), tb=None):
    box = tuple(iv(-1, 1) for _ in range(4))
    return C0Enclosure(box, box + box[:2], rt, disc, branch, tb or tube(), None, 0.0)


def c1(rt=iv(1, "1.002")):
    """흐름 관 끝을 귀환 창 위끝보다 늦게 두어, 둘째 영점 규칙만 따로 시험되게 한다."""
    box = tuple(iv(-1, 1) for _ in range(4))
    eye = tuple(tuple(iv(1, 1) if i == j else iv(0, 0) for j in range(4)) for i in range(4))
    t_end = Interval(rt.hi + Q(1, 100), rt.hi + Q(2, 100))
    return C1Enclosure(box, box + box[:2], rt, iv("0.5", "0.6"), iv(1, 2), tube(t_end=t_end), eye, eye, 0.0)


def second(rt=iv(2, "2.1")):
    return c0(rt=rt, tb=tube(t_end=iv("2.2", "2.3")))


def test_consistent_runs_are_usable():
    """네 실행이 모두 가설을 만족하고 c1 창이 둘째 영점보다 앞섬 - 실패 없음"""
    assert usage_failures(c0(), c1(), second(), c0(), DELTA) == []


def test_c1_window_reaching_the_second_zero_is_rejected():
    """c1 창이 둘째 영점 창과 겹침 - 그 규칙 하나로만 실패(c1 이 둘째 통과를 미분했을 수 있음)"""
    failures = usage_failures(c0(), c1(rt=iv(1, "2.05")), second(), c0(), DELTA)
    assert failures and all("second zero" in f for f in failures)


def test_c1_window_touching_the_second_zero_is_rejected():
    """c1 창 위끝 = 둘째 영점 창 아래끝 - 그 규칙 하나로만 실패(엄격 부등식이어야 영점 하나만 남음)"""
    failures = usage_failures(c0(), c1(rt=iv(1, 2)), second(), c0(), DELTA)
    assert failures and all("second zero" in f for f in failures)


def test_engine_failure_in_any_run_is_rejected():
    """어느 실행이든 엔진 실패 - 실패(감싸기가 없으면 근거가 없음)"""
    f = Failure("crossing: test", 0.0)
    for args in ((f, c1(), second(), c0()), (c0(), f, second(), c0()), (c0(), c1(), f, c0()), (c0(), c1(), second(), f)):
        assert usage_failures(*args, DELTA)


def test_tube_too_close_to_collision_is_rejected():
    """흐름 관 r12 하한 0.1 < δ=1/5 - 실패(함정 ④)"""
    assert usage_failures(c0(tb=tube(r12=iv("0.1", "2"))), c1(), second(), c0(), DELTA)


def test_chart_discriminant_not_positive_is_rejected():
    """중심 점 실행의 판별식이 0 을 담음 - 실패(함정 ⑤)"""
    assert usage_failures(c0(), c1(), second(), c0(disc=iv("-0.1", "0.5")), DELTA)


def test_tube_not_covering_the_return_is_rejected():
    """흐름 관 끝이 귀환 창 위끝보다 앞섬 - 실패(반환까지의 r_ij 가 증명되지 않음)"""
    assert usage_failures(c0(tb=tube(t_end=iv("1.0005", "1.0006"))), c1(), second(), c0(), DELTA)
