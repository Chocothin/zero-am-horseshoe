"""판정층 명세 시험(G9): Krawczyk 정리의 가설·결론과 부등식의 경계 사례에서 유도한다."""
from __future__ import annotations

from fractions import Fraction

import pytest

from t3cap.api import C0Enclosure, Crossings, TubeBounds
from t3cap.exact import Interval, identity, matrix_inverse
from t3cap.verdict import certify_return, krawczyk, lower_bound_exceeds, subset_interior

F = Fraction


def iv(lo, hi) -> Interval:
    return Interval(F(lo), F(hi))


CROSSED = Crossings(1, 0, 1, iv(4, 5))


def pt(q) -> Interval:
    return Interval(F(q), F(q))


def test_subset_interior_requires_strict_inclusion_in_every_coordinate():
    outer = (iv(0, 1), iv(-1, 1))
    assert subset_interior((iv("0.1", "0.9"), iv("-0.5", "0.5")), outer)
    assert not subset_interior((iv(0, "0.9"), iv("-0.5", "0.5")), outer)        # 왼쪽 끝 접촉
    assert not subset_interior((iv("0.1", "0.9"), iv("-0.5", 1)), outer)        # 오른쪽 끝 접촉
    assert not subset_interior((iv("0.1", "1.1"), iv("-0.5", "0.5")), outer)    # 밖으로 나감
    with pytest.raises(ValueError):
        subset_interior((iv(0, 1),), outer)


def test_lower_bound_exceeds_is_strict():
    assert lower_bound_exceeds(iv("0.2", 1), F(1, 10))
    assert not lower_bound_exceeds(iv("0.1", 1), F(1, 10))
    assert not lower_bound_exceeds(iv(-1, 1), F(0))


def linear_case(center, radius):
    """F(x) = A x - b, 정확한 영점 x* = (1/2, -1/3). DF = A 는 상자 전체에서 정확히 A 다."""
    A = ((F(2), F(1)), (F(1), F(3)))
    xstar = (F(1, 2), F(-1, 3))
    b = tuple(sum(A[i][j] * xstar[j] for j in range(2)) for i in range(2))
    c = tuple(F(v) for v in center)
    f_c = tuple(pt(sum(A[i][j] * c[j] for j in range(2)) - b[i]) for i in range(2))
    df = tuple(tuple(pt(A[i][j]) for j in range(2)) for i in range(2))
    box = tuple(Interval(ci - F(radius), ci + F(radius)) for ci in c)
    return c, box, f_c, df, matrix_inverse(A)


def test_krawczyk_proves_a_zero_inside_a_box_that_contains_it():
    c, box, f_c, df, C = linear_case(("0.5001", "-0.3333"), "0.01")
    v = krawczyk(c, box, f_c, df, C)
    assert v.proven and all(b.contains(x) for b, x in zip(v.image, (F(1, 2), F(-1, 3))))


def test_krawczyk_never_certifies_a_box_without_a_zero():
    """정리의 대우: 상자에 영점이 없으면 K 는 상자 내부에 들어갈 수 없다."""
    c, box, f_c, df, C = linear_case(("0.7", "-0.3333"), "0.01")
    assert not krawczyk(c, box, f_c, df, C).proven


def test_krawczyk_rejects_center_outside_box():
    c, box, f_c, df, C = linear_case(("0.5", "-0.3333"), "0.01")
    shifted = tuple(Interval(b.lo + F(1), b.hi + F(1)) for b in box)
    v = krawczyk(c, shifted, f_c, df, C)
    assert not v.proven and not v.checks[0].passed


def test_krawczyk_nonlinear_sqrt2_with_interval_derivative():
    """F(x)=x^2-2, DF(X)=2X. sqrt(2) 을 담은 상자는 증명되고, 담지 않은 상자는 실패한다."""
    def verdict(lo, hi):
        box = (Interval(F(lo), F(hi)),)
        c = ((box[0].lo + box[0].hi) / 2,)
        f_c = (pt(c[0] ** 2 - 2),)
        df = ((Interval(2 * box[0].lo, 2 * box[0].hi),),)
        C = ((1 / (2 * c[0]),),)
        return krawczyk(c, box, f_c, df, C)
    assert verdict("1.41", "1.42").proven
    assert not verdict("1.42", "1.43").proven


def test_krawczyk_result_does_not_depend_on_preconditioner_for_soundness():
    """엉터리 전처리(C=I)는 증명을 못 할 수는 있어도 거짓 증명을 내면 안 된다."""
    c, box, f_c, df, _ = linear_case(("0.7", "-0.3333"), "0.01")
    assert not krawczyk(c, box, f_c, df, identity(2)).proven


def enclosure(**override) -> C0Enclosure:
    base = dict(
        image=(iv(0, 1),) * 4, hull=(iv(0, 1),) * 6, return_time=iv(4, 5), discriminant=iv("0.5", 1),
        branch_out=iv("0.3", 1),
        tube=TubeBounds(iv("1.7", 5), iv("1.6", 5), iv("1.8", 5), 100, iv(6, 6), CROSSED), theta=None, seconds=0.0)
    base.update(override)
    return C0Enclosure(**base)


def test_certify_return_accepts_valid_enclosure():
    assert certify_return(enclosure(), min_distance=F(1, 10)).proven


@pytest.mark.parametrize("override,failed", [
    (dict(discriminant=iv(0, 1)), "chart_discriminant_positive"),
    (dict(branch_out=iv("-0.1", 1)), "target_branch_positive"),
    (dict(return_time=iv(0, 5)), "return_time_positive"),
    (dict(tube=TubeBounds(iv("0.05", 5), iv("1.6", 5), iv("1.8", 5), 100, iv(6, 6), CROSSED)), "tube_r12_clear"),
    (dict(tube=TubeBounds(iv("1.7", 5), iv("0.1", 5), iv("1.8", 5), 100, iv(6, 6), CROSSED)), "tube_r13_clear"),
    (dict(tube=TubeBounds(iv("1.7", 5), iv("1.6", 5), iv("0.01", 5), 100, iv(6, 6), CROSSED)), "tube_r23_clear"),
    (dict(tube=TubeBounds(iv("1.7", 5), iv("1.6", 5), iv("1.8", 5), 100, iv("4.5", 6), CROSSED)), "tube_covers_return"),
])
def test_certify_return_rejects_each_violated_hypothesis(override, failed):
    v = certify_return(enclosure(**override), min_distance=F(1, 10))
    assert not v.proven and failed in [c.name for c in v.failed]


def test_certify_return_threshold_is_strict():
    v = certify_return(enclosure(), min_distance=F(16, 10))
    assert not v.proven and [c.name for c in v.failed] == ["tube_r13_clear"]


def test_krawczyk_with_off_center_expansion_point():
    """정리는 c 가 X 안이면 중점이 아니어도 성립한다. 부정확한 전처리(C=0.4, DF=2)로 수축 항이 0 이 아닌 경우."""
    box = (iv("0.49", "1.0"),)
    c = (F(1, 2),)
    f_c = (pt(0),)                      # F(x)=2x-1, F(1/2)=0
    df = ((pt(2),),)
    v = krawczyk(c, box, f_c, df, ((F(2, 5),),))
    assert v.proven and v.image[0] == Interval(F(1, 2) + F(1, 5) * F(-1, 100), F(1, 2) + F(1, 5) * F(1, 2))


def test_krawczyk_rejects_center_outside_in_a_single_coordinate():
    """가설 c in X 는 모든 성분에서 필요하다. 정확한 전처리면 K={x*} 이므로 이 검사만이 증명을 막는다."""
    A = ((F(2), F(1)), (F(1), F(3)))
    xstar = (F(1, 2), F(-1, 3))
    c = (F(1, 2), F(1, 2))
    b = tuple(sum(A[i][j] * xstar[j] for j in range(2)) for i in range(2))
    f_c = tuple(pt(sum(A[i][j] * c[j] for j in range(2)) - b[i]) for i in range(2))
    df = tuple(tuple(pt(A[i][j]) for j in range(2)) for i in range(2))
    box = (iv("0.4", "0.6"), iv("-0.4", "-0.3"))   # c[1]=1/2 는 둘째 성분 밖
    v = krawczyk(c, box, f_c, df, matrix_inverse(A))
    assert subset_interior(v.image, box) and not v.proven


@pytest.mark.parametrize("bad", ["center_long", "f_center_long", "df_short_row", "df_extra_column",
                                 "df_missing_row", "c_extra_row", "c_extra_column"])
def test_krawczyk_dimension_mismatch_is_a_value_error(bad):
    """차원이 어긋난 입력을 잘라 맞추면 거짓 증명이 될 수 있으므로 ValueError 여야 한다(조용한 계산 금지)."""
    c, box, f_c, df, C = linear_case(("0.5", "-0.3333"), "0.01")
    if bad == "center_long":
        c = c + (F(0),)
    elif bad == "f_center_long":
        f_c = f_c + (pt(0),)
    elif bad == "df_short_row":
        df = (df[0], df[1][:1])
    elif bad == "df_extra_column":
        df = (df[0] + (pt(0),), df[1])
    elif bad == "df_missing_row":
        df = df[:1]
    elif bad == "c_extra_row":
        C = C + ((F(0), F(0)),)
    else:
        C = (C[0] + (F(0),), C[1])
    with pytest.raises(ValueError):
        krawczyk(c, box, f_c, df, C)


def test_verdict_of_no_checks_proves_nothing_and_one_check_suffices():
    from t3cap.verdict import Check, verdict_of
    assert not verdict_of([]).proven
    assert verdict_of([Check("only", True, "")]).proven
    assert not verdict_of([Check("only", False, "")]).proven


def test_certify_return_accepts_any_positive_return_time_and_exact_tube_cover():
    short = enclosure(return_time=iv("0.5", "0.6"),
                      tube=TubeBounds(iv("1.7", 5), iv("1.6", 5), iv("1.8", 5), 10, iv("0.6", "0.7"), CROSSED))
    assert certify_return(short, min_distance=F(1, 10)).proven
