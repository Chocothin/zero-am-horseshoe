"""API 계약 시험: 함정 ① 상수 입력, 함정 ② 출력 단면 통과 세기, 함정 ⑥ 면 전체 덮개, 틀 역행렬, JSON 끝점, 엔진 프로토콜 거부 사례."""
from __future__ import annotations

import subprocess
from fractions import Fraction

import pytest

from t3cap import CoordinateSection, Failure, Frame, Interval, SystemParams, face_boxes, poincare_c0
from t3cap.engine import EngineError, build_request, engine_binary, parse_response
from t3cap.exact import engine_token, rational
from t3cap.export import interval_from_json, interval_json, to_json
from t3cap.model import Settings

F = Fraction


def test_float_constants_are_rejected_at_the_api_boundary():
    with pytest.raises(TypeError):
        rational(0.1)
    assert rational("0.1") == F(1, 10) and rational("2/3") == F(2, 3)


def test_engine_tokens_are_exact_or_outward():
    assert engine_token(F(2, 3), upper=False) == "2/3"
    q = F(1, 3) + F(1, 10**40)  # 분모가 2^53 을 넘으면 바깥쪽 16진 double
    lo, hi = engine_token(q, upper=False), engine_token(q, upper=True)
    assert F(float.fromhex(lo)) <= q <= F(float.fromhex(hi))


def test_engine_rejects_decimal_endpoints():
    """프로토콜 수준의 함정 ①: '0.1' 같은 10진 끝점은 엔진이 거부한다."""
    request = build_request(
        "vf", [tuple(Interval.point(1) for _ in range(6))], system=SystemParams.make(),
        section_in=CoordinateSection.syzygy(), section_out=CoordinateSection.syzygy(), returns=1,
        frame_in=Frame.identity(), frame_out=Frame.identity(), settings=Settings(), rotation=False)
    bad = request.replace("energy -1/2 -1/2", "energy -0.5 -0.5")
    proc = subprocess.run([str(engine_binary("native-strict"))], input=bad, capture_output=True, text=True)
    assert proc.returncode != 0 and "hex float or p/q" in proc.stdout
    with pytest.raises(EngineError):
        parse_response(proc.stdout, 1)


# 8자 궤도의 Σ+ 통과점 근처. 이 점은 오일러 배치라 r12 가 최대(p_r≈0)이고, 반 주기(약 13.064) 뒤에 다시 최대가 된다
FIGURE8_U = (F(5148567982527431, 10**15), F(0), F(0), F(432690650760826, 10**15))
EYE = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))


def r12_max_map(pr_offset: Fraction, returns: int = 1):
    """Σ+ 위 반경 1e-6 상자 → r12 최대 단면(p_r=0 을 감소하며 지남) returns 번째 통과."""
    u = (FIGURE8_U[0], FIGURE8_U[1], FIGURE8_U[2] + pr_offset, FIGURE8_U[3])
    box = tuple(Interval(F(-1, 10**6), F(1, 10**6)) for _ in range(4))
    return poincare_c0([box], system=SystemParams.make(), section_in=CoordinateSection.syzygy(+1),
                       section_out=CoordinateSection(3, F(0), -1, 5, -1), returns=returns, frame_in=Frame.make(u, EYE),
                       frame_out=Frame.make((u[0], u[1], 0, u[3]), EYE))[0]


def test_start_straddling_the_output_section_in_the_counted_direction_is_refused():
    """함정 ② 음성 대조: 상자가 p_r=0 에 걸치면 절반은 출발 직후 세는 방향으로 통과한다. CAPD 는 단면을 떠나는 동안의
    통과를 세지 않으므로 그대로 두면 그 절반에 대해 n번째가 아닌 통과를 돌려준다(수정 전 G5: 1000점 중 491점 위반)."""
    result = r12_max_map(F(0))
    assert isinstance(result, Failure) and "straddles the output section" in result.reason


def test_start_off_the_output_section_returns_the_next_r12_maximum():
    result = r12_max_map(F(-3, 10**6))
    assert not isinstance(result, Failure)
    assert F(129, 10) < result.return_time.lo and result.return_time.hi < F(132, 10)


def test_successful_return_reports_one_window_run_and_prior_crossings_returns_minus_one():
    """성공한 감싸기의 통과 세기 결과 - 반환 창과 겹친 묶음 1개, 그 앞 지정 방향 통과 returns−1 개, 창 묶음 시각이 반환 창과
    겹침. p_r>0 에서 출발하면 출발 직후 첫 통과가 1번째이고 반 주기 뒤 r12 최대가 2번째다(8자 궤도 대칭)"""
    for returns in (1, 2):
        result = r12_max_map(F(3, 10**6), returns)
        assert not isinstance(result, Failure)
        c = result.tube.crossings
        assert (c.window_runs, c.counted_before) == (1, returns - 1)
        assert c.runs_before >= returns - 1
        assert c.window_run.lo <= result.return_time.hi and result.return_time.lo <= c.window_run.hi
    assert r12_max_map(F(3, 10**6), 2).return_time.lo > F(129, 10)


def test_py_chart_is_singular_where_y_is_extremal_and_px_chart_is_not():
    """함정 ⑤ 의 도착 쪽: r12 최소(p_r=0 증가, 약 T/4)는 이등변 배치라 y 가 극값이고 ẏ=∂H/∂p_y=0 이다.
    그 단면에서 p_y 를 풀면 근 선택(branch_out)을 확정할 수 없고, p_x 를 풀면 확정된다."""
    u = (FIGURE8_U[0], FIGURE8_U[1], FIGURE8_U[2] - F(3, 10**6), FIGURE8_U[3])
    box = tuple(Interval(F(-1, 10**6), F(1, 10**6)) for _ in range(4))

    def branch_out(section: CoordinateSection) -> Interval:
        result = poincare_c0([box], system=SystemParams.make(), section_in=CoordinateSection.syzygy(+1),
                             section_out=section, frame_in=Frame.make(u, EYE),
                             frame_out=Frame.make((u[0], u[1], 0, u[3]), EYE))[0]
        assert not isinstance(result, Failure)
        return result.branch_out

    solved_py = branch_out(CoordinateSection(3, F(0), +1, 5, +1))
    solved_px = branch_out(CoordinateSection(3, F(0), +1, 4, -1))
    assert solved_py.lo < 0 < solved_py.hi
    assert solved_px.lo > 0


def test_face_boxes_cover_the_whole_face_without_gaps():
    b = (Interval(F(0), F(1)), Interval(F(-1), F(1)), Interval(F(2), F(3)), Interval(F(0), F(1, 2)))
    pieces = face_boxes(b, coord=1, side=+1, pieces=3)
    assert len(pieces) == 27
    assert all(p[1] == Interval(F(1), F(1)) for p in pieces)
    for axis in (0, 2, 3):
        edges = sorted({(p[axis].lo, p[axis].hi) for p in pieces})
        assert edges[0][0] == b[axis].lo and edges[-1][1] == b[axis].hi
        assert all(edges[k][1] == edges[k + 1][0] for k in range(len(edges) - 1))


def test_frame_round_trip_is_exact():
    frame = Frame.make((1, 2, 3, 4), ((2, 1, 0, 0), (0, 1, F(1, 3), 0), (0, 0, 1, 5), (1, 0, 0, 1)))
    w = (F(1, 7), F(-2, 9), F(3, 11), F(5, 13))
    assert frame.to_frame(frame.from_frame(w)) == w


def test_interval_json_keeps_exact_endpoints_and_outward_decimals():
    iv = Interval(F(1, 3), F(2, 3))
    d = interval_json(iv)
    assert interval_from_json(d) == iv
    assert Fraction(d["lo_dec"]) <= iv.lo and Fraction(d["hi_dec"]) >= iv.hi
    dyadic = Interval(F(3, 8), F(5, 8))
    assert interval_json(dyadic)["lo"] == (0.375).hex()
    assert to_json({"x": [dyadic]})["x"][0]["hi"] == (0.625).hex()
