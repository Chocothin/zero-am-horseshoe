"""저장 기록 → 판정 입력 추출 명세 시험. 기대값은 certify_return 의 가설(판별식>0, 도착 근>0, 귀환>0, 흐름 관이 반환을 덮음,
r_ij>1/5)을 c0·c1·점 실행 모두에, 「반환은 t>0 의 첫 지정 방향 통과」(통과 세기: 반환 창과 겹친 닿는 묶음 하나, 앞선 지정
방향 통과 0, 구간은 닫힌 구간)를 네 실행 모두에, 「c1 귀환 창 < 둘째 영점 창」을 적용해 정했다."""
from fractions import Fraction as Q

from t3cap.exact import Interval
from t3cap.export import to_json

from t6hs import records
from t6hs.covering import covering_verdict, face_box, unit_box


def iv(lo, hi):
    return Interval(Q(lo), Q(hi))


def tube(r12=("0.3", 7), r13=("0.4", 8), r23=("0.5", 6), t_end=("1.1", "1.2")):
    return {"r12": to_json(iv(*r12)), "r13": to_json(iv(*r13)), "r23": to_json(iv(*r23)), "t_end": to_json(iv(*t_end))}


def run(return_time=(1, "1.01"), discriminant=(1, 2), branch_out=(1, 2), tb=None, window_runs=1, counted_before=0,
        window_run=None):
    """엔진 실행 하나의 기록(certify_graph.run_record 의 꼴). 창 묶음 시각은 따로 주지 않으면 반환 창과 같다."""
    return {"return_time": to_json(iv(*return_time)), "discriminant": to_json(iv(*discriminant)),
            "branch_out": to_json(iv(*branch_out)), "tube": {**(tb or tube()), "steps": 10},
            "crossings": {"window_runs": window_runs, "counted_before": counted_before, "runs_before": 1,
                          "window_run": to_json(iv(*(window_run or return_time)))},
            "image": to_json(tuple(iv(-2, 2) for _ in range(4)))}


def runs_entry(c0=None, c1=None, second_zero=None, point=None):
    return {"c0": c0 or run(), "c1": c1 or run(), "second_zero": second_zero or run((2, "2.1"), tb=tube(t_end=("2.2", "2.3"))),
            "point": point or run()}


def piece(box, entry, index=0, **over):
    """조각 기록. c0 흐름 관·귀환 창·판별식·도착 근과 c1·둘째 영점 창은 certify_graph 처럼 실행 기록에서 옮겨 적는다."""
    c0 = entry["c0"]
    p = {"piece": to_json(box), "image": to_json((iv(-2, 2), iv(-2, 2), iv(Q(-1, 2), Q(1, 2)), iv(Q(-1, 2), Q(1, 2)))),
         "certified": True, "failures": [], "tube": {k: c0["tube"][k] for k in records.TUBE_KEYS},
         "return_time": c0["return_time"], "c1_return_time": entry["c1"]["return_time"],
         "second_zero_time": entry["second_zero"]["return_time"], "discriminant": c0["discriminant"],
         "branch_out": c0["branch_out"], "seconds": 0.0, "runs_index": index}
    p.update(over)
    return p


def record(whole=None, entry=None, runs=None, **over):
    entry = entry or runs_entry()
    detail = {"whole": [piece(unit_box(), entry, **over)] if whole is None else whole}
    for j, s in records.FACE_KEYS:
        out = [iv(-1, 1), iv(-1, 1), iv(0, 0), iv(0, 0)]
        out[j] = iv(2 * s, 3 * s) if s > 0 else iv(3 * s, 2 * s)
        detail[f"face_w{j}{s:+d}"] = [piece(face_box(j, s), entry, image=to_json(tuple(out)))]
    return {"edge": "N0>N1", "signs": [1, 1], "runs": [entry] if runs is None else runs, "pieces_detail": detail}


def unusable(rec) -> bool:
    c = records.covering_input(rec)
    return bool(c.problems) and not covering_verdict(c.whole, c.faces, c.signs).proven


def test_clean_record_is_usable_and_carries_constants():
    """조건을 모두 만족하는 기록 - 문제 없음, 흐름 관 하한 0.3·상한 8·귀환 시각 하한 1 을 그대로 옮김"""
    c = records.covering_input(record())
    assert c.problems == () and all(p.certified for p in c.whole)
    assert covering_verdict(c.whole, c.faces, c.signs).proven
    assert (c.tube_lower, c.tube_upper, c.return_time_lower) == (Q(3, 10), Q(8), Q(1))
    assert set(c.faces) == set(records.FACE_KEYS)


def test_each_c0_condition_is_enforced_on_stored_data():
    """c0 실행의 가설을 하나씩 깸(조각 사본도 같이) - 문제가 기록되고, 그 기록으로 낸 판정은 증명되지 않음"""
    breaks = [run(return_time=(0, "1.01")), run(tb=tube(t_end=("1.005", "1.2"))), run(tb=tube(r12=("0.2", 7))),
              run(tb=tube(r13=("0.19", 7))), run(tb=tube(r23=("0.1", 7))), run(discriminant=(0, 2)),
              run(branch_out=(0, 2))]
    for c0 in breaks:
        assert unusable(record(entry=runs_entry(c0=c0)))


def test_piece_without_usable_engine_record_is_unusable():
    """조각 표시·상·흐름 관·실행 기록 색인 중 하나가 없거나 범위 밖 - 사용 불가(상 없는 조각은 빠져 덮음도 깨짐)"""
    for over in ({"certified": False}, {"image": None}, {"tube": None}, {"runs_index": None}, {"runs_index": 1},
                 {"runs_index": -1}):
        assert unusable(record(**over))


def test_c1_and_point_runs_must_satisfy_certify_return():
    """c1·점 실행의 certify_return 항목을 하나씩 깸 - 조각에 옮겨 적지 않은 값이라도 저장 기록에서 다시 봐서 사용 불가"""
    for key in ("c1", "point"):
        for bad in (run(return_time=(0, "1.01")), run(tb=tube(t_end=("1.005", "1.2"))), run(tb=tube(r12=("0.2", 7))),
                    run(tb=tube(r13=("0.1", 7))), run(tb=tube(r23=("0.19", 7))), run(discriminant=(-1, 2)),
                    run(branch_out=(0, 2))):
            assert unusable(record(entry=runs_entry(**{key: bad})))


def test_every_run_must_report_the_first_crossing():
    """네 실행마다 통과 세기 결과를 깸(창 묶음 2개·0개, 앞선 지정 방향 통과 1개, 창 묶음이 반환 창과 떨어짐) - 사용 불가"""
    for key in records.RUN_KEYS:
        rt = (2, "2.1") if key == "second_zero" else (1, "1.01")
        tb = tube(t_end=("2.2", "2.3")) if key == "second_zero" else None
        for bad in (run(rt, tb=tb, window_runs=2), run(rt, tb=tb, window_runs=0), run(rt, tb=tb, counted_before=1),
                    run(rt, tb=tb, window_run=("0.2", "0.5"))):
            assert unusable(record(entry=runs_entry(**{key: bad}))), key


def test_window_run_touching_the_return_window_is_enough():
    """창 묶음이 반환 창과 끝점에서만 닿음(양쪽) - 사용 가능(닫힌 구간의 겹침)"""
    for window in (("1.01", "1.2"), ("0.5", 1)):
        assert records.covering_input(record(entry=runs_entry(c0=run(window_run=window)))).problems == ()


def test_second_zero_run_is_not_held_to_certify_return():
    """둘째 영점 실행의 판별식·흐름 관은 사용 조건이 아님 - 그 값이 나빠도 사용 가능(살아 있는 경로 usage_failures 와 같은 조건)"""
    loose = run((2, "2.1"), discriminant=(-1, 2), tb=tube(r12=("0.1", 7), t_end=("2.0", "2.05")))
    assert records.covering_input(record(entry=runs_entry(second_zero=loose))).problems == ()


def test_c1_window_must_end_before_the_second_zero():
    """c1 귀환 창이 둘째 영점 창에 닿거나 겹침 - 사용 불가(엄격 부등식이어야 c1 이 첫 통과를 미분했다)"""
    for c1_rt in ((1, 2), (1, "2.05")):
        c1 = run(c1_rt, tb=tube(t_end=("2.2", "2.3")))
        assert unusable(record(entry=runs_entry(c1=c1)))


def test_piece_copy_must_match_the_run_record():
    """조각에 옮겨 적은 값 하나가 실행 기록과 다름(흐름 관 한 성분, 판별식, c1 창, 둘째 영점 창) - 사용 불가"""
    entry = runs_entry()
    for over in ({"tube": {**{k: entry["c0"]["tube"][k] for k in records.TUBE_KEYS}, "t_end": to_json(iv("1.15", "1.2"))}},
                 {"discriminant": to_json(iv("1.5", 2))}, {"c1_return_time": to_json(iv(1, "1.009"))},
                 {"second_zero_time": to_json(iv("2.01", "2.1"))}):
        assert unusable(record(entry=entry, **over))


def test_record_without_runs_block_is_unusable():
    """실행 기록이 없는 옛 기록(v2 꼴) - 모든 조각이 사용 불가"""
    rec = record()
    del rec["runs"]
    assert unusable(rec)


def test_tube_reaching_exactly_the_return_window_is_usable():
    """흐름 관 끝 시각 하한 = 귀환 창 상한 - 사용 가능(T3 certify_return 과 같이 닫힌 시간 구간 [0, t_end] 가 반환을 덮음)"""
    c = records.covering_input(record(entry=runs_entry(c0=run(tb=tube(t_end=("1.01", "1.2"))))))
    assert c.problems == () and covering_verdict(c.whole, c.faces, c.signs).proven


LEFT = (iv(-1, 0), iv(-1, 1), iv(-1, 1), iv(-1, 1))
RIGHT = (iv(0, 1), iv(-1, 1), iv(-1, 1), iv(-1, 1))


def test_constants_are_extrema_over_whole_pieces():
    """상자 조각 둘(실행 기록도 둘) - 흐름 관 하한=두 조각 r_ij 하한의 최솟값 1/4, 상한=최댓값 9, 귀환 시각 하한=최솟값 1"""
    first = runs_entry()
    second = runs_entry(c0=run(("1.2", "1.21"), tb=tube(r12=("0.25", 9), t_end=("1.3", "1.4"))),
                        c1=run(("1.2", "1.21"), tb=tube(t_end=("1.3", "1.4"))))
    rec = record(whole=[piece(LEFT, first, 0), piece(RIGHT, second, 1)], entry=first, runs=[first, second])
    c = records.covering_input(rec)
    assert c.problems == () and covering_verdict(c.whole, c.faces, c.signs).proven
    assert (c.tube_lower, c.tube_upper, c.return_time_lower) == (Q(1, 4), Q(9), Q(1))


def test_record_without_whole_pieces_is_unusable():
    """상자 조각이 하나도 없음 - 상수는 기본값 0(δ>0 가정을 못 넘김)이고 판정은 덮음 실패"""
    c = records.covering_input(record(whole=[]))
    assert (c.tube_lower, c.tube_upper, c.return_time_lower) == (0, 0, 0)
    assert not covering_verdict(c.whole, c.faces, c.signs).proven
