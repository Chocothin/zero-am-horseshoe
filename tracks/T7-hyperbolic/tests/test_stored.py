"""저장 기록 추출 명세 시험. 기대값은 「D(B) 를 원뿔 판정에 쓸 자격 = T6 사용 조건(certify_return 항목, c1 창 < 둘째 영점 창)
· 덮개 재판정 · 같은 설계 · 단일 중심 기록」이라는 정리 A′ 의 가정에서 정했다."""
import copy
import json
from fractions import Fraction as F

from t3cap.exact import Interval
from t3cap.export import interval_from_json, to_json
from t7hb import stored
from t7hb.results import T6_DESIGN, T6_RESULTS, sha256_file

DESIGN_HASH = sha256_file(T6_DESIGN / "graph_design.json")


def record(edge="N0__N1"):
    return json.loads((T6_RESULTS / "coverings" / f"{edge}.json").read_text())


def test_real_record_is_usable_and_derivative_is_parsed_exactly():
    """T6 본 실행 기록 - 사용 가능하고 D(B) 는 JSON 끝점 그대로(정확 유리수)다."""
    rec = record()
    c = stored.cone_input(rec, DESIGN_HASH)
    assert c.usable and c.edge == "N0>N1"
    assert c.derivative[2][3] == interval_from_json(rec["derivative"][2][3])


def test_other_design_is_rejected():
    """다른 설계로 낸 기록 - 틀·반경이 다르므로 쓰지 않는다."""
    assert not stored.cone_input(record(), "0" * 64).usable


def test_piece_method_record_is_rejected():
    """조각 분할 기록 - 상자 전체 D(B) 가 없으므로 쓰지 않는다."""
    rec = record()
    rec["method"] = "pieces"
    assert not stored.cone_input(rec, DESIGN_HASH).usable


def test_c1_window_reaching_second_zero_is_rejected():
    """c1 귀환 창이 둘째 영점 창에 닿으면 D(B) 가 첫 통과 사상의 미분임을 보장하지 못한다(BRIEF rev9)."""
    rec = record()
    whole = rec["pieces_detail"]["whole"][0]
    z2 = interval_from_json(whole["second_zero_time"])
    c1 = interval_from_json(whole["c1_return_time"])
    whole["c1_return_time"] = to_json(Interval(c1.lo, z2.lo))
    assert not stored.cone_input(rec, DESIGN_HASH).usable


def test_tube_below_min_distance_is_rejected():
    """흐름 관 r_ij 하한이 1/5 이하이면 사용 조건 위반이다(certify_return 의 충돌 배제 항목)."""
    rec = record()
    whole = rec["pieces_detail"]["whole"][0]
    r = interval_from_json(whole["tube"]["r12"])
    whole["tube"]["r12"] = to_json(Interval(F(1, 5), r.hi))
    assert not stored.cone_input(rec, DESIGN_HASH).usable


def test_unproven_covering_is_rejected():
    """출구 면 상이 문턱 1 을 넘지 못하면 덮개가 증명되지 않으므로 그 변의 원뿔 주장도 세우지 않는다."""
    rec = copy.deepcopy(record())
    face = rec["pieces_detail"]["face_w0+1"][0]
    img = [interval_from_json(x) for x in face["image"]]
    img[0] = Interval(F(1, 2), img[0].hi)
    face["image"] = [to_json(iv) for iv in img]
    assert not stored.cone_input(rec, DESIGN_HASH).usable


def test_missing_derivative_is_not_usable():
    """D(B) 가 없는 기록 - 판정 입력이 없다."""
    rec = record()
    rec["derivative"] = None
    c = stored.cone_input(rec, DESIGN_HASH)
    assert c.derivative is None and not c.usable


def test_compact_row_round_trip():
    """G8 압축 기록 - D(B) 와 저장된 사용 판정을 그대로 옮긴다."""
    rec = record()
    c = stored.compact_input({"edge": "N0>N1", "derivative": rec["derivative"], "problems": []})
    assert c.usable and c.derivative == stored.derivative_of(rec)
    assert not stored.compact_input({"edge": "N0>N1", "derivative": rec["derivative"], "problems": ["x"]}).usable
