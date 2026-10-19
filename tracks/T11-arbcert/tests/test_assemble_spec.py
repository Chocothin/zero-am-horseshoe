"""조립층 시험(G9 + 값 고정). 변 조건의 기대값은 (A2) 의 가설(첫 통과·횡단·판별식·도착 근, 두 창이 같은 τ(B) 를
감쌈)과 T6 REPORT §3.3 의 그래프에서 정했다. 마지막 두 시험은 실데이터 조립 결과를 조립 코드를 쓰지 않는 독립 재계산과
맞추는 값 고정 시험이다(명세 유도 시험이 아니다)."""
from __future__ import annotations

import copy
import json
import sys
from fractions import Fraction as F
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from t11 import assemble as A  # noqa: E402
from t11 import claims as C  # noqa: E402
from t11 import design as dz  # noqa: E402
from t11.paths import BOXES, RESULTS, TUBES  # noqa: E402

TUBE = {"certified": True, "disc_lower": "1/2", "window": ["2", "3"],
        "r_lower": {"r12": "1/4", "r13": "1/3", "r23": "1/2"}, "r_upper": {"r12": "5", "r13": "6", "r23": "7"}}
BOX = {"tau_hull_exact": ["5/2", "4"], "disc_hull_exact": ["1/4", "3/4"], "branch_out_hull_exact": ["1/10", "1/5"]}


def test_structure_holds_for_consistent_records():
    """모든 조건이 성립하는 흐름 관·상자 기록 - 실패 목록이 비어 있다."""
    assert A.structure_failures(TUBE, BOX) == ()


@pytest.mark.parametrize("change, expect", [
    (("tube", None, None), "tube record missing"),
    (("box", None, None), "box record missing"),
    (("tube", "certified", False), "tube not certified"),
    (("tube", "disc_lower", "0"), "V2 discriminant"),
    (("tube", "window", ["0", "3"]), "V2 return window not positive"),
    (("box", "tau_hull_exact", ["7/2", "4"]), "disjoint"),
    (("box", "disc_hull_exact", ["0", "1"]), "V5 discriminant"),
    (("box", "branch_out_hull_exact", ["0", "1"]), "V5 arrival root"),
    (("box", "tau_hull_exact", ["0", "4"]), "V5 return window not positive"),
])
def test_each_structure_condition_is_enforced(change, expect):
    """조건 하나만 깬 기록 - 그 조건의 실패가 보고된다((A2) 조건은 structure_failures 한 곳에서 선언)."""
    which, key, value = change
    tube, box = copy.deepcopy(TUBE), copy.deepcopy(BOX)
    if key is None:
        tube, box = (None, box) if which == "tube" else (tube, None)
    else:
        (tube if which == "tube" else box)[key] = value
    fails = A.structure_failures(tube, box)
    assert any(expect in f for f in fails), fails


def test_tube_lower_bound_must_be_positive():
    """흐름 관 r_ij 하한이 0 - 충돌 배제가 아니라 구조 실패."""
    tube = copy.deepcopy(TUBE)
    tube["r_lower"]["r13"] = "0"
    assert any("r_ij lower" in f for f in A.structure_failures(tube, BOX))


def test_edge_verdict_cone_uses_T7_constants_strictly():
    """원뿔 판정 - a*<μ_s=0.8944, b*>μ_u=1.1167 엄격 부등식(같으면 거짓), 차수 = s₀s₁."""
    j = {"a_star_exact": "0.8944", "b_star_exact": "1.2", "cover_margin_mv_exact": "1/10"}
    v = A.edge_verdict("A>B", TUBE, BOX, j, [1, -1])
    assert not v.cone_proven_t7 and v.cover_proven and v.degree == -1
    j2 = {"a_star_exact": "0.8943", "b_star_exact": "1.1168", "cover_margin_mv_exact": "0"}
    v2 = A.edge_verdict("A>B", TUBE, BOX, j2, [1, 1])
    assert v2.cone_proven_t7 and not v2.cover_proven


def test_expected_graph_has_two_cycles_sharing_N8():
    """기대 그래프 - 변 314개, cycle 16 + 접합 2 + 사슬 296, N8 만 후속·선행이 둘(T6 §3.3)."""
    E = A.expected_edges()
    assert len(E) == 314
    succ = {}
    pred = {}
    for s, t in E:
        succ.setdefault(s, []).append(t)
        pred.setdefault(t, []).append(s)
    assert sorted(succ["N8"]) == ["M-148", "N9"] and sorted(pred["N8"]) == ["M+148", "N7"]
    assert all(len(v) == 1 for k, v in succ.items() if k != "N8") and len(succ) == 313


def test_hull_gap_of_identical_hsets_is_not_positive():
    """같은 h-set 두 개 - 외피 분리 간격 ≤ 0(서로소 아님)."""
    d = copy.deepcopy(dz.design())
    d["vertices"]["M+10"] = copy.deepcopy(d["vertices"]["M+0"])
    gap, where = A.hull_gaps(d)["A3_gap"]
    assert gap <= 0 and where == "M+10"


def real():
    out = json.loads((RESULTS / "theorem_claims.json").read_text())
    return out


@pytest.mark.skipif(not (RESULTS / "theorem_claims.json").exists(), reason="본 실행 결과가 아직 없다")
def test_real_assembly_extremes_match_independent_recomputation():
    """값 고정: theorem_claims.json 의 δ·R·t_min·τ 상한·덮개 여유·a*·b*·g(1)·μ 여유가 변별 기록에서 조립 코드 없이 다시 잰 값과 같다."""
    out = real()
    tubes = [json.loads(p.read_text())["rows"][0] for p in sorted(TUBES.glob("*.json"))]
    judged = json.loads((RESULTS / "exact_judge.json").read_text())["v5_stage2"]["rows"]
    assert len(tubes) == 314 and len(judged) == 314
    delta = min(F(t["r_lower"][p]) for t in tubes for p in ("r12", "r13", "r23"))
    R = max(F(t["r_upper"][p]) for t in tubes for p in ("r12", "r13", "r23"))
    tmin = min(F(t["window"][0]) for t in tubes)
    tmax = max(F(t["window"][1]) for t in tubes)
    cov = min(F(j["cover_margin_mv_exact"]) for j in judged)
    a = max(F(j["a_star_exact"]) for j in judged)
    b = min(F(j["b_star_exact"]) for j in judged)
    m = out["measured"]
    assert F(m["delta"]["exact"]) == delta and F(m["R"]["exact"]) == R
    assert F(m["t_min"]["exact"]) == tmin and F(m["tau_max"]["exact"]) == tmax
    assert F(m["cover_margin"]["exact"]) == cov
    assert F(m["a_max"]["exact"]) == a and F(m["b_min"]["exact"]) == b
    assert F(m["g1"]["exact"]) == min(min(F(j["b_star_exact"]) - 1, 1 - F(j["a_star_exact"])) for j in judged)
    checks = out["checks"]
    assert F(checks["mu_s_margin"]["measured"]) == C.THEOREM["T7.mu_s"] - a
    assert F(checks["mu_u_margin"]["measured"]) == b - C.THEOREM["T7.mu_u"]


@pytest.mark.skipif(not (RESULTS / "theorem_claims.json").exists(), reason="본 실행 결과가 아직 없다")
def test_real_assembly_reproduces_stored_claims():
    """값 고정: 저장 기록으로 다시 조립하면 측정값·판정·구조 실패 목록이 theorem_claims.json 과 같다."""
    design = dz.design()
    result = A.assemble(A.load_records(TUBES, BOXES, design), design)
    rep = A.report(result)
    stored = real()
    for k in ("measured", "checks", "degrees", "structure_failed", "cover_unproven", "g1_nonpositive", "all_claims_hold"):
        assert rep[k] == stored[k], k


def test_degenerate_or_touching_windows_are_consistent():
    """점 창 [w,w] 는 허용되고, 끝점에서만 만나는 두 창은 서로소가 아니다(같은 τ(B) 를 감싸는 두 닫힌 구간)."""
    tube = copy.deepcopy(TUBE)
    tube["window"] = ["3", "3"]
    box = copy.deepcopy(BOX)
    box["tau_hull_exact"] = ["3", "4"]
    assert A.structure_failures(tube, box) == ()
    tube["window"] = ["2", "3"]
    box["tau_hull_exact"] = ["1", "2"]
    assert A.structure_failures(tube, box) == ()


def test_edge_verdict_without_judgement_is_not_proven():
    """상자 기록이 없어 판정이 없음 - 덮개·원뿔 모두 증명이 아니다."""
    v = A.edge_verdict("A>B", TUBE, None, None, [1, 1])
    assert not v.cover_proven and not v.cone_proven_t7 and v.a_star is None


def test_edge_verdict_cone_rejects_b_equal_to_mu_u():
    """b* 가 μ_u 와 같음 - 원뿔(T7 A5)은 엄격 부등식이라 거짓."""
    j = {"a_star_exact": "0.5", "b_star_exact": "1.1167", "cover_margin_mv_exact": "1/10"}
    assert not A.edge_verdict("A>B", TUBE, BOX, j, [1, 1]).cone_proven_t7


def planted(tmp_path, edge: str, deriv):
    """실데이터 기록에서 한 변의 D(B) 만 바꾼 상자 기록(중심 상 0)을 임시 폴더에 둔다."""
    design = dz.design()
    records = A.load_records(TUBES, BOXES, design)
    rec = json.loads(Path(records[edge]["box_path"]).read_text())
    rec["deriv_mine_exact"] = [[[str(deriv[i][0]), str(deriv[i][1])] if i == j else ["0", "0"] for j in range(4)]
                               for i in range(4)]
    rec["center_image_exact"] = [["0", "0"]] * 4
    p = tmp_path / f"{edge.replace('>', '__')}.json"
    p.write_text(json.dumps(rec))
    records[edge]["box_path"] = p
    return A.assemble(records, design)


@pytest.mark.skipif(not (RESULTS / "theorem_claims.json").exists(), reason="본 실행 결과가 아직 없다")
def test_g1_is_taken_per_edge_from_the_smaller_side(tmp_path):
    """g_e(1)=min(b*−1, 1−a*) - 한 변의 D(B) 를 diag(21/20, 3/2, 1/2, 1/4) 로 두면 b*−1=41/400 이 그 변과 전체의 최솟값이다."""
    res = planted(tmp_path, "N0>N1", [(F(21, 20),) * 2, (F(3, 2),) * 2, (F(1, 2),) * 2, (F(1, 4),) * 2])
    m = res["measured"]
    assert m.values["g1"] == F(41, 400) and m.at["g1"] == "N0>N1"


@pytest.mark.skipif(not (RESULTS / "theorem_claims.json").exists(), reason="본 실행 결과가 아직 없다")
def test_g1_zero_on_either_side_breaks_A5(tmp_path):
    """g_e(1)=0 은 양정치가 아니다 - b*=1(diag(1, 3/2, 1/2, 1/4)) 이나 a*=1(diag(3/2, 3/2, 1, 1/4)) 인 변이 g1≤0 목록에 든다."""
    for diag in ([(F(1),) * 2, (F(3, 2),) * 2, (F(1, 2),) * 2, (F(1, 4),) * 2],
                 [(F(3, 2),) * 2, (F(3, 2),) * 2, (F(1),) * 2, (F(1, 4),) * 2]):
        res = planted(tmp_path, "N0>N1", diag)
        assert res["measured"].g1_nonpositive == ("N0>N1",)
        assert not next(c for c in res["checks"] if c.name == "T7.A5").holds


@pytest.mark.skipif(not (RESULTS / "theorem_claims.json").exists(), reason="본 실행 결과가 아직 없다")
def test_missing_tube_with_present_box_is_a_missing_edge():
    """상자 기록은 있고 흐름 관 기록만 없음 - 그 변은 빠진 변이고 조립은 예외 없이 주장을 거짓으로 낸다."""
    design = dz.design()
    records = A.load_records(TUBES, BOXES, design)
    records["N1>N2"]["tube"] = None
    res = A.assemble(records, design)
    assert res["measured"].missing == ("N1>N2",)
    assert not any(c.holds for c in res["checks"] if c.name != "A4")


def test_graph_check_rejects_duplicated_edges():
    """설계 변 목록에 같은 변이 두 번 - 집합은 같아도 그래프 일치(A4)가 아니다."""
    design = copy.deepcopy(dz.design())
    design["edges"].append(copy.deepcopy(design["edges"][0]))
    res = A.assemble({}, design)
    assert not res["measured"].graph_matches_expected
