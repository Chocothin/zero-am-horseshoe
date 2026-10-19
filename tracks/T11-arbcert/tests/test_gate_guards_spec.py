"""게이트 가드 시험(sentinel-cap-p2 r1 권고 1~3): G5 첫 영점 전 표본이 비면 이탈, G6 심은 입력은 판정 가드 예외만 거절로
세고 코드 버그 예외는 올림, 독립성 감사의 금지 경로에 새 CAPD 결과 폴더가 든다. 기대값은 BRIEF §3 의 치명 가드에서 정했다."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import v7exact as X  # noqa: E402
from t11 import controls, g5, independence  # noqa: E402
from t11.paths import DESIGN, REPO, RESULTS  # noqa: E402


def test_guarded_returns_result_when_no_exception():
    assert controls.guarded(lambda: 7) == (7, None)


def test_guarded_turns_judge_guard_exception_into_rejection():
    """v7exact.iv 의 lo>hi 거부는 판정 경로가 일부러 내는 가드 예외다 - 거절(결과 없음 + 예외)로 돌려준다."""
    res, exc = controls.guarded(lambda: X.iv({"lo": "1", "hi": "0"}))
    assert res is None and isinstance(exc, ValueError)


@pytest.mark.parametrize("bug", [KeyError("tube"), TypeError("bad"), AttributeError("x"), ZeroDivisionError()])
def test_guarded_reraises_code_bug_exceptions(bug):
    """심기 코드의 버그가 거절로 숨으면 leaked 0 이 거짓 안심이 된다 - 그대로 올라와야 한다."""
    def action():
        raise bug
    with pytest.raises(type(bug)):
        controls.guarded(action)


HAVE_G5 = (RESULTS / "g5_points.jsonl").exists()


@pytest.mark.skipif(not HAVE_G5, reason="G5 기준 해가 아직 없다")
def test_missing_samples_before_first_zero_count_as_violation():
    """min_A_before 가 비면 (0,τ) 에 영점이 없다는 표본 확인이 없는 것이다 - 통과가 아니라 이탈(음수 여유)이어야 한다."""
    row = json.loads((RESULTS / "g5_points.jsonl").read_text().splitlines()[0])
    tube, box = g5.records(row["edge"], {})
    margin, _ = g5.judge_point(row, tube, box)["first_crossing_A_positive_before"]
    assert margin > 0
    margin, _ = g5.judge_point({**row, "min_A_before": None}, tube, box)["first_crossing_A_positive_before"]
    assert margin < 0


@pytest.mark.parametrize("path", ["results/2026-10-17-T6-horseshoe/coverings/N0__N1.json",
                                  "results/2026-10-17-T7-hyperbolic/theorem_Aprime.json",
                                  "results/2026-10-15-T6-horseshoe/coverings/N0__N1.json",
                                  "tracks/T3-cap/t3cap/api.py", "tracks/V5-indep/v5/flow.py"])
def test_independence_forbids_capd_results_and_tracks(path):
    assert independence.forbidden(str(REPO / path))


def test_independence_allows_design_and_own_results():
    assert not independence.forbidden(str(DESIGN))
    assert not independence.forbidden(str(RESULTS / "stage2" / "N0__N1.json"))
