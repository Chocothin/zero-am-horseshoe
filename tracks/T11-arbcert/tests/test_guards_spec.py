"""치명 가드 시험(R1-02): 감사 사본 stage2.py·exact_judge.py 는 판정이 거짓이거나 변이 빠져도 exit 0 이므로, T11 의 조립
(t11cli assemble)과 실행기 상태(jobs status --strict)가 314변 전부가 증명되지 않거나 빠진 변이 있으면 exit 1 로 끝나는지 고정한다.
기대값은 BRIEF §3 의 치명 가드(검증기는 실패를 삼키지 않고 반드시 실패로 끝난다)에서 정했다."""
from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction as F
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from t11 import jobs  # noqa: E402
from t11.paths import BOXES, PYTHON, RESULTS, TRACK, TUBES  # noqa: E402

HAVE_RESULTS = (RESULTS / "theorem_claims.json").exists() and len(list(BOXES.glob("*.json"))) == 314
pytestmark = pytest.mark.skipif(not HAVE_RESULTS, reason="본 실행 결과가 아직 없다")


def stage(tmp_path: Path, skip_box=None, plant_box=None, plant_tube=None):
    """변별 기록을 임시 폴더에 링크하고, 지정한 변만 빼거나 바꾼 사본으로 둔다."""
    tdir, bdir = tmp_path / "tubes", tmp_path / "stage2"
    tdir.mkdir()
    bdir.mkdir()
    for p in TUBES.glob("*.json"):
        if plant_tube and p.stem == plant_tube[0]:
            rec = json.loads(p.read_text())
            plant_tube[1](rec["rows"][0])
            (tdir / p.name).write_text(json.dumps(rec))
        else:
            (tdir / p.name).symlink_to(p)
    for p in BOXES.glob("*.json"):
        if p.stem == skip_box:
            continue
        if plant_box and p.stem == plant_box[0]:
            (bdir / p.name).write_text(json.dumps(plant_box[1](json.loads(p.read_text()))))
        else:
            (bdir / p.name).symlink_to(p)
    return tdir, bdir


def assemble_exit(tmp_path: Path, tdir: Path, bdir: Path) -> int:
    cmd = [str(PYTHON), "-I", "-B", str(TRACK / "t11cli.py"), "assemble", "--tubes", str(tdir), "--boxes", str(bdir),
           "--out", str(tmp_path / "claims.json"), "--merged-tubes", str(tmp_path / "merged.json")]
    return subprocess.run(cmd, capture_output=True, text=True).returncode


def widen(rec, delta=F(1, 10)):
    rec["center_image_exact"] = [[str(F(lo) - delta), str(F(hi) + delta)] for lo, hi in rec["center_image_exact"]]
    return rec


def test_assemble_exits_zero_on_complete_proven_records(tmp_path):
    """314변 기록이 모두 있고 모두 증명 - 조립이 exit 0."""
    assert assemble_exit(tmp_path, *stage(tmp_path)) == 0


def test_assemble_exits_one_when_a_box_record_is_missing(tmp_path):
    """상자 기록 하나(M+0⟹M+1)가 빠짐 - 조립이 exit 1(변이 빠진 채 정리를 주장하지 않는다)."""
    assert assemble_exit(tmp_path, *stage(tmp_path, skip_box="M+0__M+1")) == 1


def test_assemble_exits_one_when_a_covering_is_not_proven(tmp_path):
    """한 변(N0⟹N1)의 중심 상을 넓혀 덮개가 거짓 - 조립이 exit 1(stage2·exact_judge 는 이 경우에도 exit 0 이다)."""
    assert assemble_exit(tmp_path, *stage(tmp_path, plant_box=("N0__N1", widen))) == 1


def test_assemble_exits_one_when_a_tube_is_not_certified(tmp_path):
    """흐름 관 하나(N1⟹N2)가 인증 실패 - 조립이 exit 1."""
    tdir, bdir = stage(tmp_path, plant_tube=("N1__N2", lambda r: r.update({"certified": False, "reason": "planted"})))
    assert assemble_exit(tmp_path, tdir, bdir) == 1


def test_strict_job_status_fails_on_missing_or_failed_jobs(tmp_path, monkeypatch):
    """실행기 상태 --strict - 출력 없는 작업이나 마지막 실행이 실패한 작업이 있으면 1, 모두 끝나고 성공이면 0."""
    tdir, bdir = tmp_path / "tubes", tmp_path / "stage2"
    tdir.mkdir()
    bdir.mkdir()
    log = tmp_path / "run_log.jsonl"
    monkeypatch.setattr(jobs, "TUBES", tdir)
    monkeypatch.setattr(jobs, "BOXES", bdir)
    monkeypatch.setattr(jobs, "RUN_LOG", log)
    monkeypatch.setattr(jobs, "LOCKS", tmp_path / "locks")
    assert jobs.status(strict=True) == 1
    for j in jobs.all_jobs():
        j.output.write_text("{}")
    log.write_text("".join(json.dumps({"job": j.name, "kind": j.kind, "edge": j.edge, "exit": 0, "output_written": True,
                                       "seconds": 1}) + "\n" for j in jobs.all_jobs()))
    assert jobs.status(strict=True) == 0
    with open(log, "a") as f:
        f.write(json.dumps({"job": "box__N0__N1", "kind": "box", "edge": "N0>N1", "exit": 1, "output_written": True,
                            "seconds": 1}) + "\n")
    assert jobs.status(strict=True) == 1
    assert jobs.status(strict=False) == 0
