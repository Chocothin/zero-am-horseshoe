"""V2 흐름 관 인증기의 엄밀성 변이(R1-07)를 레포 밖 사본에 심고, 어떤 시험·대조가 잡는지 기록한다.

변이: (가) 단계 갱신의 Lagrange 나머지(remQ·remV × [0,h]^{n+1}) 삭제, (나) 교차 창의 횡단성 검사(Ȧ<0) 삭제.
시험: tests/test_tube_spec.py(기존 명세 시험), tests/test_tube_rigor_spec.py(엄밀성 시험), V2 음성 대조 rigorous_controls.py,
그리고 변이 사본으로 다시 낸 확인 변 10개의 흐름 관에 저장된 G5 무작위 점·꼭짓점을 대조한다.
사용: t11cli.py tube-mutants [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import g5
from .mutant_probe import EDGES
from .mutation import staged_copy
from .paths import BLAS_ONE, BOXES, PYTHON, RESULTS, edge_stem

OUT = RESULTS / "tube_rigor_mutants.json"
TARGET = "rigorous_tube.py"
MUTANTS = {
    "step_remainder_removed": (
        "            newq.append(Aff(accq.c, accq.g, accq.r + remQ[c] * Iord))\n"
        "            newv.append(Aff(accv.c, accv.g, accv.r + remV[c] * Iord))\n",
        "            newq.append(Aff(accq.c, accq.g, accq.r))\n"
        "            newv.append(Aff(accv.c, accv.g, accv.r))\n"),
    "transversality_check_removed": (
        "            if not adot < 0:\n"
        "                raise FlightFailure(\"crossing not certified transversal (dA/dt may vanish)\")\n",
        "            pass\n"),
}
RUNS = {
    "test_tube_spec": ["-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_tube_spec.py"],
    "test_tube_rigor_spec": ["-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_tube_rigor_spec.py"],
}


def run(cmd, cwd, timeout=1800) -> dict:
    started = time.time()
    proc = subprocess.run(["nice", "-n", "10", str(PYTHON), *cmd], cwd=cwd, capture_output=True, text=True,
                          env={**os.environ, **BLAS_ONE}, timeout=timeout)
    return {"exit": proc.returncode, "tail": (proc.stdout[-500:] + proc.stderr[-300:]).strip(),
            "seconds": round(time.time() - started, 1)}


def g5_against_mutant_tubes(track: Path, work: Path) -> dict:
    """변이 사본의 certify_tubes 로 확인 변의 흐름 관을 다시 내고, 저장된 G5 무작위 점·꼭짓점이 그 관·창 안에 드는지 센다."""
    random_rows = g5.load_rows(False)
    vertex_rows = g5.load_rows(False, True) if g5.VERTEX_POINTS.exists() else {}
    out = {"random_points": 0, "random_caught": 0, "vertex_points": 0, "vertex_caught": 0, "failed_checks": {}}
    for e in EDGES:
        p = work / f"{edge_stem(e)}.json"
        r = run(["-I", "-B", "certify_tubes.py", "--edges", e, "--out", str(p)], track)
        if not p.exists():
            out.setdefault("tube_failures", []).append({"edge": e, "tail": r["tail"][-200:]})
            continue
        tube = json.loads(p.read_text())["rows"][0]
        if not tube.get("certified"):
            out.setdefault("tube_failures", []).append({"edge": e, "reason": tube.get("reason")})
            continue
        box = json.loads((BOXES / f"{edge_stem(e)}.json").read_text())
        for kind, rows in (("random", random_rows), ("vertex", vertex_rows)):
            for pt in (x for x in rows.values() if x["edge"] == e and "error" not in x):
                fails = [k for k, (mg, _) in g5.judge_point(pt, tube, box).items() if mg < 0]
                out[f"{kind}_points"] += 1
                out[f"{kind}_caught"] += bool(fails)
                for k in fails:
                    out["failed_checks"][k.split("[")[0]] = out["failed_checks"].get(k.split("[")[0], 0) + 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    started = time.time()
    out = {"tag": "R1-07: V2 엄밀성 변이를 시험이 잡는가", "target": TARGET, "mutants": {}}
    with tempfile.TemporaryDirectory() as tmp:
        base = staged_copy(Path(tmp) / "base")
        out["original"] = {k: run(v, base) for k, v in RUNS.items()}
        out["original"]["rigorous_controls"] = run(["-I", "-B", "rigorous_controls.py", "--out",
                                                    str(Path(tmp) / "controls_base.json")], base)
        for name, (old, new) in MUTANTS.items():
            track = staged_copy(Path(tmp) / name)
            src = (track / TARGET).read_text()
            if src.count(old) != 1:
                print(f"{name}: mutation site not found exactly once", file=sys.stderr)
                return 1
            (track / TARGET).write_text(src.replace(old, new))
            runs = {k: run(v, track) for k, v in RUNS.items()}
            runs["rigorous_controls"] = run(["-I", "-B", "rigorous_controls.py", "--out",
                                             str(Path(tmp) / f"controls_{name}.json")], track)
            work = Path(tmp) / f"tubes_{name}"
            work.mkdir()
            g5_check = g5_against_mutant_tubes(track, work)
            out["mutants"][name] = {"old": old.strip(), "new": new.strip(), "runs": runs, "g5": g5_check,
                                    "caught": runs["test_tube_rigor_spec"]["exit"] != 0,
                                    "caught_by_existing_controls": runs["rigorous_controls"]["exit"] != 0,
                                    "caught_by_existing_spec": runs["test_tube_spec"]["exit"] != 0}
    out["original_pass"] = all(v["exit"] == 0 for v in out["original"].values())
    out["seconds"] = round(time.time() - started, 1)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"original_pass": out["original_pass"],
                      **{k: {kk: v[kk] for kk in ("caught", "caught_by_existing_controls", "caught_by_existing_spec")}
                         for k, v in out["mutants"].items()}}, ensure_ascii=False, indent=1))
    return 0 if out["original_pass"] and all(v["caught"] for v in out["mutants"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
