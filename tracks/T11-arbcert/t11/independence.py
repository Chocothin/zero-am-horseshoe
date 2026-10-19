"""독립성·재현 검사: 변 몇 개의 흐름 관·상자 감싸기·정확 판정을 open/import 감사 훅 아래 새 프로세스에서 다시 돌린다.

- 열린 파일과 불러온 모듈에 CAPD 경로(T3·T6·T7 코드, CAPD 덮개 기록, T7 결과)나 감사 트랙(tracks/V*, results/*-V*)이
  하나도 없어야 한다. h-set 정의(graph_design.json)는 허용한다.
- 다시 낸 출력이 저장된 T11 기록과 정확 문자열로 같아야 한다(시간 칸 제외).
사용: t11cli.py independence [--edges "N1>N2,M+146>M+147"] [--out PATH]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .paths import BLAS_ONE, BOXES, DESIGN, PYTHON, REPO, TRACK, TUBES, edge_stem

FORBIDDEN_PREFIXES = [REPO / "tracks" / n for n in ("T3-cap", "T6-horseshoe", "T7-hyperbolic")] + [
    REPO / "results" / "2026-10-15-T6-horseshoe" / "coverings", REPO / "results" / "2026-10-16-T7-hyperbolic",
    REPO / "results" / "2026-10-17-T6-horseshoe", REPO / "results" / "2026-10-17-T7-hyperbolic"]
FORBIDDEN_MODULE_ROOTS = ("t3cap", "t6hs", "t7hb", "t9cap", "t10gamma", "arb_flow", "tbpnum")
TIME_KEYS = {"seconds", "seconds_center", "seconds_hull"}


def forbidden(path: str) -> bool:
    p = Path(path).resolve() if path else None
    if p is None:
        return False
    if any(p == f or f in p.parents for f in FORBIDDEN_PREFIXES):
        return True
    rel = p.relative_to(REPO) if REPO in p.parents else None
    if rel is None:
        return False
    parts = rel.parts
    return (len(parts) > 1 and parts[0] == "tracks" and parts[1].startswith("V")) or (
        len(parts) > 1 and parts[0] == "results" and "-V" in parts[1])


def child(edges: list[str], out: Path) -> int:
    opened, imported = [], []

    def hook(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            opened.append(os.fsdecode(args[0]))
        elif event == "import":
            imported.append((args[0], args[1]))

    sys.addaudithook(hook)
    sys.path.insert(0, str(TRACK))
    import certify_tubes as ct
    import exact_judge as EJ

    spec = importlib.util.spec_from_file_location("stage2_t11", TRACK / "scripts" / "stage2.py")
    stage2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage2)
    design = json.loads(DESIGN.read_text())
    signs = EJ.design_signs(design)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for e in edges:
            tube = ct.certify_edge(e, design, 18, 0.08, 32)
            box = stage2.run_edge(e)
            p = Path(tmp) / f"{edge_stem(e)}.json"
            p.write_text(json.dumps(box))
            judged = EJ.judge_v5(p, signs)
            results[e] = {"tube": tube, "box": box,
                          "judged": {k: v for k, v in judged.items() if not k.startswith("_")}}
    out.write_text(json.dumps({"results": results, "opened": sorted(set(opened)),
                               "imported": sorted({(m, f or "") for m, f in imported})}, default=str))
    return 0


def strip_times(d):
    if isinstance(d, dict):
        return {k: strip_times(v) for k, v in d.items() if k not in TIME_KEYS}
    if isinstance(d, list):
        return [strip_times(v) for v in d]
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default="N1>N2,M+146>M+147")
    ap.add_argument("--child-out", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    edges = [e for e in args.edges.split(",") if e]
    if args.child_out:
        return child(edges, Path(args.child_out))
    started = time.time()
    import exact_judge as EJ
    with tempfile.TemporaryDirectory() as tmp:
        cout = Path(tmp) / "child.json"
        env = {**os.environ, **BLAS_ONE}
        subprocess.run(["nice", "-n", "10", str(PYTHON), "-I", "-B", str(TRACK / "t11cli.py"), "independence",
                        "--edges", args.edges, "--child-out", str(cout)], check=True, env=env)
        data = json.loads(cout.read_text())
    bad_open = [p for p in data["opened"] if forbidden(p)]
    bad_import = [(m, f) for m, f in data["imported"]
                  if m.split(".")[0] in FORBIDDEN_MODULE_ROOTS or (f and forbidden(f))]
    design_signs = EJ.design_signs(json.loads(DESIGN.read_text()))
    mismatches = []
    for e, r in data["results"].items():
        stored_tube = json.loads((TUBES / f"{edge_stem(e)}.json").read_text())["rows"][0]
        stored_box = json.loads((BOXES / f"{edge_stem(e)}.json").read_text())
        if strip_times(r["tube"]) != strip_times(stored_tube):
            mismatches.append(f"{e}: tube record differs")
        if strip_times(r["box"]) != strip_times(stored_box):
            mismatches.append(f"{e}: box record differs")
        stored_j = {k: v for k, v in EJ.judge_v5(BOXES / f"{edge_stem(e)}.json", design_signs).items()
                    if not k.startswith("_")}
        if r["judged"] != stored_j:
            mismatches.append(f"{e}: exact judgement differs")
    # 감사 이벤트는 없는 .pyc 를 여는 시도도 남기므로 실제로 있는 파일만, 가상환경 밖만 적는다
    repo_files = sorted({str(Path(p).resolve().relative_to(REPO)) for p in data["opened"]
                         if REPO in Path(p).resolve().parents and Path(p).exists() and ".venv" not in Path(p).parts})
    out = {"edges": edges, "forbidden_opened": bad_open, "forbidden_imported": bad_import,
           "repo_files_opened": repo_files, "mismatches": mismatches,
           "pass": not bad_open and not bad_import and not mismatches, "seconds": round(time.time() - started, 1)}
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
