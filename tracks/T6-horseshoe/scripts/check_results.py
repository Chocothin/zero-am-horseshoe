"""verify.sh 빠른 모드: 저장된 결과를 엔진 없이 다시 판정하고 manifest 해시를 대조한다(레포에 쓰지 않음).

- 덮개 기록마다 records.covering_input(재판정·대조·조립이 함께 쓰는 경로)으로 감싸기를 꺼내 covering_verdict 를 다시
  돌린다. 단일 중심 기록은 저장된 P(0)·D(B) 로 평균값 상을 다시 계산해 저장된 상과 정확히 같은지, P(0) 이 점 실행의 상과
  같은지 본다. 조각마다 같은 실행 넷의 c0 상(외피∩아핀)이 평균값 상과 겹치는지도 본다(같은 집합의 상을 감싸므로 서로소면
  어느 한 감싸기가 틀렸다).
- 정리 A 를 다시 조립해 가정이 모두 성립하는지 보고, 상수 δ·귀환 시각 하한이 요약 스크립트(summarize_coverings)가
  따로 낸 값과 같은지 대조한다(독립 출처 값 고정).
- manifest 의 결과 sha256 이 파일과 같고, 스크립트 sha256 이 현재 스크립트와 같으며, T3 엔진 코드 해시가 그대로인지 본다.
- 재판정이 믿는 덮개 기록 314개의 sha256 이 manifest certify_graph 항목의 records 목록·묶음 digest 와 같은지 본다.
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t6hs.results import DESIGN_DIR, REPO_DIR, RESULTS_DIR, T3_DIR, T3_FILES, sha256_file  # noqa: E402

sys.path.insert(0, str(T3_DIR))
from t3cap.export import interval_from_json  # noqa: E402

from t6hs import covering, enclose, records  # noqa: E402
import assemble_theorem  # noqa: E402
import record_hashes  # noqa: E402

def overlaps(a, b) -> bool:
    return all(max(x.lo, y.lo) <= min(x.hi, y.hi) for x, y in zip(a, b, strict=True))


def engine_images_disagree(rec: dict) -> list[str]:
    runs = rec.get("runs", [])
    out = []
    for key, plist in rec["pieces_detail"].items():
        for p in plist:
            if p["image"] is None or p.get("runs_index") is None:
                continue
            c0 = [interval_from_json(x) for x in runs[p["runs_index"]]["c0"]["image"]]
            if not overlaps(c0, [interval_from_json(x) for x in p["image"]]):
                out.append(f"{key}: c0 image and mean-value image are disjoint")
    return out


def recheck_edge(rec: dict) -> list[str]:
    c = records.covering_input(rec)
    problems = list(c.problems) + engine_images_disagree(rec)
    if rec["method"] == "single-center":
        center = tuple(interval_from_json(x) for x in rec["center_image"])
        der = tuple(tuple(interval_from_json(x) for x in row) for row in rec["derivative"])
        zero = (Fraction(0),) * 4
        point_runs = {p["runs_index"] for ps in rec["pieces_detail"].values() for p in ps if p.get("runs_index") is not None}
        if any(center != tuple(interval_from_json(x) for x in rec["runs"][i]["point"]["image"]) for i in point_runs):
            problems.append("stored centre image differs from the point run image")
        for p in c.whole + [q for ps in c.faces.values() for q in ps]:
            if enclose.mean_value_image(center, der, p.piece, zero) != p.image:
                problems.append("stored image differs from recomputed mean-value image")
    v = covering.covering_verdict(c.whole, c.faces, c.signs)
    if not v.proven:
        problems.append("verdict fails: " + ",".join(x.name for x in v.failed))
    return problems


def main() -> int:
    design = json.loads((DESIGN_DIR / "graph_design.json").read_text())
    design_hash = sha256_file(DESIGN_DIR / "graph_design.json")
    bad = []
    for e in design["edges"]:
        path = RESULTS_DIR / "coverings" / f"{e['source']}__{e['target']}.json"
        if not path.exists():
            bad.append(f"{e['source']}>{e['target']}: missing")
            continue
        rec = records.load(path)
        if rec["design_sha256"] != design_hash:
            bad.append(f"{rec['edge']}: stale design")
        bad += [f"{rec['edge']}: {p}" for p in recheck_edge(rec)]
    a, _ = assemble_theorem.build()
    if not a.certified:
        bad.append("theorem A hypotheses not certified on re-assembly")
    summary = json.loads((RESULTS_DIR / "coverings_summary.json").read_text())["total"]
    if float(a.delta) != summary["tube_min"]:
        bad.append(f"delta {float(a.delta)} differs from coverings_summary tube_min {summary['tube_min']}")
    manifest = json.loads((RESULTS_DIR / "manifest.json").read_text())
    for name, entry in manifest["entries"].items():
        out = RESULTS_DIR / entry["output"]
        if not out.exists() or sha256_file(out) != entry["output_sha256"]:
            bad.append(f"manifest {name}: output hash mismatch")
        if sha256_file(REPO_DIR / entry["script"]) != entry["script_sha256"]:
            bad.append(f"manifest {name}: script changed after the result was written")
        for rel in T3_FILES:
            key = f"tracks/T3-cap/{rel}"
            if key in entry["code_sha256"] and entry["code_sha256"][key] != sha256_file(T3_DIR / rel):
                bad.append(f"manifest {name}: T3 {rel} changed")
    stored = manifest["entries"].get(record_hashes.ENTRY, {}).get("records")
    if not stored:
        bad.append("manifest certify_graph: coverings record hashes missing (run scripts/record_hashes.py)")
    else:
        current = record_hashes.records_block(RESULTS_DIR / stored["dir"])
        if current["files"] != stored["files"] or current["digest_sha256"] != stored["digest_sha256"]:
            changed = sorted(set(current["files"].items()) ^ set(stored["files"].items()))
            bad.append(f"coverings records differ from manifest hashes ({len(changed)} entries)")
        if sha256_file(REPO_DIR / stored["hashed_by"]) != stored["hashed_by_sha256"]:
            bad.append("manifest certify_graph: record hashing script changed after hashing")
        if current["count"] != len(design["edges"]):
            bad.append(f"coverings holds {current['count']} records, design has {len(design['edges'])} edges")
    for b in bad[:40]:
        print("FAIL", b)
    print(f"edges {len(design['edges'])}, theorem A certified {a.certified}, problems {len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
