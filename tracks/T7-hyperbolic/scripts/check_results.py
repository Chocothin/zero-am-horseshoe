"""verify.sh 빠른 모드: 저장된 결과를 엔진 없이 다시 판정하고 manifest 해시를 대조한다(레포에 쓰지 않음).

- T6 덮개 기록 314개가 T6 manifest 의 sha256 목록·digest 와 같은지 본다(판정이 믿는 입력).
- 변마다 원뿔 판정을 현재 코드로 다시 내어 cones.json 의 a*·b*·여유(정확 유리수 문자열)와 같은지 본다.
- 정리 A′ 를 다시 조립해 theorem_Aprime.json 의 주장 판정·여유와 같은지 본다.
- G8 압축 기록을 다시 판정해 g8_cones.json 의 변별 값과 같은지 본다. raw-local 이 있으면 원 기록 해시도 대조한다.
- manifest 의 결과 sha256 이 파일과 같고, 스크립트 sha256 이 현재 스크립트와 같으며, T3·T6 코드 해시가 그대로인지 본다.
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t7hb.results import (REPO_DIR, RESULTS_DIR, T3_DIR, T6_DIR, T6_DESIGN, T6_RESULTS, records_block,  # noqa: E402
                          sha256_file, t6_record_problems)

sys.path.insert(0, str(T3_DIR))
sys.path.insert(0, str(T6_DIR))
from t7hb import cone, hyperbolic, stored  # noqa: E402
import assemble_aprime  # noqa: E402
import g8_cones  # noqa: E402

ONE = Fraction(1)


def recheck_cones(design: dict, design_hash: str) -> list[str]:
    rows = {r["edge"]: r for r in json.loads((RESULTS_DIR / "cones.json").read_text())["rows"]}
    bad = []
    for e in design["edges"]:
        rec = stored.load(T6_RESULTS / "coverings" / f"{e['source']}__{e['target']}.json")
        c = stored.cone_input(rec, design_hash)
        row = rows.get(c.edge)
        if row is None:
            bad.append(f"{c.edge}: missing in cones.json")
            continue
        b = cone.cone_bounds(c.derivative)
        now = {"lower_a": str(b.lower), "upper_b": str(b.upper), "margin_mu1": str(b.margin(ONE))}
        if any(row.get(k) != v for k, v in now.items()) or row["proven"] != (c.usable and b.margin(ONE) > 0):
            bad.append(f"{c.edge}: re-judged cone bounds differ from cones.json")
    return bad


def recheck_theorem() -> list[str]:
    stored_out = json.loads((RESULTS_DIR / "theorem_Aprime.json").read_text())
    src = assemble_aprime.load_sources(T6_RESULTS / "coverings")
    a = assemble_aprime.assemble_from(src)
    now = {c.name: (c.holds, None if c.margin is None else str(c.margin))
           for c in hyperbolic.judge_claims(a, assemble_aprime.CLAIMED)}
    was = {c["claim"]: (c["holds"], None if c["margin"] is None else c["margin"]["exact"]) for c in stored_out["claims"]}
    bad = [] if now == was else ["theorem A' claims differ from theorem_Aprime.json"]
    if stored_out["claimed"] != {k: str(v) for k, v in vars(assemble_aprime.CLAIMED).items()}:
        bad.append("CLAIMED constants differ from theorem_Aprime.json")
    if not all(h for h, _ in now.values()):
        bad.append("theorem A' claims do not hold on re-assembly")
    return bad


def recheck_g8(design: dict) -> list[str]:
    compact = json.loads(g8_cones.COMPACT.read_text())
    rows = {r["edge"]: r for r in json.loads((RESULTS_DIR / "g8_cones.json").read_text())["rows"]}
    bad = []
    for r in compact["rows"]:
        c = stored.compact_input(r)
        b = cone.cone_bounds(c.derivative)
        old = rows.get(c.edge)
        if old is None or old["lower_a"] != str(b.lower) or old["upper_b"] != str(b.upper):
            bad.append(f"G8 {c.edge}: re-judged bounds differ from g8_cones.json")
    if len(compact["rows"]) != len(design["edges"]):
        bad.append("G8 compact record count differs from the design")
    if g8_cones.G8_DIR.exists():
        block = records_block([g8_cones.G8_DIR / f"{e['source']}__{e['target']}.json" for e in design["edges"]])
        if block["digest_sha256"] != compact["source_digest_sha256"]:
            bad.append("raw-local G8 records differ from the digest recorded at extraction")
    return bad


def recheck_manifest() -> list[str]:
    manifest = json.loads((RESULTS_DIR / "manifest.json").read_text())
    bad = []
    for name, entry in manifest["entries"].items():
        out = RESULTS_DIR / entry["output"]
        if not out.exists() or sha256_file(out) != entry["output_sha256"]:
            bad.append(f"manifest {name}: output hash mismatch")
        if sha256_file(REPO_DIR / entry["script"]) != entry["script_sha256"]:
            bad.append(f"manifest {name}: script changed after the result was written")
        for key, h in entry["code_sha256"].items():
            # 읽기 전용 입력(T3 엔진·T6 코드)이 결과를 낸 뒤 바뀌었는지 본다. T7 코드는 위의 재판정이 현재 판으로 확인한다
            if key.startswith(("tracks/T3-cap/", "tracks/T6-horseshoe/")) and sha256_file(REPO_DIR / key) != h:
                bad.append(f"manifest {name}: {key} changed")
    cones_inputs = manifest["entries"].get("cones", {}).get("inputs", {})
    design = json.loads((T6_DESIGN / "graph_design.json").read_text())
    current = records_block([T6_RESULTS / "coverings" / f"{e['source']}__{e['target']}.json" for e in design["edges"]])
    if cones_inputs.get("t6_records_digest") != current["digest_sha256"]:
        bad.append("manifest cones: T6 records digest differs from the one the verdicts relied on")
    g5 = json.loads((RESULTS_DIR / "g5_derivative.json").read_text())["summary"]
    if g5["critical"] or g5["edges"] != len(design["edges"]):
        bad.append(f"G5 derivative: critical {g5['critical']}, edges {g5['edges']}")
    return bad


def main() -> int:
    design_path = T6_DESIGN / "graph_design.json"
    design = json.loads(design_path.read_text())
    design_hash = sha256_file(design_path)
    paths = [T6_RESULTS / "coverings" / f"{e['source']}__{e['target']}.json" for e in design["edges"]]
    bad = t6_record_problems(paths)
    bad += recheck_cones(design, design_hash)
    bad += recheck_theorem()
    bad += recheck_g8(design)
    bad += recheck_manifest()
    for b in bad[:40]:
        print("FAIL", b)
    print(f"edges {len(design['edges'])}, problems {len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
