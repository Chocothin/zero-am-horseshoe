"""verify.sh 빠른 모드: results/ 의 manifest 해시(출력·스크립트)와 저장된 핵심 주장을 대조한다. 아무것도 쓰지 않는다."""
from __future__ import annotations

import json
import sys

from common import REPO_DIR, RESULTS_DIR, TRACK_DIR, sha256_file

REQUIRED = ["figure8_newton", "milestone_figure8", "g5_soundness", "g5_prefix_trap2", "mutation_audit",
            "engine_mutation_audit", "benchmark", "arb_krawczyk", "c1_mitigation", "capd_selftests", "build_flags"]


def unchanged(rel: str, digest: str) -> bool:
    path = TRACK_DIR / rel
    return path.exists() and sha256_file(path) == digest


def main() -> int:
    manifest = json.loads((RESULTS_DIR / "manifest.json").read_text())
    problems = []
    for name in REQUIRED:
        entry = manifest["entries"].get(name)
        if entry is None:
            problems.append(f"{name}: missing from manifest")
            continue
        out = RESULTS_DIR / entry["output"]
        if sha256_file(out) != entry["output_sha256"]:
            problems.append(f"{name}: output hash mismatch")
        script = REPO_DIR / entry["script"]
        if sha256_file(script) != entry["script_sha256"]:
            problems.append(f"{name}: script changed since the recorded run ({entry['script']})")
        # 공통 코드(CODE_FILES) 외에 결과마다 따로 기대는 코드(extra_code_sha256)도 같은 잣대로 본다
        recorded = {**entry.get("code_sha256", {}), **entry["params"].get("extra_code_sha256", {})}
        stale = [rel for rel, digest in recorded.items() if not unchanged(rel, digest)]
        if not entry.get("code_sha256") or stale:
            problems.append(f"{name}: code changed since the recorded run {stale or '(no code hashes)'}")
    milestone = json.loads((RESULTS_DIR / "milestone_figure8.json").read_text())
    for case in milestone["cases"]:
        expected = not case["case"].startswith("neg")
        if case["proven"] != expected:
            problems.append(f"milestone case '{case['case']}' proven={case['proven']} (expected {expected})")
    g5 = json.loads((RESULTS_DIR / "g5_soundness.json").read_text())
    if g5["critical"] != 0 or any(c["violations"] for c in g5["configs"]):
        problems.append("g5: critical violations recorded")
    if any(c["points"] < 1000 for c in g5["configs"]):
        problems.append("g5: fewer than 1000 points in a core configuration")
    prefix = json.loads((RESULTS_DIR / "g5_prefix_trap2.json").read_text())["summary"]
    if not prefix["fixed_engine_refuses"] or prefix["violations"] == 0:
        problems.append("g5_prefix_trap2: fixed engine must refuse and the counter-off engine must show the hole")
    arb_check = json.loads((RESULTS_DIR / "arb_krawczyk.json").read_text())
    if arb_check["disagreements"] or not any(r.get("proven_arb") for r in arb_check["cases"] if r["case"].startswith("main")):
        problems.append("arb_krawczyk: independent re-verification disagrees or main case not proven")
    mutation = json.loads((RESULTS_DIR / "mutation_audit.json").read_text())
    total = sum(v["mutants"] for v in mutation["summary"].values())
    killed = sum(v["killed"] for v in mutation["summary"].values())
    if killed / total < 0.9:
        problems.append(f"mutation: kill rate {killed}/{total} below 0.9")
    build_flags = json.loads((RESULTS_DIR / "build_flags.json").read_text())
    if build_flags["problems"]:
        problems.append(f"build_flags: {build_flags['problems'][:3]}")
    engine_mutation = json.loads((RESULTS_DIR / "engine_mutation_audit.json").read_text())
    if engine_mutation["killed"] / engine_mutation["scored"] < 0.9 or engine_mutation["wrong_equivalence"]:
        problems.append(f"engine mutation: kill rate {engine_mutation['killed']}/{engine_mutation['scored']} below 0.9 "
                        f"or a declared-equivalent mutant was killed")
    for p in problems:
        print("FAIL", p)
    print(f"check_results: {len(REQUIRED)} entries, problems={len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
