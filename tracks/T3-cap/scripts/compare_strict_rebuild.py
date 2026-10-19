"""옛 T3 결과(커밋 93a1692, results/2026-10-15-T3-cap)와 융합 금지 빌드로 다시 낸 결과(RESULTS_DIR)를 맞댄다(엔진 없음).

결과마다 JSON 값을 끝까지 비교해 다른 값의 경로를 센다. 실행 시간·로그 꼬리·빌드 구성 이름처럼 빌드와 시각에 따라
당연히 달라지는 키는 뺀다. 이정표 사례·G5 설정·mutation 감사의 핵심 값은 따로 나란히 적는다. 아무것도 판정하지 않고
기록만 한다(판정은 각 결과의 check_results 소관).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from common import REPO_DIR, RESULTS_DIR, write_result

OLD_DIR = REPO_DIR / "results" / "2026-10-15-T3-cap"
NAMES = ("figure8_newton", "orbit_butterfly1", "milestone_figure8", "arb_krawczyk", "g5_soundness", "g5_prefix_trap2",
         "mutation_audit", "engine_mutation_audit", "benchmark", "c1_mitigation", "capd_selftests")


def ignored(key: str) -> bool:
    return "second" in key or key.endswith("_tail") or key in ("build", "configuration", "build_dir")


def differences(old, new, path: str = "") -> list[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        keys = sorted(set(old) | set(new))
        return [d for k in keys if not ignored(k)
                for d in (differences(old.get(k), new.get(k), f"{path}/{k}") if k in old and k in new else [f"{path}/{k}"])]
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{path}[len {len(old)}->{len(new)}]"]
        return [d for i, (a, b) in enumerate(zip(old, new)) for d in differences(a, b, f"{path}[{i}]")]
    return [] if old == new else [path]


def key_values(name: str, data: dict) -> object:
    if name == "milestone_figure8":
        return [(c["case"], c["proven"], c["k_radius_over_rho"], c["tube_min_distance_lower"]) for c in data["cases"]]
    if name == "arb_krawczyk":
        return {"disagreements": data["disagreements"],
                "cases": [(c["case"], c["agree"], c.get("k_radius_over_rho_arb")) for c in data["cases"]]}
    if name == "g5_soundness":
        return {"critical": data["critical"], "configs": [(c["config"], c["points"], c["violations"], c["min_relative_margin"])
                                                          for c in data["configs"]]}
    if name == "g5_prefix_trap2":
        return {k: data["summary"][k] for k in ("violations", "fixed_engine_refuses", "enclosure_return_time")}
    if name == "engine_mutation_audit":
        return {k: data[k] for k in ("mutants", "equivalent", "scored", "killed", "wrong_equivalence")}
    if name == "mutation_audit":
        return data["summary"]
    if name == "capd_selftests":
        return {"summary": data["summary"], "failed": data["failed_tests"]}
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    rows = []
    for name in NAMES:
        old = json.loads((OLD_DIR / f"{name}.json").read_text())
        new = json.loads((RESULTS_DIR / f"{name}.json").read_text())
        diff = differences(old, new)
        rows.append({"result": name, "differing_values": len(diff), "first_differences": diff[:12],
                     "old": key_values(name, old), "new": key_values(name, new)})
    for r in rows:
        print(r["result"], r["differing_values"], r["first_differences"][:4])
    if not args.no_write:
        write_result("compare_strict_rebuild", {"tag": "옛(93a1692)·새(융합 금지 빌드) T3 결과 대조(실행 시간·로그·빌드 이름 제외)",
                                                "old_dir": str(OLD_DIR.relative_to(REPO_DIR)), "rows": rows},
                     script=Path(__file__).resolve(), params=vars(args), started=started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
