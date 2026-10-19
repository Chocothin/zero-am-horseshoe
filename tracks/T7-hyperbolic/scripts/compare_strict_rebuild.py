"""[CAP] 옛 원뿔 결과(커밋 93a1692, results/2026-10-16-T7-hyperbolic)와 융합 금지 빌드로 다시 낸 T6 기록에서 낸 새 결과
(RESULTS_DIR)를 314변 전수로 맞댄다(엔진 없음). 변마다 원뿔 판정·a*·b*·μ=1 여유(정확 유리수)를, 정리 A′ 는 주장 판정·여유,
극값과 그 변, cycle 외피 분리, G8 요약을 비교한다. 판정이 하나라도 다르거나 CLAIMED 상수가 다르면 exit 1 이다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from t7hb.results import REPO_DIR, RESULTS_DIR, write_result  # noqa: E402

OLD_DIR = REPO_DIR / "results" / "2026-10-16-T7-hyperbolic"
ROW_KEYS = ("usable", "proven", "lower_a", "upper_b", "margin_mu1")


def theorem_of(path: Path) -> dict:
    t = json.loads(path.read_text())
    return {"hypotheses_certified": t["theorem_Aprime_hypotheses_certified"], "claimed": t["claimed"],
            "claims": {c["claim"]: {"holds": c["holds"], "margin": c["margin"]} for c in t["claims"]},
            "max_lower_a": t["max_lower_a"], "max_lower_edge": t["max_lower_edge"], "min_upper_b": t["min_upper_b"],
            "min_upper_edge": t["min_upper_edge"], "cycle_min_separation": t["cycle_min_separation"],
            "cycle_closest_pair": t["cycle_closest_pair"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    old = {r["edge"]: r for r in json.loads((OLD_DIR / "cones.json").read_text())["rows"]}
    new = {r["edge"]: r for r in json.loads((RESULTS_DIR / "cones.json").read_text())["rows"]}
    rows = [{"edge": e, **{f"{k}_same": old[e].get(k) == new[e].get(k) for k in ROW_KEYS},
             "old": {k: old[e].get(k) for k in ROW_KEYS}, "new": {k: new[e].get(k) for k in ROW_KEYS}}
            for e in new]
    theorem = {"old": theorem_of(OLD_DIR / "theorem_Aprime.json"), "new": theorem_of(RESULTS_DIR / "theorem_Aprime.json")}
    g8 = {side: json.loads((d / "g8_cones.json").read_text())["summary"] for side, d in (("old", OLD_DIR), ("new", RESULTS_DIR))
          if (d / "g8_cones.json").exists()}
    summary = {
        "edges": len(rows), "edges_missing": sorted(set(old) ^ set(new)),
        "verdicts_differing": [r["edge"] for r in rows if not (r["usable_same"] and r["proven_same"])],
        "bounds_identical_edges": sum(r["lower_a_same"] and r["upper_b_same"] and r["margin_mu1_same"] for r in rows),
        "theorem_claims_same": {k: (theorem["old"]["claims"][k]["holds"], theorem["new"]["claims"][k]["holds"],
                                    theorem["old"]["claims"][k]["margin"] == theorem["new"]["claims"][k]["margin"])
                                for k in theorem["new"]["claims"]},
        "claimed_constants_same": theorem["old"]["claimed"] == theorem["new"]["claimed"],
        "extrema_same": {k: theorem["old"][k] == theorem["new"][k]
                         for k in ("max_lower_a", "max_lower_edge", "min_upper_b", "min_upper_edge", "cycle_min_separation")},
        "g8": {side: {k: s[k] for k in ("g8_certified", "g8_claims_hold", "g8_max_lower_a", "g8_min_upper_b", "weyl_failed")}
               for side, s in g8.items()},
    }
    print(json.dumps(summary, indent=1))
    if not args.no_write:
        write_result("compare_strict_rebuild", {"tag": "[CAP] 옛(93a1692)·새(융합 금지 빌드) 원뿔 결과 314변 대조(엔진 없음)",
                                                "old_dir": str(OLD_DIR.relative_to(REPO_DIR)), "summary": summary,
                                                "theorem_Aprime": theorem, "rows": rows},
                     script=Path(__file__).resolve(), params=vars(args), started=started)
    ok = not summary["edges_missing"] and not summary["verdicts_differing"] and summary["claimed_constants_same"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
