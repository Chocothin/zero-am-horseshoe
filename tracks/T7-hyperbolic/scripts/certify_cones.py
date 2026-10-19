"""[CAP] 덮개 314개마다 원뿔 조건(Z2009 Def 11 을 Lemma 8 로 판정)을 저장된 D(B) 만으로 인증하고, 변마다 전용 대조 3개를 돌린다(G6).

입력은 T6 덮개 기록이고, 기록 314개의 sha256 이 T6 manifest(certify_graph.records)의 목록·digest 와 같아야 시작한다.
변마다 판정은 stored.cone_input(T6 사용 조건·덮개 재판정 포함) → cone.cone_verdict(μ=1, 문턱 0) 이다. 대조는 같은 경로를 지난다.
  tighten_threshold : 문턱 0 을 인증 여유 g 바로 너머(g·(1+1e-12))로 올린다
  swap_weak_rows    : 기록의 D(B) 1·2 행(상 좌표의 약불안정·약안정)을 맞바꾼다
  swap_strong_rows  : 기록의 D(B) 0·3 행(강불안정·강안정)을 맞바꾼다
모든 변이 인증되고, 변마다 거절된 대조가 2개 이상이며(controls_cover), 새어 나간 대조가 없고, 문턱 강화의 안쪽 g·(1−1e-12)
에서는 여전히 인증되어야 exit 0 이다.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from t7hb.results import T3_DIR, T6_DIR, T6_DESIGN, T6_RESULTS, records_block, sha256_file, t6_record_problems, write_result  # noqa: E402

sys.path.insert(0, str(T3_DIR))
sys.path.insert(0, str(T6_DIR))
from t7hb import cone, stored  # noqa: E402

RELATIVE = Fraction(1, 10**12)
MIN_CONTROLS = 2
ONE = Fraction(1)


def record_paths(design: dict, cover_dir: Path) -> list[Path]:
    return [cover_dir / f"{e['source']}__{e['target']}.json" for e in design["edges"]]


def swapped_rows(rec: dict, a: int, b: int) -> dict:
    out = copy.deepcopy(rec)
    out["derivative"][a], out["derivative"][b] = rec["derivative"][b], rec["derivative"][a]
    return out


def judged(c: stored.ConeInput, threshold: Fraction = Fraction(0)) -> tuple[bool, Fraction | None]:
    """변 하나의 원뿔 주장 판정: 사용 가능한 D(B) 이고 μ=1 의 Gershgorin 여유가 문턱보다 크다."""
    if c.derivative is None:
        return False, None
    v = cone.cone_verdict(c.derivative, ONE, threshold)
    return c.usable and v.proven, v.margin - threshold


def run_control(name: str, planted: str, call) -> dict:
    row = {"control": name, "planted": planted}
    try:
        proven, computed = call()
        return {**row, "computed": None if computed is None else str(computed),
                "computed_float": None if computed is None else float(computed), "judge_output": proven,
                "reason": "judge false" if not proven else "LEAKED", "rejected": not proven}
    except ValueError as exc:
        return {**row, "computed": None, "computed_float": None, "judge_output": None,
                "reason": f"guard exception: {exc}", "rejected": True}


def edge_row(rec: dict, design_hash: str) -> dict:
    c = stored.cone_input(rec, design_hash)
    proven, margin = judged(c)
    b = None if c.derivative is None else cone.cone_bounds(c.derivative)
    row = {"edge": c.edge, "usable": c.usable, "problems": list(c.problems), "proven": proven,
           "margin_mu1": None if margin is None else str(margin), "margin_float": None if margin is None else float(margin)}
    if b is not None:
        row.update({"lower_a": str(b.lower), "lower_float": float(b.lower), "upper_b": str(b.upper),
                    "upper_float": float(b.upper), "diag_lower": [float(x) for x in b.diag_lower], "radii": [float(x) for x in b.radii]})
    if margin is None or margin <= 0:
        row.update({"controls": [], "rejected_controls": 0, "sharpness_inside_margin_still_proven": False})
        return row
    beyond, inside = margin * (1 + RELATIVE), margin * (1 - RELATIVE)
    controls = [
        run_control("tighten_threshold", f"threshold g*(1+1e-12) = {float(beyond):.17g}", lambda: judged(c, beyond)),
        run_control("swap_weak_rows", "record D(B) rows 1<->2",
                    lambda: judged(stored.cone_input(swapped_rows(rec, 1, 2), design_hash))),
        run_control("swap_strong_rows", "record D(B) rows 0<->3",
                    lambda: judged(stored.cone_input(swapped_rows(rec, 0, 3), design_hash))),
    ]
    row.update({"controls": controls, "rejected_controls": sum(1 for x in controls if x["rejected"]),
                "sharpness_inside_margin_still_proven": judged(c, inside)[0]})
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    design_path = T6_DESIGN / "graph_design.json"
    design = json.loads(design_path.read_text())
    design_hash = sha256_file(design_path)
    paths = record_paths(design, T6_RESULTS / "coverings")
    integrity = t6_record_problems(paths)
    rows = [edge_row(stored.load(p), design_hash) for p in paths]
    ctrl = [x for r in rows for x in r["controls"]]
    summary = {
        "edges": len(rows), "proven": sum(r["proven"] for r in rows), "t6_record_integrity_problems": integrity,
        "controls_total": len(ctrl), "leaked": sum(1 for x in ctrl if not x["rejected"]),
        "controls_cover": all(r["rejected_controls"] >= MIN_CONTROLS for r in rows),
        "sharpness_all": all(r["sharpness_inside_margin_still_proven"] for r in rows),
        "margin_sign_agrees": all(x["judge_output"] is None or x["judge_output"] == (Fraction(x["computed"]) > 0)
                                  for x in ctrl if x["computed"] is not None),
        "computed_max": max((x["computed_float"] for x in ctrl if x["computed_float"] is not None), default=None),
        "min_margin_mu1": min((r["margin_float"] for r in rows if r["margin_float"] is not None), default=None),
        "max_lower_a": max((r["lower_float"] for r in rows if "lower_float" in r), default=None),
        "min_upper_b": min((r["upper_float"] for r in rows if "upper_float" in r), default=None),
    }
    print(json.dumps(summary))
    if not args.no_write:
        block = records_block(paths)
        write_result("cones", {"tag": "[CAP] 덮개별 원뿔 조건(μ=1, 저장 D(B)), G6 덮개별 전용 대조", "summary": summary,
                               "rows": rows},
                     script=Path(__file__).resolve(), params=vars(args), started=started,
                     extra={"inputs": {"t6_coverings": str((T6_RESULTS / "coverings").relative_to(T6_DIR.parent.parent)),
                                       "t6_design": str(design_path.relative_to(T6_DIR.parent.parent)),
                                       "t6_design_sha256": design_hash,
                                       "t6_records_count": block["count"], "t6_records_digest": block["digest_sha256"],
                                       "t6_manifest_sha256": sha256_file(T6_RESULTS / "manifest.json")}})
    ok = (not integrity and summary["proven"] == len(design["edges"]) == len(rows) and summary["controls_cover"]
          and not summary["leaked"] and summary["sharpness_all"] and summary["margin_sign_agrees"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
