"""[CAP] G8 차등 재검증: 원뿔 조건과 정리 A′ 주장을 (가) 다른 산술 계층의 D(B) 와 (나) 독립 판정법으로 다시 통과시킨다.

(가) T6 G8 실행 기록(x86filib c0·c1 + MP 96비트·차수 24 점 상, raw-local)에서 D(B) 와 사용 판정을 압축 기록
     g8_derivatives.json 으로 옮기고(--extract, 원 기록 sha256 목록·digest 를 manifest 에), 같은 판정층으로 판정한다.
     raw-local 은 gitignore 라서 깨끗한 사본의 재판정은 압축 기록만 쓴다.
(나) 본 실행 D(B) 를 중점·반경 감싸기 + Weyl 부등식 + 정확 Cholesky(t7hb/weyl.py)로 μ ∈ {μ_s, 1, μ_u} 에서 판정한다.
두 갈래 모두 314변이 통과하고 (가)에서 정리 A′ 주장 8개가 성립해야 exit 0 이다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t7hb.results import RESULTS_DIR, T3_DIR, T6_DIR, T6_DESIGN, T6_RESULTS, records_block, sha256_file, write_result  # noqa: E402

sys.path.insert(0, str(T3_DIR))
sys.path.insert(0, str(T6_DIR))
from t7hb import hyperbolic, stored, weyl  # noqa: E402
import assemble_aprime  # noqa: E402

G8_DIR = T6_RESULTS / "raw-local" / "coverings_g8_x86filib_mp96"
COMPACT = RESULTS_DIR / "g8_derivatives.json"


def extract(design: dict, design_hash: str) -> tuple[dict, dict]:
    paths = [G8_DIR / f"{e['source']}__{e['target']}.json" for e in design["edges"]]
    rows = []
    for p in paths:
        rec = stored.load(p)
        c = stored.cone_input(rec, design_hash)
        rows.append({"edge": c.edge, "derivative": rec["derivative"], "problems": list(c.problems),
                     "settings": rec["settings"], "source_sha256": sha256_file(p)})
    block = records_block(paths)
    return ({"tag": "G8 압축 기록: T6 G8(x86filib) 덮개 기록의 D(B) 와 사용 판정", "source_dir": str(G8_DIR.relative_to(
        T6_RESULTS.parent.parent)), "source_count": block["count"], "source_digest_sha256": block["digest_sha256"],
        "rows": rows}, block)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true", help="raw-local G8 기록에서 압축 기록을 다시 만든다")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    design_path = T6_DESIGN / "graph_design.json"
    design = json.loads(design_path.read_text())
    design_hash = sha256_file(design_path)
    if args.extract:
        payload, block = extract(design, design_hash)
        if not args.no_write:
            write_result("g8_derivatives", payload, script=Path(__file__).resolve(), params=vars(args), started=started,
                         extra={"records": {k: block[k] for k in ("count", "digest_sha256", "files")}})
    compact = json.loads(COMPACT.read_text())
    edges = [f"{e['source']}>{e['target']}" for e in design["edges"]]
    src = assemble_aprime.load_sources(T6_RESULTS / "coverings")
    g8 = hyperbolic.assemble([stored.compact_input(r) for r in compact["rows"]], edges, src.hulls)
    claims = hyperbolic.judge_claims(g8, assemble_aprime.CLAIMED)
    main_run = {c.edge: c for c in assemble_aprime.assemble_from(src).edges}
    diffs = [(c.edge, abs(c.bounds.lower - main_run[c.edge].bounds.lower),
              abs(c.bounds.upper - main_run[c.edge].bounds.upper)) for c in g8.edges]
    mus = (assemble_aprime.CLAIMED.mu_s, Fraction(1), assemble_aprime.CLAIMED.mu_u)
    weyl_fail = [f"{e}@mu={float(mu)}" for e, c in ((e, stored.cone_input(r, design_hash))
                                                       for e, r in src.records.items())
                 for mu in mus if not (c.usable and weyl.certifies(c.derivative, mu))]
    summary = {
        "g8_edges": len(g8.edges), "g8_certified": len(g8.edges) - len(g8.uncertified), "g8_missing": list(g8.missing),
        "g8_claims_hold": all(c.holds for c in claims),
        "g8_claims": [{"claim": c.name, "holds": c.holds, "margin_float": None if c.margin is None else float(c.margin)}
                      for c in claims],
        "g8_max_lower_a": float(g8.lower_max), "g8_min_upper_b": float(g8.upper_min),
        "max_abs_diff_lower_a": float(max(d[1] for d in diffs)), "max_abs_diff_upper_b": float(max(d[2] for d in diffs)),
        "weyl_checks": len(src.records) * len(mus), "weyl_failures": weyl_fail[:20], "weyl_failed": len(weyl_fail),
        "compact_sha256": sha256_file(COMPACT),
    }
    print(json.dumps(summary, indent=1))
    if not args.no_write:
        write_result("g8_cones", {"tag": "[CAP] G8 원뿔 조건 재통과(x86filib D(B))·독립 판정(Weyl+Cholesky)", "summary": summary,
                                  "rows": [{"edge": c.edge, "lower_a": str(c.bounds.lower), "upper_b": str(c.bounds.upper),
                                            "certified": c.certified} for c in g8.edges]},
                     script=Path(__file__).resolve(), params=vars(args), started=started)
    ok = (not g8.missing and not g8.uncertified and summary["g8_claims_hold"] and not weyl_fail
          and len(g8.edges) == len(edges))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
