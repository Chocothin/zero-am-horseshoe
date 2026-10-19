"""verify 빠른 모드의 엔진 재현 판정: 임시 디렉터리에 다시 인증한 변 기록을 재판정하고, 저장된 본 실행 기록과 감싸기가
성분마다 겹치는지 본다. 통과 기준은 「요청한 변이 모두 증명되고 겹침」이다(전체 314변 판정이 아니다).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t6hs.results import RESULTS_DIR, T3_DIR  # noqa: E402

sys.path.insert(0, str(T3_DIR))
from t6hs import records  # noqa: E402
import compare_runs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--only", required=True, help="쉼표로 구분한 변 목록, 예: N0>N1,M+0>M+1")
    args = parser.parse_args()
    edges = [e.strip() for e in args.only.split(",") if e.strip()]
    ok = 0
    for edge in edges:
        name = edge.replace(">", "__") + ".json"
        path = Path(args.dir) / name
        if not path.exists():
            print(f"{edge}: missing reproduction record")
            continue
        row = compare_runs.compare(records.covering_input(records.load(RESULTS_DIR / "coverings" / name)),
                                   records.covering_input(records.load(path)))
        good = row["both_proven"] and row["images_overlap"] and row["same_degree"]
        ok += good
        print(f"{edge}: proven {row['both_proven']}, overlaps stored record {row['images_overlap']}, "
              f"width ratio {row['width_ratio_min']:.4f}~{row['width_ratio_max']:.4f}")
    print(f"quick-mode criterion (requested edges only): {ok}/{len(edges)} proven and overlapping the stored records")
    return 0 if ok == len(edges) else 1


if __name__ == "__main__":
    raise SystemExit(main())
