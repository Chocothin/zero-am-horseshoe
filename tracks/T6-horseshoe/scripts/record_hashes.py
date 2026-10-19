"""coverings/ 덮개 기록의 sha256 목록과 묶음 digest 를 manifest 의 certify_graph 항목(records)에 넣는다.

check_results 는 기록 안의 엔진 출력(P(0)·D(B)·흐름 관)을 믿고 재판정하므로 기록 자체를 manifest 로 고정한다(CLAUDE.md
규칙 5). certify_graph 를 다시 돌리면 write_result 가 항목을 새로 쓰면서 records 가 빠지므로, 그때 이 스크립트를 다시 돌린다.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from t6hs.results import REPO_DIR, RESULTS_DIR, sha256_file  # noqa: E402

ENTRY = "certify_graph"


def records_block(cover_dir: Path) -> dict:
    files = {p.name: sha256_file(p) for p in sorted(cover_dir.glob("*.json"))}
    digest = hashlib.sha256("".join(f"{name} {h}\n" for name, h in files.items()).encode()).hexdigest()
    return {"dir": str(cover_dir.relative_to(RESULTS_DIR)), "count": len(files), "digest_sha256": digest,
            "files": files}


def main() -> int:
    manifest_path = RESULTS_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    script = Path(__file__).resolve()
    block = records_block(RESULTS_DIR / "coverings")
    manifest["entries"][ENTRY]["records"] = {
        **block, "hashed_by": str(script.relative_to(REPO_DIR)), "hashed_by_sha256": sha256_file(script),
        "hashed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in block.items() if k != "files"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
