"""결과 파일과 manifest(git SHA·스크립트·코드 sha256·매개변수·실행 시간·환경) 기록."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TRACK_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = TRACK_DIR.parent.parent
RESULTS_DIR = REPO_DIR / "results" / "2026-10-17-T7-hyperbolic"
T6_DIR = REPO_DIR / "tracks" / "T6-horseshoe"
# T6 덮개 기록은 융합 금지 빌드로 다시 낸 폴더에, h-set 설계(graph_design.json)는 처음 만든 폴더에 있다
T6_RESULTS = REPO_DIR / "results" / "2026-10-17-T6-horseshoe"
T6_DESIGN = REPO_DIR / "results" / "2026-10-15-T6-horseshoe"
T3_DIR = REPO_DIR / "tracks" / "T3-cap"
# export.py 는 D(B) JSON 끝점을 읽고 __init__.py 는 공개 API 를 묶는다 — 판정이 기대는 T3 코드라 함께 고정한다.
# 빌드 스크립트는 감싸기를 낸 빌드 플래그(-ffp-contract=off)를 정한다
T3_FILES = ("engine/t3cap_engine.cpp", "t3cap/api.py", "t3cap/verdict.py", "t3cap/exact.py", "t3cap/model.py",
            "t3cap/export.py", "t3cap/__init__.py", "build_capd.sh", "build_engine.sh")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records_block(paths: list[Path]) -> dict:
    """기록 파일들의 sha256 목록과 묶음 digest(T6 record_hashes 와 같은 식: 「이름 해시\\n」 이어 붙인 문자열의 sha256)."""
    files = {p.name: sha256_file(p) for p in sorted(paths, key=lambda p: p.name)}
    digest = hashlib.sha256("".join(f"{name} {h}\n" for name, h in files.items()).encode()).hexdigest()
    return {"count": len(files), "digest_sha256": digest, "files": files}


def t6_record_problems(paths: list[Path]) -> list[str]:
    """T6 덮개 기록이 T6 manifest(certify_graph.records)의 목록·digest 와 같은지 본다."""
    manifest = json.loads((T6_RESULTS / "manifest.json").read_text())
    stored = manifest["entries"].get("certify_graph", {}).get("records")
    if not stored:
        return ["T6 manifest has no certify_graph.records block"]
    current = records_block(paths)
    out = []
    if current["files"] != stored["files"]:
        out.append(f"T6 records differ from T6 manifest hashes "
                   f"({len(set(current['files'].items()) ^ set(stored['files'].items()))} entries)")
    if current["digest_sha256"] != stored["digest_sha256"]:
        out.append("T6 records digest differs from T6 manifest")
    return out


def git_sha() -> str:
    out = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip()


def code_hashes() -> dict[str, str]:
    """결과가 기대는 T7·T6 코드와 T3 엔진의 해시 — 코드가 바뀐 뒤 남은 낡은 결과를 검사기가 걸러낸다."""
    files = sorted((TRACK_DIR / "t7hb").glob("*.py")) + sorted((T6_DIR / "t6hs").glob("*.py"))
    out = {str(p.relative_to(REPO_DIR)): sha256_file(p) for p in files}
    for rel in T3_FILES:
        out[f"tracks/T3-cap/{rel}"] = sha256_file(T3_DIR / rel)
    return out


def environment() -> dict[str, str]:
    import mpmath
    import numpy
    import scipy

    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": numpy.__version__,
            "scipy": scipy.__version__, "mpmath": mpmath.__version__}


def write_result(name: str, payload: Any, *, script: Path, params: dict, started: float,
                 extra: dict | None = None) -> Path:
    """extra 는 manifest 항목에 덧붙일 값(판정이 믿는 기록의 sha256 목록 등)이다."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{name}.json"
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=_plain) + "\n")
    manifest_path = RESULTS_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"track": "T7-hyperbolic",
                                                                                      "entries": {}}
    manifest["entries"][name] = {
        "output": out.name, "output_sha256": sha256_file(out), "script": str(script.relative_to(REPO_DIR)),
        "script_sha256": sha256_file(script), "code_sha256": code_hashes(), "git_sha": git_sha(), "params": params,
        "seconds": round(time.time() - started, 2), "environment": environment(), **(extra or {}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    return out


def _plain(obj):
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")
