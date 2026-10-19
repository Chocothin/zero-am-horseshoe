"""스크립트 공통: 경로, 결과 디렉터리, manifest(git SHA·스크립트 sha256·매개변수·실행 시간·환경) 기록."""
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
RESULTS_DIR = REPO_DIR / "results" / "2026-10-17-T3-cap"
sys.path.insert(0, str(TRACK_DIR))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str:
    out = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip()


CODE_FILES = ["engine/t3cap_engine.cpp", "t3cap/api.py", "t3cap/engine.py", "t3cap/exact.py", "t3cap/model.py",
              "t3cap/verdict.py", "t3cap/coords.py", "t3cap/export.py", "reference/mp_reference.py", "scripts/common.py",
              "build_capd.sh", "build_engine.sh"]


def code_hashes() -> dict[str, str]:
    """결과가 기대는 코드의 해시 — 코드가 바뀐 뒤 남은 낡은 결과를 check_results 가 걸러낸다."""
    return {rel: sha256_file(TRACK_DIR / rel) for rel in CODE_FILES}


def environment() -> dict[str, str]:
    import mpmath
    import numpy

    engines = {config: TRACK_DIR / "build" / f"engine-{config}" / "t3cap_engine"
               for config in ("native-strict", "x86filib-strict")}
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mpmath": mpmath.__version__,
        "numpy": numpy.__version__,
        "capd_source": (TRACK_DIR / "third_party" / "CAPD_SOURCE.txt").read_text().strip().replace("\n", "; "),
        **{f"engine_{config.replace('-', '_')}_sha256": sha256_file(path) if path.exists() else "missing"
           for config, path in engines.items()},
    }


def write_result(name: str, payload: Any, *, script: Path, params: dict, started: float) -> Path:
    """results/<날짜>-T3-cap/<name>.json 과 manifest 항목을 함께 쓴다."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{name}.json"
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    manifest_path = RESULTS_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"track": "T3-cap", "entries": {}}
    manifest["entries"][name] = {
        "output": out.name,
        "output_sha256": sha256_file(out),
        "script": str(script.relative_to(REPO_DIR)),
        "script_sha256": sha256_file(script),
        "code_sha256": code_hashes(),
        "git_sha": git_sha(),
        "params": params,
        "seconds": round(time.time() - started, 2),
        "environment": environment(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    return out
