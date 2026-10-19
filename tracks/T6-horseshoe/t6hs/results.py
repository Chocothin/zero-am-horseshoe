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
# h-set 설계 단계(궤도 재고정·덮개 그래프 설계, 엔진과 무관한 [수치])는 처음 만든 폴더에 두고, 덮개 인증 결과는
# 융합 금지 빌드로 다시 낸 폴더에 둔다. 인증은 설계 파일을 sha256 으로 묶어 읽는다
DESIGN_DIR = REPO_DIR / "results" / "2026-10-15-T6-horseshoe"
RESULTS_DIR = REPO_DIR / "results" / "2026-10-17-T6-horseshoe"
T3_DIR = REPO_DIR / "tracks" / "T3-cap"
# 판정이 기대는 T3 코드(엔진·API·판정·정확 산술·JSON 끝점)와 빌드 플래그를 정하는 빌드 스크립트
T3_FILES = ("engine/t3cap_engine.cpp", "t3cap/api.py", "t3cap/verdict.py", "t3cap/exact.py", "t3cap/model.py",
            "t3cap/export.py", "t3cap/__init__.py", "build_capd.sh", "build_engine.sh")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str:
    out = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip()


def code_hashes() -> dict[str, str]:
    """결과가 기대는 T6 코드와 T3 엔진의 해시 — 코드가 바뀐 뒤 남은 낡은 결과를 검사기가 걸러낸다."""
    files = sorted((TRACK_DIR / "t6hs").glob("*.py"))
    out = {str(p.relative_to(REPO_DIR)): sha256_file(p) for p in files}
    for rel in T3_FILES:
        out[f"tracks/T3-cap/{rel}"] = sha256_file(T3_DIR / rel)
    return out


def environment() -> dict[str, str]:
    import mpmath
    import numpy
    import scipy

    engines = {config: T3_DIR / "build" / f"engine-{config}" / "t3cap_engine" for config in ("native-strict", "x86filib-strict")}
    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": numpy.__version__,
            "scipy": scipy.__version__, "mpmath": mpmath.__version__,
            **{f"engine_{config.replace('-', '_')}_sha256": sha256_file(path) if path.exists() else "missing"
               for config, path in engines.items()}}


def write_result(name: str, payload: Any, *, script: Path, params: dict, started: float,
                 results_dir: Path = RESULTS_DIR) -> Path:
    """results_dir 는 설계 단계 스크립트만 DESIGN_DIR 로 바꾼다."""
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{name}.json"
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=_plain) + "\n")
    manifest_path = results_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"track": "T6-horseshoe",
                                                                                      "entries": {}}
    manifest["entries"][name] = {
        "output": out.name, "output_sha256": sha256_file(out), "script": str(script.relative_to(REPO_DIR)),
        "script_sha256": sha256_file(script), "code_sha256": code_hashes(), "git_sha": git_sha(), "params": params,
        "seconds": round(time.time() - started, 2), "environment": environment(),
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
