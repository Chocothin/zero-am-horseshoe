import subprocess
import sys
from pathlib import Path

import pytest

TRACK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRACK_DIR))


def built(config: str) -> Path:
    out = subprocess.run(["bash", str(TRACK_DIR / "build_engine.sh"), config], capture_output=True, text=True)
    if out.returncode != 0:
        pytest.fail(f"engine build failed for {config}: {out.stderr[-2000:]}")
    return Path(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="session")
def native_dir() -> Path:
    return built("native-strict")


@pytest.fixture(scope="session")
def x86_dir() -> Path:
    return built("x86filib-strict")
