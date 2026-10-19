import os
import sys
from pathlib import Path

TRACK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRACK))
# mutation 감사는 임시 디렉터리에서 돌므로 T3 위치를 환경 변수나 형제 경로에서 찾는다
sys.path.insert(0, os.environ.get("T3_CAP_DIR", str(TRACK.parent / "T3-cap")))
