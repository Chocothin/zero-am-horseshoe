import sys
from pathlib import Path

TRACK = Path(__file__).resolve().parent.parent
REPO = TRACK.parent.parent
# mutation 감사는 트랙을 임시 디렉터리에 복사해 돌리므로 T3·T6 는 레포 경로로 직접 건다
for p in (TRACK, REPO / "tracks" / "T3-cap", REPO / "tracks" / "T6-horseshoe"):
    sys.path.insert(0, str(p))
