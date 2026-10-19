#!/usr/bin/env bash
# 그림 스크립트를 임시 폴더로 다시 돌려 PDF 3개가 만들어지는지와 figdata.json 이 저장본과 같은지 본다. 레포에는 쓰지 않는다.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR="$TMP/mpl"
nice -n 10 "$PY" -I -B "$ROOT/tracks/PF-figures/make_figures.py" --out "$TMP/out"
for f in fig1-orbits fig2-graph fig3-margins; do
  [ "$(head -c 4 "$TMP/out/$f.pdf")" = "%PDF" ] || { echo "FAIL: $f.pdf 이 PDF 가 아니다"; exit 1; }
  [ -s "$ROOT/paper/p1/figures/$f.pdf" ] || { echo "FAIL: 저장본 $f.pdf 없음"; exit 1; }
done
"$PY" -I -B - "$TMP/out/figdata.json" "$ROOT/paper/p1/figures/figdata.json" <<'PYEOF'
import json, sys
new, old = (json.load(open(p)) for p in sys.argv[1:3])
if new != old:
    for k in sorted(set(new) | set(old)):
        if new.get(k) != old.get(k):
            print("FAIL: figdata.json 이 저장본과 다르다:", k)
    sys.exit(1)
print("figdata.json 일치")
PYEOF
echo "PF-figures verify: OK"
