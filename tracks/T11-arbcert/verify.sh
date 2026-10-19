#!/usr/bin/env bash
# T11-arbcert 검증: 저장 판정의 정확 재판정, manifest·digest·복사 출처 대조, 2변 재실행 일치(감사 훅), G5·G6·G8 재판정, 명세 시험.
# exit 0 = 통과. 레포에는 쓰지 않는다(mktemp 만). 314변 전부가 증명되지 않거나 빠진 변이 있으면 조립 단계가 exit 1 로 끝난다.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
RES="$REPO/results/2026-10-17-T11-arbcert"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
CLI=(nice -n 10 "$PY" -I -B "$HERE/t11cli.py")
START=$(date +%s)
BEFORE="$(cd "$REPO" && find tracks/T11-arbcert results/2026-10-17-T11-arbcert -path '*/raw-local' -prune -o -type f -print0 | xargs -0 shasum -a 256 | shasum -a 256)"

echo "[1/12] 복사 출처: 감사 원본·T11 사본 sha256 과 원본→사본 diff 가 기록과 같다"
"${CLI[@]}" provenance check

echo "[2/12] manifest: 변별 기록 628개 sha256 목록·묶음 digest·출력·스크립트 해시"
"${CLI[@]}" manifest check

echo "[3/12] 실행기 상태: 628작업 모두 출력·마지막 실행 exit 0"
"${CLI[@]}" jobs status --strict > /dev/null

echo "[4/12] 정확 재판정(exact_judge)과 조립·주장 판정(314변 전부 증명이 아니면 exit 1)"
nice -n 10 "$PY" -I -B "$HERE/exact_judge.py" --out "$TMP/exact_judge.json" > /dev/null
"${CLI[@]}" assemble --out "$TMP/theorem_claims.json" --merged-tubes "$TMP/rigorous_tubes.json" > /dev/null
nice -n 10 "$PY" -I -B "$HERE/graph_and_hulls.py" --out "$TMP/graph_and_hulls.json" > /dev/null
"${CLI[@]}" compare "$RES/exact_judge.json" "$TMP/exact_judge.json" "$RES/theorem_claims.json" "$TMP/theorem_claims.json" \
  "$RES/rigorous_tubes.json" "$TMP/rigorous_tubes.json" "$RES/graph_and_hulls.json" "$TMP/graph_and_hulls.json"

echo "[5/12] G6 음성 대조 재판정(controls_cover, leaked 0, 날카로움)"
"${CLI[@]}" controls --out "$TMP/controls.json" > /dev/null
"${CLI[@]}" compare "$RES/controls.json" "$TMP/controls.json"

echo "[6/12] G8 CAPD 교차 대조 재판정(포함·부호·차수·모순 0)"
"${CLI[@]}" g8 --out "$TMP/g8.json" > /dev/null
"${CLI[@]}" compare "$RES/g8_capd_crosscheck.json" "$TMP/g8.json"
# 엄격 빌드 재인증 기록(E1)도 T11 감싸기를 담는지 다시 본다(T11.1)
"${CLI[@]}" g8 --capd-dir "$REPO/results/2026-10-17-T6-horseshoe/coverings" --out "$TMP/g8_strict.json" > /dev/null
"${CLI[@]}" compare "$RES/g8_capd_crosscheck_2026-10-17.json" "$TMP/g8_strict.json"

echo "[7/12] G5 재판정(커밋된 기준 해 1,424점)과 G5-꼭짓점 재판정(5,024점) + 기준 해 3점 재적분 일치"
"${CLI[@]}" g5 judge --out-dir "$TMP" > /dev/null
"${CLI[@]}" g5 judge --vertices --out-dir "$TMP" > /dev/null
"${CLI[@]}" compare "$RES/g5_soundness.json" "$TMP/g5_soundness.json" "$RES/g5_tube_samples.json" "$TMP/g5_tube_samples.json" \
  "$RES/g5_vertex_soundness.json" "$TMP/g5_vertex_soundness.json"
"${CLI[@]}" g5 repeat --ids "N1__N2__r0,M+146__M+147__c,N3__N4__v05"

echo "[8/12] V2 음성 대조(rigorous_controls) 재실행"
nice -n 10 "$PY" -I -B "$HERE/rigorous_controls.py" --out "$TMP/rigorous_controls.json" > /dev/null
"${CLI[@]}" compare "$RES/rigorous_controls.json" "$TMP/rigorous_controls.json"

echo "[9/12] 독립성·재현: 2변 재실행을 감사 훅 아래 돌려 CAPD 경로 접근 0, 저장 기록과 정확 일치"
"${CLI[@]}" independence --edges "N1>N2,M+146>M+147" --out "$TMP/independence.json" > /dev/null

echo "[10/12] G9 명세 시험"
"${CLI[@]}" spec --out "$TMP/spec.json" > /dev/null

echo "[11/12] 저장된 G7·변이 확인 결과"
"$PY" -I -B - "$RES" <<'PY'
import json, sys
from pathlib import Path
res = Path(sys.argv[1])
m = json.loads((res / "mutation_audit.json").read_text())
assert m["kill_rate"] >= 0.9, m["kill_rate"]
c = json.loads((res / "crossing_jets_mutant.json").read_text())
assert c["summary"]["new_G9_test_crossing_jets"] != 0, "crossing_jets 변이를 새 시험이 잡지 못했다"
t = json.loads((res / "tube_rigor_mutants.json").read_text())
assert all(v["caught"] for v in t["mutants"].values()), "V2 엄밀성 변이를 잡지 못한 시험이 있다"
r = json.loads((res / "tube_rerun_r105.json").read_text())
assert r["identical"] == 314, r
print(f"  G7 kill {m['killed']}/{m['mutants']} = {m['kill_rate']:.3f}; crossing_jets 변이 검출; V2 변이 {len(t['mutants'])}개 검출; 관 재실행 314/314 동일")
PY

echo "[12/12] 레포 무쓰기: T11 트랙·결과 폴더(raw-local 제외)의 sha256 이 실행 전과 같고 바이트코드 잔여가 없다"
AFTER="$(cd "$REPO" && find tracks/T11-arbcert results/2026-10-17-T11-arbcert -path '*/raw-local' -prune -o -type f -print0 | xargs -0 shasum -a 256 | shasum -a 256)"
[ "$BEFORE" = "$AFTER" ] || { echo "FAIL: verify wrote into the repo"; exit 1; }
[ -z "$(find "$HERE" -name __pycache__ -o -name '*.pyc' -o -name .pytest_cache)" ] || { echo "FAIL: bytecode left"; exit 1; }

echo "T11-arbcert verify: PASS ($(( $(date +%s) - START ))초)"
