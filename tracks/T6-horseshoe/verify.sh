#!/usr/bin/env bash
# T6-horseshoe 검증(exit 0 = 통과). 레포에 쓰지 않는다: 엔진 재현 결과는 mktemp 디렉터리, 바이트코드·pytest 캐시는 끈다.
# 단계: 명세 시험 → 저장된 덮개 314개 재판정·manifest 해시 → 정리 A 재조립·주장별 대조 → 덮개별 음성 대조(저장 감싸기)
#       → 옛(93a1692)·새 기록 314변 대조 → G7 → 엔진 재현(변 2개) → G5·G6 빠른 모드
# 덮개 기록은 results/2026-10-17-T6-horseshoe(융합 금지 빌드), h-set 설계는 results/2026-10-15-T6-horseshoe 에 있다(t6hs/results.py).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
# 트랙 CPU 예산(BRIEF rev3): 동시 worker 3 이하, nice 10, BLAS 1스레드
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
NICE="nice -n 10"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
start=$(date +%s)
step() { echo "== [$(( $(date +%s) - start ))s] $*"; }

step "명세 시험(판정층·평균값 조립·조각 이분·기록 추출·정리 조립)"
(cd "$HERE" && $NICE "$PY" -m pytest -q -p no:cacheprovider tests)

step "저장된 덮개 재판정(정확 유리수, 엔진 없음) + manifest 해시(결과·스크립트·T3 코드·덮개 기록 314개)"
$NICE "$PY" "$HERE/scripts/check_results.py"

step "정리 A 가정 재조립과 주장 7개 판정(δ·R·t_min·(A3)·(A4)·ρ·ln ρ), 주장마다 전용 대조 2개(controls_cover 요구)"
$NICE "$PY" "$HERE/scripts/assemble_theorem.py" --no-write

step "G6 덮개별 음성 대조(저장 감싸기, 314개 x 3종 모두 판정 거짓·덮개마다 >= 2)"
$NICE "$PY" "$HERE/scripts/stored_controls.py" --no-write

step "옛(93a1692)·새 덮개 기록 314변 대조(판정·차수·방법·조각 수 일치, 감싸기 비트 단위 비교, 정리 A 주장)"
$NICE "$PY" "$HERE/scripts/compare_strict_rebuild.py" --no-write | tail -4

step "G7 mutation 감사(판정층·추출·조립층, kill >= 0.9)"
$NICE "$PY" "$HERE/scripts/mutation_audit.py" --workers 3 --no-write | grep -E "^(mutants|SURVIVED)"

QUICK="N0>N1,M+0>M+1"
step "엔진 재현 빠른 모드: 가장 어려운 cycle 변과 y* 변(2개)만 임시 디렉터리에 다시 인증"
echo "   통과 기준: 요청한 2변이 증명되고 저장된 본 실행 기록과 감싸기가 겹침."
echo "   certify_graph 의 'all proven: False | edges done: 2 of 314' 는 314변 중 2변만 돌렸다는 뜻이다(실패 아님)."
$NICE "$PY" "$HERE/scripts/certify_graph.py" --only "$QUICK" --out-dir "$TMP/coverings" --no-write
$NICE "$PY" "$HERE/scripts/check_reproduction.py" --dir "$TMP/coverings" --only "$QUICK"

step "G5 빠른 모드(변 2개 x 상자 100점 + 면 40점, 직접 적분 1점)"
$NICE "$PY" "$HERE/scripts/g5_soundness.py" --only "N0>N1,M+148>N8" --points 100 --face-points 10 --direct 1 --no-write

step "G6 구조 대조 빠른 모드(변 2개 x 대조 4종, 엔진 재계산, 모두 실패해야 함)"
$NICE "$PY" "$HERE/scripts/negative_controls.py" --quick --no-write | tail -3

step "PASS"
