#!/usr/bin/env bash
# T7-hyperbolic 검증(exit 0 = 통과). 레포에 쓰지 않는다: 바이트코드·pytest 캐시는 끄고 결과 파일은 쓰지 않는다(--no-write).
# 단계: 명세 시험 → 저장 결과 재판정·manifest 해시 → 덮개별 원뿔(314)·전용 대조 → 정리 A′ 주장·전용 대조
#       → G8(x86filib D(B)·독립 판정) → 옛(93a1692)·새 원뿔 결과 대조 → 요청 토큰 사전 점검 → G7 → G5 빠른 모드
# 결과는 results/2026-10-17-T7-hyperbolic, 읽는 T6 기록은 results/2026-10-17-T6-horseshoe 다(t7hb/results.py).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
# 트랙 CPU 예산(BRIEF rev3): 동시 worker 3 이하, nice 10, BLAS 1스레드
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
NICE="nice -n 10"
start=$(date +%s)
step() { echo "== [$(( $(date +%s) - start ))s] $*"; }

step "명세 시험(원뿔 판정층·독립 판정·기록 추출·정리 조립)"
(cd "$HERE" && $NICE "$PY" -m pytest -q -p no:cacheprovider tests)

step "저장 결과 재판정(정확 유리수, 엔진 없음) + manifest 해시(결과·스크립트·T3·T6 코드, T6 덮개 기록 314개)"
$NICE "$PY" "$HERE/scripts/check_results.py"

step "덮개별 원뿔 조건 314개와 전용 대조 942개(controls_cover·leaked 0·날카로움 요구)"
$NICE "$PY" "$HERE/scripts/certify_cones.py" --no-write

step "정리 A′: T6 정리 A 전제 재판정, 주장 8개(A5·μ_u·μ_s·A6·λ_s·λ_u·C_s·C_u)와 주장마다 전용 대조 2개, 방향 반올림 일치"
$NICE "$PY" "$HERE/scripts/assemble_aprime.py" --no-write > /dev/null

step "G8: x86filib D(B) 314개 재판정·주장 8개, 중점·반경 감싸기 + Weyl + 정확 Cholesky 942개"
$NICE "$PY" "$HERE/scripts/g8_cones.py" --no-write > /dev/null

step "옛(93a1692)·새 원뿔 결과 314변 대조(판정 일치, a*·b*·여유 정확값, 정리 A′ 주장)"
$NICE "$PY" "$HERE/scripts/compare_strict_rebuild.py" --no-write > /dev/null

step "요청 토큰 사전 점검(T6·T7 이 만드는 엔진 요청 전수, 산술별 정밀도 기준, 심은 대조 6건, 엔진 적분 없음)"
$NICE "$PY" "$HERE/scripts/audit_request_tokens.py" --no-write > /dev/null

step "G7 mutation 감사(판정층·독립 판정·추출·조립층, kill >= 0.9)"
$NICE "$PY" "$HERE/scripts/mutation_audit.py" --workers 3 --no-write | grep -E "^(mutants|SURVIVED)"

step "G5 빠른 모드(변 2개 x 상자 100점, 직접 mpmath 야코비안 1점)"
$NICE "$PY" "$HERE/scripts/g5_derivative.py" --only "N0>N1,M-1>M+0" --points 100 --workers 2 --no-write | tail -1

step "PASS"
