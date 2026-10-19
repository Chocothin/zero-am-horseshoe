#!/usr/bin/env bash
# T3-cap 검증(exit 0 = 통과). 빌드·융합 금지 확인 → 명세 시험(컴파일러 탐침 포함) → G5·G6·이정표·G7 빠른 모드 → results/ 해시·주장 대조.
# 레포에는 쓰지 않는다: 빌드 산출물은 gitignore 된 build/, 바이트코드·pytest 캐시는 끈다.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$HERE:$HERE/scripts"
# 트랙 CPU 예산(BRIEF rev3, tbp01-p1): 동시 worker 3 이하, 빌드 -j 2, nice 10, BLAS 1스레드
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 JOBS=2
NICE="nice -n 10"
start=$(date +%s)

step() { echo "== [$(( $(date +%s) - start ))s] $*"; }

step "CAPD 소스 무결성(고정 커밋 트리 해시)"
if [ ! -d "$HERE/third_party/capd" ]; then
  echo "FAIL: third_party/capd 없음 — bash $HERE/fetch_capd.sh 로 받은 뒤 다시 실행" >&2
  exit 1
fi
bash "$HERE/fetch_capd.sh"

step "CAPD·엔진 빌드 확인(native-strict arm64, x86filib-strict x86_64, -ffp-contract=off) — build/ 에만 쓴다"
$NICE bash "$HERE/build_capd.sh" native-strict >/dev/null
$NICE bash "$HERE/build_capd.sh" x86filib-strict >/dev/null
$NICE bash "$HERE/build_engine.sh" native-strict >/dev/null
$NICE bash "$HERE/build_engine.sh" x86filib-strict >/dev/null

step "융합 금지 확인(CAPD·엔진 컴파일 명령 전수, strict 바이너리·라이브러리의 융합 명령 0)"
$NICE "$PY" "$HERE/scripts/check_build_flags.py" > /dev/null

step "명세 시험(pytest: 구간 산술 대조·컴파일러 탐침·끝점 가드·벡터장·차트·판정·API 계약·기준 해)"
(cd "$HERE" && $NICE "$PY" -m pytest -q -p no:cacheprovider tests)

step "G5 건전성 빠른 모드(설정당 12점) + 시험기 음성 대조"
$NICE "$PY" "$HERE/scripts/g5_soundness.py" --quick --workers 3

step "이정표 빠른 모드(본 증명 + 음성 대조 2건)"
$NICE "$PY" "$HERE/scripts/milestone_figure8.py" --quick --no-write

step "G7 mutation 감사(kill >= 0.9)"
$NICE "$PY" "$HERE/scripts/mutation_audit.py" --workers 3 --no-write | tail -3

step "G8 독립 재판정: 저장된 CAPD 감싸기로 Krawczyk·사용 자격 부등식을 Arb 로 다시 판정"
$NICE "$PY" "$HERE/scripts/arb_krawczyk.py" --no-write

step "results/ manifest 해시와 저장된 핵심 주장 대조"
"$PY" "$HERE/scripts/check_results.py"

step "옛(93a1692)·새 결과 대조 기록(실행 시간·로그·빌드 이름 제외, 판정 없음)"
"$PY" "$HERE/scripts/compare_strict_rebuild.py" --no-write | tail -3

step "PASS"
