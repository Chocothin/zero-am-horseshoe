#!/usr/bin/env bash
# T11-arbcert 전체 재계산: 314변 흐름 관·상자 감싸기 → 정확 판정·조립 → G5~G9 → 출처·manifest.
# 동시 계산 프로세스는 3개 이하(nice 10, BLAS 1스레드). 어느 단계든 exit 0 이 아니면 멈춘다(치명 가드).
# 기존 출력은 지우지 않는다. 처음부터 다시 만들려면 results/2026-10-17-T11-arbcert 를 비운 뒤 돌린다.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
RES="$REPO/results/2026-10-17-T11-arbcert"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
CLI=(nice -n 10 "$PY" -I -B "$HERE/t11cli.py")
mkdir -p "$RES/raw-local"

# 1. 314변 본 실행(흐름 관 314 + 상자 314, 2레인, 실측 17분)
"${CLI[@]}" jobs worker --lane 0 > "$RES/raw-local/lane0.out" 2>&1 &
"${CLI[@]}" jobs worker --lane 1 > "$RES/raw-local/lane1.out" 2>&1 &
wait
"${CLI[@]}" jobs status --strict

# 2. 정확 판정, 외피·그래프, 조립(314변 전부 증명이 아니면 exit 1), G6, G8
nice -n 10 "$PY" -I -B "$HERE/exact_judge.py"
nice -n 10 "$PY" -I -B "$HERE/graph_and_hulls.py"
"${CLI[@]}" assemble
"${CLI[@]}" controls
"${CLI[@]}" g8

# 3. G5 기준 해(1,424점, 3레인, 실측 약 1시간 반)와 판정
[ -f "$RES/g5_plan.json" ] || "${CLI[@]}" g5 plan
for k in 0 1 2; do "${CLI[@]}" g5 worker --lane "$k" > "$RES/raw-local/g5_lane$k.out" 2>&1 & done
wait
"${CLI[@]}" g5 judge --from-lanes
nice -n 10 "$PY" -I -B "$HERE/rigorous_controls.py"
# G5-꼭짓점(314변 × 16, 3레인, 약 2.5시간). 이탈이 나오면 그 레인이 exit 3 으로 멈추고 ALERT 파일을 남긴다
[ -f "$RES/g5_vertex_plan.json" ] || "${CLI[@]}" g5 plan --vertices
for k in 0 1 2; do "${CLI[@]}" g5 worker --vertices --lane "$k" > "$RES/raw-local/g5v_lane$k.out" 2>&1 & done
wait
"${CLI[@]}" g5 judge --vertices --from-lanes

# 4. G9 명세 시험·정확해·대칭 항등식, 독립성 감사 훅, 시험 공백 변이 확인(R1-01·R1-07), 관 재실행 재현(R1-05)
nice -n 10 "$PY" -I -B "$HERE/scripts/validate_engine.py" --quick
nice -n 10 "$PY" -I -B "$HERE/v5_symmetry_test.py"
"${CLI[@]}" spec
"${CLI[@]}" independence --out "$RES/independence.json"
"${CLI[@]}" crossing-mutant
"${CLI[@]}" tube-mutants
"${CLI[@]}" tube-rerun

# 5. G7(판정 함수 + 조립층, 3워커), 복사 출처, manifest
"${CLI[@]}" mutation --workers 3
"${CLI[@]}" provenance write
"${CLI[@]}" manifest write
echo "T11 run_all: done"
