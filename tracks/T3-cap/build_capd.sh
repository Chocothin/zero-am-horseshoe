#!/usr/bin/env bash
# CAPD 를 build/ 아래에 빌드·설치한다. 인자: native-strict | x86filib-strict (엔진용, -ffp-contract=off)
#   | native | x86filib (기본 fp-contract 의 기존 설치: T9 가 capd-native-install 을 링크한다)
# native 계열은 arm64·CAPD 자체 구간·MPFR, x86filib 계열은 x86_64 filib(Rosetta 2)다.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${1:-native-strict}"
SRC="$HERE/third_party/capd"
BUILD="$HERE/build/capd-$CONFIG"
PREFIX="$HERE/build/capd-$CONFIG-install"
JOBS="${JOBS:-2}"  # CPU 예산(tbp01-p1 BRIEF): make -j 2 이하

NATIVE=(-DCAPD_INTERVAL_TYPE=NATIVE -DCAPD_ENABLE_MULTIPRECISION=ON "-DCMAKE_PREFIX_PATH=/opt/homebrew"
        "-DCMAKE_EXE_LINKER_FLAGS=-L/opt/homebrew/lib" "-DCMAKE_SHARED_LINKER_FLAGS=-L/opt/homebrew/lib")
# 공식 안내(filibsrc/CMakeLists.txt): Apple Silicon 에서 filib 는 x86_64 로 빌드해 Rosetta 2 로 실행한다
X86FILIB=(-DCAPD_INTERVAL_TYPE=FILIB -DCAPD_ENABLE_MULTIPRECISION=OFF -DCMAKE_OSX_ARCHITECTURES=x86_64)
# 방향 반올림 구간의 끝점 계산에 곱셈·덧셈 융합(fmadd)이 끼지 않게 융합을 끈다. 컴파일 명령은 확인용으로 남긴다
STRICT=(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON "-DCMAKE_C_FLAGS=-ffp-contract=off")
case "$CONFIG" in
  native) ARGS=("${NATIVE[@]}" "-DCMAKE_CXX_FLAGS=-I/opt/homebrew/include") ;;
  x86filib) ARGS=("${X86FILIB[@]}") ;;
  native-strict) ARGS=("${NATIVE[@]}" "${STRICT[@]}" "-DCMAKE_CXX_FLAGS=-I/opt/homebrew/include -ffp-contract=off") ;;
  x86filib-strict) ARGS=("${X86FILIB[@]}" "${STRICT[@]}" "-DCMAKE_CXX_FLAGS=-ffp-contract=off") ;;
  *) echo "unknown config: $CONFIG" >&2; exit 2 ;;
esac

if [ -f "$PREFIX/.built" ]; then
  echo "capd-$CONFIG already built at $PREFIX"
  exit 0
fi
# 소스 무결성은 새로 빌드할 때만 대조한다(verify.sh 는 fetch_capd.sh 로 따로 대조한다)
bash "$HERE/fetch_capd.sh" >/dev/null
# PATH 의 ~/.local/bin/cc 는 Claude Code 래퍼라서 컴파일러를 절대경로로 고정한다
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=$PREFIX" \
  -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ "${ARGS[@]}"
cmake --build "$BUILD" -j "$JOBS"
cmake --install "$BUILD"
date -u +%Y-%m-%dT%H:%M:%SZ > "$PREFIX/.built"
echo "capd-$CONFIG installed at $PREFIX"
