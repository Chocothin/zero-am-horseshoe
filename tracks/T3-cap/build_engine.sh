#!/usr/bin/env bash
# engine/*.cpp 실행 파일을 build/engine-<config>/ 에 만든다. 인자: native-strict | x86filib-strict (기본 native-strict)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${1:-native-strict}"
PREFIX="$HERE/build/capd-$CONFIG-install"
# mutation 감사는 바꾼 소스를 임시 디렉터리에서 빌드한다(기본값은 engine/ → build/engine-<config>/)
SRC="${ENGINE_SRC_DIR:-$HERE/engine}"
OUT="${ENGINE_OUT_DIR:-$HERE/build/engine-$CONFIG}"

case "$CONFIG" in
  native-strict) ARCH="" ;;
  x86filib-strict) ARCH="-arch x86_64" ;;
  # 기본 fp-contract 의 build/engine-native·engine-x86filib 은 감사가 대조한 원 바이너리라 지금 소스로 덮어쓰지 않는다
  native | x86filib) echo "config $CONFIG is kept only for the audited binaries and T9; use $CONFIG-strict" >&2; exit 2 ;;
  *) echo "unknown config: $CONFIG" >&2; exit 2 ;;
esac
bash "$HERE/build_capd.sh" "$CONFIG" >/dev/null
mkdir -p "$OUT"
PC="$PREFIX/lib/pkgconfig/capd.pc"
pc_field() { grep "^$1:" "$PC" | cut -d: -f2- | sed -e "s|\${capd_prefix}|$PREFIX|g" -e "s|\${libdir}|$PREFIX/lib|g"; }
# capd.pc 의 Cflags 에는 CMake 의 CMAKE_CXX_FLAGS 가 실리지 않으므로 융합 금지를 엔진 빌드에 따로 준다
CFLAGS="$(pc_field Cflags) -ffp-contract=off"
LIBS="$(pc_field Libs)"
if [ "$CONFIG" = native-strict ]; then
  CFLAGS="$CFLAGS -I/opt/homebrew/include"
  LIBS="$LIBS -L/opt/homebrew/lib -lmpfr -lgmp"
fi

# MpStepControl.h 는 CAPD 설치 단계가 빠뜨리므로 같은 커밋의 소스 헤더를 후순위로 찾는다
for src in "$SRC"/*.cpp; do
  name="$(basename "$src" .cpp)"
  bin="$OUT/$name"
  # 다른 .cpp 를 #include 하는 음성 대조 빌드는 그 파일이 바뀌어도 다시 만든다
  deps="$src $(sed -n 's/^#include "\(.*\.cpp\)"$/\1/p' "$src" | sed "s|^|$SRC/|")"
  stale=""
  for dep in $deps; do [ "$dep" -nt "$bin" ] && stale=1; done
  if [ ! -x "$bin" ] || [ -n "$stale" ] || [ "$0" -nt "$bin" ] || [ -n "$(find "$SRC" -name '*.hpp' -newer "$bin" 2>/dev/null)" ]; then
    # 동시에 여러 프로세스가 빌드해도 반쯤 쓴 바이너리를 실행하지 않도록 임시 파일에 만든 뒤 원자적으로 바꾼다
    # shellcheck disable=SC2086
    /usr/bin/clang++ $ARCH $CFLAGS -Wall -Wno-deprecated-literal-operator -Wno-inconsistent-missing-override \
      -I"$SRC" -idirafter "$HERE/third_party/capd/capdDynSys/include" "$src" -o "$bin.tmp.$$" $LIBS
    # 컴파일 명령을 바이너리 옆에 남겨 scripts/check_build_flags.py 가 융합 금지 플래그를 확인한다
    echo "/usr/bin/clang++ $ARCH $CFLAGS $src $LIBS" > "$bin.cmd.tmp.$$"
    mv -f "$bin.cmd.tmp.$$" "$bin.cmd"
    mv -f "$bin.tmp.$$" "$bin"
  fi
done
echo "$OUT"
