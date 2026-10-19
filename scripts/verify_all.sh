#!/usr/bin/env bash
# 각 트랙의 tracks/<T>/verify.sh 를 실행한다. 하나라도 실패하면 exit 1.
set -u
cd "$(dirname "$0")/.."
fail=0; n=0
for v in tracks/*/verify.sh; do
  [ -e "$v" ] || continue
  n=$((n+1))
  echo "== $v"
  if bash "$v"; then echo "PASS $v"; else echo "FAIL $v"; fail=1; fi
done
echo "verify_all: $n scripts, fail=$fail"
exit $fail
