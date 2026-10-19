#!/usr/bin/env bash
# CAPD 소스를 고정 커밋으로 third_party/capd 에 받는다(.git 제외). 이미 있으면 트리 해시만 대조한다.
set -euo pipefail

CAPD_URL="https://github.com/CAPDGroup/CAPD"
CAPD_COMMIT="2f060980d0685cc9bf88352e08c242197cd7d686"
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/third_party/capd"
STAMP="$HERE/third_party/CAPD_SOURCE.txt"

tree_hash() {
  (cd "$1" && find . -type f ! -name '.DS_Store' -print0 | LC_ALL=C sort -z | xargs -0 shasum -a 256 | shasum -a 256 | cut -d' ' -f1)
}

if [ -d "$DEST" ]; then
  expected="$(grep '^tree_sha256=' "$STAMP" | cut -d= -f2)"
  actual="$(tree_hash "$DEST")"
  if [ "$expected" != "$actual" ]; then
    echo "FAIL: $DEST tree hash $actual != recorded $expected" >&2
    exit 1
  fi
  echo "capd source OK (commit $CAPD_COMMIT, tree $actual)"
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
git -C "$tmp" init -q
git -C "$tmp" remote add origin "$CAPD_URL"
git -C "$tmp" fetch -q --depth 1 origin "$CAPD_COMMIT"
git -C "$tmp" checkout -q FETCH_HEAD
got="$(git -C "$tmp" rev-parse HEAD)"
if [ "$got" != "$CAPD_COMMIT" ]; then
  echo "FAIL: fetched $got, expected $CAPD_COMMIT" >&2
  exit 1
fi
rm -rf "$tmp/.git"
mkdir -p "$HERE/third_party"
cp -R "$tmp" "$DEST"
cat > "$STAMP" <<EOF
name=CAPD::DynSys (Computer Assisted Proofs in Dynamics)
url=$CAPD_URL
commit=$CAPD_COMMIT
version=6.1.0-dev (CAPDVersion.txt; upstream master, no release tag)
license=GPL-3.0 (third_party/capd/COPYING)
fetched=$(date -u +%Y-%m-%dT%H:%M:%SZ)
tree_sha256=$(tree_hash "$DEST")
EOF
cat "$STAMP"
