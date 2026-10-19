"""다시 낸 JSON 이 저장 결과와 같은지 본다(실행 시간 칸은 뺀다). 다르면 첫 차이 경로를 찍고 exit 1.

사용: t11cli.py compare STORED NEW [STORED NEW ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TIME_KEYS = {"seconds", "seconds_center", "seconds_hull", "seconds_judge", "seconds_reference_total"}


def strip(d):
    if isinstance(d, dict):
        return {k: strip(v) for k, v in d.items() if k not in TIME_KEYS}
    if isinstance(d, list):
        return [strip(v) for v in d]
    return d


def first_difference(a, b, path="$"):
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: missing on one side"
            d = first_difference(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {str(a)[:80]} vs {str(b)[:80]}"


def load(p: Path):
    text = p.read_text()
    if p.suffix == ".jsonl":
        return [json.loads(x) for x in text.splitlines() if x.strip()]
    return json.loads(text)


def main(argv=None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) < 2 or len(args) % 2:
        print("usage: compare STORED NEW [STORED NEW ...]", file=sys.stderr)
        return 2
    bad = 0
    for stored, new in zip(args[::2], args[1::2]):
        d = first_difference(strip(load(Path(stored))), strip(load(Path(new))))
        print(f"  {Path(stored).name}: {'same' if d is None else 'DIFFERS ' + d}")
        bad += d is not None
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
