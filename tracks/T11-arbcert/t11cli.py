"""T11 명령 진입점. `python -I` 는 현재 폴더를 sys.path 에 넣지 않으므로 트랙 폴더를 넣고 t11 하위 명령으로 넘긴다.

사용: python -I -B tracks/T11-arbcert/t11cli.py <명령> [인자...]
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

COMMANDS = {
    "jobs": "t11.jobs",
    "assemble": "t11.assemble",
    "controls": "t11.controls",
    "g5": "t11.g5",
    "g8": "t11.g8",
    "provenance": "t11.provenance",
    "mutation": "t11.mutation",
    "manifest": "t11.manifest",
    "independence": "t11.independence",
    "spec": "t11.spec",
    "crossing-mutant": "t11.mutant_probe",
    "tube-mutants": "t11.tube_mutants",
    "compare": "t11.compare",
    "tube-rerun": "t11.tube_rerun",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"usage: t11cli.py {{{'|'.join(COMMANDS)}}} [args...]", file=sys.stderr)
        return 2
    module = importlib.import_module(COMMANDS[sys.argv[1]])
    return module.main(sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
