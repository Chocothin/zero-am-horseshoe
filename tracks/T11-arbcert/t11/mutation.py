"""G7 mutation 감사: 판정 함수(v7exact·exact_judge)와 조립층(claims·assemble)에 AST mutant 를 하나씩 심고 시험으로 죽는지 본다.

mutant 마다 트랙을 레포 밖 임시 폴더에 복사하고 results 는 읽기 전용 심볼릭 링크로 둔다(레포에 쓰지 않는다).
사용: t11cli.py mutation [--workers 3] [--out PATH]
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from .paths import BLAS_ONE, PYTHON, REPO, RESULTS, TRACK

OUT = RESULTS / "mutation_audit.json"
TARGETS = {
    "v7exact.py": None,
    "exact_judge.py": {"design_signs", "judge_v5"},
    "t11/claims.py": {"judge_claims", "epsilon"},
    "t11/assemble.py": {"structure_failures", "edge_verdict", "expected_edges", "hull_gaps", "assemble",
                        "load_records"},
}
TESTS = ["tests/test_judge_spec.py", "tests/test_claims_spec.py", "tests/test_assemble_spec.py", "tests/test_guards_spec.py"]
COMPARE_SWAPS = {ast.Lt: [ast.LtE, ast.Gt], ast.LtE: [ast.Lt, ast.GtE], ast.Gt: [ast.GtE, ast.Lt],
                 ast.GtE: [ast.Gt, ast.LtE], ast.Eq: [ast.NotEq], ast.NotEq: [ast.Eq], ast.In: [ast.NotIn],
                 ast.NotIn: [ast.In], ast.Is: [ast.IsNot], ast.IsNot: [ast.Is]}
BINOP_SWAPS = {ast.Add: [ast.Sub], ast.Sub: [ast.Add], ast.Mult: [ast.Div], ast.Div: [ast.Mult],
               ast.FloorDiv: [ast.Mult]}
CALL_SWAPS = {"min": "max", "max": "min", "all": "any", "any": "all"}


@dataclass(frozen=True)
class Mutant:
    file: str
    function: str
    line: int
    operator: str
    original: str
    mutated: str
    killed: bool


def owners(tree) -> dict:
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                out.setdefault(id(child), node.name)
    return out


def sites(tree, allowed) -> list:
    own = owners(tree)
    out = []
    for idx, node in enumerate(ast.walk(tree)):
        fn = own.get(id(node))
        if fn is None or (allowed is not None and fn not in allowed):
            continue
        if isinstance(node, ast.Compare):
            for k, op in enumerate(node.ops):
                for repl in COMPARE_SWAPS.get(type(op), []):
                    out.append((idx, fn, f"compare {type(op).__name__}->{repl.__name__}", (k, repl)))
        elif isinstance(node, ast.BoolOp):
            out.append((idx, fn, f"boolop {type(node.op).__name__}->swap", None))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub)):
            out.append((idx, fn, f"drop {type(node.op).__name__}", None))
        elif isinstance(node, ast.BinOp) and type(node.op) in BINOP_SWAPS:
            for repl in BINOP_SWAPS[type(node.op)]:
                out.append((idx, fn, f"binop {type(node.op).__name__}->{repl.__name__}", repl))
        elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
            out.append((idx, fn, f"const {node.value}->{not node.value}", not node.value))
        elif isinstance(node, ast.Constant) and isinstance(node.value, int):
            out.append((idx, fn, f"const {node.value}->{node.value + 1}", node.value + 1))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in CALL_SWAPS:
            out.append((idx, fn, f"call {node.func.id}->{CALL_SWAPS[node.func.id]}", CALL_SWAPS[node.func.id]))
    return out


def apply(tree, target, payload):
    tree = copy.deepcopy(tree)
    for idx, node in enumerate(ast.walk(tree)):
        if idx != target:
            continue
        before = ast.unparse(node)
        if isinstance(node, ast.Compare):
            k, repl = payload
            node.ops[k] = repl()
        elif isinstance(node, ast.BoolOp):
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        elif isinstance(node, ast.UnaryOp):
            repl_node = node.operand
            for parent in ast.walk(tree):
                for field, value in ast.iter_fields(parent):
                    if value is node:
                        setattr(parent, field, repl_node)
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            if item is node:
                                value[i] = repl_node
            return tree, before, ast.unparse(repl_node)
        elif isinstance(node, ast.BinOp):
            node.op = payload()
        elif isinstance(node, ast.Constant):
            node.value = payload
        elif isinstance(node, ast.Call):
            node.func.id = payload
        return tree, before, ast.unparse(node)
    raise IndexError(target)


def staged_copy(tmp: Path) -> Path:
    track = tmp / "tracks" / "T11-arbcert"
    shutil.copytree(TRACK, track, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    os.symlink(REPO / "results", tmp / "results")
    os.symlink(REPO / ".venv", tmp / ".venv")          # 사본 안의 시험이 같은 해석기를 subprocess 로 부른다
    return track


def run_mutant(rel, tree, target, fn, op, payload) -> Mutant:
    mutated, before, after = apply(tree, target, payload)
    line = getattr(next(n for i, n in enumerate(ast.walk(tree)) if i == target), "lineno", 0)
    with tempfile.TemporaryDirectory() as tmp:
        track = staged_copy(Path(tmp))
        (track / rel).write_text(ast.unparse(mutated) + "\n")
        env = {**os.environ, **BLAS_ONE}
        proc = subprocess.run(["nice", "-n", "10", str(PYTHON), "-B", "-m", "pytest", "-q", "-x", "-p",
                               "no:cacheprovider", *TESTS], cwd=track, capture_output=True, text=True, env=env,
                              timeout=900)
    return Mutant(rel, fn, line, op, before[:140], after[:140], proc.returncode != 0)


def baseline() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        track = staged_copy(Path(tmp))
        proc = subprocess.run([str(PYTHON), "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
                              cwd=track, capture_output=True, text=True, env={**os.environ, **BLAS_ONE}, timeout=900)
    print(proc.stdout[-600:])
    return proc.returncode == 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    started = time.time()
    if not baseline():
        print("baseline tests fail; mutation audit is meaningless", file=sys.stderr)
        return 1
    jobs = []
    for rel, allowed in TARGETS.items():
        tree = ast.parse((TRACK / rel).read_text())
        for target, fn, op, payload in sites(tree, allowed):
            jobs.append((rel, tree, target, fn, op, payload))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        mutants = list(pool.map(lambda j: run_mutant(*j), jobs))
    killed = sum(m.killed for m in mutants)
    survivors = [asdict(m) for m in mutants if not m.killed]
    rate = killed / max(1, len(mutants))
    per_file = {}
    for m in mutants:
        f = per_file.setdefault(m.file, {"mutants": 0, "killed": 0})
        f["mutants"] += 1
        f["killed"] += m.killed
    out = {"tag": "G7 mutation 감사(판정 함수 + 조립층)", "targets": {k: sorted(v) if v else "전체" for k, v in TARGETS.items()},
           "tests": TESTS, "mutants": len(mutants), "killed": killed, "kill_rate": rate, "per_file": per_file,
           "survivors": survivors, "seconds": round(time.time() - started, 1)}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(f"mutants {len(mutants)} killed {killed} ({rate:.3f})")
    for s in survivors:
        print("SURVIVED", s["file"], s["function"], s["line"], s["operator"], "|", s["original"], "=>", s["mutated"])
    return 0 if rate >= 0.9 else 1


if __name__ == "__main__":
    sys.exit(main())
