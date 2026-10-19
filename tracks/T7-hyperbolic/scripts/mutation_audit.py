"""G7 mutation 감사: 원뿔 판정층(cone.py), G8 독립 판정(weyl.py), 저장 기록 추출(stored.py), 정리 A′ 조립층(hyperbolic.py)에
AST mutant 를 하나씩 심고 명세 시험(G9)으로 죽는지 본다. mutant 마다 레포 배치를 흉내 낸 임시 디렉터리(T3·T6·results 는
레포로 가는 심볼릭 링크)에 t7hb·tests 를 복사해 돌린다(레포에 쓰지 않음).
"""
from __future__ import annotations

import argparse
import ast
import copy
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from t7hb.results import REPO_DIR, TRACK_DIR, write_result  # noqa: E402

PYTHON = sys.executable
TARGETS = ("t7hb/cone.py", "t7hb/weyl.py", "t7hb/stored.py", "t7hb/hyperbolic.py")
TESTS = ["tests/test_cone.py", "tests/test_weyl.py", "tests/test_stored.py", "tests/test_hyperbolic.py"]
COMPARE_SWAPS = {ast.Lt: [ast.LtE, ast.Gt], ast.LtE: [ast.Lt, ast.GtE], ast.Gt: [ast.GtE, ast.Lt],
                 ast.GtE: [ast.Gt, ast.LtE], ast.Eq: [ast.NotEq], ast.NotEq: [ast.Eq], ast.In: [ast.NotIn],
                 ast.NotIn: [ast.In]}
BINOP_SWAPS = {ast.Add: [ast.Sub], ast.Sub: [ast.Add], ast.Mult: [ast.Div], ast.Div: [ast.Mult]}
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


def owners(tree):
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                out.setdefault(id(child), node.name)
    return out


def sites(tree):
    own = owners(tree)
    out = []
    for idx, node in enumerate(ast.walk(tree)):
        fn = own.get(id(node))
        if fn is None:
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


def run_mutant(rel, source_tree, target, fn, op, payload) -> Mutant:
    mutated_tree, before, after = apply(source_tree, target, payload)
    line = getattr(next(n for i, n in enumerate(ast.walk(source_tree)) if i == target), "lineno", 0)
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        track = repo / "tracks" / "T7-hyperbolic"
        shutil.copytree(TRACK_DIR / "t7hb", track / "t7hb")
        shutil.copytree(TRACK_DIR / "tests", track / "tests")
        for name in ("T3-cap", "T6-horseshoe"):
            os.symlink(REPO_DIR / "tracks" / name, repo / "tracks" / name)
        os.symlink(REPO_DIR / "results", repo / "results")
        (track / rel).write_text(ast.unparse(mutated_tree) + "\n")
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        proc = subprocess.run([PYTHON, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *TESTS], cwd=track,
                              capture_output=True, text=True, env=env, timeout=600)
    return Mutant(rel, fn, line, op, before[:120], after[:120], proc.returncode != 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    jobs = []
    for rel in TARGETS:
        tree = ast.parse((TRACK_DIR / rel).read_text())
        for target, fn, op, payload in sites(tree):
            jobs.append((rel, tree, target, fn, op, payload))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        mutants = list(pool.map(lambda j: run_mutant(*j), jobs))
    killed = sum(m.killed for m in mutants)
    survivors = [asdict(m) for m in mutants if not m.killed]
    print(f"mutants {len(mutants)} killed {killed} ({killed / max(1, len(mutants)):.3f})")
    for s in survivors:
        print("SURVIVED", s["file"], s["function"], s["line"], s["operator"], "|", s["original"], "=>", s["mutated"])
    if not args.no_write:
        write_result("mutation_audit", {"tag": "G7", "mutants": len(mutants), "killed": killed,
                                        "kill_rate": killed / max(1, len(mutants)), "survivors": survivors},
                     script=Path(__file__).resolve(), params=vars(args), started=started)
    return 0 if killed / max(1, len(mutants)) >= 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
