"""G7 mutation 감사: 판정 핵심 코드에 AST mutant 를 하나씩 심고 명세 시험(G9)으로 죽는지 본다.

mutant 마다 패키지를 레포 밖 임시 디렉터리에 복사해 돌린다(레포에 쓰지 않음). 살아남은 mutant 는 REPORT 에서 하나씩 해명한다.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from common import TRACK_DIR, write_result

PYTHON = sys.executable
TARGETS = {
    "t3cap/verdict.py": {
        "functions": None,  # 전부
        "tests": ["tests/test_verdict.py"],
    },
    "t3cap/exact.py": {
        "functions": {"__post_init__", "contains", "contains_interval", "strictly_contains_interval", "hull",
                      "magnitude", "__add__", "__sub__", "__neg__", "__mul__", "scale", "_sum", "rmat_ivec",
                      "rmat_imat", "imat_vec", "identity", "matrix_inverse", "round_down_double",
                      "round_up_double", "exact_hex_double", "is_hex_text"},
        "tests": ["tests/test_exact.py", "tests/test_verdict.py"],
    },
}
COMPARE_SWAPS = {ast.Lt: [ast.LtE, ast.Gt], ast.LtE: [ast.Lt, ast.GtE], ast.Gt: [ast.GtE, ast.Lt],
                 ast.GtE: [ast.Gt, ast.LtE], ast.Eq: [ast.NotEq], ast.NotEq: [ast.Eq], ast.In: [ast.NotIn],
                 ast.NotIn: [ast.In], ast.Is: [ast.IsNot], ast.IsNot: [ast.Is]}
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


def enclosing_function(tree: ast.AST) -> dict[int, str]:
    owner: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                owner.setdefault(id(child), node.name)
    return owner


def sites(tree: ast.AST, allowed: set[str] | None) -> list[tuple[int, str, object]]:
    """(노드 순번, 연산자 설명, 바꿀 값) 목록. 순번은 ast.walk 순서."""
    owner = enclosing_function(tree)
    out = []
    for idx, node in enumerate(ast.walk(tree)):
        fn = owner.get(id(node))
        if fn is None or (allowed is not None and fn not in allowed):
            continue
        if isinstance(node, ast.Compare):
            for k, op in enumerate(node.ops):
                for repl in COMPARE_SWAPS.get(type(op), []):
                    out.append((idx, f"compare[{k}] {type(op).__name__}->{repl.__name__}", (k, repl)))
        elif isinstance(node, ast.BoolOp):
            out.append((idx, f"boolop {type(node.op).__name__}->swap", None))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub)):
            out.append((idx, f"drop {type(node.op).__name__}", None))
        elif isinstance(node, ast.BinOp) and type(node.op) in BINOP_SWAPS:
            for repl in BINOP_SWAPS[type(node.op)]:
                out.append((idx, f"binop {type(node.op).__name__}->{repl.__name__}", repl))
        elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
            out.append((idx, f"const {node.value}->{not node.value}", not node.value))
        elif isinstance(node, ast.Constant) and isinstance(node.value, int):
            out.append((idx, f"const {node.value}->{node.value + 1}", node.value + 1))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in CALL_SWAPS:
            out.append((idx, f"call {node.func.id}->{CALL_SWAPS[node.func.id]}", CALL_SWAPS[node.func.id]))
    return out


def apply(tree: ast.AST, target_idx: int, payload) -> tuple[ast.AST, str, str]:
    tree = copy.deepcopy(tree)
    for idx, node in enumerate(ast.walk(tree)):
        if idx != target_idx:
            continue
        before = ast.unparse(node)
        if isinstance(node, ast.Compare):
            k, repl = payload
            node.ops[k] = repl()
        elif isinstance(node, ast.BoolOp):
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        elif isinstance(node, ast.UnaryOp):
            replacement = node.operand
            for parent in ast.walk(tree):
                for field, value in ast.iter_fields(parent):
                    if value is node:
                        setattr(parent, field, replacement)
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            if item is node:
                                value[i] = replacement
            return tree, before, ast.unparse(replacement)
        elif isinstance(node, ast.BinOp):
            node.op = payload()
        elif isinstance(node, ast.Constant):
            node.value = payload
        elif isinstance(node, ast.Call):
            node.func.id = payload
        return tree, before, ast.unparse(node)
    raise IndexError(target_idx)


def run_mutant(rel: str, source_tree: ast.AST, site, tests: list[str]) -> tuple[Mutant, bool, str]:
    idx, desc, payload = site
    tree, before, after = apply(source_tree, idx, payload)
    owner = enclosing_function(source_tree)
    node = list(ast.walk(source_tree))[idx]
    mutant = Mutant(rel, owner.get(id(node), "?"), getattr(node, "lineno", 0), desc, before[:120], after[:120])
    work = Path(tempfile.mkdtemp(prefix="t3cap-mut-"))
    try:
        for sub in ("t3cap", "reference", "tests"):
            shutil.copytree(TRACK_DIR / sub, work / sub, ignore=shutil.ignore_patterns("__pycache__"))
        (work / rel).write_text(ast.unparse(tree) + "\n")
        proc = subprocess.run([PYTHON, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *tests], cwd=work,
                              capture_output=True, text=True, timeout=120)
        killed = proc.returncode != 0
        tail = (proc.stdout.strip().splitlines() or [""])[-1]
        return mutant, killed, tail
    except subprocess.TimeoutExpired:
        return mutant, True, "timeout"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)  # 트랙 CPU 예산(BRIEF rev3)
    parser.add_argument("--no-write", action="store_true", help="results/ 에 쓰지 않는다(verify.sh 빠른 모드)")
    args = parser.parse_args()
    started = time.time()
    records = []
    for rel, spec in TARGETS.items():
        tree = ast.parse((TRACK_DIR / rel).read_text())
        todo = sites(tree, spec["functions"])
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda s: run_mutant(rel, tree, s, spec["tests"]), todo))
        for mutant, killed, tail in results:
            records.append({**asdict(mutant), "killed": killed, "pytest": tail})
        n_killed = sum(1 for _, k, _ in results if k)
        print(f"{rel}: {n_killed}/{len(results)} killed", flush=True)
    summary = {}
    for rel in TARGETS:
        rows = [r for r in records if r["file"] == rel]
        killed = sum(r["killed"] for r in rows)
        summary[rel] = {"mutants": len(rows), "killed": killed,
                        "kill_rate": round(killed / len(rows), 4) if rows else None}
    survivors = [r for r in records if not r["killed"]]
    if not args.no_write:
        write_result("mutation_audit", {"tag": "G7 mutation 감사", "summary": summary, "survivors": survivors,
                                        "all": records},
                     script=Path(__file__).resolve(), params={"targets": list(TARGETS)}, started=started)
    for s in survivors:
        print("SURVIVOR", s["file"], s["function"], s["line"], s["operator"], "|", s["original"], "=>", s["mutated"])
    total = sum(v["mutants"] for v in summary.values())
    killed = sum(v["killed"] for v in summary.values())
    return 0 if total and killed / total >= 0.9 else 1


if __name__ == "__main__":
    sys.exit(main())
