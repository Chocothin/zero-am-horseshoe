"""314변 본 실행 작업 큐: 흐름 관(certify_tubes.py)과 상자 C¹ 감싸기(scripts/stage2.py)를 변마다 별도 프로세스로 돌린다.

작업은 잠금 파일(O_EXCL)로 나눠 가지므로 레인을 나중에 더 붙여도 같은 작업을 두 번 돌리지 않는다.
실패한 작업의 잠금은 지우지 않는다(원인을 보기 전에 다시 돌지 않게). 사용:
  python -I -B t11cli.py jobs worker --lane 0     # 레인마다 하나씩 띄운다
  python -I -B t11cli.py jobs status [--strict]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .paths import BLAS_ONE, BOXES, DESIGN, PYTHON, RAW, RESULTS, TRACK, TUBES, edge_stem

LOCKS = RAW / "locks"
LOGS = RAW / "logs"
RUN_LOG = RESULTS / "run_log.jsonl"


@dataclass(frozen=True)
class Job:
    kind: str          # "box" | "tube"
    edge: str

    @property
    def name(self) -> str:
        return f"{self.kind}__{edge_stem(self.edge)}"

    @property
    def output(self) -> Path:
        return (BOXES if self.kind == "box" else TUBES) / f"{edge_stem(self.edge)}.json"

    def command(self) -> tuple[list[str], dict]:
        if self.kind == "box":
            return [str(PYTHON), "-I", "-B", str(TRACK / "scripts" / "stage2.py"), self.edge], {"V5_STAGE2_OUT": str(BOXES)}
        return [str(PYTHON), "-I", "-B", str(TRACK / "certify_tubes.py"), "--edges", self.edge,
                "--out", str(self.output)], {}


def design_edges() -> list[str]:
    return [f"{e['source']}>{e['target']}" for e in json.loads(DESIGN.read_text())["edges"]]


def all_jobs() -> list[Job]:
    # 상자 작업이 평균 몇 배 길어서 먼저 돌린다(끝에 긴 작업이 남지 않게)
    edges = design_edges()
    return [Job("box", e) for e in edges] + [Job("tube", e) for e in edges]


def claim(job: Job) -> bool:
    try:
        fd = os.open(LOCKS / f"{job.name}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{os.getpid()} {time.strftime('%F %T')}\n".encode())
    os.close(fd)
    return True


def run(job: Job, lane: int) -> dict:
    cmd, extra = job.command()
    env = {**os.environ, **BLAS_ONE, **extra}
    started = time.time()
    with open(LOGS / f"{job.name}.out", "w") as out, open(LOGS / f"{job.name}.err", "w") as err:
        code = subprocess.run(["nice", "-n", "10", *cmd], stdout=out, stderr=err, env=env, cwd=TRACK).returncode
    row = {"job": job.name, "kind": job.kind, "edge": job.edge, "lane": lane, "exit": code,
           "output_written": job.output.exists(), "seconds": round(time.time() - started, 1),
           "end": time.strftime("%F %T")}
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if code == 0 and job.output.exists():
        (LOCKS / f"{job.name}.lock").unlink()
    return row


def worker(lane: int, kinds: set[str]) -> int:
    for d in (LOCKS, LOGS, TUBES, BOXES):
        d.mkdir(parents=True, exist_ok=True)
    failures = 0
    for job in all_jobs():
        if job.kind not in kinds or job.output.exists() or not claim(job):
            continue
        row = run(job, lane)
        failures += row["exit"] != 0 or not row["output_written"]
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print(f"lane {lane} done {time.strftime('%F %T')} failures={failures}", flush=True)
    return 1 if failures else 0


def status(strict: bool) -> int:
    """strict 이면 628작업이 모두 출력을 냈고 마지막 실행이 모두 exit 0 일 때만 exit 0(빠진 변·실패 변이 있으면 1)."""
    jobs = all_jobs()
    done = [j for j in jobs if j.output.exists()]
    locked = sorted(p.stem for p in LOCKS.glob("*.lock")) if LOCKS.exists() else []
    last = {}
    for x in (RUN_LOG.read_text().splitlines() if RUN_LOG.exists() else []):
        if x.strip():
            r = json.loads(x)
            last[r["job"]] = r
    failed = [r for r in last.values() if r["exit"] != 0 or not r["output_written"]]
    missing = [j.name for j in jobs if not j.output.exists()]
    print(json.dumps({"jobs": len(jobs), "done": len(done), "missing": missing[:20], "missing_count": len(missing),
                      "locked_or_running": locked[:20], "locked_count": len(locked), "failed": failed,
                      "cpu_seconds_last_runs": round(sum(r["seconds"] for r in last.values()), 1)},
                     ensure_ascii=False, indent=1))
    return 1 if strict and (missing or failed) else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--lane", type=int, required=True)
    w.add_argument("--kinds", default="box,tube")
    st = sub.add_parser("status")
    st.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "worker":
        return worker(args.lane, set(args.kinds.split(",")))
    return status(args.strict)


if __name__ == "__main__":
    sys.exit(main())
