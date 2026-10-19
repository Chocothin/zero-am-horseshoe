"""CAPD 엔진 실행 파일 호출. 요청 문자열을 만들고 응답을 상자별 레코드로 나눈다. 해석·판정은 api.py·verdict.py 몫이다."""
from __future__ import annotations

import functools
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .exact import Box
from .model import CoordinateSection, Frame, Settings, SystemParams

TRACK_DIR = Path(__file__).resolve().parent.parent


class EngineError(RuntimeError):
    """요청 자체가 잘못됐거나 엔진이 응답 형식을 지키지 않음 — 상자별 실패와 달리 호출 전체가 무효다."""


@dataclass(frozen=True)
class BoxRecord:
    index: int
    ok: bool
    failure: str | None
    fields: dict[str, list[str]]
    seconds: float


@functools.lru_cache(maxsize=None)
def engine_binary(build: str) -> Path:
    """프로세스마다 한 번 증분 빌드를 거친다 — 소스보다 오래된 바이너리로 계산하는 일을 막는다."""
    binary = TRACK_DIR / "build" / f"engine-{build}" / "t3cap_engine"
    proc = subprocess.run(["bash", str(TRACK_DIR / "build_engine.sh"), build], capture_output=True, text=True)
    if proc.returncode != 0 or not binary.exists():
        raise EngineError(f"engine build failed for {build}: {proc.stderr[-1500:]}")
    return binary


def build_request(
    mode: str,
    boxes: Sequence[Box],
    *,
    system: SystemParams,
    section_in: CoordinateSection,
    section_out: CoordinateSection,
    returns: int,
    frame_in: Frame,
    frame_out: Frame,
    settings: Settings,
    rotation: bool,
) -> str:
    dim = len(boxes[0]) if boxes else 4
    lines = [
        "T3CAP",
        f"arith {settings.arithmetic}",
        f"mode {mode}",
        settings.engine_header(),
        "masses " + " ".join(m.engine_tokens() for m in system.masses),
        f"energy {system.energy.engine_tokens()}",
        f"angmom {system.angular_momentum.engine_tokens()}",
        f"section_in {section_in.engine_tokens()}",
        f"section_out {section_out.engine_tokens()}",
        f"returns {returns}",
        f"rotation {int(rotation)}",
        f"frame_in {frame_in.engine_tokens_in()}",
        f"frame_out {frame_out.engine_tokens_out()}",
        f"boxes {len(boxes)} {dim}",
    ]
    lines += [" ".join(iv.engine_tokens() for iv in b) for b in boxes]
    lines.append("end")
    return "\n".join(lines) + "\n"


def parse_response(text: str, expected: int) -> list[BoxRecord]:
    lines = text.splitlines()
    if not lines or not lines[0].startswith("T3CAP 1"):
        raise EngineError(f"engine response malformed: {lines[:2]}")
    records: list[BoxRecord] = []
    i = 1
    while i < len(lines) and lines[i] != "done":
        head = lines[i].split(maxsplit=3)
        if head[0] != "box":
            raise EngineError(f"engine response: expected 'box', got {lines[i]!r}")
        index, status = int(head[1]), head[2]
        failure = head[3] if status == "fail" and len(head) > 3 else (None if status == "ok" else status)
        fields: dict[str, list[str]] = {}
        seconds = 0.0
        i += 1
        while lines[i] != "end":
            key, *values = lines[i].split()
            if key == "seconds":
                seconds = float(values[0])
            else:
                fields[key] = values
            i += 1
        records.append(BoxRecord(index, status == "ok", failure, fields, seconds))
        i += 1
    if len(records) != expected:
        raise EngineError(f"engine returned {len(records)} records for {expected} boxes")
    return records


def run(mode: str, boxes: Sequence[Box], **kwargs) -> list[BoxRecord]:
    settings: Settings = kwargs["settings"]
    if not boxes:
        return []
    chunks = _chunks(list(boxes), max(1, settings.workers))
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        parts = list(pool.map(lambda chunk: _run_chunk(mode, chunk, **kwargs), chunks))
    records: list[BoxRecord] = []
    for part in parts:
        offset = len(records)
        records += [BoxRecord(offset + r.index, r.ok, r.failure, r.fields, r.seconds) for r in part]
    return records


def _run_chunk(mode: str, chunk: Sequence[Box], **kwargs) -> list[BoxRecord]:
    settings: Settings = kwargs["settings"]
    request = build_request(mode, chunk, **kwargs)
    try:
        proc = subprocess.run([str(engine_binary(settings.build))], input=request, capture_output=True, text=True,
                              timeout=settings.timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise EngineError(f"engine timeout after {settings.timeout_seconds}s") from exc
    if proc.returncode != 0:
        raise EngineError(f"engine exit {proc.returncode}: {proc.stdout.strip()[:500]} {proc.stderr.strip()[:500]}")
    return parse_response(proc.stdout, len(chunk))


def _chunks(items: list, n: int) -> list[list]:
    size = -(-len(items) // n)
    return [items[k : k + size] for k in range(0, len(items), size)]
