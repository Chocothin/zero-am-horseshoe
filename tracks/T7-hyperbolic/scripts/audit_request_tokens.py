"""엔진 요청 토큰 사전 점검(엔진 적분 없음): T6·T7 스크립트가 엔진에 보내는 요청을 그 스크립트의 함수로 만들어 모으고,
끝점 토큰을 두 해석기로 다시 읽어 거부 수를 센다.

(가) 파이썬: p/q 는 |p|, q ≤ 2^53, 16진은 그 요청의 산술로 정확히 나타나야 한다(double 은 t3cap.exact.exact_hex_double,
     mp:<b> 는 홀수 가수 b 비트 이하).
(나) 엔진: engine_tests/request_parse_check 가 엔진과 같은 경로(산술 설정 → parseRequest)로 읽는다. double 요청은
     native-strict 와 x86filib-strict 두 빌드에서, MP 요청은 native-strict 에서 읽는다.
모으는 요청: T6 덮개 인증 본 실행·G8(단일 중심의 c0·c1·둘째 영점 c0·점), 조각 분할 첫 단계, 구조 대조 4종, T7 정리 B
시제품의 매개변수 변형 점 상. 엔진 호출(t3cap.engine._run_chunk)만 가로채 요청 문자열을 남기고 실패 기록을 돌려준다.
대조: 산술마다 첫 요청의 frame_in 첫 구간을 그 산술 정밀도를 1비트 넘는 16진 토큰으로 바꾸면 두 해석기가 모두 거부하고,
정밀도 안의 경계 토큰(double 53비트, mp:96 은 93비트, mp:128 은 113비트)으로 바꾸면 모두 받아야 한다.
거부가 하나라도 있거나 대조가 어긋나면 exit 1.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t7hb.results import T3_DIR, T6_DESIGN, T6_DIR, write_result  # noqa: E402

sys.path.insert(0, str(T3_DIR))
sys.path.insert(0, str(T6_DIR))
sys.path.insert(0, str(T6_DIR / "scripts"))
from t3cap import Interval, engine  # noqa: E402
from t3cap.exact import exact_hex_double  # noqa: E402
from t6hs import enclose  # noqa: E402
import certify_graph  # noqa: E402
import negative_controls  # noqa: E402
import probe_sensitivity  # noqa: E402

LIMIT = 1 << 53
# certify_graph 의 기본값(본 실행)과 G8 명령(--build x86filib-strict --point-arith mp:96 --point-order 24)
MAIN = certify_graph.engine_settings(20, "native-strict", "mp:128", 30, "native-strict")
G8 = certify_graph.engine_settings(20, "x86filib-strict", "mp:96", 24, "native-strict")
# (정밀도를 1비트 넘는 토큰, 정밀도 안의 토큰): 1+2^-53·1+2^-96·1+2^-128 과 1+2^-52·1+2^-92·1+2^-112
PLANTED = {"double": ("0x1.00000000000008p+0", "0x1.0000000000001p+0"),
           "mp:96": ("0x1.000000000000000000000001p+0", "0x1.00000000000000000000001p+0"),
           "mp:128": ("0x1.00000000000000000000000000000001p+0", "0x1.0000000000000000000000000001p+0")}
HEX = re.compile(r"([+-]?)0[xX]([0-9a-fA-F]*)(?:\.([0-9a-fA-F]*))?(?:[pP]([+-]?[0-9]{1,7}))?")


def capture(store: list[str]):
    def run_chunk(mode, chunk, **kwargs):
        store.append(engine.build_request(mode, chunk, **kwargs))
        return [engine.BoxRecord(i, False, "audit: request built, engine not run", {}, 0.0) for i in range(len(chunk))]
    return run_chunk


def hex_odd_bits(text: str) -> int:
    """16진 토큰 값을 홀수 × 2^e 로 쓸 때 홀수의 비트 수(0 이면 0)."""
    m = HEX.fullmatch(text)
    if m is None:
        raise ValueError(f"not a hex float: {text!r}")
    mantissa = int((m.group(2) or "") + (m.group(3) or "") or "0", 16)
    while mantissa and mantissa % 2 == 0:
        mantissa //= 2
    return mantissa.bit_length()


def token_problem(token: str, arith: str) -> str | None:
    """끝점 토큰 하나를 그 요청의 산술로 정확히 읽을 수 없으면 사유, 읽을 수 있으면 None."""
    if "/" in token:
        p, q = (int(v) for v in token.split("/"))
        return None if q != 0 and abs(p) <= LIMIT and abs(q) <= LIMIT else "p/q outside 2^53"
    if arith == "double":
        try:
            exact_hex_double(token)
        except ValueError as exc:
            return str(exc)
        return None
    return None if hex_odd_bits(token) <= int(arith.split(":")[1]) else f"more than {arith} bits"


def requests_by_source(design: dict) -> dict[str, list[str]]:
    vertices = design["vertices"]
    hset = lambda name: enclose.hset_from_design(name, vertices[name])
    kw = dict(min_distance=certify_graph.MIN_DISTANCE)
    out: dict[str, list[str]] = {}

    def collect(source: str, calls) -> None:
        store: list[str] = []
        original, engine._run_chunk = engine._run_chunk, capture(store)
        try:
            for call in calls:
                call()
        finally:
            engine._run_chunk = original
        out[source] = store

    edges = design["edges"]
    pieces = [p for _, p, _ in certify_graph.first_round_pieces()]
    collect("T6 certify_graph main", [lambda e=e: enclose.enclose_hset(hset(e["source"]), hset(e["target"]), certify_graph.FACES,
                                                                       settings=MAIN, **kw) for e in edges])
    collect("T6 certify_graph G8", [lambda e=e: enclose.enclose_hset(hset(e["source"]), hset(e["target"]), certify_graph.FACES,
                                                                     settings=G8, **kw) for e in edges])
    collect("T6 pieces round 1", [lambda e=e: enclose.enclose_pieces(hset(e["source"]), hset(e["target"]), pieces,
                                                                     settings=MAIN, **kw) for e in edges])
    collect("T6 negative controls", [
        lambda spec=spec: enclose.enclose_hset(spec[0], spec[1], negative_controls.FACES, section_out=spec[2],
                                               settings=negative_controls.CONTROL_SETTINGS, **kw)
        for e in negative_controls.control_edges(design, quick=False)
        for spec in negative_controls.variants(e, design).values()])
    eps = Fraction(1, 10**13)
    systems = ([probe_sensitivity.system_for("", None)]
               + [probe_sensitivity.system_for(p, probe_sensitivity.DELTA) for p in probe_sensitivity.PARAMS]
               + [probe_sensitivity.system_for("", Interval(-eps, eps))])
    collect("T7 theorem B probe", [lambda e=e, s=s: probe_sensitivity.point_image(hset(e["source"]), hset(e["target"]), s)
                                   for e in edges for s in systems])
    return out


def engine_check(binary: Path, requests: list[str]) -> list[str]:
    if not requests:
        return []
    proc = subprocess.run([str(binary)], input="".join(requests), capture_output=True, text=True, timeout=1800)
    lines = proc.stdout.splitlines()
    if len(lines) != len(requests):
        raise SystemExit(f"{binary.name}: {len(lines)} verdict lines for {len(requests)} requests")
    return [line for line in lines if not line.startswith("ok ")]


def planted_request(request: str, token: str) -> str:
    """frame_in 첫 구간(중심 첫 성분, 점 구간)의 두 끝점을 token 으로 바꾼 요청."""
    lines = request.split("\n")
    k = next(i for i, line in enumerate(lines) if line.startswith("frame_in "))
    parts = lines[k].split()
    lines[k] = " ".join([parts[0], token, token] + parts[3:])
    return "\n".join(lines)


def controls(first: dict[str, str], native: Path, x86: Path) -> list[dict]:
    rows = []
    for arith, request in sorted(first.items()):
        over, within = PLANTED[arith]
        for kind, token, expect_rejected in (("over", over, True), ("within", within, False)):
            r = planted_request(request, token)
            python = sum(token_problem(t, arith) is not None for t in r.split() if "/" in t or "0x" in t.lower())
            cpp = [len(engine_check(native, [r]))] + ([len(engine_check(x86, [r]))] if arith == "double" else [])
            observed = python > 0 and all(cpp)
            rows.append({"arith": arith, "control": f"frame_in centre[0] = {token} ({kind} precision)",
                         "python_rejected_tokens": python, "engine_rejected_requests": cpp,
                         "expected_rejected": expect_rejected,
                         "agrees": observed == expect_rejected and (python == 0) == (not any(cpp))})
    return rows


def built_checker(config: str) -> Path:
    out = T3_DIR / "build" / f"engine-tests-{config}"
    env = dict(os.environ, ENGINE_SRC_DIR=str(T3_DIR / "engine_tests"), ENGINE_OUT_DIR=str(out))
    subprocess.run(["bash", str(T3_DIR / "build_engine.sh"), config], env=env, check=True, capture_output=True, text=True)
    return out / "request_parse_check"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    started = time.time()
    design = json.loads((T6_DESIGN / "graph_design.json").read_text())
    native, x86 = built_checker("native-strict"), built_checker("x86filib-strict")
    rows, rejected, first = {}, [], {}
    for source, requests in requests_by_source(design).items():
        by_arith: dict[str, list[str]] = {}
        for r in requests:
            by_arith.setdefault(r.split("\n", 2)[1].split()[1], []).append(r)
        for arith, reqs in sorted(by_arith.items()):
            first.setdefault(arith, reqs[0])
            tokens = [t for r in reqs for t in r.split() if "/" in t or "0x" in t.lower()]
            hexes = [t for t in tokens if "/" not in t]
            python = [(t, why) for t in tokens if (why := token_problem(t, arith)) is not None]
            cpp_native = engine_check(native, reqs)
            cpp_x86 = engine_check(x86, reqs) if arith == "double" else None
            rows[f"{source} | {arith}"] = {
                "source": source, "arith": arith, "requests": len(reqs), "endpoint_tokens": len(tokens),
                "hex_tokens": len(hexes), "pq_tokens": len(tokens) - len(hexes),
                "max_hex_odd_bits": max((hex_odd_bits(t) for t in hexes), default=0),
                "python_rejected": len(python), "engine_native_strict_rejected": len(cpp_native),
                "engine_x86filib_strict_rejected": None if cpp_x86 is None else len(cpp_x86),
                "samples": [list(x) for x in python[:3]] + cpp_native[:3] + (cpp_x86 or [])[:3]}
            rejected += python + cpp_native + (cpp_x86 or [])
    by_arith_total: dict[str, dict] = {}
    for r in rows.values():
        t = by_arith_total.setdefault(r["arith"], {"requests": 0, "endpoint_tokens": 0, "hex_tokens": 0, "rejected": 0,
                                                    "max_hex_odd_bits": 0})
        t["requests"] += r["requests"]
        t["endpoint_tokens"] += r["endpoint_tokens"]
        t["hex_tokens"] += r["hex_tokens"]
        t["rejected"] += r["python_rejected"] + r["engine_native_strict_rejected"] + (r["engine_x86filib_strict_rejected"] or 0)
        t["max_hex_odd_bits"] = max(t["max_hex_odd_bits"], r["max_hex_odd_bits"])
    planted = controls(first, native, x86)
    summary = {"requests": sum(r["requests"] for r in rows.values()),
               "endpoint_tokens": sum(r["endpoint_tokens"] for r in rows.values()),
               "rejected": len(rejected), "by_arith": by_arith_total,
               "controls": len(planted), "controls_agree": all(c["agrees"] for c in planted)}
    print(json.dumps({"summary": summary, "rows": list(rows.values()), "controls": planted}, indent=1, ensure_ascii=False))
    if not args.no_write:
        write_result("request_token_audit", {"tag": "엔진 요청 토큰 사전 점검(T6·T7 이 만드는 요청, 엔진 적분 없음)",
                                             "summary": summary, "rows": list(rows.values()), "controls": planted},
                     script=Path(__file__).resolve(), params=vars(args), started=started)
    return 1 if rejected or not summary["controls_agree"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
