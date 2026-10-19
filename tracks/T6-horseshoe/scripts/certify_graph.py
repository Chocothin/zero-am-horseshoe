"""[CAP] 덮개 그래프의 모든 덮개 관계를 인증한다(c1 도함수 + MP 점 상 평균값 형식, c0 로 통과 세기·흐름 관).

기본은 단일 중심 방식(상자 B 의 c0·c1 과 중심 MP 점 상 1개로 B 와 출구 면 4개를 함께 감쌈)이고, 판정이 실패한
변만 조각 분할(조각 중심마다 MP 점 상)로 다시 감싼다. 변마다 결과를 coverings/<변>.json 에 정확 끝점으로 저장하고
설계 해시·설정이 같으면 건너뛴다. 동시 엔진 프로세스는 변 단위 스레드 3개 × 호출당 1개다(BRIEF rev3).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from t6hs.results import DESIGN_DIR, RESULTS_DIR, T3_DIR, sha256_file, write_result  # noqa: E402

sys.path.insert(0, str(T3_DIR))
from t3cap.exact import Interval  # noqa: E402
from t3cap.export import to_json  # noqa: E402

from t6hs import covering, enclose  # noqa: E402

MIN_DISTANCE = Fraction(1, 5)
MAX_DEPTH = 2
RECORD_VERSION = 3  # 3: 엔진 실행 넷의 사용 조건 항목·통과 세기 결과·상(runs)과 조각별 runs_index 를 함께 기록
FACES = [(j, s) for j in covering.UNSTABLE for s in (-1, 1)]
COVER_DIR = RESULTS_DIR / "coverings"
FAR = tuple(Interval(Fraction(-10**9), Fraction(10**9)) for _ in range(4))


def piece_image(r: enclose.PieceResult) -> covering.PieceImage:
    return covering.PieceImage(r.piece, r.image if r.image is not None else FAR, r.certified)


def piece_record(r: enclose.PieceResult, runs_index: dict[int, int]) -> dict:
    opt = lambda v: None if v is None else to_json(v)
    return {"piece": to_json(r.piece), "image": opt(r.image), "certified": r.certified, "failures": list(r.failures),
            "tube": None if r.tube is None else {k: to_json(getattr(r.tube, k)) for k in ("r12", "r13", "r23", "t_end")},
            "return_time": opt(r.return_time), "c1_return_time": opt(r.c1_return_time),
            "second_zero_time": opt(r.second_zero_time), "discriminant": opt(r.discriminant),
            "branch_out": opt(r.branch_out), "seconds": r.seconds,
            "runs_index": None if r.runs is None else runs_index[id(r.runs)]}


def run_record(enc) -> dict:
    """엔진 실행 하나의 저장값: certify_return 항목, 통과 세기 결과, 상(c0·점은 외피∩아핀 상, c1 은 외피 상)."""
    t, c = enc.tube, enc.tube.crossings
    return {"return_time": to_json(enc.return_time), "discriminant": to_json(enc.discriminant),
            "branch_out": to_json(enc.branch_out),
            "tube": {**{k: to_json(getattr(t, k)) for k in ("r12", "r13", "r23", "t_end")}, "steps": t.steps},
            "crossings": {"window_runs": c.window_runs, "counted_before": c.counted_before, "runs_before": c.runs_before,
                          "window_run": to_json(c.window_run)},
            "image": to_json(enc.image)}


def runs_table(done: dict) -> tuple[list[dict], dict[int, int]]:
    """조각들이 기대는 실행 넷을 한 번씩만 적는다(단일 중심은 상자와 출구 면 넷이 실행 넷 하나를 함께 쓴다)."""
    distinct = list({id(r.runs): r.runs for results in done.values() for r in results if r.runs is not None}.values())
    table = [{k: run_record(getattr(runs, k)) for k in ("c0", "c1", "second_zero", "point")} for runs in distinct]
    return table, {id(runs): i for i, runs in enumerate(distinct)}


def piece_ok(res: enclose.PieceResult, kind, signs) -> bool:
    if res.image is None or not res.certified:
        return False
    if kind == "whole":
        return covering.outside_unstable(res.image) or covering.inside_stable(res.image)
    return covering.exit_face_out(res.image, kind[0], kind[1], signs[kind[0]])


def engine_settings(order: int, build: str, point_arith: str, point_order: int, point_build: str) -> enclose.EngineSettings:
    """덮개 인증의 엔진 설정(본 실행과 G8 이 같은 꼴). 요청 토큰 사전 점검도 이 함수로 같은 요청을 만든다."""
    return enclose.EngineSettings(
        c0=enclose.Settings(order=order, workers=1, build=build),
        c1=enclose.Settings(order=order, workers=1, build=build),
        point=enclose.Settings(order=point_order, arithmetic=point_arith, workers=1, build=point_build))


def first_round_pieces() -> list[tuple]:
    """조각 분할의 첫 단계: 상자와 출구 면 넷을 퇴화하지 않은 모든 축으로 이분한 (종류, 조각, 깊이 1)."""
    return ([("whole", p, 1) for p in covering.bisect_box(covering.unit_box())]
            + [(k, p, 1) for k in FACES for p in covering.bisect_box(covering.face_box(*k))])


def by_pieces(src, dst, signs, settings):
    """조각 분할: 판정을 못 넘은 조각만 모든 축으로 이분해 다시 감싼다(조각 중심마다 MP 점 상)."""
    done = {k: [] for k in ["whole"] + FACES}
    pending = first_round_pieces()
    while pending:
        batch, pending = pending, []
        results = enclose.enclose_pieces(src, dst, [p for _, p, _ in batch], min_distance=MIN_DISTANCE,
                                         settings=settings)
        for (kind, piece, depth), res in zip(batch, results):
            if piece_ok(res, kind, signs) or depth >= MAX_DEPTH:
                done[kind].append(res)
            else:
                pending += [(kind, q, depth + 1) for q in covering.bisect_box(piece)]
    return done


def certify_edge(src, dst, signs, settings) -> tuple[covering.CoveringVerdict, dict]:
    hs = enclose.enclose_hset(src, dst, FACES, min_distance=MIN_DISTANCE, settings=settings)
    done = {"whole": [hs.whole], **{k: [hs.faces[k]] for k in FACES}}
    verdict = covering.covering_verdict([piece_image(r) for r in done["whole"]],
                                        {k: [piece_image(r) for r in done[k]] for k in FACES}, signs)
    method = "single-center"
    extra = {"center_image": None if hs.center_image is None else to_json(hs.center_image),
             "derivative": None if hs.derivative is None else to_json(hs.derivative)}
    if not verdict.proven:
        done = by_pieces(src, dst, signs, settings)
        verdict = covering.covering_verdict([piece_image(r) for r in done["whole"]],
                                            {k: [piece_image(r) for r in done[k]] for k in FACES}, signs)
        method, extra = "pieces", {}
    runs, runs_index = runs_table(done)
    record = {"method": method, **extra, "runs": runs,
              "pieces_detail": {("whole" if k == "whole" else f"face_w{k[0]}{k[1]:+d}"):
                                [piece_record(r, runs_index) for r in v] for k, v in done.items()}}
    return verdict, record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="쉼표로 구분한 변(예: N0>N1). 비우면 전부")
    parser.add_argument("--build", default="native-strict")
    parser.add_argument("--point-arith", default="mp:128")
    parser.add_argument("--point-order", type=int, default=30)
    parser.add_argument("--order", type=int, default=20)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--tag", default="", help="G8 재검증처럼 다른 설정으로 돌릴 때 결과 하위 폴더 이름")
    parser.add_argument("--point-build", default="native-strict", help="MP 점 상 엔진 빌드(x86filib-strict 은 MPFR 없음)")
    parser.add_argument("--out-dir", default="", help="결과 폴더를 레포 밖으로(verify.sh 빠른 모드)")
    parser.add_argument("--no-write", action="store_true", help="요약·manifest 를 쓰지 않는다")
    args = parser.parse_args()
    started = time.time()
    design_path = DESIGN_DIR / "graph_design.json"
    design_hash = sha256_file(design_path)
    design = json.loads(design_path.read_text())
    settings = engine_settings(args.order, args.build, args.point_arith, args.point_order, args.point_build)
    out_dir = Path(args.out_dir) if args.out_dir else (COVER_DIR / args.tag if args.tag else COVER_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {tuple(s.split(">")) for s in args.only.split(",") if s}
    code = {p.name: sha256_file(p) for p in sorted((Path(__file__).resolve().parent.parent / "t6hs").glob("*.py"))}

    def job(e) -> dict:
        path = out_dir / f"{e['source']}__{e['target']}.json"
        if path.exists():
            old = json.loads(path.read_text())
            if (old.get("design_sha256") == design_hash and old.get("settings") == repr(settings)
                    and old.get("record_version") == RECORD_VERSION):
                return {k: old[k] for k in ("edge", "proven", "degree", "method", "seconds")}
        t0 = time.time()
        src = enclose.hset_from_design(e["source"], design["vertices"][e["source"]])
        dst = enclose.hset_from_design(e["target"], design["vertices"][e["target"]])
        signs = tuple(int(np.sign(e["linear"][j][j])) for j in covering.UNSTABLE)
        verdict, record = certify_edge(src, dst, signs, settings)
        row = {"edge": f"{e['source']}>{e['target']}", "proven": verdict.proven, "degree": verdict.degree,
               "method": record["method"], "seconds": round(time.time() - t0, 1)}
        payload = {**row, "record_version": RECORD_VERSION, "design_sha256": design_hash, "settings": repr(settings),
                   "signs": list(signs),
                   "failed_checks": [c.name for c in verdict.failed], "code_sha256": code, **record}
        path.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
        print(json.dumps(row), flush=True)
        return row

    edges = [e for e in design["edges"] if not wanted or (e["source"], e["target"]) in wanted]
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        summary = list(pool.map(job, edges))
    all_proven = all(r["proven"] for r in summary) and len(summary) == len(design["edges"])
    if not args.no_write:
        write_result(f"certify_graph{('_' + args.tag) if args.tag else ''}",
                     {"tag": "[CAP] 덮개 관계 인증 요약", "design": str(design_path.relative_to(DESIGN_DIR.parent.parent)),
                      "design_sha256": design_hash, "settings": repr(settings),
                      "rows": summary, "all_proven": all_proven},
                     script=Path(__file__).resolve(), params=vars(args), started=started)
    print("all proven:", all_proven, "| edges done:", len(summary), "of", len(design["edges"]))
    return 0 if all(r["proven"] for r in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
