"""G5 건전성 표본 대조 [수치: G5 표본 대조]: 무작위 점을 mpmath(45자리, 중심 60자리)로 적분해 T11 감싸기 안에 드는지 본다.

기준 해는 t11/g5ref.py(Jacobi 좌표, 감사 사본과 코드 공유 없음)다. 판정 항목은 점마다 다음과 같다.
- 흐름 관: τ(w) ∈ V2 창·V5 외피 창, 표본 r_ij 최소·최대 ∈ V2 관·V5 외피 관, (0,τ) 격자 A>0, Ȧ(τ)<0, 판별식·도착 근 ∈ 감싸기
- 상: P̂(w) ∈ P̂(0)+D(B)·w(정확 유리수 구간 산술), P̂(w) ∈ V5 외피 직접 상
- 도함수: DP̂(w) ∈ D(B), DP̂(w) ∈ V5 외피 직접 도함수
- 중심 점(w=0, 60자리): P̂(0) ∈ 중심 상 감싸기, DP̂(0) ∈ 중심 도함수 감싸기
G5-꼭짓점(--vertices): 314변 × 꼭짓점 16개(45자리 적분, 32자리 저장)에 같은 판정을 하고, 이탈이 나오면 그 레인을 바로 멈춘다.
사용: t11cli.py g5 plan [--vertices] | worker [--vertices] --lane K | judge [--vertices] [--from-lanes] [--out-dir DIR]
      | repeat --ids ID,...
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from fractions import Fraction
from pathlib import Path

import mpmath as mp

from . import design as dz
from . import g5ref as G
from .paths import BOXES, RAW, RESULTS, TUBES, edge_stem

PLAN = RESULTS / "g5_plan.json"
POINTS = RESULTS / "g5_points.jsonl"
SUMMARY = RESULTS / "g5_soundness.json"
TUBE_SAMPLES = RESULTS / "g5_tube_samples.json"
LANES = RAW / "g5"
VERTEX_PLAN = RESULTS / "g5_vertex_plan.json"
VERTEX_POINTS = RESULTS / "g5_vertex_points.jsonl"
VERTEX_SUMMARY = RESULTS / "g5_vertex_soundness.json"
VLANES = RAW / "g5v"
DIGITS_VERTEX = 32        # 꼭짓점 여유(도함수 절대 여유 ~1e-17)보다 훨씬 작은 저장 오차로 충분하고 파일을 10MB 아래로 둔다
SEED = 20261017
DENOM = 2 ** 20
DPS_RANDOM, DPS_CENTER = 45, 60
DPS_CONV_RANDOM, DPS_CONV_CENTER = 55, 70
PER_EDGE = {"cycle": 4, "junction_out": 50, "junction_in": 50, "back_tail": 3, "excursion": 3, "front_tail": 3}
N_CONV_RANDOM, N_CONV_CENTER = 30, 20
PAIRS = G.PAIR_NAMES


# ---------------------------------------------------------------- 계획

def make_plan() -> dict:
    rnd = random.Random(SEED)
    points = []
    for e in dz.edges():
        for k in range(PER_EDGE[dz.edge_type(e)]):
            w = [str(Fraction(rnd.randint(-DENOM, DENOM), DENOM)) for _ in range(4)]
            points.append({"id": f"{edge_stem(e)}__r{k}", "edge": e, "label": f"random{k}", "w": w, "dps": DPS_RANDOM})
    for j in range(4):
        for s in (1, -1):
            w = ["0"] * 4
            w[j] = str(s)
            points.append({"id": f"N1__N2__axis{j}{s:+d}", "edge": "N1>N2", "label": f"axis{j}{s:+d}", "w": w,
                           "dps": DPS_RANDOM})
    for e in dz.edges():
        points.append({"id": f"{edge_stem(e)}__c", "edge": e, "label": "center", "w": ["0"] * 4, "dps": DPS_CENTER})
    randoms = [p for p in points if p["label"].startswith("random")]
    centers = [p for p in points if p["label"] == "center"]
    for p in rnd.sample(randoms, N_CONV_RANDOM):
        points.append({**p, "id": p["id"] + "__conv", "label": p["label"] + "_conv", "dps": DPS_CONV_RANDOM})
    for p in rnd.sample(centers, N_CONV_CENTER):
        points.append({**p, "id": p["id"] + "__conv", "label": "center_conv", "dps": DPS_CONV_CENTER})
    return {"seed": SEED, "denominator": DENOM, "per_edge": PER_EDGE, "points": points}


def vertex_points(edges: list[str]) -> list[dict]:
    """변마다 출발 상자 꼭짓점 16개(w∈{±1}⁴). 평균값 감싸기가 가장 빠듯한 곳이라 2계 통과 사상의 결함에 민감하다."""
    out = []
    for e in edges:
        for k in range(16):
            w = [str(1 if (k >> j) & 1 else -1) for j in range(4)]
            out.append({"id": f"{edge_stem(e)}__v{k:02d}", "edge": e, "label": f"vertex{k:02d}", "w": w, "dps": DPS_RANDOM,
                        "digits": DIGITS_VERTEX})
    return out


# ---------------------------------------------------------------- 기준 해

def nstr(x, dps) -> str:
    return mp.nstr(x, dps, min_fixed=1, max_fixed=0)


def reference(point: dict) -> dict:
    mp.mp.dps = point["dps"]
    s, t = point["edge"].split(">")
    cv, Mv = dz.hset(s)
    cw, Mw = dz.hset(t)
    w = [Fraction(x) for x in point["w"]]
    u0 = [G.frac_to_mpf(cv[i] + sum(Mv[i][j] * w[j] for j in range(4))) for i in range(4)]
    cols = [[G.frac_to_mpf(Mv[i][j]) for j in range(4)] for i in range(4)]
    started = time.time()
    fl = G.fly(u0, cols)
    Minv = mp.inverse(mp.matrix([[G.frac_to_mpf(Mw[i][j]) for j in range(4)] for i in range(4)]))
    wp = Minv * mp.matrix([fl.u_out[i] - G.frac_to_mpf(cw[i]) for i in range(4)])
    DP = Minv * mp.matrix(fl.D_chart)
    d = point.get("digits", point["dps"])
    return {**point, "tau": nstr(fl.tau, d), "wprime": [nstr(wp[i], d) for i in range(4)],
            "DP": [[nstr(DP[i, j], d) for j in range(4)] for i in range(4)],
            "adot_tau": nstr(fl.adot_tau, d), "branch_out": nstr(-fl.adot_tau / fl.u_out[0], d),
            "rmin": {k: nstr(v, d) for k, v in fl.rmin.items()}, "rmax": {k: nstr(v, d) for k, v in fl.rmax.items()},
            "min_A_before": nstr(fl.min_A_before, d) if mp.isfinite(fl.min_A_before) else None,
            "disc": nstr(fl.disc, d), "steps": fl.steps, "seconds": round(time.time() - started, 2)}


def claim(lanes: Path, pid: str) -> bool:
    try:
        fd = os.open(lanes / "locks" / f"{pid}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.close(fd)
    return True


def suite(vertices: bool) -> dict:
    """G5 본 표본과 G5-꼭짓점의 계획·레인·합본·요약 경로."""
    if vertices:
        return {"plan": VERTEX_PLAN, "lanes": VLANES, "points": VERTEX_POINTS, "summary": VERTEX_SUMMARY}
    return {"plan": PLAN, "lanes": LANES, "points": POINTS, "summary": SUMMARY}


def records(edge: str, cache: dict) -> tuple[dict, dict]:
    if edge not in cache:
        cache[edge] = (json.loads((TUBES / f"{edge_stem(edge)}.json").read_text())["rows"][0],
                       json.loads((BOXES / f"{edge_stem(edge)}.json").read_text()))
    return cache[edge]


def worker(lane: int, vertices: bool) -> int:
    """기준 해를 레인 기록에 쌓는다. 꼭짓점 묶음은 점마다 바로 판정하고, 이탈이 나오면 경보 파일을 쓰고 exit 3 으로 멈춘다."""
    st = suite(vertices)
    lanes = st["lanes"]
    (lanes / "locks").mkdir(parents=True, exist_ok=True)
    plan = json.loads(st["plan"].read_text())
    done = set(load_rows(True, vertices))
    cache = {}
    failures = 0
    with open(lanes / f"lane{lane}.jsonl", "a") as out:
        for p in plan["points"]:
            if p["id"] in done or not claim(lanes, p["id"]):
                continue
            try:
                row = reference(p)
            except (RuntimeError, ValueError, ZeroDivisionError) as exc:
                row = {**p, "error": f"{type(exc).__name__}: {exc}"}
                failures += 1
            bad = []
            if vertices and "error" not in row:
                tube, box = records(p["edge"], cache)
                bad = [k for k, (m, _) in judge_point(row, tube, box).items() if m < 0]
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if bad:
                (lanes / f"ALERT_{p['id']}.json").write_text(json.dumps({"id": p["id"], "failed": bad}) + "\n")
                print(f"CRITICAL g5 vertex {p['id']} outside T11 enclosure: {bad}", flush=True)
                return 3
    print(f"g5 lane {lane} done {time.strftime('%F %T')} failures={failures}", flush=True)
    return 1 if failures else 0


# ---------------------------------------------------------------- 판정(정확 유리수)

def iv(pair) -> tuple[Fraction, Fraction]:
    return Fraction(pair[0]), Fraction(pair[1])


def margin(x: Fraction, box) -> Fraction:
    """x 가 [lo,hi] 안이면 가까운 끝까지의 거리(≥0), 밖이면 음수."""
    lo, hi = box
    return min(x - lo, hi - x)


def affine_image(center, D, w):
    """center + D·w (w 는 정확 유리수 점, D·center 는 구간). 끝점별 정확 계산."""
    out = []
    for i in range(4):
        lo, hi = center[i]
        for j in range(4):
            a, b = D[i][j]
            p, q = a * w[j], b * w[j]
            lo, hi = lo + min(p, q), hi + max(p, q)
        out.append((lo, hi))
    return out


def judge_point(p: dict, tube: dict, box: dict) -> dict:
    """점 하나의 판정 항목별 여유(정확 유리수, 음수면 이탈)와 상대 여유(여유/폭)."""
    w = [Fraction(x) for x in p["w"]]
    tau = Fraction(p["tau"])
    wp = [Fraction(x) for x in p["wprime"]]
    DP = [[Fraction(x) for x in row] for row in p["DP"]]
    checks = {}

    def put(name, x, box_):
        lo, hi = box_
        checks[name] = (margin(x, box_), hi - lo)

    put("tau_in_v2_window", tau, iv(tube["window"]))
    put("tau_in_v5_hull_window", tau, iv(box["tau_hull_exact"]))
    put("disc_in_v5_hull", Fraction(p["disc"]), iv(box["disc_hull_exact"]))
    checks["disc_above_v2_lower"] = (Fraction(p["disc"]) - Fraction(tube["disc_lower"]), Fraction(1))
    put("branch_out_in_v5_hull", Fraction(p["branch_out"]), iv(box["branch_out_hull_exact"]))
    for k, name in enumerate(PAIRS):
        lo2, hi2 = Fraction(tube["r_lower"][name]), Fraction(tube["r_upper"][name])
        lo5, hi5 = Fraction(box["tube_outer_exact"]["lo"][k]), Fraction(box["tube_outer_exact"]["hi"][k])
        rmin, rmax = Fraction(p["rmin"][name]), Fraction(p["rmax"][name])
        checks[f"r_min_above_v2_lower[{name}]"] = (rmin - lo2, hi2 - lo2)
        checks[f"r_max_below_v2_upper[{name}]"] = (hi2 - rmax, hi2 - lo2)
        checks[f"r_min_above_v5_lower[{name}]"] = (rmin - lo5, hi5 - lo5)
        checks[f"r_max_below_v5_upper[{name}]"] = (hi5 - rmax, hi5 - lo5)
    # 첫 영점 전 격자점이 없으면 「앞에 영점 없음」을 표본으로 확인하지 못한 것이라 이탈로 센다(치명 가드)
    checks["first_crossing_A_positive_before"] = (Fraction(p["min_A_before"]) if p["min_A_before"] is not None
                                                  else Fraction(-1), Fraction(1))
    checks["descending_crossing"] = (-Fraction(p["adot_tau"]), Fraction(1))
    c5 = [iv(x) for x in box["center_image_exact"]]
    D5 = [[iv(box["deriv_mine_exact"][i][j]) for j in range(4)] for i in range(4)]
    img = affine_image(c5, D5, w)
    hull_img = [iv(x) for x in box["hull_image_exact"]]
    hull_D = [[iv(box["hull_deriv_exact"][i][j]) for j in range(4)] for i in range(4)]
    for i in range(4):
        put(f"image_in_mean_value[{i}]", wp[i], img[i])
        put(f"image_in_hull[{i}]", wp[i], hull_img[i])
        for j in range(4):
            put(f"deriv_in_DB[{i}{j}]", DP[i][j], D5[i][j])
            put(f"deriv_in_hull[{i}{j}]", DP[i][j], hull_D[i][j])
    if p["label"].startswith("center"):
        cc = [iv(x) for x in box["center_image_exact"]]
        cd = [[iv(box["center_deriv_exact"][i][j]) for j in range(4)] for i in range(4)]
        for i in range(4):
            put(f"center_image[{i}]", wp[i], cc[i])
            for j in range(4):
                put(f"center_deriv[{i}{j}]", DP[i][j], cd[i][j])
    return checks


def load_rows(from_lanes: bool, vertices: bool = False) -> dict:
    """기준 해 행: 실행 직후에는 레인 기록(raw-local), 그 뒤에는 커밋된 합본에서 읽는다.
    꼭짓점 묶음은 변이 확인(crossing-mutant --vertices)이 먼저 계산해 둔 합본도 레인 기록과 함께 읽는다."""
    st = suite(vertices)
    if from_lanes:
        files = ([st["points"]] if vertices and st["points"].exists() else []) + sorted(st["lanes"].glob("lane*.jsonl"))
    else:
        files = [st["points"]]
    rows = {}
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def judge(from_lanes: bool, out_dir: Path, vertices: bool = False) -> int:
    started = time.time()
    st = suite(vertices)
    plan = json.loads(st["plan"].read_text())
    rows = load_rows(from_lanes, vertices)
    missing = [p["id"] for p in plan["points"] if p["id"] not in rows]
    errors = [r for r in rows.values() if "error" in r]
    ordered = [rows[p["id"]] for p in plan["points"] if p["id"] in rows]
    (out_dir / st["points"].name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered))
    cache = {}
    violations = []
    per_type = {t: {"points": 0, "random_points": 0, "vertex_points": 0, "violations": 0, "min_rel_margin": None}
                for t in dz.EDGE_TYPES}
    per_check = {}
    tube_samples = {}
    for r in ordered:
        if "error" in r:
            continue
        e = r["edge"]
        tube, box = records(e, cache)
        checks = judge_point(r, tube, box)
        t = dz.edge_type(e)
        per_type[t]["points"] += 1
        per_type[t]["random_points"] += r["label"].startswith("random") and not r["label"].endswith("_conv")
        per_type[t]["vertex_points"] += r["label"].startswith("vertex")
        for name, (m, width) in checks.items():
            rel = m / width if width > 0 else m
            key = name.split("[")[0]
            pc = per_check.setdefault(key, {"n": 0, "violations": 0, "min_margin": None, "min_rel": None, "at": None})
            pc["n"] += 1
            if pc["min_rel"] is None or rel < pc["min_rel"]:
                pc["min_rel"], pc["min_margin"], pc["at"] = rel, m, r["id"]
            if m < 0:
                pc["violations"] += 1
                violations.append({"id": r["id"], "check": name, "margin": float(m), "width": float(width)})
                per_type[t]["violations"] += 1
            mt = per_type[t]["min_rel_margin"]
            per_type[t]["min_rel_margin"] = rel if mt is None or rel < mt else mt
        if r["label"].startswith(("random", "axis")) and not r["label"].endswith("_conv"):
            tube_samples.setdefault(e, []).append(
                {"label": r["label"], "w": r["w"], "tau": r["tau"], "rmin": r["rmin"], "rmax": r["rmax"]})
    conv = convergence(rows)
    out = {
        "tag": ("[수치: G5 표본 대조] G5-꼭짓점: 314변 × 꼭짓점 16개, mpmath Jacobi 좌표 Taylor 45자리" if vertices else
                "[수치: G5 표본 대조] mpmath Jacobi 좌표 Taylor(45자리, 중심 60자리)와 T11 감싸기의 포함"),
        "plan": {k: v for k, v in plan.items() if k != "points"} | {"points": len(plan["points"])},
        "points_judged": sum(1 for r in ordered if "error" not in r), "missing": missing,
        "errors": [{"id": r["id"], "error": r["error"]} for r in errors],
        "random_points": sum(v["random_points"] for v in per_type.values()),
        "vertex_points": sum(v["vertex_points"] for v in per_type.values()),
        "per_type": {t: {**v, "min_rel_margin": None if v["min_rel_margin"] is None else float(v["min_rel_margin"])}
                     for t, v in per_type.items()},
        "per_check": {k: {"n": v["n"], "violations": v["violations"], "min_margin": float(v["min_margin"]),
                          "min_rel_margin": float(v["min_rel"]), "at": v["at"]} for k, v in sorted(per_check.items())},
        "violations": violations, "critical": len(violations),
        "convergence": conv,
        "seconds_reference_total": round(sum(r.get("seconds", 0) for r in ordered), 1),
        "seconds_judge": round(time.time() - started, 1),
    }
    if vertices:
        out["pass"] = not violations and not missing and not errors and out["vertex_points"] == 16 * len(dz.edges())
    else:
        out["pass"] = (not violations and not missing and not errors and out["random_points"] >= 1000
                       and all(v["random_points"] >= 50 for v in per_type.values()))
    (out_dir / st["summary"].name).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    brief = {k: (len(v) if k == "missing" else v) for k, v in out.items() if k not in ("per_check", "violations")}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    if not vertices:
        (out_dir / TUBE_SAMPLES.name).write_text(json.dumps({"tag": "G5 기준 해 표본(V2 표본 형식, rigorous_controls 입력)",
                                                             "results": [{"summary": {"edge": e}, "rows": rs}
                                                                         for e, rs in tube_samples.items()]},
                                                            ensure_ascii=False, indent=1) + "\n")
    return 0 if out["pass"] else 1


def repeat(ids: list[str]) -> int:
    """저장된 기준 해 몇 점을 현재 코드로 다시 적분해 정확 문자열로 비교한다(시간 칸 제외)."""
    stored = load_rows(False)
    if VERTEX_POINTS.exists():
        stored.update(load_rows(False, True))
    bad = []
    for pid in ids:
        old = stored[pid]
        new = reference({k: old[k] for k in ("id", "edge", "label", "w", "dps", "digits") if k in old})
        diff = [k for k in new if k != "seconds" and new[k] != old.get(k)]
        if diff:
            bad.append((pid, diff))
        print(pid, "same" if not diff else f"differs in {diff}")
    return 1 if bad else 0


def convergence(rows: dict) -> dict:
    """같은 점을 더 높은 자릿수로 다시 적분한 값과의 차(기준 해 오차 추정). 상·도함수는 정규 좌표."""
    worst = {"tau": Fraction(0), "wprime": Fraction(0), "DP": Fraction(0), "r": Fraction(0)}
    pairs = 0
    for pid, r in rows.items():
        if not pid.endswith("__conv") or "error" in r:
            continue
        base = rows.get(pid[: -len("__conv")])
        if base is None or "error" in base:
            continue
        pairs += 1
        worst["tau"] = max(worst["tau"], abs(Fraction(r["tau"]) - Fraction(base["tau"])))
        worst["wprime"] = max(worst["wprime"], max(abs(Fraction(a) - Fraction(b))
                                                   for a, b in zip(r["wprime"], base["wprime"])))
        worst["DP"] = max(worst["DP"], max(abs(Fraction(a) - Fraction(b)) for ra, rb in zip(r["DP"], base["DP"])
                                           for a, b in zip(ra, rb)))
        worst["r"] = max(worst["r"], max(abs(Fraction(r[k][n]) - Fraction(base[k][n])) for k in ("rmin", "rmax")
                                         for n in PAIRS))
    return {"pairs": pairs, **{k: float(v) for k, v in worst.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--vertices", action="store_true")
    wk = sub.add_parser("worker")
    wk.add_argument("--lane", type=int, required=True)
    wk.add_argument("--vertices", action="store_true")
    jd = sub.add_parser("judge")
    jd.add_argument("--vertices", action="store_true")
    jd.add_argument("--from-lanes", action="store_true", help="raw-local 레인 기록에서 읽어 합본을 만든다")
    jd.add_argument("--out-dir", default=str(RESULTS))
    rp = sub.add_parser("repeat")
    rp.add_argument("--ids", required=True, help="다시 적분해 저장값과 정확 문자열로 비교할 점 id(쉼표 구분)")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        if args.vertices:
            plan = {"tag": "G5-꼭짓점", "digits": DIGITS_VERTEX, "points": vertex_points(dz.edges())}
        else:
            plan = make_plan()
        suite(args.vertices)["plan"].write_text(json.dumps(plan, ensure_ascii=False, indent=1) + "\n")
        print(len(plan["points"]), "points")
        return 0
    if args.cmd == "worker":
        return worker(args.lane, args.vertices)
    if args.cmd == "repeat":
        return repeat(args.ids.split(","))
    return judge(args.from_lanes, Path(args.out_dir), args.vertices)


if __name__ == "__main__":
    sys.exit(main())
