"""V3-cone 독립 재계산: (1) 313개 h-set 의 축 정렬 외피와 겹침 쌍, (2) 실제 평행체끼리의 서로소 여부(분리 방향의 정확
유리수 인증서), (3) 설계 그래프 314변의 인접 행렬 특성 다항식(flint 정수), 기약성·주기·순환류, ρ·ln ρ 의 Arb 근 분리.
원 트랙 코드를 import 하지 않는다. 입력은 graph_design.json 과 coverings/ 파일 이름뿐이다.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from collections import deque
from fractions import Fraction
from pathlib import Path

import flint

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "results/2026-10-15-T6-horseshoe/graph_design.json"
COVER_DIR = ROOT / "results/2026-10-15-T6-horseshoe/coverings"
LN_RHO_CLAIM = Fraction("0.0073696936384425343859428")
RHO_CLAIM = Fraction(1217868143156043955884963, 2 ** 80)


def hset(vertex):
    c = [Fraction(float.fromhex(h)) for h in vertex["center_hex"]]
    F = [[Fraction(float.fromhex(h)) for h in row] for row in vertex["frame_hex"]]
    r = [Fraction(float(v)) for v in vertex["radii"]]  # JSON 의 double 반경을 그대로 유리수로(T6 정의와 같은 수)
    M = [[F[i][j] * r[j] for j in range(4)] for i in range(4)]
    return c, M


def hull(c, M):
    return [(c[i] - sum(abs(x) for x in M[i]), c[i] + sum(abs(x) for x in M[i])) for i in range(4)]


def separation(a, b):
    return max(max(y[0] - x[1], x[0] - y[1]) for x, y in zip(a, b))


def support(n, c, M):
    """{n·x : x ∈ c + M[−1,1]⁴} = [n·c − Σ_j |n·M_j|, n·c + Σ_j |n·M_j|] (정확)."""
    nc = sum(ni * ci for ni, ci in zip(n, c))
    s = sum(abs(sum(n[i] * M[i][j] for i in range(4))) for j in range(4))
    return nc - s, nc + s


def inverse_rows(M):
    """정확 유리수 역행렬의 행(정규 좌표 함수 κ 의 기울기) — 분리 방향 후보."""
    n = 4
    a = [list(M[i]) + [Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[piv] = a[piv], a[col]
        inv = 1 / a[col][col]
        a[col] = [v * inv for v in a[col]]
        for r in range(n):
            if r != col and a[r][col] != 0:
                f = a[r][col]
                a[r] = [v - f * w for v, w in zip(a[r], a[col])]
    return [row[n:] for row in a]


def separating_certificate(h1, h2):
    """평행체 둘을 가르는 방향 n 을 후보(각 틀의 역행렬 행, 중심 차)에서 찾는다. 찾으면 (방향 이름, 간격>0)."""
    (c1, M1), (c2, M2) = h1, h2
    cands = []
    for tag, rows in (("inv1", inverse_rows(M1)), ("inv2", inverse_rows(M2))):
        cands += [(f"{tag}[{k}]", rows[k]) for k in range(4)]
    cands.append(("center_diff", [c2[i] - c1[i] for i in range(4)]))
    best = None
    for name, n in cands:
        lo1, hi1 = support(n, c1, M1)
        lo2, hi2 = support(n, c2, M2)
        gap = max(lo2 - hi1, lo1 - hi2)
        scale = math.sqrt(sum(float(x) ** 2 for x in n))
        if gap > 0:
            g = float(gap) / scale
            if best is None or g > best[1]:
                best = (name, g)
    return best


def kind(v):
    return "cycle" if v.startswith("N") else "chain"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/2026-10-16-V3-cone/graph_and_hulls.json"))
    args = ap.parse_args()
    t0 = time.time()
    design = json.loads(DESIGN.read_text())
    V = design["vertices"]
    names = sorted(V)
    H = {v: hset(V[v]) for v in names}
    hulls = {v: hull(*H[v]) for v in names}

    # (1) 외피 겹침 전수(313C2 쌍)
    overlaps = []
    for a, b in itertools.combinations(names, 2):
        s = separation(hulls[a], hulls[b])
        if not s > 0:
            overlaps.append((a, b, float(s)))
    by_kind = {}
    for a, b, _ in overlaps:
        k = "-".join(sorted((kind(a), kind(b))))
        by_kind[k] = by_kind.get(k, 0) + 1
    m0 = {v: separation(hulls["M+0"], hulls[v]) for v in names if v != "M+0"}
    m0_min = min(m0.items(), key=lambda kv: kv[1])
    cyc = [f"N{i}" for i in range(16)]
    cyc_pairs = {(a, b): separation(hulls[a], hulls[b]) for a, b in itertools.combinations(cyc, 2)}
    cyc_min = min(cyc_pairs.items(), key=lambda kv: kv[1])
    # 외피가 h-set 을 덮는지: 꼭짓점 16개(극점)가 외피 안인지 정확히 본다(평행체는 꼭짓점의 볼록 껍질)
    hull_covers = True
    for v in names:
        c, M = H[v]
        for signs in itertools.product((-1, 1), repeat=4):
            p = [c[i] + sum(M[i][j] * signs[j] for j in range(4)) for i in range(4)]
            hull_covers &= all(hulls[v][i][0] <= p[i] <= hulls[v][i][1] for i in range(4))
    # (2) 외피가 겹치는 쌍의 실제 평행체 서로소 인증서
    certs = []
    for a, b, s in overlaps:
        certs.append(dict(pair=[a, b], hull_separation=s, certificate=separating_certificate(H[a], H[b])))

    # (3) 그래프
    edges = [(e["source"], e["target"]) for e in design["edges"]]
    files = sorted(p.name for p in COVER_DIR.glob("*.json"))
    edge_files = sorted(f"{s}__{t}.json" for s, t in edges)
    idx = {v: i for i, v in enumerate(names)}
    n = len(names)
    A = flint.fmpz_mat(n, n)
    for s, t in edges:
        A[idx[s], idx[t]] = 1
    succ = {v: [t for s, t in edges if s == v] for v in names}
    pred = {v: [s for s, t in edges if t == v] for v in names}

    def reach(start, nb):
        seen, dq = {start}, deque([start])
        while dq:
            x = dq.popleft()
            for y in nb[x]:
                if y not in seen:
                    seen.add(y)
                    dq.append(y)
        return seen
    strongly_connected = len(reach("N8", succ)) == n and len(reach("N8", pred)) == n
    # 기대 구조(이 파일에서 직접 씀): cycle N_i→N_{i+1}, N8→M−148, M_k→M_{k+1}(k=−148..147), M+148→N8
    expected = {(f"N{i}", f"N{(i + 1) % 16}") for i in range(16)} | {("N8", "M-148"), ("M+148", "N8")}
    expected |= {(f"M{k:+d}", f"M{k + 1:+d}") for k in range(-148, 148)}
    degree_ok = all(len(succ[v]) == (2 if v == "N8" else 1) and len(pred[v]) == (2 if v == "N8" else 1) for v in names)
    structure = dict(edge_set_equals_expected=set(edges) == expected, degrees_ok=degree_ok,
                     n8_successors=sorted(succ["N8"]), n8_predecessors=sorted(pred["N8"]))
    # 주기: BFS 깊이 차의 gcd(강연결 그래프)
    depth = {"N8": 0}
    dq = deque(["N8"])
    while dq:
        x = dq.popleft()
        for y in succ[x]:
            if y not in depth:
                depth[y] = depth[x] + 1
                dq.append(y)
    period = 0
    for s, t in edges:
        period = math.gcd(period, depth[s] + 1 - depth[t])
    classes = {}
    for v, d in depth.items():
        classes[d % period] = classes.get(d % period, 0) + 1
    cp = A.charpoly()
    x = flint.fmpz_poly([0, 1])
    target = x ** 313 - x ** 297 - x ** 15
    charpoly_matches = cp == target
    # ρ: q(x)=x^298 − x^282 − 1 의 (1,2) 근(유일한 양의 근). Arb 이분법으로 분리
    flint.ctx.prec = 300
    lo, hi = flint.arb(1), flint.arb(2)
    q = lambda z: z ** 298 - z ** 282 - 1
    for _ in range(200):
        mid = (lo + hi) / 2
        mid = flint.arb(mid.mid())
        if q(mid) > 0:
            hi = mid
        elif q(mid) < 0:
            lo = mid
        else:
            break
    rho = lo.union(hi)
    q_lo_negative = bool(q(lo) < 0)
    q_hi_positive = bool(q(hi) > 0)
    lnrho = rho.log()
    ln_claim = flint.arb(flint.fmpq(LN_RHO_CLAIM.numerator, LN_RHO_CLAIM.denominator))
    rho_claim = flint.arb(flint.fmpq(RHO_CLAIM.numerator, RHO_CLAIM.denominator))
    out = dict(
        description="V3-cone 외피·그래프 독립 재계산",
        vertices=n, edges=len(edges), edge_set_equals_record_files=(files == edge_files),
        hull_overlap_pairs=len(overlaps), overlap_pairs_by_kind=by_kind,
        overlap_pairs=[o[:2] for o in overlaps],
        hull_covers_hset_vertices=hull_covers,
        m0_min_separation=dict(vertex=m0_min[0], value=float(m0_min[1]), exact=str(m0_min[1])),
        m0_disjoint_all=all(s > 0 for s in m0.values()),
        cycle_min_separation=dict(pair=list(cyc_min[0]), value=float(cyc_min[1]), exact=str(cyc_min[1])),
        cycle_disjoint_all=all(s > 0 for s in cyc_pairs.values()),
        overlap_parallelepiped_certificates=certs,
        all_overlapping_pairs_certified_disjoint=all(c["certificate"] is not None for c in certs),
        strongly_connected=strongly_connected, graph_structure=structure, period=period,
        class_sizes=sorted(classes.values()),
        charpoly_equals_x313_minus_x297_minus_x15=charpoly_matches,
        rho_enclosure=rho.str(40, radius=True), q_sign_change=q_lo_negative and q_hi_positive,
        ln_rho_enclosure=lnrho.str(40, radius=True),
        ln_rho_lower_ge_claim=bool(lnrho > ln_claim), rho_ge_claim=bool(rho > rho_claim),
        seconds=round(time.time() - t0, 2))
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k not in ("overlap_pairs", "overlap_parallelepiped_certificates")},
                     indent=1, ensure_ascii=False))
    print("overlap pairs:", out["overlap_pairs"])
    ok = (charpoly_matches and strongly_connected and period == 2 and out["m0_disjoint_all"]
          and structure["edge_set_equals_expected"] and structure["degrees_ok"]
          and out["cycle_disjoint_all"] and hull_covers and out["ln_rho_lower_ge_claim"] and out["q_sign_change"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
