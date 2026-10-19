"""[수치] 덮개 그래프와 h-set 반경 설계(선형 모형). CAP 판정은 이 설계를 입력으로 받을 뿐 결과를 믿지 않는다.

그래프: cycle N_0..N_15(주기 사슬)와 호모클리닉 사슬 N_jd ⟹ M_{−K+1} ⟹ … ⟹ M_{K−1} ⟹ N_ja.
h-set 은 c + F·diag(r)·[−1,1]^4 이고 F 의 열은 단위 방향 (강불안정, 약불안정, 약안정, 강안정)이다.
변 v→w 의 선형 모형 D=F_w⁻¹ DP̃(c_v) F_v, 중심 오프셋 δ=F_w⁻¹(P̃(c_v)−c_w) 로 다음을 여유 m 과 함께 요구한다.
  불안정 k: |D_kk| r_k(v) − Σ_{l≠k}|D_kl| r_l(v) − |δ_k| ≥ (1+m) r_k(w)   (출구 면이 밖으로)
  안정 k:   |D_kk| r_k(v) + Σ_{l≠k}|D_kl| r_l(v) + |δ_k| ≤ (1−m) r_k(w)   (상 전체가 안쪽으로)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mpmath as mp
import numpy as np

from .results import DESIGN_DIR

UNSTABLE, STABLE = (0, 1), (2, 3)
PERIOD = 16


@dataclass(frozen=True)
class Vertex:
    name: str
    center: np.ndarray
    frame: np.ndarray          # 단위 열 (uu, wu, ws, ss)
    radii: np.ndarray          # 성분별 반경


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    linear: np.ndarray         # D (단위 틀)
    offset: np.ndarray         # δ (단위 틀)


@dataclass(frozen=True)
class Graph:
    vertices: dict[str, Vertex]
    edges: list[Edge]
    k_junction: int
    departure: int             # N_jd: u_{−K} 가 따라가는 cycle 점
    arrival: int               # N_ja: u_K 가 따라가는 cycle 점
    margin: float
    center_error: float        # 중심 P̃(c_v) 의 float64 오차 허용치(절대값) — δ 에 더해 판정한다
    meta: dict = field(default_factory=dict)


def cycle_name(i: int) -> str:
    return f"N{i % PERIOD}"


def hom_name(k: int) -> str:
    return f"M{k:+d}"


def load_refit() -> dict:
    return json.loads((DESIGN_DIR / "orbit_refit.json").read_text())


def load_refit_mp() -> dict | None:
    path = DESIGN_DIR / "orbit_refit_mp.json"
    return json.loads(path.read_text()) if path.exists() else None


def _offset(image, target_center, target_frame) -> np.ndarray:
    """δ=F_w⁻¹(P̃(c_v)−c_w). mp 상이면 차를 mp 로 뺀 뒤 float 로 옮긴다(두 값이 1e-15 안에서 같아 float 뺄셈은 자릿수를 잃는다)."""
    diff = np.array([float(mp.mpf(a) - mp.mpf(float(b))) for a, b in zip(image, target_center)])
    return np.linalg.solve(target_frame, diff)


def _edge(src_f, src_jac, image, tgt_c, tgt_f, s, t) -> Edge:
    """image 는 P̃(c_v) 의 설계용 값(mp 재고정이 있으면 mp 상). 엄밀한 상은 CAP 이 따로 감싼다."""
    return Edge(s, t, np.linalg.solve(tgt_f, src_jac @ src_f), _offset(image, tgt_c, tgt_f))


def build_edges(refit: dict, k_junction: int, refit_mp: dict | None = None) -> tuple[list[Edge], dict, int, int]:
    cyc, hom = refit["cycle"], refit["homoclinic"]
    fr, jac = np.array(cyc["eigenframes"]), np.array(cyc["jacobians"])
    k_max = hom["k_max"]
    if not 1 <= k_junction <= k_max - 1:
        raise ValueError("junction index must lie inside the refitted orbit")
    hf, hj = np.array(hom["frames"]), np.array(hom["jacobians"])
    src = refit_mp or refit
    pts = np.array([[float.fromhex(h) for h in row] for row in src["cycle"]["points_hex"]])
    hp = np.array([[float.fromhex(h) for h in row] for row in src["homoclinic"]["points_hex"]])
    if refit_mp:
        cyc_img, hom_img = refit_mp["cycle"]["images_mp"], refit_mp["homoclinic"]["images_mp"]
    else:
        # float64 재고정만 있으면 다음 점을 P̃(c_v) 로 쓴다(결손 3e-13 은 중심 오차 허용치로 덮는다)
        cyc_img = [[str(x) for x in pts[(i + 1) % PERIOD]] for i in range(PERIOD)]
        hom_img = [[str(x) for x in hp[i + 1]] for i in range(len(hp) - 1)]
    idx = {k: i for i, k in enumerate(hom["k"])}
    shadow = dict(zip(hom["k"], hom["shadow_cycle_index"]))
    dep, arr = shadow[-k_junction], shadow[k_junction]
    frames = {cycle_name(i): (pts[i], fr[i]) for i in range(PERIOD)}
    for k in range(-k_junction + 1, k_junction):
        frames[hom_name(k)] = (hp[idx[k]], hf[idx[k]])
    edges = []
    for i in range(PERIOD):
        j = (i + 1) % PERIOD
        edges.append(_edge(fr[i], jac[i], cyc_img[i], pts[j], fr[j], cycle_name(i), cycle_name(i + 1)))
    k0 = -k_junction + 1
    edges.append(_edge(fr[dep], jac[dep], cyc_img[dep], hp[idx[k0]], hf[idx[k0]], cycle_name(dep), hom_name(k0)))
    for k in range(k0, k_junction - 1):
        edges.append(_edge(hf[idx[k]], hj[idx[k]], hom_img[idx[k]], hp[idx[k + 1]], hf[idx[k + 1]],
                           hom_name(k), hom_name(k + 1)))
    k1 = k_junction - 1
    edges.append(_edge(hf[idx[k1]], hj[idx[k1]], hom_img[idx[k1]], pts[arr], fr[arr], hom_name(k1), cycle_name(arr)))
    return edges, frames, dep, arr


def cycle_radii(step_mult: np.ndarray, period_mult: np.ndarray, anchors: dict[int, tuple[int, float]]) -> np.ndarray:
    """주기 사슬 반경. 성분 k 의 확대비를 단계마다 |λ_k|^{1/16} 로 고르게 나누고, anchors[k]=(점, 반경)으로 크기를 정한다."""
    e = np.abs(period_mult) ** (1.0 / PERIOD)
    radii = np.ones((PERIOD, 4))
    for i in range(1, PERIOD):
        radii[i] = radii[i - 1] * np.abs(step_mult[i - 1]) / e
    for k, (point, value) in anchors.items():
        radii[:, k] *= value / radii[point, k]
    return radii


def chain_radii(edges_chain: list[Edge], start_stable: np.ndarray, end_unstable: np.ndarray, margin: float,
                floor: float, center_error: float) -> np.ndarray:
    """사슬 정점 반경. 안정 반경은 출발 접합에서 앞으로, 불안정 반경은 도착 접합에서 뒤로 최소값을 전파하고 floor 로 받친다."""
    n = len(edges_chain) - 1  # 사슬 정점 수(양 끝 cycle 정점 제외)
    radii = np.full((n, 4), floor)
    # 교차항(|D_kl|≈1e-8)은 전파에서 빼고 slacks() 로 다시 확인한다
    for v in range(n):
        e, rs = edges_chain[v], (start_stable if v == 0 else radii[v - 1])
        for k in STABLE:
            radii[v, k] = max(floor, 1.001 * (abs(e.linear[k, k]) * rs[k] + abs(e.offset[k]) + center_error) / (1 - margin))
    for v in range(n - 1, -1, -1):
        e, rw = edges_chain[v + 1], (end_unstable if v == n - 1 else radii[v + 1])
        for k in UNSTABLE:
            radii[v, k] = max(floor, 1.001 * ((1 + margin) * rw[k] + abs(e.offset[k]) + center_error) / abs(e.linear[k, k]))
    return radii


@dataclass(frozen=True)
class Slack:
    edge: str
    component: int
    kind: str
    value: float               # >0 이면 여유 m 을 넘어 성립(정규화 단위)


def slacks(graph: Graph) -> list[Slack]:
    out = []
    for e in graph.edges:
        rv, rw = graph.vertices[e.source].radii, graph.vertices[e.target].radii
        for k in range(4):
            cross = sum(abs(e.linear[k, l]) * rv[l] for l in range(4) if l != k)
            off = abs(e.offset[k]) + graph.center_error
            if k in UNSTABLE:
                value = (abs(e.linear[k, k]) * rv[k] - cross - off) / rw[k] - (1 + graph.margin)
                kind = "exit-face-out"
            else:
                value = (1 - graph.margin) - (abs(e.linear[k, k]) * rv[k] + cross + off) / rw[k]
                kind = "image-inside"
            out.append(Slack(f"{e.source}->{e.target}", k, kind, float(value)))
    return out


def design(k_junction: int, margin: float = 0.06, dep_fraction: float = 0.5, floor_fraction: float = 0.1,
           center_error: float = 1e-14, refit: dict | None = None, refit_mp: dict | None = None) -> Graph:
    """접합 오프셋 d 를 cycle 반경의 dep_fraction 에 두고, 사슬 반경 floor 는 d·floor_fraction 으로 둔다."""
    refit = refit or load_refit()
    refit_mp = refit_mp if refit_mp is not None else load_refit_mp()
    edges, frames, dep, arr = build_edges(refit, k_junction, refit_mp)
    src = refit_mp or refit
    hp = {k: np.array([float.fromhex(h) for h in row]) for k, row in zip(refit["homoclinic"]["k"], src["homoclinic"]["points_hex"])}
    fr = np.array(refit["cycle"]["eigenframes"])
    cp = [np.array([float.fromhex(h) for h in row]) for row in src["cycle"]["points_hex"]]
    d_dep = abs(np.linalg.solve(fr[dep], hp[-k_junction] - cp[dep])[1])
    d_arr = abs(np.linalg.solve(fr[arr], hp[k_junction] - cp[arr])[2])
    cyc = refit["cycle"]
    sm, pm = np.array(cyc["step_multipliers"]), np.array(cyc["period_multipliers"])
    rad_dep, rad_arr = d_dep / dep_fraction, d_arr / dep_fraction
    anchors = {1: (dep, rad_dep), 2: (arr, rad_arr), 0: (dep, rad_dep), 3: (arr, rad_arr)}
    rc = cycle_radii(sm, pm, anchors)
    chain_edges = edges[PERIOD:]
    floor = floor_fraction * min(d_dep, d_arr)
    rh = chain_radii(chain_edges, rc[dep], rc[arr], margin, floor, center_error)
    vertices = {cycle_name(i): Vertex(cycle_name(i), frames[cycle_name(i)][0], frames[cycle_name(i)][1], rc[i])
                for i in range(PERIOD)}
    for n, k in enumerate(range(-k_junction + 1, k_junction)):
        name = hom_name(k)
        vertices[name] = Vertex(name, frames[name][0], frames[name][1], rh[n])
    return Graph(vertices, edges, k_junction, dep, arr, margin, center_error,
                 {"d_departure": d_dep, "d_arrival": d_arr, "floor": floor, "dep_fraction": dep_fraction,
                  "centers": "mpmath 다중 사격(orbit_refit_mp.json)" if refit_mp else "float64 다중 사격"})


def loop_lengths(graph: Graph) -> tuple[int, int]:
    """(주기 순환 길이, 호모클리닉 순환 길이) — 두 순환은 N_ja..N_jd 구간을 공유한다."""
    excursion = 2 * graph.k_junction
    back = (graph.departure - graph.arrival) % PERIOD
    return PERIOD, excursion + back


def entropy_lower_bound(l1: int, l2: int) -> float:
    """첫 귀환 고리가 길이 l1, l2 두 개인 그래프의 스펙트럼 반지름 ρ: ρ^{−l1}+ρ^{−l2}=1 의 근(이분법)."""
    lo, hi = 1.0 + 1e-15, 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid ** (-l1) + mid ** (-l2) > 1:
            lo = mid
        else:
            hi = mid
    return float(np.log(0.5 * (lo + hi)))
