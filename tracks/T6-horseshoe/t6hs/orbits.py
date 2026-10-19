"""[수치] HH2024-18 의 16점 cycle 과 대칭 호모클리닉 점열을 다중 사격으로 다시 고정한다.

한 번의 긴 적분은 강불안정 승수(주기당 1022)로 오차를 키우므로 쓰지 않는다. 마디마다 P̃ 한 단계만 적분하고
P̃(u_k)=u_{k+1} 과 경계 조건을 함께 Newton 으로 푼다. 호모클리닉의 뒤쪽 절반은 가역자로 u_{−k}=R u_k 이다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import flow

REPO = Path(__file__).resolve().parents[3]
T2_PACKAGE = REPO / "results" / "2026-10-15-T2-num" / "homoclinic" / "HH2024-18.json"
PERIOD = 16
# 앞쪽 꼬리 u_k(k≫0) 는 p_{(k−SHIFT) mod 16} 를 따라간다(T2 패키지: k=39 의 착지 cycle 점 첨자 10)
FORWARD_SHIFT = 29


def load_package() -> dict:
    return json.loads(T2_PACKAGE.read_text())["t6_package"]


@dataclass(frozen=True)
class Cycle:
    points: np.ndarray        # (16, 4)
    jacobians: np.ndarray     # (16, 4, 4) DP̃(p_i)
    flight_times: np.ndarray
    min_distances: np.ndarray
    residual: float           # max_i |P̃(p_i) − p_{i+1}|


def refine_cycle(points: np.ndarray, iterations: int = 4) -> Cycle:
    """16마디 다중 사격 Newton. 미지수는 16점 전부, 방정식은 P̃(p_i)−p_{i+1}=0(64개)이다."""
    pts = np.array(points, dtype=float)
    m = len(pts)
    for _ in range(iterations + 1):
        steps = [flow.next_syzygy(p, with_jacobian=True) for p in pts]
        res = np.concatenate([steps[i].u - pts[(i + 1) % m] for i in range(m)])
        if np.max(np.abs(res)) < 1e-14 or _ == iterations:
            break
        jac = np.zeros((4 * m, 4 * m))
        for i in range(m):
            jac[4 * i:4 * i + 4, 4 * i:4 * i + 4] = steps[i].jacobian
            j = (i + 1) % m
            jac[4 * i:4 * i + 4, 4 * j:4 * j + 4] -= np.eye(4)
        # 시간 이동 방향이 없는 사상이라 정칙이다(주기점이 쌍곡)
        pts = pts - np.linalg.solve(jac, res).reshape(m, 4)
    return Cycle(pts, np.array([s.jacobian for s in steps]), np.array([s.time for s in steps]),
                 np.array([s.min_distance for s in steps]), float(np.max(np.abs(res))))


@dataclass(frozen=True)
class EigenFrame:
    """주기점 p_i 의 단위 고유벡터 열 (강불안정, 약불안정, 약안정, 강안정)과 단계 승수(틀 사이 대각)."""

    vectors: np.ndarray       # (16, 4, 4), 열이 고유벡터
    step_multipliers: np.ndarray  # (16, 4): F_{i+1}^{-1} DP̃(p_i) F_i 의 대각
    period_multipliers: np.ndarray


def eigenframes(cycle: Cycle) -> EigenFrame:
    """점마다 그 점에서 시작한 주기 곱을 직접 고유분해한다(앞으로 수송하면 강안정 방향이 오염된다)."""
    m = len(cycle.points)
    frames, vals0 = [], None
    for i in range(m):
        prod = np.eye(4)
        for l in range(m):
            prod = cycle.jacobians[(i + l) % m] @ prod
        vals, vecs = np.linalg.eig(prod)
        order = np.argsort(-np.abs(vals))
        vals, vecs = np.real(vals[order]), np.real(vecs[:, order])
        vecs = vecs / np.linalg.norm(vecs, axis=0)
        if frames:
            # 앞 점 틀을 한 단계 민 방향과 부호를 맞춘다(단계 승수의 부호가 사슬에서 일관되게)
            pushed = cycle.jacobians[i - 1] @ frames[-1]
            vecs = vecs * np.sign(np.sum(vecs * pushed, axis=0))
        frames.append(vecs)
        vals0 = vals if vals0 is None else vals0
    frames = np.array(frames)
    mult = np.array([np.diag(np.linalg.solve(frames[(i + 1) % m], cycle.jacobians[i] @ frames[i])) for i in range(m)])
    return EigenFrame(frames, mult, vals0)


@dataclass(frozen=True)
class Homoclinic:
    """대칭 호모클리닉 점열 u_k, k=0..K_land(앞쪽). u_{−k}=R u_k 로 뒤쪽을 만든다."""

    forward: np.ndarray       # (K_land+1, 4)
    jacobians: np.ndarray     # (K_land+1, 4, 4) DP̃(u_k)
    flight_times: np.ndarray
    min_distances: np.ndarray
    defects: np.ndarray       # |P̃(u_k) − u_{k+1}|_∞, k=0..K_land−1
    landing_unstable: np.ndarray  # u_K 의 착지 cycle 점 고유좌표 불안정 성분(경계 조건 잔차)
    k_land: int


def cycle_index_forward(k: int) -> int:
    return (k - FORWARD_SHIFT) % PERIOD


def solve_homoclinic(y_guess, tail_guess: list[np.ndarray], cycle: Cycle, frames: EigenFrame,
                     iterations: int = 8) -> Homoclinic:
    """미지수 (a, e, u_1..u_K), 방정식 P̃(u_k)=u_{k+1}(k=0..K−1) 과 착지점 불안정 성분 2개=0."""
    k_land = len(tail_guess)
    j_land = cycle_index_forward(k_land)
    finv = np.linalg.inv(frames.vectors[j_land])
    nodes = [np.array([y_guess[0], 0.0, 0.0, y_guess[1]])] + [np.array(v, dtype=float) for v in tail_guess]
    n_unknown = 2 + 4 * k_land
    for it in range(iterations + 1):
        steps = [flow.next_syzygy(nodes[k], with_jacobian=True) for k in range(k_land + 1)]
        res = [steps[k].u - nodes[k + 1] for k in range(k_land)]
        landing = (finv @ (nodes[k_land] - cycle.points[j_land]))[:2]
        r = np.concatenate(res + [landing])
        if np.max(np.abs(r)) < 1e-14 or it == iterations:
            break
        jac = np.zeros((n_unknown, n_unknown))
        jac[0:4, 0:2] = steps[0].jacobian[:, [0, 3]]
        for k in range(k_land):
            row = 4 * k
            if k >= 1:
                jac[row:row + 4, 2 + 4 * (k - 1):2 + 4 * k] = steps[k].jacobian
            jac[row:row + 4, 2 + 4 * k:2 + 4 * (k + 1)] = -np.eye(4)
        jac[4 * k_land:, 2 + 4 * (k_land - 1):] = finv[:2]
        delta = np.linalg.solve(jac, r)
        nodes[0] = nodes[0] - np.array([delta[0], 0.0, 0.0, delta[1]])
        for k in range(1, k_land + 1):
            nodes[k] = nodes[k] - delta[2 + 4 * (k - 1):2 + 4 * k]
    defects = np.array([np.max(np.abs(steps[k].u - nodes[k + 1])) for k in range(k_land)])
    return Homoclinic(np.array(nodes), np.array([s.jacobian for s in steps]), np.array([s.time for s in steps]),
                      np.array([s.min_distance for s in steps]), defects, landing, k_land)


def tail_initial_guess(stored_forward: np.ndarray, k_land: int, cycle: Cycle, frames: EigenFrame) -> list[np.ndarray]:
    """k≤39 는 T2 저장 점열, 그 뒤는 착지 안정 성분만 cycle 단계 승수로 밀어 만든 근사(불안정 성분 0)."""
    guess = [np.array(v) for v in stored_forward[1:]]
    k = len(stored_forward) - 1
    j = cycle_index_forward(k)
    coords = np.linalg.solve(frames.vectors[j], stored_forward[-1] - cycle.points[j])
    coords[:2] = 0.0
    while k < k_land:
        coords = coords * frames.step_multipliers[j]
        k, j = k + 1, (j + 1) % PERIOD
        guess.append(cycle.points[j] + frames.vectors[j] @ coords)
    return guess[:k_land]
