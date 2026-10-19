"""[수치] 호모클리닉 점열을 따라가는 4방향 불변 직선장 틀 (강불안정, 약불안정, 약안정, 강안정).

T^u 는 과거 꼬리에서 앞으로 밀어(QR), T^s 는 미래 꼬리에서 뒤로 당겨 얻는다. 2차원 부분공간 안에서
강불안정·약안정은 앞으로, 약불안정·강안정은 뒤로 반복하면 그 방향으로 수렴한다. 이 틀에서 DP̃ 는 대각에 가깝다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OrbitFrames:
    vectors: np.ndarray          # (n, 4, 4) 열 = 단위 방향 (uu, wu, ws, ss)
    step_matrices: np.ndarray    # (n−1, 4, 4) F_{k+1}^{-1} DP̃(u_k) F_k
    offdiag: np.ndarray          # (n−1,) 각 단계 비대각 최대 절댓값
    angle_u_s_deg: np.ndarray    # (n,) span(uu,wu) 와 span(ws,ss) 의 최소 주각
    angle_in_u_deg: np.ndarray   # (n,) uu 와 wu 사이 각
    angle_in_s_deg: np.ndarray   # (n,) ws 와 ss 사이 각


def _unit(v):
    return v / np.linalg.norm(v)


def _orth(m):
    return np.linalg.qr(m)[0]


def _min_angle_deg(a, b):
    qa, qb = _orth(a), _orth(b)
    cos = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0.0, 1.0)
    return float(np.degrees(np.arccos(cos.max())))


def _angle_deg(v, w):
    c = abs(_unit(v) @ _unit(w))
    return float(np.degrees(np.arccos(min(c, 1.0))))


def invariant_frames(jacobians: np.ndarray, start_frame: np.ndarray, end_frame: np.ndarray) -> OrbitFrames:
    """jacobians[k]=DP̃(u_k), k=0..n−2. start_frame/end_frame 는 양 끝이 따라가는 cycle 점의 고유틀(열 uu,wu,ws,ss)."""
    n = len(jacobians) + 1
    tu = [_orth(start_frame[:, :2])]
    for k in range(n - 1):
        tu.append(_orth(jacobians[k] @ tu[-1]))
    ts = [None] * n
    ts[-1] = _orth(end_frame[:, 2:])
    for k in range(n - 2, -1, -1):
        ts[k] = _orth(np.linalg.solve(jacobians[k], ts[k + 1]))
    uu = [_unit(start_frame[:, 0])]
    ws = [_unit(start_frame[:, 2])]
    for k in range(n - 1):
        uu.append(_unit(_project_into(tu[k + 1], jacobians[k] @ uu[-1])))
        ws.append(_unit(_project_into(ts[k + 1], jacobians[k] @ ws[-1])))
    wu = [None] * n
    ss = [None] * n
    wu[-1] = _unit(_project_into(tu[-1], end_frame[:, 1]))
    ss[-1] = _unit(_project_into(ts[-1], end_frame[:, 3]))
    for k in range(n - 2, -1, -1):
        wu[k] = _unit(_project_into(tu[k], np.linalg.solve(jacobians[k], wu[k + 1])))
        ss[k] = _unit(_project_into(ts[k], np.linalg.solve(jacobians[k], ss[k + 1])))
    vectors = np.array([np.column_stack([uu[k], wu[k], ws[k], ss[k]]) for k in range(n)])
    steps = np.array([np.linalg.solve(vectors[k + 1], jacobians[k] @ vectors[k]) for k in range(n - 1)])
    offd = np.array([np.max(np.abs(s - np.diag(np.diag(s)))) for s in steps])
    return OrbitFrames(vectors, steps, offd,
                       np.array([_min_angle_deg(tu[k], ts[k]) for k in range(n)]),
                       np.array([_angle_deg(uu[k], wu[k]) for k in range(n)]),
                       np.array([_angle_deg(ws[k], ss[k]) for k in range(n)]))


def _project_into(basis, v):
    """2차원 부분공간(정규직교 기저) 위로의 직교 사영 — 수송 중 쌓이는 반대쪽 부분공간 성분을 지운다."""
    return basis @ (basis.T @ v)
