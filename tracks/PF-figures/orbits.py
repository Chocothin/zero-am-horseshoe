"""단면 점열에서 짧은 float 적분으로 배위 공간 곡선을 만든다(설명용 수치 적분)."""
import json
import numpy as np
from scipy.integrate import solve_ivp

MU1, MU2, H = 0.5, 2.0 / 3.0, -0.5


def hex_vec(row):
    return np.array([float.fromhex(s) for s in row])


def lift(u):
    r, x, pr, px = u
    qa = 1 / (2 * MU2) + x * x / (2 * MU1 * r * r)
    qc = pr * pr / (2 * MU1) + px * px / (2 * MU2) - 1 / r - 1 / abs(x + r / 2) - 1 / abs(x - r / 2)
    py = np.sqrt(-4 * qa * (qc - H)) / (2 * qa)
    return np.array([r, 0.0, x, 0.0, pr, -x * py / r, px, py])  # rho1, rho2, pi1, pi2


def rhs(_, s):
    r1, r2, p1, p2 = s[0:2], s[2:4], s[4:6], s[6:8]
    a, b = r2 + r1 / 2, r2 - r1 / 2  # rho13, rho23
    na, nb, n1 = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(r1)
    # dH/d rho: 1/r12 -> -rho1/r^3 ; 1/r13 -> (-a/na^3)*(dr13) ...
    g1 = r1 / n1**3 + a / (2 * na**3) - b / (2 * nb**3)
    g2 = a / na**3 + b / nb**3
    return np.concatenate([p1 / MU1, p2 / MU2, -g1, -g2])


def syz(_, s):
    return s[0] * s[3] - s[1] * s[2]


syz.terminal, syz.direction = True, -1


def flight(u):
    s0 = lift(u)
    sol = solve_ivp(rhs, (0, 20), s0, method="DOP853", rtol=1e-13, atol=1e-14,
                    events=syz, dense_output=True)
    te = sol.t_events[0][0]
    t = np.linspace(0, te, 200)
    return sol.sol(t), sol.y_events[0][0], te


def bodies(s):
    r1, r2 = s[0:2], s[2:4]
    return np.array([-r1 / 2 - r2 / 3, r1 / 2 - r2 / 3, 2 * r2 / 3])  # q1,q2,q3


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def chain(points):
    """flight 마다 이전 도착 상태와 반사 S·회전을 이어 붙인 세 물체 궤적과 이음 오차."""
    A = np.eye(2)
    paths, gaps, times = [], [], []
    prev_end = None
    for u in points:
        traj, end, te = flight(u)
        q = np.array([bodies(traj[:, i]) for i in range(traj.shape[1])])  # (n,3,2)
        if prev_end is not None:
            A = prev_A @ rot(prev_th) @ np.diag([1.0, -1.0])
            gaps.append(float(np.abs(A @ q[0].T - prev_q.T).max()))
        paths.append(q @ A.T)
        th = np.arctan2(end[1], end[0])
        prev_A, prev_th, prev_end, prev_q = A, th, end, (q[-1] @ A.T)
        times.append(te)
    return np.array(paths), gaps, times
