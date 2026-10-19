"""G5 기준 해: 표준 8차원 Jacobi 좌표 뉴턴 방정식(CLAUDE.md)을 mpmath 고정밀 Taylor 급수법으로 적분한다.

CAPD 엔진의 축소 좌표 벡터장과 독립이므로, 축소 방정식 유도 오류도 G5 에서 함께 드러난다. 비엄밀(정밀도 검사용)이다.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import mpmath as mp

ALPHA = mp.mpf(-3) / 2  # r^-3 = (r^2)^(-3/2)


@dataclass(frozen=True)
class MassData:
    """질량은 정확 유리수로 들고, 파생 상수는 쓰는 순간의 mp 정밀도로 만든다.

    모듈 적재 시점(기본 15자리)에 mu2=2/3 을 만들어 두면 질량 치환 대칭이 1e-17 에서 깨진다(REPORT 정정 기록 1).
    """

    masses: tuple[Fraction, Fraction, Fraction]

    @staticmethod
    def of(masses: Sequence) -> "MassData":
        return MassData(tuple(Fraction(m) for m in masses))  # type: ignore[arg-type]

    def _q(self, q: Fraction) -> mp.mpf:
        return mp.mpf(q.numerator) / q.denominator

    @property
    def m1(self) -> mp.mpf:
        return self._q(self.masses[0])

    @property
    def m2(self) -> mp.mpf:
        return self._q(self.masses[1])

    @property
    def m3(self) -> mp.mpf:
        return self._q(self.masses[2])

    @property
    def a(self) -> mp.mpf:
        return self._q(self.masses[1] / (self.masses[0] + self.masses[1]))

    @property
    def b(self) -> mp.mpf:
        return self._q(self.masses[0] / (self.masses[0] + self.masses[1]))

    @property
    def mu1(self) -> mp.mpf:
        m1, m2, _ = self.masses
        return self._q(m1 * m2 / (m1 + m2))

    @property
    def mu2(self) -> mp.mpf:
        m1, m2, m3 = self.masses
        return self._q((m1 + m2) * m3 / (m1 + m2 + m3))


def taylor_coefficients(state: Sequence[mp.mpf], md: MassData, order: int) -> list[list[mp.mpf]]:
    """X=(rho1x,rho1y,rho2x,rho2y,pi1x,pi1y,pi2x,pi2y) 의 Taylor 계수 [성분][차수] (0..order)."""
    c = [[mp.mpf(v)] for v in state]
    a, b, mu1, mu2 = md.a, md.b, md.mu1, md.mu2
    pairs = (("12", 1, 0, 0), ("13", a, 1, 1), ("23", -b, 1, 1))
    g = {"12": md.m1 * md.m2, "13": md.m1 * md.m3, "23": md.m2 * md.m3}
    d = {k: ([], []) for k, *_ in pairs}
    s = {k: [] for k, *_ in pairs}
    w = {k: [] for k, *_ in pairs}
    wd = {k: ([], []) for k, *_ in pairs}
    for k in range(order):
        for key, coef1, coef2, _ in pairs:
            dx, dy = d[key]
            # d12 = rho1, d13 = rho2 + a rho1, d23 = rho2 - b rho1
            dx.append(coef1 * c[0][k] + coef2 * c[2][k])
            dy.append(coef1 * c[1][k] + coef2 * c[3][k])
            s[key].append(mp.fsum(dx[l] * dx[k - l] + dy[l] * dy[k - l] for l in range(k + 1)))
            sk, wk = s[key], w[key]
            if k == 0:
                wk.append(sk[0] ** ALPHA)
            else:
                wk.append(mp.fsum((ALPHA * j - (k - j)) * sk[j] * wk[k - j] for j in range(1, k + 1)) / (k * sk[0]))
            wdx, wdy = wd[key]
            wdx.append(mp.fsum(wk[l] * dx[k - l] for l in range(k + 1)))
            wdy.append(mp.fsum(wk[l] * dy[k - l] for l in range(k + 1)))
        f12x, f12y = (g["12"] * v[k] for v in wd["12"])
        f13x, f13y = (g["13"] * v[k] for v in wd["13"])
        f23x, f23y = (g["23"] * v[k] for v in wd["23"])
        inv = mp.mpf(1) / (k + 1)
        c[0].append(c[4][k] / mu1 * inv)
        c[1].append(c[5][k] / mu1 * inv)
        c[2].append(c[6][k] / mu2 * inv)
        c[3].append(c[7][k] / mu2 * inv)
        c[4].append((-f12x - a * f13x + b * f23x) * inv)
        c[5].append((-f12y - a * f13y + b * f23y) * inv)
        c[6].append((-f13x - f23x) * inv)
        c[7].append((-f13y - f23y) * inv)
    return c


def evaluate(coeffs: list[list[mp.mpf]], tau: mp.mpf) -> list[mp.mpf]:
    out = []
    for comp in coeffs:
        acc = mp.mpf(0)
        for a in reversed(comp):
            acc = acc * tau + a
        out.append(acc)
    return out


def step_size(coeffs: list[list[mp.mpf]], tolerance: mp.mpf) -> mp.mpf:
    """마지막 두 계수로 수렴 반경을 어림해 마지막 항이 허용오차 아래가 되게 한다."""
    n = len(coeffs[0]) - 1
    radii = []
    for k in (n, n - 1):
        mag = max(abs(comp[k]) for comp in coeffs)
        if mag > 0:
            radii.append((tolerance / mag) ** (mp.mpf(1) / k))
    return min(radii) if radii else mp.mpf(1)


def reduced_coordinate(state: Sequence[mp.mpf], k: int) -> mp.mpf:
    """8차원 상태의 축소 좌표 성분 k (0=r,1=x,2=y,3=pr,4=px,5=py) — t3cap.coords 와 같은 정의를 여기서 독립 구현한다."""
    r = mp.sqrt(state[0] ** 2 + state[1] ** 2)
    ex, ey = state[0] / r, state[1] / r
    if k == 0:
        return r
    vx, vy = {1: (state[2], state[3]), 2: (state[2], state[3]), 3: (state[4], state[5]),
              4: (state[6], state[7]), 5: (state[6], state[7])}[k]
    return ex * vx + ey * vy if k in (1, 3, 4) else ex * vy - ey * vx


def distances(state: Sequence[mp.mpf], md: MassData) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    r12 = mp.sqrt(state[0] ** 2 + state[1] ** 2)
    r13 = mp.sqrt((state[2] + md.a * state[0]) ** 2 + (state[3] + md.a * state[1]) ** 2)
    r23 = mp.sqrt((state[2] - md.b * state[0]) ** 2 + (state[3] - md.b * state[1]) ** 2)
    return r12, r13, r23


@dataclass(frozen=True)
class Crossing:
    state: list[mp.mpf]
    time: mp.mpf
    min_distances: tuple[mp.mpf, mp.mpf, mp.mpf]
    max_distances: tuple[mp.mpf, mp.mpf, mp.mpf]


def flow_to_section(
    state: Sequence[mp.mpf],
    md: MassData,
    *,
    coord: int,
    value: Fraction,
    direction: int,
    returns: int,
    order: int = 32,
    t_max: float = 1e4,
    samples_per_step: int = 12,
) -> Crossing:
    """단면 s = 축소좌표_k - value 를 direction 방향으로 returns 번째 지나는 상태. 각 단계를 표본 점으로 훑어 부호 변화를 찾는다."""
    tolerance = mp.mpf(10) ** (-(mp.mp.dps - 2))
    target = mp.mpf(value.numerator) / value.denominator
    x = [mp.mpf(v) for v in state]
    t = mp.mpf(0)
    count = 0
    mins = list(distances(x, md))
    maxs = list(mins)

    def s_of(st):
        return reduced_coordinate(st, coord) - target

    s_prev = s_of(x)
    while t < t_max:
        coeffs = taylor_coefficients(x, md, order)
        h = step_size(coeffs, tolerance)
        taus = [h * j / samples_per_step for j in range(1, samples_per_step + 1)]
        tau_prev = mp.mpf(0)
        for tau in taus:
            st = evaluate(coeffs, tau)
            s_now = s_of(st)
            # 엄격 부등식이라 단면 위 출발점(s=0)은 통과로 세지 않는다
            crossed = (s_prev < 0 < s_now) if direction > 0 else (s_prev > 0 > s_now)
            if crossed:
                count += 1
                if count == returns:
                    root = mp.findroot(lambda z: s_of(evaluate(coeffs, z)), (tau_prev, tau), solver="anderson")
                    end = evaluate(coeffs, root)
                    # 거리 표본은 반환 시각까지만 — 단면 너머 표본은 흐름 관 [0, T] 밖이다
                    for i, dist in enumerate(distances(end, md)):
                        mins[i] = min(mins[i], dist)
                        maxs[i] = max(maxs[i], dist)
                    return Crossing(end, t + root, tuple(mins), tuple(maxs))
            for i, dist in enumerate(distances(st, md)):
                mins[i] = min(mins[i], dist)
                maxs[i] = max(maxs[i], dist)
            s_prev, tau_prev = s_now, tau
        x = evaluate(coeffs, h)
        t += h
    raise RuntimeError("section not reached before t_max")


def jacobi_energy(state: Sequence[mp.mpf], md: MassData) -> mp.mpf:
    kinetic = (state[4] ** 2 + state[5] ** 2) / (2 * md.mu1) + (state[6] ** 2 + state[7] ** 2) / (2 * md.mu2)
    r12, r13, r23 = distances(state, md)
    return kinetic - md.m1 * md.m2 / r12 - md.m1 * md.m3 / r13 - md.m2 * md.m3 / r23


def angular_momentum(state: Sequence[mp.mpf]) -> mp.mpf:
    return state[0] * state[5] - state[1] * state[4] + state[2] * state[7] - state[3] * state[6]


def reduced_to_state(y: Sequence[mp.mpf], J: mp.mpf) -> list[mp.mpf]:
    """theta=0 게이지: rho1=(r,0), rho2=(x,y), pi1=(pr,(J-x py+y px)/r), pi2=(px,py) — J 의 정의식만 쓴다."""
    r, x, yy, pr, px, py = y
    return [r, mp.mpf(0), x, yy, pr, (J - x * py + yy * px) / r, px, py]


def to_mpf(v) -> mp.mpf:
    return mp.mpf(v.numerator) / v.denominator if isinstance(v, Fraction) else mp.mpf(v)


def chart_state(u: Sequence, *, coord: int, value: Fraction, solved: int, branch: int,
                free: Sequence[int], md: MassData, h, J) -> list[mp.mpf]:
    """차트 u -> 8차원 상태. H 는 y_m 의 2차식이므로 표준 좌표 에너지를 세 점에서 재어 계수를 정확히 맞춘다(엔진 유도와 독립)."""
    h, J = to_mpf(h), to_mpf(J)
    y = [mp.mpf(0)] * 6
    y[coord] = to_mpf(value)
    for i, k in enumerate(free):
        y[k] = to_mpf(u[i])

    def energy_at(v):
        yy = list(y)
        yy[solved] = mp.mpf(v)
        return jacobi_energy(reduced_to_state(yy, J), md)

    e_m, e_0, e_p = energy_at(-1), energy_at(0), energy_at(1)
    qa = (e_p + e_m) / 2 - e_0
    qb = (e_p - e_m) / 2
    qc = e_0 - h
    disc = qb ** 2 - 4 * qa * qc
    if disc <= 0:
        raise ValueError("chart point off the energy surface branch (disc <= 0)")
    y[solved] = (-qb + branch * mp.sqrt(disc)) / (2 * qa)
    return reduced_to_state(y, J)


@dataclass(frozen=True)
class SectionData:
    coord: int
    value: Fraction
    direction: int
    solved: int
    branch: int

    @property
    def free(self) -> tuple[int, ...]:
        return tuple(i for i in range(6) if i not in (self.coord, self.solved))


def reduced_map(u: Sequence, *, section_in: SectionData, section_out: SectionData, returns: int, md: MassData,
                h, J, order: int = 32) -> tuple[list[mp.mpf], Crossing]:
    """축소 Poincaré 사상 u -> u' (도착 단면의 자유 좌표 4개)."""
    st = chart_state(u, coord=section_in.coord, value=section_in.value, solved=section_in.solved,
                     branch=section_in.branch, free=section_in.free, md=md, h=h, J=J)
    c = flow_to_section(st, md, coord=section_out.coord, value=section_out.value, direction=section_out.direction,
                        returns=returns, order=order)
    return [reduced_coordinate(c.state, k) for k in section_out.free], c
