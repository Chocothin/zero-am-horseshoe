"""관성 좌표 <-> 표준 Jacobi 좌표(CLAUDE.md) <-> 축소 좌표 y=(r,x,y,pr,px,py) 변환과 에너지 크기 조절. mpmath 고정밀(비엄밀)."""
from __future__ import annotations

from typing import Sequence

import mpmath as mp

Vec2 = tuple[mp.mpf, mp.mpf]


def _v(a: Sequence) -> Vec2:
    return (mp.mpf(a[0]), mp.mpf(a[1]))


def cross(a: Vec2, b: Vec2) -> mp.mpf:
    return a[0] * b[1] - a[1] * b[0]


def dot(a: Vec2, b: Vec2) -> mp.mpf:
    return a[0] * b[0] + a[1] * b[1]


def mass_constants(masses: Sequence = (1, 1, 1)) -> dict[str, mp.mpf]:
    m1, m2, m3 = (mp.mpf(m) for m in masses)
    m12 = m1 + m2
    return {"m1": m1, "m2": m2, "m3": m3, "a": m2 / m12, "b": m1 / m12, "mu1": m1 * m2 / m12, "mu2": m12 * m3 / (m12 + m3)}


def inertial_to_jacobi(q: Sequence[Sequence], v: Sequence[Sequence], masses: Sequence = (1, 1, 1)) -> tuple[Vec2, ...]:
    """rho1=q2-q1, rho2=q3-(m1q1+m2q2)/(m1+m2), pi_i = mu_i * d/dt rho_i. 질량중심 운동은 버린다."""
    c = mass_constants(masses)
    q1, q2, q3 = (_v(p) for p in q)
    v1, v2, v3 = (_v(p) for p in v)
    m1, m2, m12 = c["m1"], c["m2"], c["m1"] + c["m2"]
    rho1 = (q2[0] - q1[0], q2[1] - q1[1])
    rho2 = tuple(q3[i] - (m1 * q1[i] + m2 * q2[i]) / m12 for i in range(2))
    pi1 = tuple(c["mu1"] * (v2[i] - v1[i]) for i in range(2))
    pi2 = tuple(c["mu2"] * (v3[i] - (m1 * v1[i] + m2 * v2[i]) / m12) for i in range(2))
    return rho1, rho2, pi1, pi2  # type: ignore[return-value]


def jacobi_to_reduced(rho1: Vec2, rho2: Vec2, pi1: Vec2, pi2: Vec2) -> tuple[tuple[mp.mpf, ...], mp.mpf, mp.mpf]:
    """(y, J, theta): e=rho1/|rho1|, e_perp=(-e_y,e_x); x=e.rho2, y=e_perp.rho2, pr=e.pi1, px=e.pi2, py=e_perp.pi2."""
    r = mp.sqrt(dot(rho1, rho1))
    e = (rho1[0] / r, rho1[1] / r)
    ep = (-e[1], e[0])
    y = (r, dot(e, rho2), dot(ep, rho2), dot(e, pi1), dot(e, pi2), dot(ep, pi2))
    J = cross(rho1, pi1) + cross(rho2, pi2)
    return y, J, mp.atan2(e[1], e[0])


def reduced_to_jacobi(y: Sequence, J, theta=0) -> tuple[Vec2, Vec2, Vec2, Vec2]:
    """역변환: rho1=r e, rho2=x e + y e_perp, pi1=pr e + ((J - x py + y px)/r) e_perp, pi2=px e + py e_perp."""
    r, x, yy, pr, px, py = (mp.mpf(v) for v in y)
    J, theta = mp.mpf(J), mp.mpf(theta)
    e = (mp.cos(theta), mp.sin(theta))
    ep = (-e[1], e[0])
    pperp = (J - x * py + yy * px) / r
    rho1 = (r * e[0], r * e[1])
    rho2 = (x * e[0] + yy * ep[0], x * e[1] + yy * ep[1])
    pi1 = (pr * e[0] + pperp * ep[0], pr * e[1] + pperp * ep[1])
    pi2 = (px * e[0] + py * ep[0], px * e[1] + py * ep[1])
    return rho1, rho2, pi1, pi2


def jacobi_energy(rho1: Vec2, rho2: Vec2, pi1: Vec2, pi2: Vec2, masses: Sequence = (1, 1, 1)) -> mp.mpf:
    c = mass_constants(masses)
    d13 = (rho2[0] + c["a"] * rho1[0], rho2[1] + c["a"] * rho1[1])
    d23 = (rho2[0] - c["b"] * rho1[0], rho2[1] - c["b"] * rho1[1])
    kinetic = dot(pi1, pi1) / (2 * c["mu1"]) + dot(pi2, pi2) / (2 * c["mu2"])
    potential = (c["m1"] * c["m2"] / mp.sqrt(dot(rho1, rho1)) + c["m1"] * c["m3"] / mp.sqrt(dot(d13, d13))
                 + c["m2"] * c["m3"] / mp.sqrt(dot(d23, d23)))
    return kinetic - potential


def scale_jacobi(state: tuple[Vec2, Vec2, Vec2, Vec2], lam) -> tuple[Vec2, ...]:
    """q -> lam q, p -> lam^(-1/2) p 이면 H -> H/lam, J -> lam^(1/2) J, T -> lam^(3/2) T."""
    lam = mp.mpf(lam)
    s = 1 / mp.sqrt(lam)
    rho1, rho2, pi1, pi2 = state
    return ((lam * rho1[0], lam * rho1[1]), (lam * rho2[0], lam * rho2[1]),
            (s * pi1[0], s * pi1[1]), (s * pi2[0], s * pi2[1]))


def scale_factor_to_energy(h_from, h_to=mp.mpf(-1) / 2) -> mp.mpf:
    return mp.mpf(h_from) / mp.mpf(h_to)
