"""V7 정확 유리수 구간 산술과 끝점 파서. 원 트랙 코드와 감사자 코드를 import 하지 않는다.

끝점 문자열은 16진 double(0x1.8p+0), p/q, 10진 세 꼴이다. 16진은 float.fromhex 를 쓰지 않고 직접 풀어
53비트를 넘는 가수도 반올림 없이 정확한 유리수로 바꾼다.
"""
from __future__ import annotations

import re
from fractions import Fraction

_HEX = re.compile(r"^([+-]?)0[xX]([0-9a-fA-F]*)(?:\.([0-9a-fA-F]*))?[pP]([+-]?\d+)$")


def parse(tok) -> Fraction:
    if isinstance(tok, (int, Fraction)):
        return Fraction(tok)
    if isinstance(tok, float):
        return Fraction(tok)
    t = str(tok).strip()
    m = _HEX.match(t)
    if m:
        sign, ip, fp, ex = m.groups()
        fp = fp or ""
        mant = int((ip or "0") + fp, 16)
        val = Fraction(mant) * Fraction(2) ** (int(ex) - 4 * len(fp))
        return -val if sign == "-" else val
    return Fraction(t)


def iv(d) -> tuple[Fraction, Fraction]:
    """{"lo","hi"} 사전이나 [lo, hi] 목록을 정확 구간으로."""
    if isinstance(d, dict):
        lo, hi = parse(d["lo"]), parse(d["hi"])
    else:
        lo, hi = parse(d[0]), parse(d[1])
    if lo > hi:
        raise ValueError("lo > hi")
    return lo, hi


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def neg(a):
    return -a[1], -a[0]


def mul(a, b):
    p = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return min(p), max(p)


def sq(a):
    lo, hi = a
    if lo >= 0:
        return lo * lo, hi * hi
    if hi <= 0:
        return hi * hi, lo * lo
    return Fraction(0), max(lo * lo, hi * hi)


def mag(a):
    return max(abs(a[0]), abs(a[1]))


def contains(outer, inner) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def point(q):
    q = Fraction(q)
    return q, q


UNIT = (Fraction(-1), Fraction(1))


def affine_image(center, D, piece):
    """center + D·piece (평균값 형식의 상). center 는 4 구간, D 는 4×4 구간, piece 는 4 구간."""
    out = []
    for i in range(4):
        acc = center[i]
        for j in range(4):
            acc = add(acc, mul(D[i][j], piece[j]))
        out.append(acc)
    return out


def gershgorin(D):
    """S = DᵀJD, J = diag(1,1,−1,−1). λ_min(AᵀJA − μJ) ≥ min(b* − μ, μ − a*) (A ∈ D) 의 a*, b*.

    i∈{0,1}: S_ii − μ − R_i ≥ (S_ii.lo − R_i) − μ, i∈{2,3}: S_ii + μ − R_i ≥ μ − (R_i − S_ii.lo).
    """
    J = (1, 1, -1, -1)
    S = [[None] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(i, 4):
            acc = (Fraction(0), Fraction(0))
            for k in range(4):
                t = sq(D[k][i]) if i == j else mul(D[k][i], D[k][j])
                acc = add(acc, t if J[k] > 0 else neg(t))
            S[i][j] = S[j][i] = acc
    R = [sum((mag(S[i][j]) for j in range(4) if j != i), Fraction(0)) for i in range(4)]
    a_star = max(R[i] - S[i][i][0] for i in (2, 3))
    b_star = min(S[i][i][0] - R[i] for i in (0, 1))
    return a_star, b_star


def cover_margins(img_box, img_faces, signs):
    """ZG2004 Thm 15(q₀=0) 충분조건의 여유. (W): 상자 상에서 불안정 성분 하나가 통째로 |·|>1 이거나 안정 둘이 (−1,1) 안.
    (F): 출구 면 {w_j=σ} 상에서 s_j·σ·w'_j > 1. 반환: (W 여유, W 안정쪽 여유, 면별 여유 사전)."""
    out = max(max(img_box[j][0] - 1, -1 - img_box[j][1]) for j in (0, 1))
    inside = min(min(img_box[k][0] + 1, 1 - img_box[k][1]) for k in (2, 3))
    faces = {}
    for (j, s), img in img_faces.items():
        faces[(j, s)] = img[j][0] - 1 if signs[j] * s > 0 else -1 - img[j][1]
    return max(out, inside), inside, faces


def face_piece(j, s):
    return [point(s) if i == j else UNIT for i in range(4)]


def fstr(q: Fraction, digits=15, mode="down") -> str:
    """방향 반올림 10진 문자열(mode=down 은 내림, up 은 올림)."""
    scale = 10 ** digits
    n = q * scale
    k = n.numerator // n.denominator
    if mode == "up" and k != n:
        k += 1
    sign = "-" if k < 0 else ""
    k = abs(k)
    s = str(k).rjust(digits + 1, "0")
    return f"{sign}{s[:-digits]}.{s[-digits:]}"


def relpath(d, repo) -> str:
    """레포 안이면 레포 상대 경로, 밖(mktemp 등)이면 절대 경로."""
    from pathlib import Path
    p = Path(d).resolve()
    try:
        return str(p.relative_to(Path(repo).resolve()))
    except ValueError:
        return str(p)
