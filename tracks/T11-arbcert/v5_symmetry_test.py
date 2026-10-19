"""[재계산·엄밀, 시험 대상 = 감사자 V5 엔진] 대칭 항등식으로 V5 의 1·2계 변분 감싸기를 시험한다.

평면 3체 흐름 φ_t 의 정확한 항등식(x=(q₁,q₂,q₃,v₁,v₂,v₃), J 는 각 평면 벡터의 90° 회전, f 는 벡터장):
- 회전: φ_t(R_θx)=R_θφ_t(x) → Dφ_t(x)·Jx = Jφ_t(x),  D²φ_t(x)[Jx,Jx] = −φ_t(x) + Dφ_t(x)·x
- 크기 조절: φ_{λ^{3/2}t}(S_λx)=S_λφ_t(x), S_λ=(λq, λ^{−1/2}v) → Dφ_t(x)·s(x) = s(φ_t(x)) − (3/2)t·f(φ_t(x)), s(x)=(q, −v/2)
- 갈릴레이: φ_t(x+ε(0,w)) = φ_t(x)+ε(wt, w) → Dφ_t·(0,w) = (wt, w),  D²φ_t[(0,w),(0,w)] = 0
V5 의 FirstOrder/SecondOrder 계를 flow_to_time 으로 고정 시각까지 적분하고, 감싸기가 항등식 값의 감싸기와 겹치는지(서로소면 결함) 본다.
의도적으로 틀린 항등식(부호 반전) 대조가 서로소로 거절되는지도 본다. 사용: python -I -B v5_symmetry_test.py [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flint import arb, ctx  # noqa: E402

from v5.flow import SecondOrder, flow_to_time  # noqa: E402
from v5.taylor import PAIRS2, field  # noqa: E402

ctx.prec = 128


def rot(v):
    """각 평면 벡터 (a,b) → (−b, a)."""
    out = []
    for i in range(0, len(v), 2):
        out += [-v[i + 1], v[i]]
    return out


def overlaps(a: arb, b: arb) -> bool:
    return a.overlaps(b)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results" / "2026-10-17-T11-arbcert" / "v5_symmetry_test.json"))
    ap.add_argument("--t", default="3/2")
    a = ap.parse_args(argv)
    t0 = time.time()
    # 충돌에서 떨어진 일반 상태(총운동량 0), 정확 유리수
    q = [arb(-1), arb(1) / 10, arb(1), arb(-1) / 5, arb(1) / 20, arb(1) / 10]
    v = [arb(3) / 10, arb(1) / 5, arb(-1) / 10, arb(1) / 4, arb(-1) / 5, arb(-9) / 20]
    x0 = q + v
    w = [arb(1) / 3, arb(-2) / 7]
    tangents = [rot(x0), list(x0), q + [-c / 2 for c in v], [arb(0)] * 6 + w * 3]
    seconds = [[arb(0)] * 12 for _ in PAIRS2]
    z0 = x0 + [c for t in tangents for c in t] + [c for s in seconds for c in s]
    num, den = (int(s) for s in a.t.split("/"))
    T = arb(num) / den
    z, steps = flow_to_time(z0, SecondOrder(), T, order=24, tol=1e-28)
    x = z[:12]
    TT = [z[12 + 12 * k: 24 + 12 * k] for k in range(4)]
    W = {ab: z[60 + 12 * i: 72 + 12 * i] for i, ab in enumerate(PAIRS2)}
    f, _ = field(x, [])
    checks = {}

    def check(name, got, want):
        ov = all(overlaps(g, h) for g, h in zip(got, want))
        rad = max(float(g.rad()) for g in got)
        checks[name] = {"overlap": ov, "max_rad_enclosure": rad}

    check("rotation_first: T0(t) = J x(t)", TT[0], rot(x))
    check("scaling_first: T2(t) = s(x(t)) - 1.5 t f(x(t))", TT[2],
          [x[i] - arb(3) / 2 * T * f[i] for i in range(6)] + [-x[6 + i] / 2 - arb(3) / 2 * T * f[6 + i] for i in range(6)])
    check("boost_first: T3(t) = (w t, w)", TT[3], [w[i % 2] * T for i in range(6)] + [w[i % 2] for i in range(6)])
    check("rotation_second: W00(t) = -x(t) + T1(t)", W[(0, 0)], [-x[i] + TT[1][i] for i in range(12)])
    check("boost_second: W33(t) = 0", W[(3, 3)], [arb(0)] * 12)
    # 음성 대조: 틀린 항등식은 서로소여야 한다
    neg = {}
    neg["rotation_first_sign_flip"] = not all(overlaps(g, h) for g, h in zip(TT[0], [-c for c in rot(x)]))
    neg["rotation_second_wrong(+x)"] = not all(overlaps(g, h) for g, h in zip(W[(0, 0)], [x[i] + TT[1][i] for i in range(12)]))
    neg["scaling_first_wrong_exponent(-1/2 t)"] = not all(overlaps(g, h) for g, h in zip(
        TT[2], [x[i] - T / 2 * f[i] for i in range(6)] + [-x[6 + i] / 2 - T / 2 * f[6 + i] for i in range(6)]))
    out = {"tag": "재계산·엄밀(대칭 항등식, 시험 대상 V5 엔진)", "t_end": a.t, "steps": steps, "prec_bits": 128,
           "order": 24, "checks": checks, "negative_controls_rejected": neg,
           "all_pass": all(c["overlap"] for c in checks.values()) and all(neg.values()),
           "seconds": round(time.time() - t0, 1)}
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
