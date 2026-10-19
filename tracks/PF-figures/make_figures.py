"""논문 1 그림 1~3 과 그림 데이터 요약(figdata.json)을 만든다. 레포 안 결과 JSON 만 읽는다."""
import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I 는 스크립트 폴더를 경로에 넣지 않는다
import orbits as ob  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
REFIT = R / "2026-10-15-T6-horseshoe/orbit_refit_mp.json"
DESIGN = R / "2026-10-15-T6-horseshoe/graph_design.json"
COVER = R / "2026-10-17-T6-horseshoe/coverings_summary.json"
CONES = R / "2026-10-17-T7-hyperbolic/cones.json"
THM2 = R / "2026-10-17-T7-hyperbolic/theorem_Aprime.json"
HOM_RANGE = 8  # 그림 1(b)에 그리는 호모클리닉 점 번호 -K..K

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "pdf.fonttype": 42,
    "axes.linewidth": 0.6, "lines.linewidth": 0.8,
})
STY = [dict(color="0.0", ls="-", lw=0.8), dict(color="0.35", ls=(0, (4, 1.5)), lw=0.8),
       dict(color="0.55", ls=(0, (1, 1.2)), lw=1.1)]  # 물체 1,2,3: 흑백에서도 구분
NAMES = ["body 1", "body 2", "body 3"]


def load(p):
    return json.load(open(p))


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None, "Producer": None})
    if os.environ.get("FIG_PREVIEW_PNG"):  # 눈으로 점검할 때만
        fig.savefig(out / f"{name}.png", dpi=130)


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None, "Producer": None})
    if os.environ.get("FIG_PREVIEW_PNG"):  # 눈으로 점검할 때만
        fig.savefig(out / f"{name}.png", dpi=130)


def sig(x, n=9):
    return float(f"{x:.{n}g}")


def draw_bodies(ax, paths, alpha=1.0, zorder=2):
    for b in range(3):
        for seg in paths:
            ax.plot(seg[:, b, 0], seg[:, b, 1], alpha=alpha, zorder=zorder, solid_capstyle="round", **STY[b])


def start_marks(ax, paths):
    for b in range(3):
        ax.plot(*paths[0][0, b], "o", ms=3.2, mfc="white", mec="0", mew=0.7, zorder=5)


def g_kmax():
    return load(DESIGN)["summary"]["k_junction"] - 1


def fig_orbits(out, data):
    o = load(REFIT)
    cyc = np.array([ob.hex_vec(r) for r in o["cycle"]["points_hex"]])
    hom = np.array([ob.hex_vec(r) for r in o["homoclinic"]["points_hex"]])
    ks = o["homoclinic"]["k"]
    cpaths, cgaps, ctimes = ob.chain(list(cyc))
    sel = [i for i, k in enumerate(ks) if -HOM_RANGE <= k < HOM_RANGE]  # 16번의 비행 = 한 주기만큼
    hpaths, hgaps, htimes = ob.chain([hom[i] for i in sel])
    dist = [float(np.min(np.linalg.norm(cyc - p, axis=1))) for p in hom]

    fig = plt.figure(figsize=(6.3, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.62], hspace=0.38, wspace=0.12,
                          left=0.15, right=0.985, bottom=0.085, top=0.93)
    axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    draw_bodies(axa, cpaths)
    start_marks(axa, cpaths)
    for b in range(3):  # 주기 궤도를 옅게 깔아 이탈한 곳이 보이게 한다
        for seg in cpaths:
            axb.plot(seg[:, b, 0], seg[:, b, 1], color="0.82", lw=0.6, zorder=1)
    draw_bodies(axb, hpaths)
    start_marks(axb, hpaths)
    lim = max(np.abs(np.concatenate([p.reshape(-1, 2) for p in list(cpaths) + list(hpaths)])).max() * 1.05, 1)
    for ax, t in ((axa, "(a)"), (axb, "(b)")):
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("$x$")
        ax.set_title(t, loc="left", fontsize=8)
    axa.set_ylabel("$y$")
    axb.set_yticklabels([])
    fig.legend(handles=[Line2D([], [], label=NAMES[b], **STY[b]) for b in range(3)] +
               [Line2D([], [], ls="", marker="o", ms=3.2, mfc="white", mec="0", label="start")],
               loc="upper center", ncol=4, frameon=False, handlelength=2.6, bbox_to_anchor=(0.5, 1.0))
    kk = np.array(ks)
    axc.semilogy(kk, dist, "-", color="0.55", lw=0.6, zorder=1)
    axc.semilogy(kk, dist, "s", ms=1.8, color="0", zorder=2)
    kmax = g_kmax()
    axc.axvspan(-kmax, kmax, color="0.94", zorder=0)
    axc.axvline(-kmax, color="0.5", lw=0.6, ls=(0, (3, 2)))
    axc.axvline(kmax, color="0.5", lw=0.6, ls=(0, (3, 2)))
    axc.text(kmax - 3, 1e-12, r"chain vertices of $G$: $|k|\leq L-1$", ha="right", va="center", fontsize=7)
    axc.axvspan(-HOM_RANGE, HOM_RANGE, color="0.8", zorder=0)
    axc.annotate("flights drawn in (b)", (HOM_RANGE, 1e-9), (HOM_RANGE + 25, 1e-9), fontsize=7, va="center",
                 arrowprops=dict(arrowstyle="->", lw=0.6))
    axc.set_xlabel("index $k$ of the homoclinic section point")
    axc.set_ylabel("distance to the\nperiodic points", fontsize=7.5)
    axc.set_xlim(kk.min(), kk.max())
    axc.set_title("(c)", loc="left", fontsize=8)
    save(fig, out, "fig1-orbits")
    plt.close(fig)
    data["fig1"] = {
        "sources": [str(REFIT.relative_to(ROOT))],
        "method": "stored points (hex) -> float DOP853 flight to next collinear configuration; numerical integration for illustration",
        "cycle_flights": len(cpaths), "cycle_period_sum_flight_times": sig(sum(ctimes)),
        "cycle_max_junction_gap": float(f"{max(cgaps):.1g}") if max(cgaps) > 1e-9 else "<1e-9",
        "cycle_closing_gap_below_1e-9": bool(np.abs(cpaths[0][0] - cpaths[-1][-1]).max() < 1e-9),
        "homoclinic_k_range": [-HOM_RANGE, HOM_RANGE], "homoclinic_flights": len(hpaths),
        "homoclinic_max_junction_gap": "<1e-9" if max(hgaps) < 1e-9 else sig(max(hgaps), 2),
        "homoclinic_point_count": len(ks), "graph_k_range": [-g_kmax(), g_kmax()], "homoclinic_k_min_max": [min(ks), max(ks)],
        "dist_k0": sig(dist[ks.index(0)]), "dist_min_over_k": sig(min(dist)),
        "axis_half_width": sig(lim, 4),
    }


def loop_order(edges):
    nxt = {}
    for e in edges:
        nxt.setdefault(e["source"], []).append(e["target"])
    cyc = ["N8"]
    while True:
        n = [t for t in nxt[cyc[-1]] if t.startswith("N")][0]
        if n == "N8":
            break
        cyc.append(n)
    loop = ["N8", [t for t in nxt["N8"] if t.startswith("M")][0]]
    while loop[-1] != "N8":
        loop.append(nxt[loop[-1]][0])
    return cyc, loop[:-1]


def fig_graph(out, data):
    g = load(DESIGN)
    cov = load(COVER)
    seg = {r["edge"]: r["segment"] for r in cov["rows"]}
    cyc, loop = loop_order(g["edges"])
    n1, n2 = len(cyc), len(loop)
    assert [n1, n2] == g["summary"]["loops"]
    r1, r2 = 1.0, 3.2
    fig, ax = plt.subplots(figsize=(6.3, 3.35))
    c1, c2 = np.array([-r1, 0.0]), np.array([r2, 0.0])

    def pos(i, n, c, r, a0):
        a = a0 + 2 * np.pi * i / n
        return c + r * np.array([np.cos(a), np.sin(a)])

    def arc(i, n, c, r, a0, m=12):
        a = a0 + 2 * np.pi * np.linspace(i, i + 1, m) / n
        return c[0] + r * np.cos(a), c[1] + r * np.sin(a)

    # 짧은 순환
    for i in range(n1):
        ax.plot(*arc(i, n1, c1, r1, 0.0), color="0", lw=1.6, solid_capstyle="butt")
    P1 = np.array([pos(i, n1, c1, r1, 0.0) for i in range(n1)])
    ax.plot(P1[1:, 0], P1[1:, 1], "o", ms=3.6, color="0", zorder=4)
    for i in range(1, n1):  # 순환 정점 p_j 는 원 바깥에 적는다
        d = (P1[i] - c1) / r1
        ax.text(*(c1 + 1.27 * d * r1), f"$p_{{{int(cyc[i][1:])}}}$", ha="center", va="center", fontsize=6.5)
    # 긴 순환: 변마다 구간 이름에 따라 선 모양
    sty = {"departure": dict(color="0", lw=6), "arrival": dict(color="0", lw=6),
           "backward tail": dict(color="0.55", lw=1.2), "forward tail": dict(color="0.55", lw=1.2),
           "excursion |k|<20": dict(color="0", lw=2.4, ls=(0, (0.5, 0.5)))}
    for i in range(n2):
        a, b = loop[i], loop[(i + 1) % n2]
        ax.plot(*arc(i, n2, c2, r2, np.pi), solid_capstyle="butt", **sty[seg[f"{a}>{b}"]])
    P2 = np.array([pos(i, n2, c2, r2, np.pi) for i in range(n2)])
    # 공유 정점과 표지 정점
    ax.plot(0, 0, "o", ms=8, mfc="white", mec="0", mew=1.4, zorder=6)
    ax.plot(0, 0, "+", ms=6, mec="0", mew=1.0, zorder=7)
    im = loop.index("M+0")
    ax.plot(*P2[im], "*", ms=11, mfc="white", mec="0", mew=1.0, zorder=6)
    ax.annotate("shared vertex $p_8$", (0, 0), (-0.2, 1.95), ha="center", fontsize=7.5,
                arrowprops=dict(arrowstyle="-", lw=0.6, shrinkA=1, shrinkB=5))
    ax.annotate(r"marker vertex $\ell_0=v_*$", P2[im], (P2[im][0] - 0.2, P2[im][1] + 1.55), ha="center", va="center",
                fontsize=7.5, arrowprops=dict(arrowstyle="-", lw=0.6, shrinkA=1, shrinkB=6))
    for i, txt, ty in ((n2 - 1, "arriving edge", 2.1), (0, "departing edge", -2.1)):
        mid = (P2[i] + P2[(i + 1) % n2]) / 2
        ax.annotate(txt, mid, (1.9, ty), ha="center", va="center", fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", lw=0.6, shrinkA=1, shrinkB=2))
    k0 = int(loop[1][1:])
    for i, dy in ((1, -0.55), (n2 - 1, 0.55)):
        ax.text(P2[i][0] + 0.45, dy, rf"$\ell_{{{int(loop[i][1:])}}}$", ha="left", va="center", fontsize=7)
    ax.text(c1[0], c1[1], f"cycle of\nlength {n1}", ha="center", va="center", fontsize=8.5)

    def count(name):
        return sum(1 for r in cov["rows"] if r["segment"] == name)

    cnt = {k: count(k) for k in ("departure", "backward tail", "excursion |k|<20", "forward tail", "arrival")}
    ax.text(c2[0] + 0.5, c2[1] + 0.2,
            f"cycle of length {n2}\n\n{cnt['departure']} departing edge\n{cnt['backward tail']} edges approaching the excursion\n"
            f"{cnt['excursion |k|<20']} edges of the excursion\n{cnt['forward tail']} edges leaving the excursion\n{cnt['arrival']} arriving edge",
            ha="center", va="center", fontsize=7.5, linespacing=1.35)
    for (c, r, n, a0, i0, dl) in ((c1, r1, n1, 0.0, 4, 1), (c2, r2, n2, np.pi, 60, 6), (c2, r2, n2, np.pi, 230, 6)):
        p, q = pos(i0, n, c, r, a0), pos(i0 + dl, n, c, r, a0)
        ax.annotate("", q, p, arrowprops=dict(arrowstyle="-|>", lw=0, mutation_scale=16, color="0", shrinkA=0, shrinkB=0))
    ax.legend(handles=[Line2D([], [], color="0", lw=1.6, label="cycle edge"),
                       Line2D([], [], color="0", lw=2.4, ls=(0, (0.5, 0.5)), label="excursion edges"),
                       Line2D([], [], color="0.55", lw=1.2, label="tail edges"),
                       Line2D([], [], color="0", lw=6, label="departing / arriving edge")],
              loc="lower left", fontsize=7, frameon=False, bbox_to_anchor=(0.0, 0.0))
    ax.set_aspect("equal")
    ax.set_xlim(-2.5, 6.6)
    ax.set_ylim(-3.45, 3.45)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, out, "fig2-graph")
    plt.close(fig)
    data["fig2"] = {
        "sources": [str(DESIGN.relative_to(ROOT)), str(COVER.relative_to(ROOT))],
        "method": "schematic of stored graph edges; vertices placed on two circles",
        "vertices": g["summary"]["vertices"], "edges": g["summary"]["edges"], "loops": g["summary"]["loops"],
        "shared_vertex_count": len(set(cyc) & set(loop)), "marker_position_in_long_cycle": im,
        "edges_per_segment": cnt, "cycle_edges": count("cycle"), "junction_depth_L": g["summary"]["k_junction"],
    }


def fig_margins(out, data):
    cov = load(COVER)
    cones = load(CONES)
    g = load(DESIGN)
    mus, muu = (float(eval(v)) for v in (load(THM2)["claimed"]["mu_s"], load(THM2)["claimed"]["mu_u"]))
    cyc, loop = loop_order(g["edges"])
    ordered = [f"{cyc[i]}>{cyc[(i + 1) % len(cyc)]}" for i in range(len(cyc))] + \
              [f"{loop[i]}>{loop[(i + 1) % len(loop)]}" for i in range(len(loop))]
    rows = {r["edge"]: r for r in cov["rows"]}
    crow = {r["edge"]: r for r in cones["rows"]}
    gap = 6
    x = np.array([i if i < len(cyc) else i + gap for i in range(len(ordered))], float)
    exitm = np.array([rows[e]["exit_face_margin"] for e in ordered])
    stab = np.array([rows[e]["stable_margin"] for e in ordered])
    a = np.array([crow[e]["lower_float"] for e in ordered])
    b = np.array([crow[e]["upper_float"] for e in ordered])
    xc = x[len(cyc) - 1] + (gap + 1) / 2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.3, 4.1), sharex=True, gridspec_kw=dict(hspace=0.14))
    ax1.semilogy(x, exitm, "-", color="0", lw=0.7, label="exit-face margin")
    ax1.semilogy(x, stab, color="0.55", lw=1.0, ls=(0, (4, 1.5)), label="stability margin")
    ax1.set_ylabel("margin (normal coordinates)")
    ax1.legend(loc="upper left", bbox_to_anchor=(0.12, 1.0), frameon=True, framealpha=0.9, edgecolor="0.6")
    ax2.plot(x, b, "-", color="0", lw=0.7, label="$b^*_e$")
    ax2.plot(x, a, color="0.55", lw=1.0, ls=(0, (4, 1.5)), label="$a^*_e$")
    for v, t in ((muu, "$\\mu_u$"), (mus, "$\\mu_s$")):
        ax2.axhline(v, color="0", lw=0.6, ls=(0, (1, 1.5)))
        ax2.text(1.003, v, t, transform=ax2.get_yaxis_transform(), va="center", ha="left", fontsize=8)
    ax2.axhline(1.0, color="0.75", lw=0.5)
    ax2.set_ylim(0.65, 1.6)  # 위로 잘린 변 수는 figdata.json 에 기록한다
    ib = int(np.argmax(b))
    ax2.annotate(f"$b^*_e\\leq {math.ceil(b[ib] * 100) / 100:.2f}$ on the departing edge", (x[ib], 1.6), (x[ib] + 25, 1.46), fontsize=7, va="center",
                 bbox=dict(fc="white", ec="none", pad=1.5), arrowprops=dict(arrowstyle="->", lw=0.6))
    ax2.set_ylabel("Gershgorin constants")
    ax2.legend(loc="lower right", ncol=2, frameon=True, framealpha=0.9, edgecolor="0.6")
    for ax in (ax1, ax2):
        ax.axvline(xc, color="0.75", lw=0.6)
        ax.margins(x=0.01)
    # 구간 이름: 긴 순환의 변 번호 위치
    start = len(cyc)
    seg = [rows[e]["segment"] for e in ordered]
    names = {"cycle": "cycle", "departure": "", "backward tail": "backward tail", "excursion |k|<20": "excursion",
             "forward tail": "forward tail", "arrival": ""}
    for nm in ("cycle", "backward tail", "excursion |k|<20", "forward tail"):
        idx = [i for i, s in enumerate(seg) if s == nm]
        ax1.text((x[idx[0]] + x[idx[-1]]) / 2, 1.02, names[nm], transform=ax1.get_xaxis_transform(), ha="center", va="bottom", fontsize=7)
    ex = [i for i, s in enumerate(seg) if s == "excursion |k|<20"]
    ks_ = [int(t[1:]) for i in ex for t in ordered[i].split(">")]
    exc_k = [min(ks_), max(ks_)]
    for ax in (ax1, ax2):
        ax.axvspan(x[ex[0]] - .5, x[ex[-1]] + .5, color="0.9", zorder=0)
    ax2.set_xlabel("edge of $G$ (the cycle of length %d, then the cycle of length %d from the shared vertex)" % (len(cyc), len(loop)))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.94, bottom=0.11)
    save(fig, out, "fig3-margins")
    plt.close(fig)
    data["fig3"] = {
        "sources": [str(COVER.relative_to(ROOT)), str(CONES.relative_to(ROOT)), str(THM2.relative_to(ROOT))],
        "method": "stored per-edge values, no recomputation",
        "edges_plotted": len(ordered), "mu_s": mus, "mu_u": muu,
        "exit_margin_min": sig(exitm.min()), "stable_margin_min": sig(stab.min()),
        "a_star_max": sig(a.max()), "b_star_min": sig(b.min()),
        "b_star_upper_label": math.ceil(b.max() * 100) / 100, "excursion_edges_k_range": exc_k, "excursion_edge_count": len(ex), "excursion_shaded_x_range_in_fig3": [float(x[ex[0]] - .5), float(x[ex[-1]] + .5)], "b_star_cut_above": 1.6, "b_star_cut_count": int((b > 1.6).sum()), "b_star_max": float(b.max()), "b_star_max_edge_is_departing": bool(rows[ordered[int(np.argmax(b))]]["segment"] == "departure"),
        "a_star_below_mu_s": bool((a < mus).all()), "b_star_above_mu_u": bool((b > muu).all()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper/p1/figures"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    fig_orbits(out, data)
    fig_graph(out, data)
    fig_margins(out, data)
    (out / "figdata.json").write_text(json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
